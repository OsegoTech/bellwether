"""Bellwether's HTTP API (BUILD_SPEC §3.9). Served by ``bellwether serve``.

JSON only. The interactive reference is at ``/docs`` (Swagger UI) and
``/redoc``; the schema is at ``/openapi.json`` (``bellwether.api_models``).

Read endpoints return what the audit store holds: proposals with their state
and audit history, findings (including INFO ones that were only noted), and
pipeline runs.

**Two write paths, each independently gated, both audited.**

1. ``POST /slack/actions`` — Slack interactive callbacks. Gated by the Slack
   signature (Slack's v0 scheme: HMAC-SHA256 over ``v0:<ts>:<raw body>`` with
   the signing secret, a constant-time compare, a five-minute replay window;
   anything that fails is a 401 and changes nothing) and then by
   ``approval.approver_ids``.
2. ``POST /api/proposals/{id}/decision`` — the JSON decision endpoint. Gated by
   the bind address (403 unless ``approval.ui_approval_enabled`` is true AND
   the server is bound to a loopback host) and then by the same
   ``approval.approver_ids``.

Why loopback: the decision endpoint is for callers inside the trust boundary —
an operator on the host, or a frontend proxied through it — never the public
internet. Slack proves who clicked with a signature; this path proves presence
on the host instead. The ``approver_id`` in the body is an attestation checked
against the allowlist, and host access is the authentication — which is why a
publicly bound server refuses it.

After their gates both paths share one implementation: an approver outside the
allowlist is refused with 403, changes nothing, and is recorded as an
``unauthorized_decision`` audit event; an allowlisted decision re-reads the
proposal's state from the store and moves only a PENDING proposal; an approved
EXECUTABLE proposal is handed to the executor after the response, through the
same code path either way. The executor records EXECUTED or FAILED itself.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import ipaddress
import json
import logging
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from bellwether.api_models import (
    ApprovalRecordOut,
    AuditEventOut,
    AuditTrailOut,
    DecisionRequest,
    DecisionResponse,
    ErrorOut,
    FindingOut,
    ProposalDetailOut,
    ProposalListItem,
    ProposalOut,
    RunOut,
    TransitionOut,
)
from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.notify.slack import APPROVE_ACTION_ID, REJECT_ACTION_ID
from bellwether.store.sqlite import InvalidTransition, SqliteStore

logger = logging.getLogger(__name__)

MAX_SKEW_SECONDS = 300

API_DESCRIPTION = """\
Bellwether watches a self-hosted MongoDB deployment, turns what it measures into
findings, has an AI propose one remediation per finding, and lets a human approve
or reject it. Only two bounded, reversible actions can ever be executed, and only
after approval.

**Reading.** `GET /api/proposals`, `/api/findings` and `/api/runs` return what the
audit store holds.

**Deciding.** There are two write paths, each independently gated and audited:

* `POST /slack/actions` — Slack interactive callbacks, verified by Slack's request
  signature, then checked against the approver allowlist.
* `POST /api/proposals/{proposal_id}/decision` — for callers inside the trust
  boundary. It only works when the server is bound to a **loopback** address
  (reach it over an SSH tunnel or a local proxy) and `approval.ui_approval_enabled`
  is true, and the approver must be on the allowlist.
"""

DECISION_DESCRIPTION = """\
Approve or reject a pending proposal.

**Gates, in order.** Refused with `403 approval_not_permitted` unless the server is
bound to a **loopback** host (127.0.0.1, ::1, localhost) and
`approval.ui_approval_enabled` is true — this endpoint is for callers inside the
trust boundary (an operator on the host, or a frontend proxied through it); the
public approval path is Slack, which is signature-verified. Then the `approver_id`
must be on the **allowlist** `approval.approver_ids`; otherwise `403
not_authorized`, nothing changes, and an `unauthorized_decision` audit event is
recorded.

**Effect.** The proposal's state is re-read from the store: only a `pending`
proposal moves (`409 already_decided` otherwise). Approving an `executable`
proposal hands it to the executor after the response — the same code path as the
Slack approval; read the proposal back to see `executed` or `failed`. Approving
requires `"acknowledged": true`.
"""

OPENAPI_TAGS = [
    {"name": "proposals", "description": "AI proposals, their approval state and audit trail."},
    {"name": "decisions", "description": "Approve or reject a proposal (loopback only)."},
    {"name": "findings", "description": "What the detectors found, including noted INFO findings."},
    {"name": "runs", "description": "Pipeline runs."},
    {"name": "slack", "description": "Slack interactive callbacks (signature-verified)."},
    {"name": "health", "description": "Liveness."},
]

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorOut, "description": "No such proposal."},
}


class SupportsExecute(Protocol):
    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord: ...


@dataclass(frozen=True)
class Decision:
    proposal_id: str
    approve: bool
    user_id: str  # the approver id checked against approval.approver_ids
    actor: str  # how the decision is attributed in the audit trail
    channel: str = "slack"  # "slack" | "api"


@dataclass(frozen=True)
class Outcome:
    """What a decision did, independent of how the channel reports it."""

    ok: bool
    status: int
    proposal_id: str
    message: str
    state: str | None = None


def verify_slack_signature(
    secret: str, body: bytes, timestamp: str | None, signature: str | None, now: float
) -> bool:
    if not timestamp or not signature or not signature.startswith("v0="):
        return False
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    if abs(now - sent_at) > MAX_SKEW_SECONDS:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def is_loopback_host(host: str | None) -> bool:
    """Whether the server's bind address is loopback-only (127.0.0.0/8, ::1, localhost)."""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def create_app(
    store: SqliteStore,
    *,
    signing_secret: str,
    approver_ids: Collection[str],
    executor: SupportsExecute | None = None,
    clock: Callable[[], float] = time.time,
    bind_host: str | None = None,
    ui_approval_enabled: bool = True,
) -> FastAPI:
    if not signing_secret:
        raise ValueError("a Slack signing secret is required")
    if not approver_ids:
        raise ValueError("an approver allowlist (approval.approver_ids) is required; it is empty")
    approvers = frozenset(approver_ids)
    # The decision endpoint needs both: the operator's switch, and a loopback-only
    # bind. An unknown bind address (None) fails closed.
    decisions_permitted = ui_approval_enabled and is_loopback_host(bind_host)

    version = _package_version()
    app = FastAPI(
        title="Bellwether",
        version=version,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
    )
    app.state.bind_host = bind_host
    app.state.ui_approval_permitted = decisions_permitted

    def openapi() -> dict[str, Any]:
        # The decision body is parsed by hand (to answer malformed input with a
        # JSON 400 rather than FastAPI's 422), so register its schema explicitly.
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
                tags=OPENAPI_TAGS,
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            components["DecisionRequest"] = DecisionRequest.model_json_schema()
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]

    # --- read endpoints -------------------------------------------------------------------

    @app.get(
        "/api/proposals",
        response_model=list[ProposalListItem],
        tags=["proposals"],
        summary="List proposals with their current state",
    )
    def list_proposals(
        state: ApprovalState | None = Query(None, description="Only proposals in this state."),
        limit: int = Query(50, ge=1, le=1000, description="At most this many, newest first."),
    ) -> list[ProposalListItem]:
        items: list[ProposalListItem] = []
        for proposal, record in reversed(store.list_proposals()):
            if state is not None and record.state is not state:
                continue
            items.append(
                ProposalListItem(
                    proposal=ProposalOut.of(proposal),
                    state=record.state.value,
                    created_at=proposal.created_at,
                )
            )
            if len(items) >= limit:
                break
        return items

    @app.get(
        "/api/proposals/{proposal_id}",
        response_model=ProposalDetailOut,
        responses=ERROR_RESPONSES,
        tags=["proposals"],
        summary="One proposal in full, with its state and audit history",
    )
    def get_proposal(proposal_id: str) -> Any:
        proposal = store.get_proposal(proposal_id)
        if proposal is None:
            return _error(404, "not_found", f"no proposal {proposal_id}")
        return ProposalDetailOut(
            proposal=ProposalOut.of(proposal),
            state=ApprovalRecordOut.of(store.current_state(proposal_id)),
            history=[TransitionOut.of(t) for t in store.history(proposal_id)],
            audit_events=[AuditEventOut.of(e) for e in store.list_audit_events(proposal_id)],
        )

    @app.get(
        "/api/proposals/{proposal_id}/audit",
        response_model=AuditTrailOut,
        responses=ERROR_RESPONSES,
        tags=["proposals"],
        summary="The full audit trail for one proposal",
    )
    def get_audit(proposal_id: str) -> Any:
        if store.get_proposal(proposal_id) is None:
            return _error(404, "not_found", f"no proposal {proposal_id}")
        record = store.current_state(proposal_id)
        return AuditTrailOut(
            proposal_id=proposal_id,
            state=record.state.value,
            transitions=[TransitionOut.of(t) for t in store.history(proposal_id)],
            audit_events=[AuditEventOut.of(e) for e in store.list_audit_events(proposal_id)],
            execution_result=record.execution_result,
        )

    @app.get(
        "/api/findings",
        response_model=list[FindingOut],
        tags=["findings"],
        summary="List findings, including INFO ones that were only noted",
    )
    def list_findings(
        run_id: str | None = Query(None, description="Only findings from this pipeline run."),
        limit: int = Query(50, ge=1, le=1000, description="At most this many, newest first."),
    ) -> list[FindingOut]:
        proposal_for = {p.finding_id: p.proposal_id for p, _ in store.list_proposals()}
        rows = list(reversed(store.list_findings(run_id)))[:limit]
        return [FindingOut.of(row, proposal_for.get(row.finding_id)) for row in rows]

    @app.get(
        "/api/runs",
        response_model=list[RunOut],
        tags=["runs"],
        summary="List pipeline runs",
    )
    def list_runs(
        limit: int = Query(50, ge=1, le=1000, description="At most this many, newest first."),
    ) -> list[RunOut]:
        return [RunOut.of(run) for run in list(reversed(store.list_runs()))[:limit]]

    # --- write path 1: Slack ---------------------------------------------------------------

    @app.post("/slack/actions", tags=["slack"], summary="Slack interactive callback")
    async def slack_actions(request: Request, background: BackgroundTasks) -> JSONResponse:
        body = await request.body()
        if not verify_slack_signature(
            signing_secret,
            body,
            request.headers.get("X-Slack-Request-Timestamp"),
            request.headers.get("X-Slack-Signature"),
            clock(),
        ):
            logger.warning(
                "slack callback rejected: bad signature",
                extra={"client": request.client.host if request.client else None},
            )
            raise HTTPException(status_code=401, detail="invalid Slack signature")
        decision = _parse_slack_decision(body)
        if decision.user_id not in approvers:
            outcome = await run_in_threadpool(_refuse, store, decision)
        else:
            outcome = await run_in_threadpool(_decide, store, executor, decision, background)
        return _slack_response(outcome)

    # --- write path 2: the JSON decision endpoint (loopback only) ---------------------------

    @app.post(
        "/api/proposals/{proposal_id}/decision",
        response_model=DecisionResponse,
        tags=["decisions"],
        summary="Approve or reject a proposal (loopback only)",
        description=DECISION_DESCRIPTION,
        responses={
            400: {"model": ErrorOut, "description": "Malformed body, or approve without acknowledgement."},
            403: {"model": ErrorOut, "description": "Not a loopback-bound server, or approver not on the allowlist."},
            404: {"model": ErrorOut, "description": "No such proposal."},
            409: {"model": ErrorOut, "description": "The proposal is no longer pending."},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/DecisionRequest"}}
                },
            }
        },
    )
    async def decide_proposal(proposal_id: str, request: Request, background: BackgroundTasks) -> Any:
        if not decisions_permitted:
            # A publicly bound server must not accept decisions here: this path
            # authenticates by host access, which only holds on loopback.
            reason = (
                "ui_approval_enabled is false"
                if not ui_approval_enabled
                else f"server is bound to {bind_host!r}, not a loopback host"
            )
            logger.warning(
                "API decision refused: approval requires a loopback-bound server with "
                "ui_approval_enabled",
                extra={
                    "proposal_id": proposal_id,
                    "reason": reason,
                    "client": request.client.host if request.client else None,
                },
            )
            return _error(
                403,
                "approval_not_permitted",
                "decisions over the API require a loopback-bound server with "
                "approval.ui_approval_enabled; approve or reject from Slack instead",
            )
        try:
            data = json.loads(await request.body())
        except ValueError:
            return _error(400, "malformed_request", "the body is not JSON")
        try:
            body = DecisionRequest.model_validate(data)
        except ValidationError as exc:
            return _error(400, "malformed_request", _validation_detail(exc))
        approver = body.approver_id.strip()
        if not approver:
            return _error(400, "malformed_request", "approver_id is blank")
        if body.decision == "approve" and not body.acknowledged:
            return _error(
                400,
                "acknowledgement_required",
                'approving requires "acknowledged": true — it may let Bellwether change the '
                "production database",
            )
        decision = Decision(
            proposal_id=proposal_id,
            approve=body.decision == "approve",
            user_id=approver,
            actor=f"api:{approver}",
            channel="api",
        )
        if approver not in approvers:
            outcome = await run_in_threadpool(_refuse, store, decision)
        else:
            outcome = await run_in_threadpool(_decide, store, executor, decision, background)
        if outcome.ok and outcome.state is not None:
            return DecisionResponse(
                proposal_id=proposal_id,
                state=outcome.state,  # type: ignore[arg-type]
                message=outcome.message,
            )
        codes = {403: "not_authorized", 404: "not_found", 409: "already_decided"}
        return _error(outcome.status, codes.get(outcome.status, "refused"), outcome.message)

    @app.get("/healthz", tags=["health"], summary="Liveness")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


def _package_version() -> str:
    try:
        return importlib.metadata.version("bellwether")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def _validation_detail(exc: ValidationError) -> str:
    """Field paths and messages only — never the submitted values."""
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "body"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def _parse_slack_decision(body: bytes) -> Decision:
    fields = parse_qs(body.decode("utf-8", "replace"))
    raw = fields.get("payload")
    if not raw:
        raise HTTPException(status_code=400, detail="missing payload")
    try:
        payload = json.loads(raw[0])
    except ValueError:
        raise HTTPException(status_code=400, detail="payload is not JSON") from None
    if not isinstance(payload, dict) or payload.get("type") != "block_actions":
        raise HTTPException(status_code=400, detail="expected a block_actions payload")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], dict):
        raise HTTPException(status_code=400, detail="no action in payload")
    action: dict[str, Any] = actions[0]
    if action.get("action_id") not in (APPROVE_ACTION_ID, REJECT_ACTION_ID):
        raise HTTPException(status_code=400, detail="unknown action")
    proposal_id = action.get("value")
    user = payload.get("user")
    user_id = user.get("id") if isinstance(user, dict) else None
    if not isinstance(proposal_id, str) or not proposal_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="payload lacks proposal or user")
    username = user.get("username") or user.get("name") if isinstance(user, dict) else None
    actor = f"slack:{username} ({user_id})" if username else f"slack:{user_id}"
    return Decision(
        proposal_id=proposal_id,
        approve=action["action_id"] == APPROVE_ACTION_ID,
        user_id=user_id,
        actor=actor,
        channel="slack",
    )


def _slack_response(outcome: Outcome) -> JSONResponse:
    body: dict[str, Any] = {
        "ok": outcome.ok,
        "proposal_id": outcome.proposal_id,
        "message": outcome.message,
    }
    if outcome.state is not None:
        body["state"] = outcome.state
    return JSONResponse(body, status_code=outcome.status)


def _refuse(store: SqliteStore, decision: Decision) -> Outcome:
    """A decision from someone outside the allowlist: record it, change nothing."""
    verb = "approve" if decision.approve else "reject"
    store.record_audit_event(
        "unauthorized_decision",
        proposal_id=decision.proposal_id,
        actor=decision.actor,
        detail=(
            f"{verb} refused: {decision.channel} approver {decision.user_id} is not in "
            "approval.approver_ids"
        ),
    )
    logger.warning(
        "decision refused: approver not in allowlist",
        extra={
            "proposal_id": decision.proposal_id,
            "actor": decision.actor,
            "decision": verb,
            "channel": decision.channel,
        },
    )
    return Outcome(
        ok=False,
        status=403,
        proposal_id=decision.proposal_id,
        message=f"{decision.actor} is not an authorized approver; nothing was changed",
    )


def _decide(
    store: SqliteStore,
    executor: SupportsExecute | None,
    decision: Decision,
    background: BackgroundTasks,
) -> Outcome:
    """Apply an allowlisted decision. State is re-read from the store, never trusted."""
    proposal_id = decision.proposal_id
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        return Outcome(ok=False, status=404, proposal_id=proposal_id, message="unknown proposal")
    target = ApprovalState.APPROVED if decision.approve else ApprovalState.REJECTED
    try:
        record = store.record_approval_transition(proposal_id, target, by=decision.actor)
    except InvalidTransition:
        current = store.current_state(proposal_id).state.value
        return Outcome(
            ok=False,
            status=409,
            proposal_id=proposal_id,
            message=f"already {current}",
            state=current,
        )
    logger.info(
        "decision recorded",
        extra={
            "proposal_id": proposal_id,
            "state": target.value,
            "actor": decision.actor,
            "channel": decision.channel,
        },
    )

    message = f"{target.value} by {decision.actor}"
    if target is ApprovalState.APPROVED:
        if proposal.action.kind is ActionKind.EXECUTABLE and executor is not None:
            background.add_task(_execute, executor, proposal, record)
            message += f"; executing {proposal.action.executor_op}"
        elif proposal.action.kind is ActionKind.EXECUTABLE:
            message += "; executor not enabled, run the command manually"
        else:
            message += "; propose-only, run the command manually"
    return Outcome(
        ok=True, status=200, proposal_id=proposal_id, message=message, state=record.state.value
    )


def _execute(executor: SupportsExecute, proposal: Proposal, record: ApprovalRecord) -> None:
    try:
        executor.execute(proposal, record)
    except Exception:  # the executor has already recorded FAILED where it applies
        logger.exception(
            "execution after approval did not complete",
            extra={"proposal_id": proposal.proposal_id},
        )
