"""Approval endpoint and operations UI (BUILD_SPEC §3.9). Served by ``bellwether serve``.

**Two write paths, each independently gated, both audited.**

1. ``POST /slack/actions`` — Slack interactive callbacks. Gated by the Slack
   signature (Slack's v0 scheme: HMAC-SHA256 over ``v0:<ts>:<raw body>`` with
   the signing secret, a constant-time compare, a five-minute replay window;
   anything that fails is a 401 and changes nothing) and then by
   ``approval.approver_ids``.
2. ``POST /proposals/{id}/decision`` — the in-UI approve/reject form. Gated by
   the bind address (it refuses with 403 unless ``approval.ui_approval_enabled``
   is true AND the server is bound to a loopback host) and then by the same
   ``approval.approver_ids``.

Why loopback: in-UI approval is for an operator inside the trust boundary —
SSH-tunnelled to the host — never for the public internet. The public
approval path is Slack, which proves who clicked with a signature; the UI path
proves presence on the host instead. An approver id typed into the form is an
attestation checked against the allowlist, and host access is the
authentication — which is exactly why a publicly bound server must not accept it.

Both paths share one implementation after their gates: an approver outside the
allowlist is refused with 403, changes nothing, and is recorded as an
``unauthorized_decision`` audit event; an allowlisted decision re-reads the
proposal's state from the store (never trusting a page's view of it) and moves
only a PENDING proposal; an approved EXECUTABLE proposal is handed to the
executor after the response, through the same code path either way. The
executor records EXECUTED or FAILED itself.

``GET /`` and ``GET /proposals/{id}`` render the operations UI (bellwether.ui).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from bellwether import ui
from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.notify.slack import APPROVE_ACTION_ID, REJECT_ACTION_ID
from bellwether.store.sqlite import InvalidTransition, SqliteStore

logger = logging.getLogger(__name__)

MAX_SKEW_SECONDS = 300


class SupportsExecute(Protocol):
    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord: ...


@dataclass(frozen=True)
class Decision:
    proposal_id: str
    approve: bool
    user_id: str  # the approver id checked against approval.approver_ids
    actor: str  # how the decision is attributed in the audit trail
    channel: str = "slack"  # "slack" | "ui"


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
    provider_labels: Mapping[str, str] | None = None,
    cluster_label: str = "rs0",
) -> FastAPI:
    if not signing_secret:
        raise ValueError("a Slack signing secret is required")
    if not approver_ids:
        raise ValueError("an approver allowlist (approval.approver_ids) is required; it is empty")
    approvers = frozenset(approver_ids)
    labels = dict(provider_labels or {})
    # In-UI approval needs both: the operator's switch, and a loopback-only bind.
    # An unknown bind address (None) fails closed.
    loopback = is_loopback_host(bind_host)
    ui_permitted = ui_approval_enabled and loopback
    ui_approval = "permitted" if ui_permitted else ("disabled" if not ui_approval_enabled else "not_loopback")

    app = FastAPI(title="Bellwether", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bind_host = bind_host
    app.state.ui_approval_permitted = ui_permitted

    def now() -> datetime:
        return datetime.fromtimestamp(clock(), timezone.utc)

    def updated_note(moment: datetime) -> str:
        runs = store.list_runs()
        last = f"last run {ui.relative(runs[-1].finished_at, moment)}" if runs else "no runs yet"
        return f"updated {moment:%H:%M} UTC · {last}"

    def html_error(title: str, message: str, status: int, back: str = "/") -> HTMLResponse:
        moment = now()
        page = ui.error_page(
            title,
            message,
            status=status,
            back_href=back,
            cluster_label=cluster_label,
            updated_note=updated_note(moment),
        )
        return HTMLResponse(page, status_code=status)

    # --- write path 1: Slack ------------------------------------------------------------

    @app.post("/slack/actions")
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

    # --- write path 2: the in-UI form (loopback only) -------------------------------------

    @app.post("/proposals/{proposal_id}/decision")
    async def ui_decision(
        proposal_id: str, request: Request, background: BackgroundTasks
    ) -> Response:
        back = f"/proposals/{proposal_id}"
        if not ui_permitted:
            # A publicly bound server must not accept decisions from a web form:
            # the UI path authenticates by host access, which only holds on loopback.
            reason = (
                "ui_approval_enabled is false"
                if not ui_approval_enabled
                else f"server is bound to {bind_host!r}, not a loopback host"
            )
            logger.warning(
                "in-UI decision refused: approval requires a loopback-bound server with "
                "ui_approval_enabled",
                extra={
                    "proposal_id": proposal_id,
                    "reason": reason,
                    "client": request.client.host if request.client else None,
                },
            )
            return html_error(
                "In-UI approval is not available here",
                "In-UI approval only works on a loopback-bound server with "
                "approval.ui_approval_enabled. Approve or reject from Slack instead.",
                403,
                back,
            )
        fields = parse_qs((await request.body()).decode("utf-8", "replace"))
        approver = _first(fields, "approver_id").strip()
        verb = _first(fields, "decision")
        if verb not in ("approve", "reject") or not approver:
            return html_error(
                "Incomplete decision", "A decision needs an approver ID and approve or reject.", 400, back
            )
        if verb == "approve" and _first(fields, "acknowledge") != "yes":
            return html_error(
                "Confirmation required",
                "Tick the confirmation to approve: approving may let Bellwether change the "
                "production database.",
                400,
                back,
            )
        decision = Decision(
            proposal_id=proposal_id,
            approve=verb == "approve",
            user_id=approver,
            actor=f"ui:{approver}",
            channel="ui",
        )
        if approver not in approvers:
            outcome = await run_in_threadpool(_refuse, store, decision)
        else:
            outcome = await run_in_threadpool(_decide, store, executor, decision, background)
        if outcome.ok:
            return RedirectResponse(back, status_code=303)  # post/redirect/get
        titles = {403: "Not an authorized approver", 404: "No such proposal", 409: "Already decided"}
        return html_error(titles.get(outcome.status, "Decision refused"), outcome.message, outcome.status, back)

    # --- the operations UI (read-only views) ---------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def overview() -> HTMLResponse:
        moment = now()
        page = ui.overview_page(
            _overview_rows(store),
            now=moment,
            cluster_label=cluster_label,
            updated_note=updated_note(moment),
        )
        return HTMLResponse(page)

    @app.get("/proposals/{proposal_id}", response_class=HTMLResponse)
    def detail(proposal_id: str) -> HTMLResponse:
        proposal = store.get_proposal(proposal_id)
        if proposal is None:
            return html_error("No such proposal", f"There is no proposal {proposal_id}.", 404)
        finding = next((f for f in store.list_findings() if f.finding_id == proposal.finding_id), None)
        moment = now()
        page = ui.detail_page(
            proposal,
            store.current_state(proposal_id),
            severity=finding.severity.value if finding else None,
            summary=finding.summary if finding else None,
            history=store.history(proposal_id),
            events=store.list_audit_events(proposal_id),
            provider_label=labels.get(proposal.provider, proposal.provider.capitalize()),
            ui_approval=ui_approval,
            now=moment,
            cluster_label=cluster_label,
            updated_note=updated_note(moment),
        )
        return HTMLResponse(page)

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


def _overview_rows(store: SqliteStore) -> list[ui.OverviewRow]:
    """Proposals with their state, plus the latest run's findings that never escalated."""
    findings = {f.finding_id: f for f in store.list_findings()}
    rows = []
    for proposal, record in store.list_proposals():
        finding = findings.get(proposal.finding_id)
        rows.append(
            ui.OverviewRow(
                href=f"/proposals/{proposal.proposal_id}",
                severity=finding.severity.value if finding else None,
                failure_mode=proposal.failure_mode,
                node=proposal.node,
                summary=finding.summary if finding else proposal.action.title,
                state=record.state.value,
                at=proposal.created_at,
            )
        )
    runs = store.list_runs()
    if runs:
        for finding in store.list_findings(runs[-1].run_id):
            if not finding.escalated:
                rows.append(
                    ui.OverviewRow(
                        href=None,
                        severity=finding.severity.value,
                        failure_mode=finding.failure_mode,
                        node=finding.node,
                        summary=finding.summary,
                        state="noted",
                        at=finding.detected_at,
                    )
                )
    return rows


def _first(fields: Mapping[str, list[str]], name: str) -> str:
    values = fields.get(name)
    return values[0] if values else ""


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
