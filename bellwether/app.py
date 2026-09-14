"""Approval endpoint and read-only UI (BUILD_SPEC §3.9). Served by ``bellwether serve``.

``POST /slack/actions`` receives Slack interactive callbacks. The Slack
signature is verified before anything else — an unverified approval endpoint
is a hole — using Slack's v0 scheme (HMAC-SHA256 over ``v0:<ts>:<raw body>``
with the signing secret), a constant-time compare, and a five-minute replay
window. Anything that fails is a 401 and changes nothing.

A verified Approve records the APPROVED transition. If the proposal is
EXECUTABLE, the executor runs after the response is sent (Slack wants an answer
within three seconds); the executor records EXECUTED or FAILED itself. A
verified Reject records REJECTED.

``GET /`` and ``GET /proposals/{id}`` are a minimal read-only UI. They have no
forms: the Slack callback is the only route that accepts a write.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.notify.slack import APPROVE_ACTION_ID, REJECT_ACTION_ID
from bellwether.notify.stdout import render_text
from bellwether.store.sqlite import InvalidTransition, SqliteStore

logger = logging.getLogger(__name__)

MAX_SKEW_SECONDS = 300


class SupportsExecute(Protocol):
    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord: ...


@dataclass(frozen=True)
class Decision:
    proposal_id: str
    approve: bool
    actor: str


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


def create_app(
    store: SqliteStore,
    *,
    signing_secret: str,
    executor: SupportsExecute | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    if not signing_secret:
        raise ValueError("a Slack signing secret is required")
    app = FastAPI(title="Bellwether", docs_url=None, redoc_url=None, openapi_url=None)

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
        decision = _parse_decision(body)
        return await run_in_threadpool(_decide, store, executor, decision, background)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        rows = "".join(
            "<tr>"
            f"<td><a href=\"/proposals/{_e(p.proposal_id)}\"><code>{_e(p.proposal_id)}</code></a></td>"
            f"<td>{_e(p.created_at.strftime('%Y-%m-%d %H:%M'))}</td>"
            f"<td>{_e(p.failure_mode)}</td><td>{_e(p.node)}</td>"
            f"<td>{_e(p.action.kind.value)}</td><td>{_e(p.provider)}</td>"
            f"<td class=\"{_e(r.state.value)}\">{_e(r.state.value)}</td>"
            "</tr>"
            for p, r in reversed(store.list_proposals())
        )
        table = (
            "<table><thead><tr><th>Proposal</th><th>Created (UTC)</th><th>Failure mode</th>"
            "<th>Node</th><th>Action</th><th>Provider</th><th>State</th></tr></thead>"
            f"<tbody>{rows or '<tr><td colspan=7>No proposals yet.</td></tr>'}</tbody></table>"
        )
        return HTMLResponse(_page("Bellwether proposals", table))

    @app.get("/proposals/{proposal_id}", response_class=HTMLResponse)
    def detail(proposal_id: str) -> HTMLResponse:
        proposal = store.get_proposal(proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="unknown proposal")
        record = store.current_state(proposal_id)
        history = "".join(
            f"<tr><td>{_e(t.at.strftime('%Y-%m-%d %H:%M:%S'))}</td><td>{_e(t.state.value)}</td>"
            f"<td>{_e(t.actor or '')}</td><td>{_e(t.result or '')}</td></tr>"
            for t in store.history(proposal_id)
        )
        body = (
            '<p><a href="/">&larr; all proposals</a></p>'
            f"<pre>{_e(render_text(proposal, record))}</pre>"
            "<h2>Audit trail</h2><table><thead><tr><th>At (UTC)</th><th>State</th>"
            f"<th>By</th><th>Result</th></tr></thead><tbody>{history}</tbody></table>"
        )
        return HTMLResponse(_page(f"Proposal {proposal_id}", body))

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


def _parse_decision(body: bytes) -> Decision:
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
        proposal_id=proposal_id, approve=action["action_id"] == APPROVE_ACTION_ID, actor=actor
    )


def _decide(
    store: SqliteStore,
    executor: SupportsExecute | None,
    decision: Decision,
    background: BackgroundTasks,
) -> JSONResponse:
    proposal_id = decision.proposal_id
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        return JSONResponse({"ok": False, "message": "unknown proposal"}, status_code=404)
    target = ApprovalState.APPROVED if decision.approve else ApprovalState.REJECTED
    try:
        record = store.record_approval_transition(proposal_id, target, by=decision.actor)
    except InvalidTransition:
        current = store.current_state(proposal_id).state.value
        return JSONResponse(
            {
                "ok": False,
                "proposal_id": proposal_id,
                "state": current,
                "message": f"already {current}",
            },
            status_code=409,
        )
    logger.info(
        "slack decision recorded",
        extra={"proposal_id": proposal_id, "state": target.value, "actor": decision.actor},
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
    return JSONResponse(
        {"ok": True, "proposal_id": proposal_id, "state": record.state.value, "message": message}
    )


def _execute(executor: SupportsExecute, proposal: Proposal, record: ApprovalRecord) -> None:
    try:
        executor.execute(proposal, record)
    except Exception:  # the executor has already recorded FAILED where it applies
        logger.exception(
            "execution after approval did not complete",
            extra={"proposal_id": proposal.proposal_id},
        )


def _e(value: str) -> str:
    return html.escape(value, quote=True)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{_e(title)}</title><style>"
        "body{font:14px/1.45 system-ui,sans-serif;margin:2rem;color:#1d1d1f}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #ddd}"
        "pre{background:#f6f6f6;padding:1rem;overflow-x:auto;white-space:pre-wrap}"
        ".pending{color:#9a6700}.approved,.executed{color:#1a7f37}"
        ".rejected,.failed,.expired{color:#cf222e}"
        f"</style></head><body><h1>{_e(title)}</h1>{body}</body></html>"
    )
