"""Acceptance tests for the approval endpoint — BUILD_SPEC §3.9.

Spec acceptance:
  - a request with a bad signature is rejected 401
  - a valid Approve on an EXECUTABLE proposal transitions state and triggers
    the (mocked) executor
  - a valid Approve on a propose-only proposal transitions to APPROVED and
    does not call the executor
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from bellwether.app import create_app
from bellwether.executor.executor import ExecutionRefused
from bellwether.models import (
    ActionKind,
    ApprovalRecord,
    ApprovalState,
    Evidence,
    Proposal,
    RemediationAction,
)
from bellwether.notify.slack import APPROVE_ACTION_ID, REJECT_ACTION_ID
from bellwether.store.sqlite import SqliteStore

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
NOW = 1_757_851_200
APPROVERS = frozenset({"U0OSEGO"})
MALLORY = {"id": "U0MALLORY", "username": "mallory"}
T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_proposal(kind: ActionKind = ActionKind.EXECUTABLE, **overrides: Any) -> Proposal:
    if kind is ActionKind.EXECUTABLE:
        action = RemediationAction(
            kind=kind,
            title="Index transactions.account_id",
            command="db.transactions.createIndex({account_id: 1})",
            rationale="Hot filter.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "meetadev_ledger",
                "collection": "transactions",
                "keys": [{"field": "account_id", "direction": 1}],
                "estimated_docs": 40_000,
            },
        )
    else:
        action = RemediationAction(
            kind=kind,
            title="Grow the oplog",
            command="db.adminCommand({replSetResizeOplog: 1, size: 51200})",
            rationale="Restore the window.",
            reversible=True,
        )
    values: dict[str, Any] = {
        "finding_id": "f" * 32,
        "failure_mode": "oplog_window_below_resync",
        "node": "node-backup.mongo.internal:27017",
        "diagnosis": "Window 40 min < resync 60 min.",
        "mechanism": "Capped oplog.",
        "impact_if_ignored": "Initial sync after maintenance.",
        "action": action,
        "confidence": 0.8,
        "provider": "claude",
        "evidence_refs": (Evidence("oplog_window_seconds", 2400, "s", observed_at=T0),),
        "created_at": T0,
    }
    values.update(overrides)
    return Proposal(**values)


class FakeExecutor:
    """Stands in for Executor: records calls, marks the proposal EXECUTED."""

    def __init__(self, store: SqliteStore, error: Exception | None = None) -> None:
        self.store = store
        self.error = error
        self.calls: list[tuple[Proposal, ApprovalRecord]] = []

    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord:
        self.calls.append((proposal, approval_record))
        if self.error is not None:
            raise self.error
        return self.store.record_approval_transition(
            proposal.proposal_id, ApprovalState.EXECUTED, result="done"
        )


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bellwether.db")


@pytest.fixture
def executor(store: SqliteStore) -> FakeExecutor:
    return FakeExecutor(store)


@pytest.fixture
def client(store: SqliteStore, executor: FakeExecutor) -> TestClient:
    app = create_app(
        store, signing_secret=SECRET, approver_ids=APPROVERS, executor=executor, clock=lambda: NOW
    )
    return TestClient(app)


def slack_body(action_id: str, proposal_id: str, **payload_overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "type": "block_actions",
        "user": {"id": "U0OSEGO", "username": "osego"},
        "actions": [{"action_id": action_id, "value": proposal_id, "type": "button"}],
        "response_url": "https://hooks.slack.com/actions/T0/1/abc",
    }
    payload.update(payload_overrides)
    return urlencode({"payload": json.dumps(payload)}).encode()


def signed(body: bytes, *, timestamp: int = NOW, secret: str = SECRET) -> dict[str, str]:
    base = b"v0:" + str(timestamp).encode() + b":" + body
    digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return {
        "X-Slack-Request-Timestamp": str(timestamp),
        "X-Slack-Signature": f"v0={digest}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def post(client: TestClient, body: bytes, headers: dict[str, str]) -> Any:
    return client.post("/slack/actions", content=body, headers=headers)


# --- Signature verification ------------------------------------------------------


def test_bad_signature_is_rejected_401(
    client: TestClient, store: SqliteStore, executor: FakeExecutor
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body, secret="not-the-secret"))

    assert response.status_code == 401
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert executor.calls == []
    assert store.list_audit_events() == []  # unverified requests carry no trustworthy identity


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda h: {}, id="no headers"),
        pytest.param(lambda h: {**h, "X-Slack-Signature": "v0=deadbeef"}, id="wrong digest"),
        pytest.param(
            lambda h: {**h, "X-Slack-Signature": h["X-Slack-Signature"][3:]}, id="no v0 prefix"
        ),
        pytest.param(lambda h: {**h, "X-Slack-Request-Timestamp": "soon"}, id="non-numeric ts"),
        pytest.param(
            lambda h: {k: v for k, v in h.items() if k != "X-Slack-Signature"}, id="no signature"
        ),
    ],
)
def test_malformed_signature_headers_are_rejected(
    client: TestClient, store: SqliteStore, mangle: Any
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, mangle(signed(body)))

    assert response.status_code == 401
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


def test_stale_timestamp_is_rejected(client: TestClient, store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body, timestamp=NOW - 301))  # replayed

    assert response.status_code == 401


def test_signature_over_a_different_body_is_rejected(
    client: TestClient, store: SqliteStore
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    signed_for = slack_body(REJECT_ACTION_ID, proposal.proposal_id)
    tampered = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, tampered, signed(signed_for))

    assert response.status_code == 401
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


# --- Approve / Reject -----------------------------------------------------------------


def test_approve_executable_transitions_and_triggers_executor(
    client: TestClient, store: SqliteStore, executor: FakeExecutor
) -> None:
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body))

    assert response.status_code == 200
    assert len(executor.calls) == 1
    called_proposal, called_record = executor.calls[0]
    assert called_proposal == proposal
    assert called_record.state is ApprovalState.APPROVED
    assert called_record.decided_by is not None and "U0OSEGO" in called_record.decided_by
    states = [t.state for t in store.history(proposal.proposal_id)]
    assert states == [ApprovalState.PENDING, ApprovalState.APPROVED, ApprovalState.EXECUTED]


def test_approve_propose_only_does_not_call_executor(
    client: TestClient, store: SqliteStore, executor: FakeExecutor
) -> None:
    proposal = make_proposal(ActionKind.PROPOSE_ONLY)
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body))

    assert response.status_code == 200
    assert response.json()["state"] == "approved"
    assert store.current_state(proposal.proposal_id).state is ApprovalState.APPROVED
    assert executor.calls == []


def test_reject_records_rejection(
    client: TestClient, store: SqliteStore, executor: FakeExecutor
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(REJECT_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body))

    assert response.status_code == 200
    record = store.current_state(proposal.proposal_id)
    assert record.state is ApprovalState.REJECTED
    assert record.decided_by is not None and "osego" in record.decided_by
    assert executor.calls == []


def test_second_decision_is_refused(
    client: TestClient, store: SqliteStore, executor: FakeExecutor
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    reject = slack_body(REJECT_ACTION_ID, proposal.proposal_id)
    post(client, reject, signed(reject))
    approve = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, approve, signed(approve))

    assert response.status_code == 409
    assert store.current_state(proposal.proposal_id).state is ApprovalState.REJECTED
    assert executor.calls == []


def test_executor_refusal_leaves_the_approval_standing(store: SqliteStore) -> None:
    refusing = FakeExecutor(store, error=ExecutionRefused("executor is disabled"))
    client = TestClient(
        create_app(
            store,
            signing_secret=SECRET,
            approver_ids=APPROVERS,
            executor=refusing,
            clock=lambda: NOW,
        )
    )
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body))

    assert response.status_code == 200
    assert len(refusing.calls) == 1
    assert store.current_state(proposal.proposal_id).state is ApprovalState.APPROVED


def test_executable_approve_without_executor_is_recorded_for_manual_run(
    store: SqliteStore,
) -> None:
    client = TestClient(
        create_app(
            store, signing_secret=SECRET, approver_ids=APPROVERS, executor=None, clock=lambda: NOW
        )
    )
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id)

    response = post(client, body, signed(body))

    assert response.status_code == 200
    assert "manual" in response.json()["message"]
    assert store.current_state(proposal.proposal_id).state is ApprovalState.APPROVED


def test_unknown_proposal_is_404(client: TestClient) -> None:
    body = slack_body(APPROVE_ACTION_ID, "0" * 32)

    assert post(client, body, signed(body)).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"payload=not-json", id="payload not json"),
        pytest.param(b"nothing=here", id="no payload"),
        pytest.param(
            slack_body(APPROVE_ACTION_ID, "x", type="view_submission"), id="wrong payload type"
        ),
        pytest.param(slack_body("bellwether_execute_now", "x"), id="unknown action"),
        pytest.param(slack_body(APPROVE_ACTION_ID, "x", actions=[]), id="no actions"),
    ],
)
def test_malformed_payload_is_400(client: TestClient, body: bytes) -> None:
    assert post(client, body, signed(body)).status_code == 400


# --- Correction B: approver allowlist ----------------------------------------------------


@pytest.mark.parametrize(("action_id", "verb"), [(APPROVE_ACTION_ID, "approve"), (REJECT_ACTION_ID, "reject")])
def test_non_allowlisted_user_is_refused_and_recorded(
    client: TestClient,
    store: SqliteStore,
    executor: FakeExecutor,
    action_id: str,
    verb: str,
) -> None:
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)
    body = slack_body(action_id, proposal.proposal_id, user=MALLORY)

    response = post(client, body, signed(body))

    assert response.status_code == 403
    assert "not an authorized approver" in response.json()["message"]
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert len(store.history(proposal.proposal_id)) == 1  # no transition row
    assert executor.calls == []
    [event] = store.list_audit_events(proposal.proposal_id)
    assert event.kind == "unauthorized_decision"
    assert event.actor is not None and "U0MALLORY" in event.actor
    assert verb in event.detail


def test_allowlisted_approver_proceeds(store: SqliteStore, executor: FakeExecutor) -> None:
    client = TestClient(
        create_app(
            store,
            signing_secret=SECRET,
            approver_ids={"U0OSEGO", "U0TEAMMATE"},
            executor=executor,
            clock=lambda: NOW,
        )
    )
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)
    body = slack_body(
        APPROVE_ACTION_ID, proposal.proposal_id, user={"id": "U0TEAMMATE", "username": "teammate"}
    )

    response = post(client, body, signed(body))

    assert response.status_code == 200
    assert len(executor.calls) == 1
    record = store.current_state(proposal.proposal_id)
    assert record.state is ApprovalState.EXECUTED
    assert record.decided_by is not None and "U0TEAMMATE" in record.decided_by
    assert store.list_audit_events() == []


def test_unauthorized_click_does_not_reveal_whether_a_proposal_exists(
    client: TestClient, store: SqliteStore
) -> None:
    body = slack_body(APPROVE_ACTION_ID, "0" * 32, user=MALLORY)

    response = post(client, body, signed(body))

    assert response.status_code == 403
    assert len(store.list_audit_events("0" * 32)) == 1


def test_refused_attempt_appears_in_the_audit_trail(
    client: TestClient, store: SqliteStore
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    body = slack_body(APPROVE_ACTION_ID, proposal.proposal_id, user=MALLORY)
    post(client, body, signed(body))

    page = client.get(f"/proposals/{proposal.proposal_id}").text

    assert "unauthorized_decision" in page
    assert "U0MALLORY" in page


def test_an_approver_allowlist_is_required(store: SqliteStore) -> None:
    with pytest.raises(ValueError, match="approver"):
        create_app(store, signing_secret=SECRET, approver_ids=set())


# --- Read-only UI ---------------------------------------------------------------------


def test_ui_lists_proposals_and_state(client: TestClient, store: SqliteStore) -> None:
    a = make_proposal()
    b = make_proposal(ActionKind.PROPOSE_ONLY, diagnosis="<script>alert(1)</script>")
    store.record_proposal(a)
    store.record_proposal(b)
    store.record_approval_transition(b.proposal_id, ApprovalState.REJECTED, by="x")

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert a.proposal_id in page and b.proposal_id in page
    assert "pending" in page and "rejected" in page
    assert "<form" not in page


def test_ui_detail_page_escapes_model_text(client: TestClient, store: SqliteStore) -> None:
    proposal = make_proposal(diagnosis="<script>alert(1)</script>")
    store.record_proposal(proposal)

    response = client.get(f"/proposals/{proposal.proposal_id}")

    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text
    assert proposal.action.command in response.text


def test_ui_unknown_proposal_is_404(client: TestClient) -> None:
    assert client.get(f"/proposals/{'0' * 32}").status_code == 404


def test_the_only_write_routes_are_the_two_gated_decision_paths(client: TestClient) -> None:
    writes = {
        (route.path, method)
        for route in client.app.routes  # type: ignore[attr-defined]
        for method in getattr(route, "methods", set())
        if method not in {"GET", "HEAD"}
    }

    # Slack (signature + allowlist) and the in-UI form (loopback + allowlist).
    assert writes == {("/slack/actions", "POST"), ("/proposals/{proposal_id}/decision", "POST")}


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"ok": True}
