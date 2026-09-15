"""The JSON API: read endpoints over the store, the loopback-gated decision
endpoint, and the OpenAPI documentation that describes both."""

from __future__ import annotations

import importlib.metadata
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bellwether.app import create_app
from bellwether.models import (
    ActionKind,
    ApprovalRecord,
    ApprovalState,
    Evidence,
    Finding,
    Proposal,
    RemediationAction,
    Severity,
    SignalClass,
)
from bellwether.store.sqlite import RunRecord, SqliteStore

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
APPROVERS = frozenset({"U0OSEGO"})
PRIMARY = "mongo-2.example.internal:27017"
HIDDEN = "mongo-hidden.example.internal:27017"
CANDIDATE = [{"field": "account_id", "direction": 1}, {"field": "posted_at", "direction": -1}]
EXECUTED_RESULT = "created index account_id_1_posted_at_-1"


class FakeExecutor:
    def __init__(self, store: SqliteStore) -> None:
        self.store = store
        self.calls: list[tuple[Proposal, ApprovalRecord]] = []

    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord:
        self.calls.append((proposal, approval_record))
        return self.store.record_approval_transition(
            proposal.proposal_id, ApprovalState.EXECUTED, result=EXECUTED_RESULT
        )


# --- fixtures ------------------------------------------------------------------------------


def index_finding() -> Finding:
    return Finding(
        signal_class=SignalClass.PERFORMANCE,
        failure_mode="missing_index_collscan",
        severity=Severity.WARNING,
        node=PRIMARY,
        summary="Query shape {account_id: eq, posted_at: range} on appdb.transactions examined "
        "11,800,000 documents to return 12,400.",
        evidence=(
            Evidence("docs_examined", 11_800_000, "docs"),
            Evidence("candidate_index", CANDIDATE),
            Evidence("collection_doc_count", 40_000, "docs"),
        ),
        detected_at=NOW - timedelta(minutes=10),
    )


def oplog_finding() -> Finding:
    return Finding(
        signal_class=SignalClass.REPLICATION,
        failure_mode="oplog_window_below_resync",
        severity=Severity.CRITICAL,
        node=HIDDEN,
        summary="Oplog window is 38 min, below the 60 min resync estimate.",
        evidence=(Evidence("oplog_window_seconds", 2280, "s"),),
        detected_at=NOW - timedelta(hours=3),
    )


def info_finding() -> Finding:
    return Finding(
        signal_class=SignalClass.PERFORMANCE,
        failure_mode="profiler_disabled",
        severity=Severity.INFO,
        node=HIDDEN,
        summary="Profiling is off on appdb.",
        evidence=(Evidence("db", "appdb"),),
        detected_at=NOW - timedelta(minutes=5),
    )


def proposal_for(finding: Finding, *, executable: bool) -> Proposal:
    if executable:
        action = RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Build ESR index {account_id: 1, posted_at: -1} on appdb.transactions",
            command="db.transactions.createIndex({ account_id: 1, posted_at: -1 })",
            rationale="The read saving outweighs one small index's write cost.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "appdb",
                "collection": "transactions",
                "keys": CANDIDATE,
                "estimated_docs": 40_000,
            },
        )
    else:
        action = RemediationAction(
            kind=ActionKind.PROPOSE_ONLY,
            title="Grow the oplog",
            command="db.adminCommand({ replSetResizeOplog: 1, size: 2560 })",
            rationale="Restore the window.",
            reversible=True,
        )
    return Proposal(
        finding_id=finding.finding_id,
        failure_mode=finding.failure_mode,
        node=finding.node,
        diagnosis=f"Diagnosis for {finding.failure_mode}.",
        mechanism=f"Mechanism for {finding.failure_mode}.",
        impact_if_ignored=f"Impact for {finding.failure_mode}.",
        action=action,
        confidence=0.88,
        provider="claude",
        evidence_refs=finding.evidence,
        created_at=finding.detected_at + timedelta(seconds=30),
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bellwether.db")


def seed(store: SqliteStore) -> dict[str, Proposal]:
    """Run r0: an escalated oplog finding (its proposal later rejected).
    Run r1: an escalated index finding (proposal pending) and a noted INFO finding."""
    index, oplog, info = index_finding(), oplog_finding(), info_finding()
    store.record_finding(oplog, run_id="r0", escalated=True)
    store.record_run(
        RunRecord(
            "r0",
            NOW - timedelta(hours=3),
            NOW - timedelta(hours=3) + timedelta(seconds=40),
            1,
            1,
            ("collector query_profile: NoReachableNode: no replica set member reachable",),
        )
    )
    store.record_finding(index, run_id="r1", escalated=True)
    store.record_finding(info, run_id="r1", escalated=False)
    store.record_run(RunRecord("r1", NOW - timedelta(minutes=10), NOW - timedelta(minutes=9), 2, 1, ()))
    rejected = proposal_for(oplog, executable=False)
    store.record_proposal(rejected, at=rejected.created_at)
    store.record_approval_transition(
        rejected.proposal_id, ApprovalState.REJECTED, by="slack:osego (U0OSEGO)",
        at=rejected.created_at + timedelta(minutes=5),
    )
    pending = proposal_for(index, executable=True)
    store.record_proposal(pending, at=pending.created_at)
    return {"pending": pending, "rejected": rejected}


def client_for(
    store: SqliteStore,
    *,
    bind: str | None = "127.0.0.1",
    enabled: bool = True,
    executor: FakeExecutor | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            store,
            signing_secret=SECRET,
            approver_ids=APPROVERS,
            executor=executor,
            clock=NOW.timestamp,
            bind_host=bind,
            ui_approval_enabled=enabled,
        )
    )


def decide(client: TestClient, proposal_id: str, **body: Any) -> Any:
    payload = {"decision": "approve", "approver_id": "U0OSEGO", "acknowledged": True, **body}
    return client.post(f"/api/proposals/{proposal_id}/decision", json=payload)


def json_error(response: Any, status: int, error: str) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    body: dict[str, Any] = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == error
    assert body["detail"]
    return body


# --- OpenAPI and the docs ----------------------------------------------------------------------

API_PATHS = (
    "/healthz",
    "/api/proposals",
    "/api/proposals/{proposal_id}",
    "/api/proposals/{proposal_id}/audit",
    "/api/proposals/{proposal_id}/decision",
    "/api/findings",
    "/api/runs",
)


def test_openapi_describes_the_api(store: SqliteStore) -> None:
    spec = client_for(store).get("/openapi.json").json()

    assert spec["info"]["title"] == "Bellwether"
    assert spec["info"]["version"] == importlib.metadata.version("bellwether")
    assert "loopback" in spec["info"]["description"]
    for path in API_PATHS:
        assert path in spec["paths"], path
    decision = spec["paths"]["/api/proposals/{proposal_id}/decision"]["post"]
    assert "loopback" in decision["description"]
    assert "allowlist" in decision["description"]
    assert decision["requestBody"]["content"]["application/json"]["schema"]
    assert {"200", "400", "403", "404", "409"} <= set(decision["responses"])


@pytest.mark.parametrize(
    "schema",
    [
        "ProposalOut",
        "EvidenceOut",
        "RemediationActionOut",
        "FindingOut",
        "RunOut",
        "ApprovalRecordOut",
        "AuditEventOut",
        "TransitionOut",
        "ProposalListItem",
        "ProposalDetailOut",
        "AuditTrailOut",
        "DecisionRequest",
        "DecisionResponse",
        "ErrorOut",
    ],
)
def test_every_documented_field_has_a_description(store: SqliteStore, schema: str) -> None:
    spec = client_for(store).get("/openapi.json").json()

    properties = spec["components"]["schemas"][schema]["properties"]

    for field, prop in properties.items():
        assert prop.get("description"), f"{schema}.{field} has no description"


def test_interactive_docs_are_served(store: SqliteStore) -> None:
    client = client_for(store)

    docs = client.get("/docs")
    redoc = client.get("/redoc")

    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    assert redoc.status_code == 200 and "redoc" in redoc.text.lower()


@pytest.mark.parametrize("path", ["/", "/proposals/abc"])
def test_there_is_no_html_ui(store: SqliteStore, path: str) -> None:
    response = client_for(store).get(path)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_nothing_imports_the_removed_ui() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "bellwether" / "ui.py").exists()
    for path in [*root.glob("bellwether/**/*.py"), *root.glob("tests/*.py")]:
        if path == Path(__file__).resolve():
            continue
        source = path.read_text()
        assert "bellwether.ui" not in source, path.name
        assert "from bellwether import ui" not in source, path.name


def test_healthz(store: SqliteStore) -> None:
    assert client_for(store).get("/healthz").json() == {"ok": True}


# --- GET /api/proposals ----------------------------------------------------------------------


def test_list_proposals_newest_first_with_state(store: SqliteStore) -> None:
    seeded = seed(store)

    items = client_for(store).get("/api/proposals").json()

    assert [i["proposal"]["proposal_id"] for i in items] == [
        seeded["pending"].proposal_id,
        seeded["rejected"].proposal_id,
    ]
    first = items[0]
    assert set(first) == {"proposal", "state", "created_at"}
    assert first["state"] == "pending"
    proposal = first["proposal"]
    for key in (
        "proposal_id", "finding_id", "failure_mode", "node", "diagnosis", "mechanism",
        "impact_if_ignored", "action", "confidence", "provider", "evidence_refs", "created_at",
    ):
        assert key in proposal, key
    assert proposal["action"]["kind"] == "executable"
    assert proposal["action"]["executor_args"]["keys"] == CANDIDATE
    evidence = {e["name"]: e for e in proposal["evidence_refs"]}
    assert evidence["docs_examined"]["value"] == 11_800_000
    assert evidence["docs_examined"]["unit"] == "docs"
    assert evidence["candidate_index"]["value"] == CANDIDATE


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("?state=pending", ["pending"]),
        ("?state=rejected", ["rejected"]),
        ("?state=executed", []),
        ("?limit=1", ["pending"]),
        ("?state=rejected&limit=5", ["rejected"]),
    ],
)
def test_list_proposals_filters(store: SqliteStore, query: str, expected: list[str]) -> None:
    seed(store)

    items = client_for(store).get(f"/api/proposals{query}").json()

    assert [i["state"] for i in items] == expected


@pytest.mark.parametrize("query", ["?state=bogus", "?limit=0", "?limit=5000"])
def test_list_proposals_rejects_bad_filters(store: SqliteStore, query: str) -> None:
    response = client_for(store).get(f"/api/proposals{query}")

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


# --- GET /api/proposals/{id} and /audit ------------------------------------------------------------


def test_proposal_detail(store: SqliteStore) -> None:
    pending = seed(store)["pending"]

    body = client_for(store).get(f"/api/proposals/{pending.proposal_id}").json()

    assert set(body) == {"proposal", "state", "history", "audit_events"}
    assert body["proposal"]["diagnosis"] == pending.diagnosis
    assert body["proposal"]["confidence"] == pytest.approx(0.88)
    assert body["state"]["state"] == "pending"
    assert body["state"]["decided_by"] is None
    assert [t["state"] for t in body["history"]] == ["pending"]
    assert body["audit_events"] == []


def test_unknown_proposals_are_json_404s(store: SqliteStore) -> None:
    client = client_for(store)

    json_error(client.get(f"/api/proposals/{'0' * 32}"), 404, "not_found")
    json_error(client.get(f"/api/proposals/{'0' * 32}/audit"), 404, "not_found")


def test_audit_trail_of_an_executed_proposal(store: SqliteStore) -> None:
    pending = seed(store)["pending"]
    client = client_for(store, executor=FakeExecutor(store))
    decide(client, pending.proposal_id)

    body = client.get(f"/api/proposals/{pending.proposal_id}/audit").json()

    assert body["proposal_id"] == pending.proposal_id
    assert body["state"] == "executed"
    assert [t["state"] for t in body["transitions"]] == ["pending", "approved", "executed"]
    assert body["transitions"][1]["actor"] == "api:U0OSEGO"
    assert body["execution_result"] == EXECUTED_RESULT


def test_audit_trail_includes_refused_attempts(store: SqliteStore) -> None:
    pending = seed(store)["pending"]
    client = client_for(store)
    decide(client, pending.proposal_id, approver_id="U0MALLORY")

    body = client.get(f"/api/proposals/{pending.proposal_id}/audit").json()

    [event] = body["audit_events"]
    assert event["kind"] == "unauthorized_decision"
    assert event["actor"] == "api:U0MALLORY"
    assert body["execution_result"] is None


# --- GET /api/findings and /api/runs --------------------------------------------------------------


def test_list_findings_includes_the_noted_ones(store: SqliteStore) -> None:
    seeded = seed(store)

    items = client_for(store).get("/api/findings").json()

    by_mode = {f["failure_mode"]: f for f in items}
    noted = by_mode["profiler_disabled"]
    assert (noted["state"], noted["severity"], noted["proposal_id"]) == ("noted", "info", None)
    index = by_mode["missing_index_collscan"]
    assert (index["state"], index["proposal_id"]) == ("escalated", seeded["pending"].proposal_id)
    assert by_mode["oplog_window_below_resync"]["proposal_id"] == seeded["rejected"].proposal_id
    assert items[0]["failure_mode"] == "profiler_disabled"  # newest first
    assert set(items[0]) == {
        "finding_id", "run_id", "failure_mode", "severity", "node", "summary",
        "horizon_seconds", "state", "proposal_id", "detected_at",
    }


@pytest.mark.parametrize(
    ("query", "count"),
    [("?run_id=r1", 2), ("?run_id=r0", 1), ("?run_id=nope", 0), ("?limit=1", 1), ("", 3)],
)
def test_list_findings_filters(store: SqliteStore, query: str, count: int) -> None:
    seed(store)

    assert len(client_for(store).get(f"/api/findings{query}").json()) == count


def test_list_runs_newest_first(store: SqliteStore) -> None:
    seed(store)

    items = client_for(store).get("/api/runs").json()

    assert [r["run_id"] for r in items] == ["r1", "r0"]
    assert set(items[0]) == {"run_id", "started_at", "finished_at", "findings", "proposals", "errors"}
    assert (items[0]["findings"], items[0]["proposals"], items[0]["errors"]) == (2, 1, [])
    assert items[1]["errors"] == ["collector query_profile: NoReachableNode: no replica set member reachable"]
    assert [r["run_id"] for r in client_for(store).get("/api/runs?limit=1").json()] == ["r1"]


# --- POST /api/proposals/{id}/decision -------------------------------------------------------------


def test_decision_on_loopback_with_an_allowlisted_approver(store: SqliteStore) -> None:
    proposal = proposal_for(oplog_finding(), executable=False)
    store.record_proposal(proposal)
    executor = FakeExecutor(store)

    response = decide(client_for(store, executor=executor), proposal.proposal_id)

    assert response.status_code == 200
    body = response.json()
    assert (body["proposal_id"], body["state"]) == (proposal.proposal_id, "approved")
    assert body["message"]
    record = store.current_state(proposal.proposal_id)
    assert (record.state, record.decided_by) == (ApprovalState.APPROVED, "api:U0OSEGO")
    assert executor.calls == []  # propose-only: a human runs the command


def test_approving_an_executable_proposal_runs_the_executor(store: SqliteStore) -> None:
    pending = seed(store)["pending"]
    executor = FakeExecutor(store)

    response = decide(client_for(store, executor=executor), pending.proposal_id)

    assert response.json()["state"] == "approved"  # execution runs after the response
    assert len(executor.calls) == 1
    assert store.current_state(pending.proposal_id).state is ApprovalState.EXECUTED


def test_reject_needs_no_acknowledgement(store: SqliteStore) -> None:
    pending = seed(store)["pending"]
    executor = FakeExecutor(store)

    response = decide(
        client_for(store, executor=executor), pending.proposal_id, decision="reject", acknowledged=False
    )

    assert response.json()["state"] == "rejected"
    assert executor.calls == []


@pytest.mark.parametrize("bind", ["0.0.0.0", "10.0.0.5", "::", None])
def test_decision_is_refused_unless_bound_to_loopback(
    store: SqliteStore, bind: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    pending = seed(store)["pending"]
    executor = FakeExecutor(store)

    with caplog.at_level(logging.WARNING, logger="bellwether.app"):
        response = decide(client_for(store, bind=bind, executor=executor), pending.proposal_id)

    json_error(response, 403, "approval_not_permitted")
    assert store.current_state(pending.proposal_id).state is ApprovalState.PENDING
    assert executor.calls == []
    assert any("loopback" in r.getMessage() for r in caplog.records)


def test_decision_is_refused_when_disabled(store: SqliteStore) -> None:
    pending = seed(store)["pending"]

    json_error(decide(client_for(store, enabled=False), pending.proposal_id), 403, "approval_not_permitted")
    assert store.current_state(pending.proposal_id).state is ApprovalState.PENDING


def test_a_non_allowlisted_approver_is_refused_and_audited(store: SqliteStore) -> None:
    pending = seed(store)["pending"]
    executor = FakeExecutor(store)

    response = decide(client_for(store, executor=executor), pending.proposal_id, approver_id="U0MALLORY")

    body = json_error(response, 403, "not_authorized")
    assert "not an authorized approver" in body["detail"]
    assert store.current_state(pending.proposal_id).state is ApprovalState.PENDING
    assert executor.calls == []
    [event] = store.list_audit_events(pending.proposal_id)
    assert (event.kind, event.actor) == ("unauthorized_decision", "api:U0MALLORY")


def test_an_already_decided_proposal_is_409(store: SqliteStore) -> None:
    rejected = seed(store)["rejected"]

    body = json_error(decide(client_for(store), rejected.proposal_id), 409, "already_decided")

    assert "rejected" in body["detail"]
    assert store.current_state(rejected.proposal_id).state is ApprovalState.REJECTED


def test_an_unknown_proposal_decision_is_404(store: SqliteStore) -> None:
    json_error(decide(client_for(store), "0" * 32), 404, "not_found")


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b"not json", "malformed_request"),
        (b'["approve"]', "malformed_request"),
        (b'{"decision": "execute_now", "approver_id": "U0OSEGO"}', "malformed_request"),
        (b'{"decision": "approve", "approver_id": ""}', "malformed_request"),
        (b'{"decision": "approve"}', "malformed_request"),
        (b'{"decision": "approve", "approver_id": "U0OSEGO", "acknowledged": false}', "acknowledgement_required"),
        (b'{"decision": "approve", "approver_id": "U0OSEGO"}', "acknowledgement_required"),
    ],
    ids=["not json", "not an object", "unknown decision", "empty approver", "no approver",
         "not acknowledged", "acknowledgement missing"],
)
def test_malformed_decisions_are_json_400s(store: SqliteStore, body: bytes, error: str) -> None:
    pending = seed(store)["pending"]

    response = client_for(store).post(
        f"/api/proposals/{pending.proposal_id}/decision",
        content=body,
        headers={"content-type": "application/json"},
    )

    json_error(response, 400, error)
    assert store.current_state(pending.proposal_id).state is ApprovalState.PENDING
