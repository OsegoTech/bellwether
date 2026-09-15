"""The operations UI and its localhost-gated in-UI approval path.

Views: the overview (GET /), proposal detail (GET /proposals/{id}) in its
pending and decided states. Security: POST /proposals/{id}/decision only works
when UI approval is enabled AND the server is bound to a loopback host, and
only for an approver on approval.approver_ids — refusals are audited exactly
as the Slack path audits them.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bellwether.app import create_app, is_loopback_host
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
UAE = "node-uae.mongo.internal:27017"
BACKUP = "node-backup.mongo.internal:27017"
LABELS = {"claude": "Claude Opus 5", "openai": "GPT-5.6"}
UNKNOWN_LEVEL = "unknown (not readable without dbAdmin; not necessarily off)"
CANDIDATE = [{"field": "account_id", "direction": 1}, {"field": "posted_at", "direction": -1}]


class FakeExecutor:
    def __init__(self, store: SqliteStore) -> None:
        self.store = store
        self.calls: list[tuple[Proposal, ApprovalRecord]] = []

    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord:
        self.calls.append((proposal, approval_record))
        return self.store.record_approval_transition(
            proposal.proposal_id, ApprovalState.EXECUTED, result="created index account_id_1_posted_at_-1"
        )


# --- fixtures ----------------------------------------------------------------------------


def index_finding() -> Finding:
    return Finding(
        signal_class=SignalClass.PERFORMANCE,
        failure_mode="missing_index_collscan",
        severity=Severity.WARNING,
        node=UAE,
        summary="Query shape {account_id: eq, posted_at: range} on meetadev_ledger.transactions "
        "examined 11,800,000 documents to return 12,400.",
        evidence=(
            Evidence("namespace", "meetadev_ledger.transactions"),
            Evidence("docs_examined", 11_800_000, "docs"),
            Evidence("docs_returned", 12_400, "docs"),
            Evidence("targeting_ratio", 951.6),
            Evidence("wasted_bytes", 6_035_251_200, "bytes"),
            Evidence("avg_object_size", 512.0, "bytes"),
            Evidence("candidate_index", CANDIDATE),
            Evidence("collection_doc_count", 40_000, "docs"),
            Evidence("profiler_levels", {UAE: UNKNOWN_LEVEL}),
        ),
        detected_at=NOW - timedelta(minutes=10),
    )


def oplog_finding() -> Finding:
    return Finding(
        signal_class=SignalClass.REPLICATION,
        failure_mode="oplog_window_below_resync",
        severity=Severity.CRITICAL,
        node=BACKUP,
        summary="Oplog window on node-backup is 38 min (2280 s), below the 60 min resync estimate.",
        evidence=(Evidence("oplog_window_seconds", 2280, "s"), Evidence("resync_seconds", 3600, "s")),
        detected_at=NOW - timedelta(hours=3),
    )


def info_finding(mode: str = "profiler_disabled") -> Finding:
    return Finding(
        signal_class=SignalClass.PERFORMANCE,
        failure_mode=mode,
        severity=Severity.INFO,
        node=BACKUP,
        summary=f"{mode} summary on node-backup.",
        evidence=(Evidence("db", "meetadev_ledger"),),
        detected_at=NOW - timedelta(minutes=5),
    )


def proposal_for(finding: Finding, *, executable: bool) -> Proposal:
    if executable:
        action = RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Build ESR index {account_id: 1, posted_at: -1} on meetadev_ledger.transactions",
            command='db.getSiblingDB("meetadev_ledger").transactions.createIndex({ account_id: 1, posted_at: -1 })',
            rationale="The read saving outweighs the write cost of one small index.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "meetadev_ledger",
                "collection": "transactions",
                "keys": CANDIDATE,
                "estimated_docs": 40_000,
            },
        )
    else:
        action = RemediationAction(
            kind=ActionKind.PROPOSE_ONLY,
            title="Grow the oplog to 2560 MB on every member",
            command="db.adminCommand({ replSetResizeOplog: 1, size: 2560 })",
            rationale="A larger oplog restores the window.",
            reversible=True,
        )
    return Proposal(
        finding_id=finding.finding_id,
        failure_mode=finding.failure_mode,
        node=finding.node,
        diagnosis=f"Diagnosis for {finding.failure_mode}: the evidence shows the cause.",
        mechanism=f"Mechanism for {finding.failure_mode}: the MongoDB internal at work.",
        impact_if_ignored=f"Impact for {finding.failure_mode}: what happens if nothing changes.",
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
    """A run with an escalated index finding (pending proposal), an escalated oplog
    finding (rejected proposal), and one INFO finding that was only noted."""
    index, oplog, info = index_finding(), oplog_finding(), info_finding()
    store.record_finding(oplog, run_id="r0", escalated=True)
    store.record_run(RunRecord("r0", NOW - timedelta(hours=3), NOW - timedelta(hours=3), 1, 1, ()))
    for finding, escalated in ((index, True), (info, False)):
        store.record_finding(finding, run_id="r1", escalated=escalated)
    store.record_run(RunRecord("r1", NOW - timedelta(minutes=10), NOW - timedelta(minutes=9), 2, 1, ()))
    pending = proposal_for(index, executable=True)
    rejected = proposal_for(oplog, executable=False)
    store.record_proposal(rejected, at=rejected.created_at)
    store.record_approval_transition(
        rejected.proposal_id, ApprovalState.REJECTED, by="slack:osego (U0OSEGO)",
        at=rejected.created_at + timedelta(minutes=5),
    )
    store.record_proposal(pending, at=pending.created_at)
    return {"pending": pending, "rejected": rejected}


def client_for(
    store: SqliteStore,
    *,
    bind: str | None = "127.0.0.1",
    enabled: bool = True,
    executor: FakeExecutor | None = None,
) -> TestClient:
    app = create_app(
        store,
        signing_secret=SECRET,
        approver_ids=APPROVERS,
        executor=executor,
        clock=NOW.timestamp,
        bind_host=bind,
        ui_approval_enabled=enabled,
        provider_labels=LABELS,
        cluster_label="rs0 · 4 members",
    )
    return TestClient(app)


def decide(client: TestClient, proposal_id: str, **form: str) -> Any:
    fields = {"approver_id": "U0OSEGO", "decision": "approve", "acknowledge": "yes", **form}
    return client.post(f"/proposals/{proposal_id}/decision", data=fields)


# --- View 1: the overview ------------------------------------------------------------------


def test_overview_lists_real_rows_pending_first(store: SqliteStore) -> None:
    seeded = seed(store)

    response = client_for(store).get("/")

    assert response.status_code == 200
    page = response.text
    assert page.index(seeded["pending"].proposal_id) < page.index(seeded["rejected"].proposal_id)
    for label in ("Missing index", "Oplog window shrinking", "Profiling disabled"):
        assert label in page
    for css in ("sev-warning", "sev-critical", "sev-info"):
        assert css in page
    for css in ("state-pending", "state-rejected", "state-noted"):
        assert css in page
    assert page.count('class="row row-pending') == 1  # the one row needing action stands out
    assert "1 needs review · 1 noted" in page
    assert f'href="/proposals/{seeded["pending"].proposal_id}"' in page
    assert "9 min ago" in page


def test_overview_frame(store: SqliteStore) -> None:
    seed(store)

    page = client_for(store).get("/").text

    assert "Bellwether" in page
    assert "rs0 · 4 members" in page
    assert "last run 9 min ago" in page
    assert "<script" not in page  # no JS on the overview


def test_all_clear_when_nothing_needs_review(store: SqliteStore) -> None:
    for mode in ("profiler_disabled", "redundant_index"):
        store.record_finding(info_finding(mode), run_id="r1", escalated=False)
    store.record_run(RunRecord("r1", NOW - timedelta(minutes=6), NOW - timedelta(minutes=5), 2, 0, ()))

    page = client_for(store).get("/").text

    assert "All clear — 2 signals noted, none need action" in page
    assert "Redundant index" in page and "Profiling disabled" in page
    assert 'class="row row-pending' not in page


def test_an_empty_store_is_calm_too(store: SqliteStore) -> None:
    page = client_for(store).get("/").text

    assert "All clear" in page
    assert "no signals noted yet" in page


# --- View 2: proposal detail --------------------------------------------------------------


def test_detail_renders_every_proposal_field(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]

    page = client_for(store).get(f"/proposals/{proposal.proposal_id}").text

    for text in (proposal.diagnosis, proposal.mechanism, proposal.impact_if_ignored, proposal.action.title):
        assert html.escape(text) in page
    assert html.escape(proposal.action.command) in page
    assert "Missing index" in page and UAE in page and "sev-warning" in page
    assert "Claude Opus 5" in page
    assert "88%" in page
    assert "Executable" in page and "Reversible" in page
    for value in ("6,035,251,200", "11,800,000", "12,400", "40,000", "951.6", "5.6 GiB"):
        assert value in page, value
    assert "{account_id: 1, posted_at: -1}" in page
    assert UNKNOWN_LEVEL in page


def test_grounding_callout_is_shown_for_an_executable_index_only(store: SqliteStore) -> None:
    seeded = seed(store)
    client = client_for(store)

    executable = client.get(f"/proposals/{seeded['pending'].proposal_id}").text
    manual = client.get(f"/proposals/{seeded['rejected'].proposal_id}").text

    assert "computed by deterministic code" in executable
    assert "computed by deterministic code" not in manual
    assert "Manual" in manual


def test_decision_block_on_a_pending_proposal_when_permitted(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]

    page = client_for(store).get(f"/proposals/{proposal.proposal_id}").text

    assert f'action="/proposals/{proposal.proposal_id}/decision"' in page
    assert 'method="post"' in page
    assert 'name="approver_id"' in page
    assert 'name="acknowledge"' in page
    assert "confirm(" in page  # the weighty approve asks twice
    assert "formnovalidate" in page  # reject does not need the acknowledgement


def test_no_decision_block_when_bound_publicly(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]

    page = client_for(store, bind="0.0.0.0").get(f"/proposals/{proposal.proposal_id}").text

    assert "<form" not in page
    assert "Slack" in page  # tells the reader where approval happens instead


def test_no_decision_block_on_a_decided_proposal(store: SqliteStore) -> None:
    proposal = seed(store)["rejected"]

    page = client_for(store).get(f"/proposals/{proposal.proposal_id}").text

    assert "<form" not in page


# --- View 3: the decided state -------------------------------------------------------------


def test_decided_proposal_shows_the_audit_timeline(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]
    client = client_for(store, executor=FakeExecutor(store))
    decide(client, proposal.proposal_id)

    page = client.get(f"/proposals/{proposal.proposal_id}").text

    assert "timeline" in page
    for step in ("Proposed", "Approved", "Executed"):
        assert step in page
    assert "ui:U0OSEGO" in page
    assert "created index account_id_1_posted_at_-1" in page
    assert "<form" not in page


def test_a_rejection_is_attributed_in_the_timeline(store: SqliteStore) -> None:
    proposal = seed(store)["rejected"]

    page = client_for(store).get(f"/proposals/{proposal.proposal_id}").text

    assert "Rejected" in page
    assert html.escape("slack:osego (U0OSEGO)") in page


# --- Security: loopback + enabled + allowlist ----------------------------------------------


@pytest.mark.parametrize("bind", ["0.0.0.0", "10.3.2.4", "20.86.152.132", "::", None])
def test_in_ui_decision_is_refused_unless_bound_to_loopback(
    store: SqliteStore, bind: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    proposal = seed(store)["pending"]
    executor = FakeExecutor(store)

    with caplog.at_level(logging.WARNING, logger="bellwether.app"):
        response = decide(client_for(store, bind=bind, executor=executor), proposal.proposal_id)

    assert response.status_code == 403
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert executor.calls == []
    assert any("loopback" in r.getMessage() for r in caplog.records)


def test_in_ui_decision_is_refused_when_disabled(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]

    response = decide(client_for(store, enabled=False), proposal.proposal_id)

    assert response.status_code == 403
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.3.2.4", False),
        ("node-uae.mongo.internal", False),
        ("", False),
        (None, False),
    ],
)
def test_loopback_detection(host: str | None, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


def test_allowlisted_approver_on_loopback_approves(store: SqliteStore) -> None:
    finding = oplog_finding()
    proposal = proposal_for(finding, executable=False)
    store.record_proposal(proposal)
    executor = FakeExecutor(store)

    response = decide(client_for(store, executor=executor), proposal.proposal_id)

    assert response.status_code == 200  # 303 back to the detail page, followed
    assert response.url.path == f"/proposals/{proposal.proposal_id}"
    record = store.current_state(proposal.proposal_id)
    assert record.state is ApprovalState.APPROVED
    assert record.decided_by == "ui:U0OSEGO"
    assert executor.calls == []  # propose-only: a human runs the command


def test_approving_an_executable_proposal_runs_the_executor(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]
    executor = FakeExecutor(store)

    decide(client_for(store, executor=executor), proposal.proposal_id)

    assert len(executor.calls) == 1
    assert executor.calls[0][1].state is ApprovalState.APPROVED
    assert store.current_state(proposal.proposal_id).state is ApprovalState.EXECUTED


def test_a_non_allowlisted_approver_is_refused_and_audited(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]
    executor = FakeExecutor(store)

    response = decide(client_for(store, executor=executor), proposal.proposal_id, approver_id="U0MALLORY")

    assert response.status_code == 403
    assert "not an authorized approver" in response.text
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert len(store.history(proposal.proposal_id)) == 1
    assert executor.calls == []
    [event] = store.list_audit_events(proposal.proposal_id)
    assert event.kind == "unauthorized_decision"
    assert event.actor == "ui:U0MALLORY"
    assert "approve" in event.detail


def test_reject(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]
    executor = FakeExecutor(store)

    decide(client_for(store, executor=executor), proposal.proposal_id, decision="reject", acknowledge="")

    assert store.current_state(proposal.proposal_id).state is ApprovalState.REJECTED
    assert executor.calls == []


def test_approve_needs_the_acknowledgement(store: SqliteStore) -> None:
    proposal = seed(store)["pending"]

    response = decide(client_for(store), proposal.proposal_id, acknowledge="")

    assert response.status_code == 400
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


def test_deciding_a_decided_proposal_is_a_clean_error_page(store: SqliteStore) -> None:
    proposal = seed(store)["rejected"]

    response = decide(client_for(store), proposal.proposal_id)

    assert response.status_code == 409
    assert response.headers["content-type"].startswith("text/html")
    assert "already rejected" in response.text
    assert store.current_state(proposal.proposal_id).state is ApprovalState.REJECTED


def test_deciding_an_unknown_proposal_is_404(store: SqliteStore) -> None:
    response = decide(client_for(store), "0" * 32)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize(
    "form",
    [{"decision": "execute_now"}, {"approver_id": ""}, {"approver_id": "   "}],
    ids=["unknown decision", "empty approver", "blank approver"],
)
def test_malformed_decisions_are_400(store: SqliteStore, form: dict[str, str]) -> None:
    proposal = seed(store)["pending"]

    response = decide(client_for(store), proposal.proposal_id, **form)

    assert response.status_code == 400
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


def test_ui_pages_escape_model_text(store: SqliteStore) -> None:
    finding = oplog_finding()
    proposal = Proposal(
        **{
            **proposal_for(finding, executable=False).__dict__,
            "diagnosis": "<script>alert(1)</script>",
        }
    )
    store.record_proposal(proposal)

    page = client_for(store).get(f"/proposals/{proposal.proposal_id}").text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
