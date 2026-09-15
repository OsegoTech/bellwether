"""Acceptance tests for the append-only audit store — BUILD_SPEC §3.6.

Spec acceptance:
  - a proposal round-trips
  - an approval transition sequence (pending -> approved -> executed)
    reconstructs to EXECUTED
  - listing pending excludes decided proposals
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from bellwether.models import (
    ActionKind,
    ApprovalState,
    Evidence,
    Finding,
    Proposal,
    RemediationAction,
    Severity,
    SignalClass,
)
from bellwether.store.sqlite import InvalidTransition, ProposalNotFound, RunRecord, SqliteStore

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
NODE = "mongo-hidden.example.internal:27017"


def make_proposal(**overrides: Any) -> Proposal:
    values: dict[str, Any] = {
        "finding_id": "f" * 32,
        "failure_mode": "oplog_window_below_resync",
        "node": NODE,
        "diagnosis": "Window 40 min < resync 60 min.",
        "mechanism": "Capped oplog truncates oldest entries.",
        "impact_if_ignored": "Secondary needs initial sync after maintenance.",
        "action": RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Index transactions.account_id",
            command="db.transactions.createIndex({account_id: 1})",
            rationale="Slow queries filter on account_id.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "appdb",
                "collection": "transactions",
                "keys": [{"field": "account_id", "direction": 1}],
                "estimated_docs": 40_000,
            },
        ),
        "confidence": 0.82,
        "provider": "claude",
        "evidence_refs": (
            Evidence("oplog_window_seconds", 2400, "s", observed_at=T0),
            Evidence("write_rate_source", "oplog_mean", observed_at=T0),
            Evidence("safety_factor", 2.0, observed_at=T0),
        ),
        "created_at": T0,
    }
    values.update(overrides)
    return Proposal(**values)


def make_finding(**overrides: Any) -> Finding:
    values: dict[str, Any] = {
        "signal_class": SignalClass.REPLICATION,
        "failure_mode": "oplog_window_below_resync",
        "severity": Severity.WARNING,
        "node": NODE,
        "summary": "Oplog window is 90 min.",
        "evidence": (Evidence("oplog_window_seconds", 5400, "s", observed_at=T0),),
        "horizon_seconds": 900,
    }
    values.update(overrides)
    return Finding(**values)


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bellwether.db")


# --- Spec acceptance ---------------------------------------------------------


def test_proposal_round_trips(store: SqliteStore) -> None:
    proposal = make_proposal()

    store.record_proposal(proposal)

    assert store.get_proposal(proposal.proposal_id) == proposal


def test_propose_only_proposal_round_trips(store: SqliteStore) -> None:
    proposal = make_proposal(
        action=RemediationAction(
            kind=ActionKind.PROPOSE_ONLY,
            title="Grow the oplog",
            command="db.adminCommand({replSetResizeOplog: 1, size: 51200})",
            rationale="Restore the window.",
            reversible=True,
        ),
        evidence_refs=(),
    )

    store.record_proposal(proposal)

    assert store.get_proposal(proposal.proposal_id) == proposal


def test_pending_approved_executed_reconstructs_executed(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    store.record_approval_transition(proposal.proposal_id, ApprovalState.APPROVED, by="cli:osego")
    store.record_approval_transition(
        proposal.proposal_id, ApprovalState.EXECUTED, result="index account_id_1 built"
    )

    record = store.current_state(proposal.proposal_id)
    assert record.state is ApprovalState.EXECUTED
    assert record.proposal_id == proposal.proposal_id
    assert record.decided_by == "cli:osego"
    assert record.decided_at is not None
    assert record.executed_at is not None
    assert record.executed_at >= record.decided_at
    assert record.execution_result == "index account_id_1 built"


def test_list_pending_excludes_decided(store: SqliteStore) -> None:
    pending = make_proposal(created_at=T0)
    approved = make_proposal(created_at=T0 + timedelta(seconds=1))
    rejected = make_proposal(created_at=T0 + timedelta(seconds=2))
    executed = make_proposal(created_at=T0 + timedelta(seconds=3))
    later_pending = make_proposal(created_at=T0 + timedelta(seconds=4))
    for p in (pending, approved, rejected, executed, later_pending):
        store.record_proposal(p)
    store.record_approval_transition(approved.proposal_id, ApprovalState.APPROVED, by="a")
    store.record_approval_transition(rejected.proposal_id, ApprovalState.REJECTED, by="b")
    store.record_approval_transition(executed.proposal_id, ApprovalState.APPROVED, by="c")
    store.record_approval_transition(executed.proposal_id, ApprovalState.EXECUTED, result="ok")

    listed = store.list_pending()

    assert [p.proposal_id for p in listed] == [pending.proposal_id, later_pending.proposal_id]


# --- Append-only semantics ------------------------------------------------------


def test_new_proposal_starts_pending(store: SqliteStore) -> None:
    proposal = make_proposal()

    record = store.record_proposal(proposal)

    assert record.state is ApprovalState.PENDING
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert record.decided_by is None


def test_transitions_are_new_rows_in_order(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    store.record_approval_transition(proposal.proposal_id, ApprovalState.APPROVED, by="x")
    store.record_approval_transition(proposal.proposal_id, ApprovalState.FAILED, result="boom")

    history = store.history(proposal.proposal_id)

    assert [t.state for t in history] == [
        ApprovalState.PENDING,
        ApprovalState.APPROVED,
        ApprovalState.FAILED,
    ]
    assert history[1].actor == "x"
    assert history[2].result == "boom"
    assert store.get_proposal(proposal.proposal_id) == proposal  # untouched


def test_latest_timestamp_wins(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal, at=T0)
    store.record_approval_transition(
        proposal.proposal_id, ApprovalState.APPROVED, by="x", at=T0 + timedelta(minutes=5)
    )

    record = store.current_state(proposal.proposal_id)

    assert record.state is ApprovalState.APPROVED
    assert record.decided_at == T0 + timedelta(minutes=5)
    assert record.created_at == T0


def test_transition_cannot_predate_the_latest(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal, at=T0 + timedelta(hours=1))

    with pytest.raises(InvalidTransition):
        store.record_approval_transition(
            proposal.proposal_id, ApprovalState.APPROVED, by="x", at=T0
        )


@pytest.mark.parametrize("table", ["proposals", "approvals", "findings", "runs", "audit_events"])
def test_tables_reject_update_and_delete(store: SqliteStore, tmp_path: Path, table: str) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    store.record_finding(make_finding(), run_id="r1", escalated=True)
    store.record_run(
        RunRecord(run_id="r1", started_at=T0, finished_at=T0, findings=1, proposals=1, errors=())
    )
    store.record_audit_event("unauthorized_decision", proposal_id=proposal.proposal_id, detail="x")

    conn = sqlite3.connect(tmp_path / "bellwether.db")
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"UPDATE {table} SET rowid = rowid")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    finally:
        conn.close()


def test_audit_events_are_recorded_without_changing_state(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    event = store.record_audit_event(
        "unauthorized_decision",
        proposal_id=proposal.proposal_id,
        actor="slack:mallory (U0MALLORY)",
        detail="approve refused: U0MALLORY is not in approval.approver_ids",
    )
    store.record_audit_event("unauthorized_decision", detail="no proposal attached")

    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert len(store.history(proposal.proposal_id)) == 1
    [recorded] = store.list_audit_events(proposal.proposal_id)
    assert recorded == event
    assert recorded.kind == "unauthorized_decision"
    assert recorded.actor == "slack:mallory (U0MALLORY)"
    assert len(store.list_audit_events()) == 2


def test_store_has_no_mutating_api() -> None:
    public = [name for name in dir(SqliteStore) if not name.startswith("_")]

    for name in public:
        assert not any(verb in name for verb in ("update", "delete", "remove", "drop", "edit")), name


# --- Transition rules ------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        ([], ApprovalState.EXECUTED),  # pending -> executed skips approval
        ([], ApprovalState.FAILED),
        ([], ApprovalState.PENDING),
        ([ApprovalState.REJECTED], ApprovalState.APPROVED),
        ([ApprovalState.EXPIRED], ApprovalState.APPROVED),
        ([ApprovalState.APPROVED], ApprovalState.REJECTED),
        ([ApprovalState.APPROVED, ApprovalState.EXECUTED], ApprovalState.EXECUTED),
        ([ApprovalState.APPROVED, ApprovalState.FAILED], ApprovalState.EXECUTED),
    ],
)
def test_illegal_transitions_raise(
    store: SqliteStore, path: list[ApprovalState], illegal: ApprovalState
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    for state in path:
        store.record_approval_transition(proposal.proposal_id, state, by="x")

    with pytest.raises(InvalidTransition):
        store.record_approval_transition(proposal.proposal_id, illegal, by="x")

    assert len(store.history(proposal.proposal_id)) == 1 + len(path)


@pytest.mark.parametrize("state", [ApprovalState.APPROVED, ApprovalState.REJECTED])
def test_decisions_require_a_decider(store: SqliteStore, state: ApprovalState) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    with pytest.raises(InvalidTransition, match="by"):
        store.record_approval_transition(proposal.proposal_id, state)


def test_unknown_proposal(store: SqliteStore) -> None:
    assert store.get_proposal("missing") is None
    with pytest.raises(ProposalNotFound):
        store.current_state("missing")
    with pytest.raises(ProposalNotFound):
        store.record_approval_transition("missing", ApprovalState.APPROVED, by="x")


def test_duplicate_proposal_is_refused(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    with pytest.raises(sqlite3.IntegrityError):
        store.record_proposal(proposal)


def test_concurrent_approvals_admit_exactly_one(store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    barrier = threading.Barrier(4)
    outcomes: list[str] = []
    lock = threading.Lock()

    def approve(who: str) -> None:
        barrier.wait()
        try:
            store.record_approval_transition(proposal.proposal_id, ApprovalState.APPROVED, by=who)
            result = "ok"
        except InvalidTransition:
            result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=approve, args=(f"u{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["ok", "refused", "refused", "refused"]
    assert len(store.history(proposal.proposal_id)) == 2


# --- Listings, findings, runs --------------------------------------------------


def test_list_proposals_with_state(store: SqliteStore) -> None:
    a = make_proposal(created_at=T0)
    b = make_proposal(created_at=T0 + timedelta(seconds=1))
    store.record_proposal(a)
    store.record_proposal(b)
    store.record_approval_transition(b.proposal_id, ApprovalState.REJECTED, by="x")

    listed = store.list_proposals()

    assert [(p.proposal_id, r.state) for p, r in listed] == [
        (a.proposal_id, ApprovalState.PENDING),
        (b.proposal_id, ApprovalState.REJECTED),
    ]


def test_has_pending_for_failure_mode_and_node(store: SqliteStore) -> None:
    proposal = make_proposal()
    assert not store.has_pending("oplog_window_below_resync", NODE)

    store.record_proposal(proposal)
    assert store.has_pending("oplog_window_below_resync", NODE)
    assert not store.has_pending("oplog_window_below_resync", "other:27017")

    store.record_approval_transition(proposal.proposal_id, ApprovalState.REJECTED, by="x")
    assert not store.has_pending("oplog_window_below_resync", NODE)


def test_has_pending_can_be_narrowed_to_a_subject(store: SqliteStore) -> None:
    # One node can carry many index findings: one pending proposal per query
    # shape, not one per failure mode.
    proposal = make_proposal(
        failure_mode="missing_index_collscan",
        evidence_refs=(Evidence("subject", "appdb.transactions#aaaa", observed_at=T0),),
    )
    store.record_proposal(proposal)

    assert store.has_pending("missing_index_collscan", NODE)
    assert store.has_pending("missing_index_collscan", NODE, subject="appdb.transactions#aaaa")
    assert not store.has_pending("missing_index_collscan", NODE, subject="appdb.transactions#bbbb")


def test_a_subject_identifies_the_finding_whatever_the_node(store: SqliteStore) -> None:
    # A query shape or an index belongs to the replica set: the member a finding
    # was placed on can change between runs without it being a new finding.
    proposal = make_proposal(
        failure_mode="missing_index_collscan",
        node="mongo-2.example.internal:27017",
        evidence_refs=(Evidence("subject", "appdb.transactions#aaaa", observed_at=T0),),
    )
    store.record_proposal(proposal)

    assert store.has_pending(
        "missing_index_collscan", NODE, subject="appdb.transactions#aaaa"
    )
    assert not store.has_pending("missing_index_collscan", NODE)  # no subject: node still counts


def test_findings_are_recorded(store: SqliteStore) -> None:
    finding = make_finding()

    store.record_finding(finding, run_id="r1", escalated=False)

    rows = store.list_findings()
    assert len(rows) == 1
    row = rows[0]
    assert row.finding_id == finding.finding_id
    assert row.run_id == "r1"
    assert row.severity is Severity.WARNING
    assert row.escalated is False
    assert row.summary == finding.summary


def test_runs_round_trip(store: SqliteStore) -> None:
    run = RunRecord(
        run_id="r1",
        started_at=T0,
        finished_at=T0 + timedelta(seconds=12),
        findings=2,
        proposals=1,
        errors=("collector cache: timeout",),
    )

    store.record_run(run)

    assert store.list_runs() == [run]


def test_data_survives_a_new_store_instance(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "bellwether.db"
    proposal = make_proposal()
    SqliteStore(path).record_proposal(proposal)

    reopened = SqliteStore(path)

    assert reopened.get_proposal(proposal.proposal_id) == proposal
    assert reopened.current_state(proposal.proposal_id).state is ApprovalState.PENDING
