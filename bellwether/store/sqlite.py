"""Append-only SQLite audit store.

Tables: ``proposals``, ``approvals``, ``runs`` (BUILD_SPEC §3.6), plus
``findings`` so the pipeline can record findings that were not analyzed (below
the token gate, or with every provider down) — §3.5/§3.10 require them stored —
and ``audit_events`` for things worth recording that change no state, such as
a Slack click refused because the user is not in ``approval.approver_ids``.

Append-only is enforced by the database, not by convention: every table has
triggers that abort any UPDATE or DELETE. A proposal is written once. Its
approval state is a sequence of rows in ``approvals``; the current state is
the latest by timestamp (ties broken by insertion order). Legal moves:

    pending  -> approved | rejected | expired
    approved -> executed | failed

Each transition is checked and inserted inside ``BEGIN IMMEDIATE``, so two
concurrent approvers cannot both move the same proposal out of PENDING.
Connections are per operation, which keeps the store safe to share across the
approval endpoint's worker threads and the timer-driven ``run`` process.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bellwether.models import ApprovalRecord, ApprovalState, Finding, Proposal, Severity
from bellwether.store.codec import (
    dt_from_str,
    dt_to_str,
    finding_to_json,
    proposal_from_json,
    proposal_to_json,
)

logger = logging.getLogger(__name__)

_TABLES = ("proposals", "approvals", "findings", "runs", "audit_events")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id  TEXT PRIMARY KEY,
    finding_id   TEXT NOT NULL,
    failure_mode TEXT NOT NULL,
    node         TEXT NOT NULL,
    provider     TEXT NOT NULL,
    action_kind  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    body         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   TEXT NOT NULL,
    proposal_id TEXT NOT NULL REFERENCES proposals (proposal_id),
    state       TEXT NOT NULL,
    actor       TEXT,
    at          TEXT NOT NULL,
    result      TEXT
);
CREATE INDEX IF NOT EXISTS approvals_by_proposal ON approvals (proposal_id, at, seq);
CREATE TABLE IF NOT EXISTS findings (
    finding_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    failure_mode    TEXT NOT NULL,
    severity        TEXT NOT NULL,
    node            TEXT NOT NULL,
    summary         TEXT NOT NULL,
    horizon_seconds INTEGER,
    escalated       INTEGER NOT NULL,
    detected_at     TEXT NOT NULL,
    body            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    findings    INTEGER NOT NULL,
    proposals   INTEGER NOT NULL,
    errors      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    proposal_id TEXT,
    actor       TEXT,
    detail      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_events_by_proposal ON audit_events (proposal_id, at, seq);
""" + "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, 'append-only table: {table}'); END;
CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, 'append-only table: {table}'); END;
"""
    for table in _TABLES
)

_LEGAL: dict[ApprovalState, frozenset[ApprovalState]] = {
    ApprovalState.PENDING: frozenset(
        {ApprovalState.APPROVED, ApprovalState.REJECTED, ApprovalState.EXPIRED}
    ),
    ApprovalState.APPROVED: frozenset({ApprovalState.EXECUTED, ApprovalState.FAILED}),
}
_NEEDS_DECIDER = frozenset({ApprovalState.APPROVED, ApprovalState.REJECTED})
_DECISIONS = frozenset({ApprovalState.APPROVED, ApprovalState.REJECTED, ApprovalState.EXPIRED})
_OUTCOMES = frozenset({ApprovalState.EXECUTED, ApprovalState.FAILED})


class InvalidTransition(Exception):
    """The requested approval transition is not allowed from the current state."""


class ProposalNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Transition:
    seq: int
    record_id: str
    state: ApprovalState
    actor: str | None
    at: datetime
    result: str | None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    started_at: datetime
    finished_at: datetime
    findings: int
    proposals: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FindingRow:
    finding_id: str
    run_id: str
    failure_mode: str
    severity: Severity
    node: str
    summary: str
    horizon_seconds: int | None
    escalated: bool
    detected_at: datetime


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    at: datetime
    kind: str
    proposal_id: str | None
    actor: str | None
    detail: str


class SqliteStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # --- proposals and approvals ------------------------------------------------

    def record_proposal(self, proposal: Proposal, *, at: datetime | None = None) -> ApprovalRecord:
        """Store a proposal with its initial PENDING transition, atomically."""
        record = ApprovalRecord(proposal_id=proposal.proposal_id)
        when = dt_to_str(at or datetime.now(timezone.utc))
        with self._write() as conn:
            conn.execute(
                "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal.proposal_id,
                    proposal.finding_id,
                    proposal.failure_mode,
                    proposal.node,
                    proposal.provider,
                    proposal.action.kind.value,
                    dt_to_str(proposal.created_at),
                    proposal_to_json(proposal),
                ),
            )
            conn.execute(
                "INSERT INTO approvals (record_id, proposal_id, state, actor, at, result) "
                "VALUES (?, ?, ?, NULL, ?, NULL)",
                (record.record_id, proposal.proposal_id, ApprovalState.PENDING.value, when),
            )
            rows = _transitions(conn, proposal.proposal_id)
        logger.info(
            "proposal recorded",
            extra={"proposal_id": proposal.proposal_id, "finding_id": proposal.finding_id},
        )
        return _reconstruct(proposal.proposal_id, rows)

    def record_approval_transition(
        self,
        proposal_id: str,
        state: ApprovalState,
        *,
        by: str | None = None,
        result: str | None = None,
        at: datetime | None = None,
    ) -> ApprovalRecord:
        when = at or datetime.now(timezone.utc)
        with self._write() as conn:
            rows = _transitions(conn, proposal_id)
            if not rows:
                raise ProposalNotFound(proposal_id)
            latest = rows[-1]
            if state not in _LEGAL.get(latest.state, frozenset()):
                raise InvalidTransition(
                    f"proposal {proposal_id}: {latest.state.value} -> {state.value} is not allowed"
                )
            if state in _NEEDS_DECIDER and not by:
                raise InvalidTransition(f"{state.value} requires by=<who decided>")
            if when < latest.at:
                raise InvalidTransition(
                    f"transition at {dt_to_str(when)} predates the latest ({dt_to_str(latest.at)})"
                )
            conn.execute(
                "INSERT INTO approvals (record_id, proposal_id, state, actor, at, result) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (latest.record_id, proposal_id, state.value, by, dt_to_str(when), result),
            )
            rows = _transitions(conn, proposal_id)
        logger.info(
            "approval transition",
            extra={
                "proposal_id": proposal_id,
                "from_state": latest.state.value,
                "to_state": state.value,
                "actor": by,
            },
        )
        return _reconstruct(proposal_id, rows)

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT body FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        return None if row is None else proposal_from_json(row["body"])

    def current_state(self, proposal_id: str) -> ApprovalRecord:
        with self._read() as conn:
            rows = _transitions(conn, proposal_id)
        if not rows:
            raise ProposalNotFound(proposal_id)
        return _reconstruct(proposal_id, rows)

    def history(self, proposal_id: str) -> list[Transition]:
        with self._read() as conn:
            return _transitions(conn, proposal_id)

    def list_proposals(self) -> list[tuple[Proposal, ApprovalRecord]]:
        """Every proposal with its current state, oldest first."""
        return self._proposals_where("", ())

    def list_pending(self) -> list[Proposal]:
        return [p for p, record in self.list_proposals() if record.state is ApprovalState.PENDING]

    def has_pending(self, failure_mode: str, node: str, subject: str | None = None) -> bool:
        """Whether an undecided proposal already covers this failure mode on this node.

        With `subject` (a finding's ``subject`` evidence — one query shape, one
        index), only a pending proposal for that same subject counts: a node can
        carry many index findings at once, and each deserves its own proposal.
        """
        matches = self._proposals_where("WHERE failure_mode = ? AND node = ?", (failure_mode, node))
        return any(
            record.state is ApprovalState.PENDING
            and (subject is None or _subject(proposal) == subject)
            for proposal, record in matches
        )

    # --- findings and runs --------------------------------------------------------

    def record_finding(self, finding: Finding, *, run_id: str, escalated: bool) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding.finding_id,
                    run_id,
                    finding.failure_mode,
                    finding.severity.value,
                    finding.node,
                    finding.summary,
                    finding.horizon_seconds,
                    int(escalated),
                    dt_to_str(finding.detected_at),
                    finding_to_json(finding),
                ),
            )

    def list_findings(self, run_id: str | None = None) -> list[FindingRow]:
        query = "SELECT * FROM findings"
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        with self._read() as conn:
            rows = conn.execute(query + " ORDER BY detected_at, rowid", params).fetchall()
        return [
            FindingRow(
                finding_id=r["finding_id"],
                run_id=r["run_id"],
                failure_mode=r["failure_mode"],
                severity=Severity(r["severity"]),
                node=r["node"],
                summary=r["summary"],
                horizon_seconds=r["horizon_seconds"],
                escalated=bool(r["escalated"]),
                detected_at=dt_from_str(r["detected_at"]),
            )
            for r in rows
        ]

    def record_run(self, run: RunRecord) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    dt_to_str(run.started_at),
                    dt_to_str(run.finished_at),
                    run.findings,
                    run.proposals,
                    json.dumps(list(run.errors)),
                ),
            )

    def list_runs(self) -> list[RunRecord]:
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at, rowid").fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                started_at=dt_from_str(r["started_at"]),
                finished_at=dt_from_str(r["finished_at"]),
                findings=r["findings"],
                proposals=r["proposals"],
                errors=tuple(json.loads(r["errors"])),
            )
            for r in rows
        ]

    # --- audit events ---------------------------------------------------------------

    def record_audit_event(
        self,
        kind: str,
        *,
        detail: str,
        proposal_id: str | None = None,
        actor: str | None = None,
        at: datetime | None = None,
    ) -> AuditEvent:
        """Record something that happened without changing any proposal's state."""
        when = dt_to_str(at or datetime.now(timezone.utc))
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO audit_events (at, kind, proposal_id, actor, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (when, kind, proposal_id, actor, detail),
            )
            seq = cursor.lastrowid
        if seq is None:
            raise RuntimeError("audit event insert returned no row id")
        logger.info(
            "audit event recorded",
            extra={"event_kind": kind, "proposal_id": proposal_id, "actor": actor},
        )
        return AuditEvent(
            seq=seq,
            at=dt_from_str(when),
            kind=kind,
            proposal_id=proposal_id,
            actor=actor,
            detail=detail,
        )

    def list_audit_events(self, proposal_id: str | None = None) -> list[AuditEvent]:
        query = "SELECT * FROM audit_events"
        params: tuple[str, ...] = ()
        if proposal_id is not None:
            query += " WHERE proposal_id = ?"
            params = (proposal_id,)
        with self._read() as conn:
            rows = conn.execute(query + " ORDER BY at, seq", params).fetchall()
        return [
            AuditEvent(
                seq=r["seq"],
                at=dt_from_str(r["at"]),
                kind=r["kind"],
                proposal_id=r["proposal_id"],
                actor=r["actor"],
                detail=r["detail"],
            )
            for r in rows
        ]

    # --- plumbing -------------------------------------------------------------------

    def _proposals_where(
        self, clause: str, params: Sequence[str]
    ) -> list[tuple[Proposal, ApprovalRecord]]:
        with self._read() as conn:
            proposals = conn.execute(
                f"SELECT proposal_id, body FROM proposals {clause} ORDER BY created_at, rowid",
                tuple(params),
            ).fetchall()
            return [
                (
                    proposal_from_json(row["body"]),
                    _reconstruct(row["proposal_id"], _transitions(conn, row["proposal_id"])),
                )
                for row in proposals
            ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()


def _subject(proposal: Proposal) -> object:
    return next((e.value for e in proposal.evidence_refs if e.name == "subject"), None)


def _transitions(conn: sqlite3.Connection, proposal_id: str) -> list[Transition]:
    rows = conn.execute(
        "SELECT seq, record_id, state, actor, at, result FROM approvals "
        "WHERE proposal_id = ? ORDER BY at, seq",
        (proposal_id,),
    ).fetchall()
    return [
        Transition(
            seq=r["seq"],
            record_id=r["record_id"],
            state=ApprovalState(r["state"]),
            actor=r["actor"],
            at=dt_from_str(r["at"]),
            result=r["result"],
        )
        for r in rows
    ]


def _reconstruct(proposal_id: str, rows: Sequence[Transition]) -> ApprovalRecord:
    """Current state = the latest transition; decision and outcome from their rows."""
    decision = next((t for t in reversed(rows) if t.state in _DECISIONS), None)
    outcome = next((t for t in reversed(rows) if t.state in _OUTCOMES), None)
    return ApprovalRecord(
        proposal_id=proposal_id,
        state=rows[-1].state,
        decided_by=decision.actor if decision else None,
        decided_at=decision.at if decision else None,
        executed_at=outcome.at if outcome else None,
        execution_result=outcome.result if outcome else None,
        created_at=rows[0].at,
        record_id=rows[0].record_id,
    )
