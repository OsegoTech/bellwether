"""Acceptance tests for the Index Advisor in the pipeline — INDEX_ADVISOR_SPEC §G.

Spec acceptance:
  - a pipeline run against a mocked cluster with a COLLSCAN-heavy profiler
    produces a stored missing_index_collscan proposal
  - a run with profiling off produces a stored profiler_disabled finding that
    is NOT analyzed
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from bellwether import pipeline
from bellwether.analysis.analyst import Analyst
from bellwether.analysis.provider import ProviderChain
from bellwether.collectors.query_profile import QueryProfileCollector
from bellwether.config import BellwetherConfig, load_config
from bellwether.detectors.index_advisor import (
    MISSING_INDEX_COLLSCAN,
    PROFILER_DISABLED,
    REDUNDANT_INDEX,
    IndexAdvisorDetector,
)
from bellwether.models import ApprovalState, Severity
from bellwether.mongo import ReadOnlyMongo
from bellwether.pipeline import run_once
from bellwether.store.sqlite import SqliteStore
from tests.fakes import PROVIDER_KEYS, FakeCluster, StaticProvider, oplog_cluster, write_config

DB = "meetadev_ledger"
HEALTHY_OPLOG = 6 * 3600

INDEX_PAYLOAD: dict[str, Any] = {
    "diagnosis": "Every ledger lookup by account and status scans the whole collection.",
    "mechanism": "No index serves the equality predicates or the sort, so the query is a COLLSCAN.",
    "impact_if_ignored": "Latency grows with the collection and the scans compete for cache.",
    "action": {
        "kind": "propose_only",
        "title": "Build the ESR candidate index",
        "command": "db.transactions.createIndex({ account_id: 1, status: 1, posted_at: -1 })",
        "rationale": "2.4M documents: over the executor threshold, so a human builds it off-peak.",
        "reversible": True,
        "executor_op": None,
        "executor_args": None,
    },
    "confidence": 0.8,
}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("BELLWETHER_"):
            monkeypatch.delenv(name)
    for name, value in PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)


def find_doc(coll: str, filter: dict[str, Any], sort: dict[str, Any]) -> dict[str, Any]:
    """A MongoDB 7.0 system.profile entry for a slow COLLSCAN find."""
    return {
        "op": "query",
        "ns": f"{DB}.{coll}",
        "command": {"find": coll, "filter": filter, "sort": sort, "limit": 50, "$db": DB},
        "keysExamined": 0,
        "docsExamined": 1_250_000,
        "nreturned": 50,
        "planSummary": "COLLSCAN",
        "millis": 1_840,
        "responseLength": 21_450,
        "ts": datetime(2026, 9, 15, 11, 0),
        "client": "10.1.1.4",
    }


HOT_QUERY = find_doc(
    "transactions",
    {"account_id": "ACC-9931", "status": "posted", "posted_at": {"$gte": datetime(2026, 8, 17)}},
    {"posted_at": -1},
)
OWNER_QUERY = find_doc("accounts", {"owner": "Jane Doe"}, {})


def profiled_cluster(
    *,
    level: int = 1,
    docs: list[dict[str, Any]] | None = None,
    index_stats: list[dict[str, Any]] | None = None,
) -> FakeCluster:
    cluster = oplog_cluster(HEALTHY_OPLOG)
    cluster.reply(
        "listDatabases",
        {"databases": [{"name": "admin"}, {"name": "local"}, {"name": DB}], "ok": 1.0},
    )
    cluster.reply(f"{DB}.profile", {"was": level, "slowms": 100, "ok": 1.0})
    cluster.collections[f"{DB}.system.profile"] = list(docs if docs is not None else [HOT_QUERY])
    stats = index_stats or [
        {"name": "_id_", "key": {"_id": 1}, "accesses": {"ops": 900, "since": datetime(2026, 8, 1)}, "spec": {}}
    ]
    cluster.aggregations["$indexStats"] = lambda node, ns, pipeline: [dict(s) for s in stats]

    def coll_stats(node: str, ns: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if ns == "local.oplog.rs":
            return [{"storageStats": {"maxSize": 990 * 2**20, "size": 512 * 2**20}}]
        return [{"ns": ns, "storageStats": {"count": 2_400_000, "avgObjSize": 436, "size": 1_046_400_000}}]

    cluster.aggregations["$collStats"] = coll_stats
    return cluster


def config_in(tmp_path: Path) -> BellwetherConfig:
    return load_config(write_config(tmp_path))


def run(config: BellwetherConfig, cluster: FakeCluster, provider: StaticProvider) -> pipeline.RunSummary:
    return run_once(
        config,
        mongo=ReadOnlyMongo(config.mongo, client_factory=cluster.factory()),
        analyst=Analyst(ProviderChain([provider], max_retries=1)),
        notifiers=[],
    )


# --- Wiring ----------------------------------------------------------------------------


def test_the_index_advisor_is_registered_beside_the_oplog_pair(tmp_path: Path) -> None:
    config = config_in(tmp_path)

    assert QueryProfileCollector in {type(c) for c in pipeline.build_collectors(config)}
    assert IndexAdvisorDetector in {type(d) for d in pipeline.build_detectors(config)}
    assert len(pipeline.build_collectors(config)) == 2
    assert len(pipeline.build_detectors(config)) == 2


# --- Spec acceptance -----------------------------------------------------------------------


def test_a_collscan_heavy_profiler_produces_a_stored_missing_index_proposal(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", INDEX_PAYLOAD)

    summary = run(config, profiled_cluster(), claude)

    assert summary.errors == []
    assert [f.failure_mode for f in summary.findings] == [MISSING_INDEX_COLLSCAN]
    [proposal] = summary.proposals
    assert proposal.failure_mode == MISSING_INDEX_COLLSCAN
    assert claude.calls == 1
    store = SqliteStore(config.store.sqlite_path)
    assert store.get_proposal(proposal.proposal_id) == proposal
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    evidence = {e.name: e.value for e in proposal.evidence_refs}
    assert evidence["candidate_index"] == [
        {"field": "account_id", "direction": 1},
        {"field": "status", "direction": 1},
        {"field": "posted_at", "direction": -1},
    ]
    [row] = store.list_findings(summary.run_id)
    assert (row.failure_mode, row.escalated) == (MISSING_INDEX_COLLSCAN, True)


def test_profiling_off_is_stored_and_not_analyzed(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", INDEX_PAYLOAD)

    summary = run(config, profiled_cluster(level=0), claude)

    assert claude.calls == 0  # INFO: under the token gate
    assert summary.proposals == []
    [row] = SqliteStore(config.store.sqlite_path).list_findings(summary.run_id)
    assert (row.failure_mode, row.severity, row.escalated) == (PROFILER_DISABLED, Severity.INFO, False)


# --- Token gate and dedup ----------------------------------------------------------------


def test_a_redundant_index_is_stored_and_not_analyzed(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", INDEX_PAYLOAD)
    stats = [
        {"name": "_id_", "key": {"_id": 1}, "accesses": {"ops": 900, "since": datetime(2026, 8, 1)}, "spec": {}},
        {
            "name": "legacy_status_1",
            "key": {"status": 1},
            "accesses": {"ops": 0, "since": datetime(2026, 8, 1)},
            "spec": {"key": {"status": 1}},
        },
    ]

    summary = run(config, profiled_cluster(index_stats=stats), claude)

    rows = {r.failure_mode: r for r in SqliteStore(config.store.sqlite_path).list_findings()}
    assert set(rows) == {MISSING_INDEX_COLLSCAN, REDUNDANT_INDEX}
    assert rows[REDUNDANT_INDEX].escalated is False
    assert claude.calls == 1  # only the missing index reached the model
    assert [p.failure_mode for p in summary.proposals] == [MISSING_INDEX_COLLSCAN]


def test_each_query_shape_gets_its_own_proposal_and_is_deduped_per_shape(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", INDEX_PAYLOAD)
    cluster = profiled_cluster(docs=[HOT_QUERY, OWNER_QUERY])

    first = run(config, cluster, claude)
    second = run(config, cluster, claude)

    assert len(first.proposals) == 2  # two shapes on one node: not collapsed into one
    assert second.proposals == []  # both still pending: not re-analyzed
    assert claude.calls == 2
    assert len(SqliteStore(config.store.sqlite_path).list_findings()) == 4


def test_a_decided_shape_can_be_proposed_again(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", INDEX_PAYLOAD)
    first = run(config, profiled_cluster(), claude)
    SqliteStore(config.store.sqlite_path).record_approval_transition(
        first.proposals[0].proposal_id, ApprovalState.REJECTED, by="x"
    )

    second = run(config, profiled_cluster(), claude)

    assert len(second.proposals) == 1


def test_no_literals_reach_the_store(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    run(config, profiled_cluster(docs=[HOT_QUERY, OWNER_QUERY]), StaticProvider("claude", INDEX_PAYLOAD))

    db_path = Path(config.store.sqlite_path)
    # WAL mode: recent rows may live in the -wal file, not yet the main database.
    files = sorted(db_path.parent.glob(db_path.name + "*"))
    raw = b"".join(f.read_bytes() for f in files)

    assert raw  # something was actually stored
    for literal in (b"ACC-9931", b"Jane Doe", b"10.1.1.4", b"2026-08-17"):
        assert literal not in raw
