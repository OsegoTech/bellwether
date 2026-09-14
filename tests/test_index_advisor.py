"""Acceptance tests for the Index Advisor detector — INDEX_ADVISOR_SPEC §E.

Spec acceptance:
  - targeting 1000 with no covering index -> missing_index_collscan
    WARNING/CRITICAL with the ESR candidate in evidence
  - a shape already covered by an existing index -> no finding
  - profiler level 0 -> profiler_disabled INFO
  - an index with 0 accesses that is not _id/shardkey/unique -> redundant_index INFO
  - a prefix index -> redundant_index naming the covering index
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from bellwether.config import IndexAdvisorDetectorConfig
from bellwether.detectors.base import Detector
from bellwether.detectors.index_advisor import (
    MISSING_INDEX_COLLSCAN,
    PROFILER_DISABLED,
    REDUNDANT_INDEX,
    IndexAdvisorDetector,
)
from bellwether.models import (
    ActionKind,
    Evidence,
    Finding,
    Proposal,
    RemediationAction,
    Severity,
    Signal,
    SignalClass,
)
from bellwether.store.sqlite import SqliteStore

NODE = "node-backup.mongo.internal:27017"
DB, COLL = "meetadev_ledger", "transactions"
NS = f"{DB}.{COLL}"
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
OLD = "2026-08-01T00:00:00+00:00"  # 45 days of index stats
RECENT = "2026-09-14T00:00:00+00:00"  # 36 hours: too young to call an index unused

HOT_FILTER = {"account_id": "eq", "status": "eq", "posted_at": "range"}
HOT_SORT = [["posted_at", -1]]
HOT_CANDIDATE = [
    {"field": "account_id", "direction": 1},
    {"field": "status", "direction": 1},
    {"field": "posted_at", "direction": -1},
]


def op(
    filter: dict[str, str] | None = None,
    sort: list[list[Any]] | None = None,
    *,
    examined: int = 1_000_000,
    returned: int = 1_000,
    millis: int = 1_840,
) -> dict[str, Any]:
    return {
        "op": "query",
        "ns": NS,
        "command": "find",
        "filter": HOT_FILTER if filter is None else filter,
        "sort": HOT_SORT if sort is None else sort,
        "millis": millis,
        "docsExamined": examined,
        "keysExamined": 0,
        "nreturned": returned,
        "responseLength": 436 * returned,
        "planSummary": "COLLSCAN",
        "ts": "2026-09-15T11:00:00+00:00",
    }


def index(
    name: str,
    key: Sequence[tuple[str, int | str]],
    *,
    ops: int = 100,
    since: str | None = OLD,
    **flags: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "key": [{"field": f, "direction": d} for f, d in key],
        "accesses_ops": ops,
        "accesses_since": since,
        "unique": False,
        "partial": False,
        "sparse": False,
        "ttl": False,
        "hidden": False,
    }
    record.update(flags)
    return record


ID_INDEX = index("_id_", [("_id", 1)], ops=900)


def profile_signal(
    ops: list[dict[str, Any]] | None = None,
    indexes: list[dict[str, Any]] | None = None,
    *,
    level: int | None = 1,
    count: int | None = 2_400_000,
    avg: float | None = 436.0,
    collection: str = COLL,
    node: str = NODE,
    swept: list[str] | None = None,
    unreachable: list[str] | None = None,
) -> Signal:
    carried = [op()] if ops is None else ops
    return Signal(
        signal_class=SignalClass.PERFORMANCE,
        source="query_profile",
        node=node,
        evidence=(
            Evidence("db", DB),
            Evidence("collection", collection),
            Evidence("profiler_level", level),
            Evidence("slow_ms", 100, "ms"),
            Evidence("slow_ops_seen", len(carried)),
            Evidence("ops", carried),
            Evidence("index_stats", [ID_INDEX] if indexes is None else indexes),
            Evidence("collection_doc_count", count, "docs"),
            Evidence("avg_object_size", avg, "bytes"),
            Evidence("members_swept", swept if swept is not None else [node]),
            Evidence("members_unreachable", unreachable or []),
        ),
        collected_at=NOW,
    )


def disabled_signal(db: str = DB, node: str = NODE) -> Signal:
    return Signal(
        signal_class=SignalClass.PERFORMANCE,
        source="query_profile",
        node=node,
        evidence=(Evidence("db", db), Evidence("profiler_level", 0), Evidence("slow_ms", 100, "ms")),
        collected_at=NOW,
    )


def detector(**config: Any) -> IndexAdvisorDetector:
    return IndexAdvisorDetector(IndexAdvisorDetectorConfig(**config))


def run(*signals: Signal, **config: Any) -> list[Finding]:
    return detector(**config).evaluate_all(list(signals))


def of_mode(findings: list[Finding], mode: str) -> list[Finding]:
    return [f for f in findings if f.failure_mode == mode]


def ev(finding: Finding) -> dict[str, Any]:
    return {e.name: e.value for e in finding.evidence}


# --- Spec acceptance: missing_index_collscan ------------------------------------------


def test_poor_targeting_without_a_covering_index_is_a_missing_index_finding() -> None:
    [finding] = of_mode(run(profile_signal()), MISSING_INDEX_COLLSCAN)

    assert finding.severity is Severity.WARNING  # ratio 1,000: under the CRITICAL bar
    assert finding.signal_class is SignalClass.PERFORMANCE
    assert finding.node == NODE
    assert finding.horizon_seconds is None
    e = ev(finding)
    assert (e["db"], e["collection"]) == (DB, COLL)
    assert e["query_shape"] == {"filter": HOT_FILTER, "sort": HOT_SORT}
    assert e["targeting_ratio"] == 1000.0
    assert (e["docs_examined"], e["docs_returned"]) == (1_000_000, 1_000)
    assert e["wasted_bytes"] == 999_000 * 436
    assert (e["avg_object_size"], e["avg_object_size_source"]) == (436.0, "collstats")
    assert e["candidate_index"] == HOT_CANDIDATE  # computed by ESR, carried for the model
    assert e["existing_indexes"] == [{"name": "_id_", "key": [{"field": "_id", "direction": 1}]}]
    assert (e["op_count"], e["worst_millis"]) == (1, 1_840)
    assert e["collection_doc_count"] == 2_400_000
    assert e["subject"] == f"{NS}#{e['shape_key']}"
    assert NS in finding.summary
    assert "1,000" in finding.summary
    assert "{account_id: 1, status: 1, posted_at: -1}" in finding.summary


def test_extreme_waste_is_critical() -> None:
    signal = profile_signal([op(examined=50_000_000, returned=50)])

    [finding] = of_mode(run(signal), MISSING_INDEX_COLLSCAN)

    assert finding.severity is Severity.CRITICAL


@pytest.mark.parametrize(
    "key",
    [
        [("account_id", 1), ("status", 1), ("posted_at", -1)],  # the candidate itself
        [("status", 1), ("account_id", -1), ("posted_at", -1)],  # equality fields in any order
        [("account_id", 1), ("status", 1), ("posted_at", 1)],  # the sort, fully reversed
        [("account_id", 1), ("status", 1), ("posted_at", -1), ("amount", 1)],  # a wider index
    ],
    ids=["exact", "equality reordered", "sort reversed", "wider"],
)
def test_a_shape_covered_by_an_existing_index_is_not_a_finding(key: list[tuple[str, int]]) -> None:
    signal = profile_signal(indexes=[ID_INDEX, index("existing", key)])

    assert of_mode(run(signal), MISSING_INDEX_COLLSCAN) == []


@pytest.mark.parametrize(
    "existing",
    [
        index("account_id_1", [("account_id", 1)]),  # only a prefix of what the shape needs
        index("hidden", [("account_id", 1), ("status", 1), ("posted_at", -1)], hidden=True),
        index("partial", [("account_id", 1), ("status", 1), ("posted_at", -1)], partial=True),
        index("wrong_sort", [("account_id", 1), ("posted_at", -1), ("status", 1)]),
    ],
    ids=["prefix only", "hidden", "partial", "sort before equality"],
)
def test_indexes_that_do_not_cover_the_shape_leave_the_finding(existing: dict[str, Any]) -> None:
    [finding] = of_mode(run(profile_signal(indexes=[ID_INDEX, existing])), MISSING_INDEX_COLLSCAN)

    assert existing["name"] in [i["name"] for i in ev(finding)["existing_indexes"]]


# --- Spec acceptance: profiler_disabled ------------------------------------------------


def test_profiler_level_zero_is_a_profiler_disabled_info_finding() -> None:
    [finding] = run(disabled_signal())

    assert finding.failure_mode == PROFILER_DISABLED
    assert finding.severity is Severity.INFO
    assert not finding.escalates
    assert ev(finding)["db"] == DB
    assert ev(finding)["subject"] == DB
    assert finding.summary == (
        f"Profiling is off on {DB}, so slow queries cannot be analysed; enabling it "
        "(level 1, slowms=100) has a small overhead."
    )


# --- Spec acceptance: redundant_index ---------------------------------------------------


def test_an_unused_index_is_a_redundant_index_info_finding() -> None:
    legacy = index("legacy_status_1", [("status", 1)], ops=0, since=OLD)

    [finding] = of_mode(run(profile_signal(indexes=[ID_INDEX, legacy])), REDUNDANT_INDEX)

    assert finding.severity is Severity.INFO
    e = ev(finding)
    assert e["reason"] == "unused"
    assert e["index_name"] == "legacy_status_1"
    assert e["index_key"] == [{"field": "status", "direction": 1}]
    assert (e["accesses_ops"], e["accesses_since"]) == (0, OLD)
    assert e["subject"] == f"{NS}#legacy_status_1"
    assert "legacy_status_1" in finding.summary


@pytest.mark.parametrize(
    "protected",
    [
        index("_id_", [("_id", 1)], ops=0),
        index("email_1", [("email", 1)], ops=0, unique=True),
        index("expire_1", [("expires_at", 1)], ops=0, ttl=True),
        index("hidden_1", [("status", 1)], ops=0, hidden=True),
        index("young_1", [("status", 1)], ops=0, since=RECENT),
        index("unknown_age_1", [("status", 1)], ops=0, since=None),
    ],
    ids=["_id", "unique", "ttl", "hidden", "stats too young", "stats age unknown"],
)
def test_protected_or_unproven_indexes_are_never_called_unused(protected: dict[str, Any]) -> None:
    assert of_mode(run(profile_signal(indexes=[protected])), REDUNDANT_INDEX) == []


def test_a_prefix_index_is_redundant_and_names_its_covering_index() -> None:
    prefix = index("account_id_1", [("account_id", 1)], ops=500)
    wider = index("account_id_1_posted_at_-1", [("account_id", 1), ("posted_at", -1)])

    [finding] = of_mode(run(profile_signal(indexes=[ID_INDEX, prefix, wider])), REDUNDANT_INDEX)

    e = ev(finding)
    assert e["reason"] == "prefix"
    assert e["index_name"] == "account_id_1"
    assert e["covering_index"] == {
        "name": "account_id_1_posted_at_-1",
        "key": [{"field": "account_id", "direction": 1}, {"field": "posted_at", "direction": -1}],
    }
    assert "account_id_1_posted_at_-1" in finding.summary


def test_a_reversed_single_field_prefix_is_redundant() -> None:
    prefix = index("a_-1", [("a", -1)])
    wider = index("a_1_b_1", [("a", 1), ("b", 1)])

    [finding] = of_mode(run(profile_signal(indexes=[prefix, wider])), REDUNDANT_INDEX)

    assert ev(finding)["covering_index"]["name"] == "a_1_b_1"


@pytest.mark.parametrize(
    ("prefix", "wider"),
    [
        (index("a_1", [("a", 1)], unique=True), index("a_1_b_1", [("a", 1), ("b", 1)])),
        (index("a_1", [("a", 1)]), index("a_1_b_1", [("a", 1), ("b", 1)], partial=True)),
        (index("a_1", [("a", 1)]), index("a_1_b_1", [("a", 1), ("b", 1)], sparse=True)),
        (index("t", [("a", "text")]), index("t_b", [("a", "text"), ("b", 1)])),
    ],
    ids=["unique prefix", "partial cover", "sparse cover", "text index"],
)
def test_prefixes_that_are_not_safely_covered_are_left_alone(
    prefix: dict[str, Any], wider: dict[str, Any]
) -> None:
    assert of_mode(run(profile_signal(indexes=[prefix, wider])), REDUNDANT_INDEX) == []


def test_an_unused_prefix_index_is_reported_once() -> None:
    prefix = index("a_1", [("a", 1)], ops=0, since=OLD)
    wider = index("a_1_b_1", [("a", 1), ("b", 1)])

    findings = of_mode(run(profile_signal(indexes=[prefix, wider])), REDUNDANT_INDEX)

    assert [ev(f)["reason"] for f in findings] == ["prefix"]


# --- Ranking, caps, thresholds ----------------------------------------------------------


def test_missing_index_findings_rank_by_impact_and_are_capped() -> None:
    ops = [
        op({"a": "eq"}, [], examined=2_000_000, returned=10),
        op({"b": "eq"}, [], examined=8_000_000, returned=10),
        op({"c": "eq"}, [], examined=500_000, returned=10),
    ]

    ranked = of_mode(run(profile_signal(ops)), MISSING_INDEX_COLLSCAN)
    capped = of_mode(run(profile_signal(ops), max_suggestions=2), MISSING_INDEX_COLLSCAN)

    assert [ev(f)["candidate_index"][0]["field"] for f in ranked] == ["b", "a", "c"]
    assert [ev(f)["candidate_index"][0]["field"] for f in capped] == ["b", "a"]


def test_ops_of_one_shape_are_scored_together() -> None:
    ops = [op(examined=400_000, returned=100, millis=900), op(examined=600_000, returned=900, millis=700)]

    [finding] = of_mode(run(profile_signal(ops)), MISSING_INDEX_COLLSCAN)

    e = ev(finding)
    assert e["op_count"] == 2
    assert (e["docs_examined"], e["docs_returned"]) == (1_000_000, 1_000)
    assert (e["total_millis"], e["worst_millis"]) == (1_600, 900)


def test_equality_order_uses_field_frequency_on_the_collection() -> None:
    ops = [
        op({"branch": "eq", "zone": "eq"}, [], examined=3_000_000, returned=10),
        op({"zone": "eq", "amount": "range"}, [], examined=10_000, returned=5),
    ]

    findings = of_mode(run(profile_signal(ops)), MISSING_INDEX_COLLSCAN)

    candidate = ev(findings[0])["candidate_index"]
    assert [k["field"] for k in candidate] == ["zone", "branch"]  # zone used by 2 shapes


@pytest.mark.parametrize(
    "config",
    [{"targeting_ratio_threshold": 2_000}, {"wasted_bytes_floor": 10**12}],
    ids=["ratio under threshold", "waste under floor"],
)
def test_thresholds_are_configurable(config: dict[str, Any]) -> None:
    assert of_mode(run(profile_signal(), **config), MISSING_INDEX_COLLSCAN) == []


def test_a_shape_with_nothing_indexable_is_not_a_finding() -> None:
    signal = profile_signal([op({"a": "or", "b": "or"}, [])])

    assert of_mode(run(signal), MISSING_INDEX_COLLSCAN) == []


def test_an_over_wide_shape_reports_the_fields_cut() -> None:
    wide = {f"f{i:02d}": "eq" for i in range(18)}

    [finding] = of_mode(run(profile_signal([op(wide, [])])), MISSING_INDEX_COLLSCAN)

    e = ev(finding)
    assert len(e["candidate_index"]) == 16
    assert e["candidate_index_dropped_fields"] == ["f16", "f17"]


def test_unknown_profiler_level_still_analyses_the_ops() -> None:
    assert of_mode(run(profile_signal(level=None)), MISSING_INDEX_COLLSCAN)


def test_an_unknown_profiler_level_never_fires_profiler_disabled() -> None:
    # The accepted blind spot: {profile: -1} needs enableProfiler (dbAdmin only),
    # which the read identity does not hold. Unknown is not "off".
    findings = run(profile_signal(level=None), profile_signal(level=None, node="node-uae.mongo.internal:27017"))

    assert of_mode(findings, PROFILER_DISABLED) == []
    assert of_mode(findings, MISSING_INDEX_COLLSCAN)


# --- Members: signals from every node are merged per collection ----------------------------

PRIMARY = "node-uae.mongo.internal:27017"
SECONDARY = "node-westeurope.mongo.internal:27017"
PAIR = [NODE, PRIMARY]


def test_one_shape_on_several_members_is_one_finding_on_the_busiest() -> None:
    signals = [
        profile_signal([op(examined=200_000, returned=100)], node=NODE, swept=PAIR),
        profile_signal([op(examined=1_000_000, returned=900)], node=PRIMARY, swept=PAIR),
    ]

    [finding] = of_mode(run(*signals), MISSING_INDEX_COLLSCAN)

    assert finding.node == PRIMARY  # where the waste is
    e = ev(finding)
    assert e["op_count"] == 2
    assert (e["docs_examined"], e["docs_returned"]) == (1_200_000, 1_000)
    assert e["observed_on"] == {NODE: 1, PRIMARY: 1}
    assert len(finding.signals) == 2


def test_an_index_used_on_any_member_is_not_unused() -> None:
    legacy_unused = index("legacy_status_1", [("status", 1)], ops=0)
    legacy_used = index("legacy_status_1", [("status", 1)], ops=500)
    signals = [
        profile_signal(indexes=[ID_INDEX, legacy_unused], node=NODE, swept=PAIR),
        profile_signal(indexes=[ID_INDEX, legacy_used], node=PRIMARY, swept=PAIR),
    ]

    assert of_mode(run(*signals), REDUNDANT_INDEX) == []


def test_an_index_unused_on_every_member_is_redundant_once() -> None:
    legacy = index("legacy_status_1", [("status", 1)], ops=0)
    signals = [
        profile_signal(indexes=[ID_INDEX, legacy], node=NODE, swept=PAIR),
        profile_signal(indexes=[ID_INDEX, legacy], node=PRIMARY, swept=PAIR),
    ]

    [finding] = of_mode(run(*signals), REDUNDANT_INDEX)

    e = ev(finding)
    assert e["accesses_ops"] == 0
    assert e["accesses_by_node"] == {NODE: 0, PRIMARY: 0}
    assert e["members"] == PAIR


def test_an_index_is_not_called_unused_while_a_member_is_unreachable() -> None:
    legacy = index("legacy_status_1", [("status", 1)], ops=0)
    signals = [
        profile_signal(indexes=[ID_INDEX, legacy], node=NODE, swept=PAIR, unreachable=[SECONDARY]),
        profile_signal(indexes=[ID_INDEX, legacy], node=PRIMARY, swept=PAIR, unreachable=[SECONDARY]),
    ]

    assert of_mode(run(*signals), REDUNDANT_INDEX) == []


def test_an_index_missing_from_a_members_stats_is_not_called_unused() -> None:
    legacy = index("legacy_status_1", [("status", 1)], ops=0)
    signals = [
        profile_signal(indexes=[ID_INDEX, legacy], node=NODE, swept=PAIR),
        profile_signal(indexes=[ID_INDEX], node=PRIMARY, swept=PAIR),  # still building there
    ]

    assert of_mode(run(*signals), REDUNDANT_INDEX) == []


def test_the_youngest_counters_decide_whether_an_index_is_unused() -> None:
    signals = [
        profile_signal(indexes=[index("legacy", [("s", 1)], ops=0, since=OLD)], node=NODE, swept=PAIR),
        profile_signal(indexes=[index("legacy", [("s", 1)], ops=0, since=RECENT)], node=PRIMARY, swept=PAIR),
    ]

    assert of_mode(run(*signals), REDUNDANT_INDEX) == []  # the primary restarted 36 hours ago


def test_a_prefix_index_seen_on_every_member_is_reported_once() -> None:
    indexes = [ID_INDEX, index("a_1", [("a", 1)]), index("a_1_b_1", [("a", 1), ("b", 1)])]
    signals = [
        profile_signal(indexes=indexes, node=NODE, swept=PAIR),
        profile_signal(indexes=indexes, node=PRIMARY, swept=PAIR),
    ]

    assert [ev(f)["index_name"] for f in of_mode(run(*signals), REDUNDANT_INDEX)] == ["a_1"]


def test_a_collection_signal_with_profiling_off_is_merged_not_disabled() -> None:
    signals = [
        profile_signal([], node=NODE, level=0, swept=PAIR),  # the backup: profiler off, index stats in
        profile_signal(node=PRIMARY, swept=PAIR),
    ]

    findings = run(*signals)

    assert of_mode(findings, PROFILER_DISABLED) == []
    assert of_mode(findings, MISSING_INDEX_COLLSCAN)


def test_profiler_disabled_is_reported_per_member() -> None:
    findings = run(disabled_signal(node=NODE), disabled_signal(node=PRIMARY))

    assert [(f.failure_mode, f.node) for f in findings] == [
        (PROFILER_DISABLED, NODE),
        (PROFILER_DISABLED, PRIMARY),
    ]


# --- Shape of the result ------------------------------------------------------------------


def test_findings_across_modes_and_collections() -> None:
    signals = [
        profile_signal(),
        profile_signal(
            [op({"owner": "eq"}, [], examined=900_000, returned=3)],
            [ID_INDEX, index("legacy", [("x", 1)], ops=0)],
            collection="accounts",
        ),
        disabled_signal("reports"),
        Signal(SignalClass.REPLICATION, "oplog_window", NODE, (Evidence("oplog_window_seconds", 1),)),
    ]

    findings = run(*signals)

    assert [f.failure_mode for f in findings] == [
        MISSING_INDEX_COLLSCAN,
        MISSING_INDEX_COLLSCAN,
        PROFILER_DISABLED,
        REDUNDANT_INDEX,
    ]


def test_nothing_to_report() -> None:
    assert run() == []
    assert detector().evaluate([]) is None


def test_evaluate_returns_the_most_important_finding() -> None:
    finding = detector().evaluate([disabled_signal(), profile_signal()])

    assert finding is not None and finding.failure_mode == MISSING_INDEX_COLLSCAN


def test_it_is_a_detector_with_three_failure_modes() -> None:
    d = detector()

    assert isinstance(d, Detector)
    assert set(d.failure_modes) == {PROFILER_DISABLED, MISSING_INDEX_COLLSCAN, REDUNDANT_INDEX}


def test_findings_are_deterministic() -> None:
    a = run(profile_signal())
    b = run(profile_signal())

    assert [(f.summary, ev(f)) for f in a] == [(f.summary, ev(f)) for f in b]


def test_evidence_survives_the_store_so_the_executor_can_verify_it(tmp_path: Path) -> None:
    [finding] = of_mode(run(profile_signal()), MISSING_INDEX_COLLSCAN)
    values = [e.value for e in finding.evidence]
    assert json.loads(json.dumps(values)) == values
    proposal = Proposal(
        finding_id=finding.finding_id,
        failure_mode=finding.failure_mode,
        node=finding.node,
        diagnosis="d",
        mechanism="m",
        impact_if_ignored="i",
        action=RemediationAction(
            kind=ActionKind.PROPOSE_ONLY, title="t", command="c", rationale="r", reversible=True
        ),
        confidence=0.5,
        provider="claude",
        evidence_refs=finding.evidence,
    )
    store = SqliteStore(tmp_path / "bellwether.db")

    store.record_proposal(proposal)

    assert store.get_proposal(proposal.proposal_id) == proposal
