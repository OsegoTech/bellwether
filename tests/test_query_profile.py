"""Acceptance tests for the query profile collector — INDEX_ADVISOR_SPEC §A.

Spec acceptance (mocked mongo, real profiler-document fixtures):
  - a profiling-off DB yields a level-0 signal
  - profiling-on with COLLSCAN ops yields a signal carrying the shape-normalised
    ops and the collection's index stats
  - literal values are redacted
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from bson.int64 import Int64
from pymongo.errors import OperationFailure

from bellwether.collectors.base import Collector
from bellwether.collectors.query_profile import QueryProfileCollector
from bellwether.config import MongoConfig, QueryProfileCollectorConfig
from bellwether.models import Signal, SignalClass
from bellwether.mongo import ReadOnlyMongo
from tests.fakes import FALLBACKS, READ_URI, TARGET, FakeCluster

DB = "meetadev_ledger"
TS = datetime(2026, 9, 14, 12, 0)  # pymongo returns naive UTC datetimes
SINCE = datetime(2026, 9, 1)  # $indexStats accesses.since: metadata, may be carried
FILTER_DATE = datetime(2026, 8, 17)  # a literal inside a query filter: must never be carried

# Literals in the fixtures below that are unique strings: none may appear anywhere.
UNIQUE_LITERALS = ("ACC-9931", "2026-08-17", "10.1.1.4", "ledger-api", "c0ffee", "2F1AB33C")
# Literal values that are also substrings of field names ("posted" in "posted_at"):
# they may not appear as a JSON string value.
WORD_LITERALS = ("posted", "pending")
OPERATOR_CLASSES = {"eq", "range", "regex", "negation", "exists", "or", "other"}


def find_doc(
    *,
    millis: int = 1840,
    examined: int = 1_250_000,
    returned: int = 50,
    plan: str = "COLLSCAN",
    coll: str = "transactions",
    filter: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A MongoDB 7.0 system.profile entry for a find."""
    return {
        "op": "query",
        "ns": f"{DB}.{coll}",
        "command": {
            "find": coll,
            "filter": filter
            if filter is not None
            else {"account_id": "ACC-9931", "status": "posted", "posted_at": {"$gte": FILTER_DATE}},
            "sort": sort if sort is not None else {"posted_at": -1},
            "limit": 50,
            "$db": DB,
            "lsid": {"id": "c0ffee"},
        },
        "keysExamined": 0,
        "docsExamined": examined,
        "cursorExhausted": True,
        "numYield": 1250,
        "nreturned": returned,
        "queryHash": "2F1AB33C",
        "planCacheKey": "7E3D1C2A",
        "planSummary": plan,
        "millis": millis,
        "responseLength": 21450,
        "protocol": "op_msg",
        "ts": TS,
        "client": "10.1.1.4",
        "appName": "ledger-api",
        "allUsers": [{"user": "ledger-api", "db": "$external"}],
        "user": "ledger-api@$external",
    }


def aggregate_doc() -> dict[str, Any]:
    return {
        "op": "command",
        "ns": f"{DB}.transactions",
        "command": {
            "aggregate": "transactions",
            "pipeline": [
                {"$match": {"account_id": "ACC-9931"}},
                {"$match": {"amount": {"$gt": 5000}}},
                {"$sort": {"posted_at": -1}},
                {"$limit": 10},
            ],
            "cursor": {},
            "$db": DB,
        },
        "keysExamined": 0,
        "docsExamined": 800_000,
        "nreturned": 10,
        "planSummary": "COLLSCAN",
        "millis": 950,
        "responseLength": 4200,
        "ts": TS,
        "client": "10.1.1.4",
    }


def getmore_doc() -> dict[str, Any]:
    return {
        "op": "getmore",
        "ns": f"{DB}.transactions",
        "command": {"getMore": Int64(12345), "collection": "transactions", "$db": DB},
        "originatingCommand": {"find": "transactions", "filter": {"status": "pending"}, "$db": DB},
        "keysExamined": 0,
        "docsExamined": 400_000,
        "nreturned": 101,
        "planSummary": "COLLSCAN",
        "millis": 600,
        "responseLength": 40_000,
        "ts": TS,
    }


INDEX_STATS: list[dict[str, Any]] = [
    {
        "name": "_id_",
        "key": {"_id": 1},
        "host": "node-backup.mongo.internal:27017",
        "accesses": {"ops": Int64(912), "since": SINCE},
        "spec": {"v": 2, "key": {"_id": 1}, "name": "_id_"},
    },
    {
        "name": "account_id_1",
        "key": {"account_id": 1},
        "host": "node-backup.mongo.internal:27017",
        "accesses": {"ops": Int64(0), "since": SINCE},
        "spec": {"v": 2, "key": {"account_id": 1}, "name": "account_id_1"},
    },
    {
        "name": "status_1",
        "key": {"status": 1.0},
        "host": "node-backup.mongo.internal:27017",
        "accesses": {"ops": Int64(3), "since": SINCE},
        "spec": {
            "v": 2,
            "key": {"status": 1.0},
            "name": "status_1",
            "partialFilterExpression": {"status": "posted"},
        },
    },
]


def cluster_with(
    *,
    level: int = 1,
    docs: list[dict[str, Any]] | None = None,
    index_stats: list[dict[str, Any]] | None = None,
) -> FakeCluster:
    cluster = FakeCluster()
    cluster.reply(f"{DB}.profile", {"was": level, "slowms": 100, "sampleRate": 1.0, "ok": 1.0})
    cluster.collections[f"{DB}.system.profile"] = list(docs if docs is not None else [find_doc()])
    stats = INDEX_STATS if index_stats is None else index_stats
    cluster.aggregations["$indexStats"] = lambda node, ns, pipeline: [dict(d) for d in stats]
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"ns": ns, "storageStats": {"count": 2_400_000, "avgObjSize": 436, "size": 1_046_400_000}}
    ]
    return cluster


def read_client(cluster: FakeCluster) -> ReadOnlyMongo:
    config = MongoConfig(
        uri=READ_URI,
        tls_ca_file=Path("/etc/mongodb/tls/ca-chain.cert.pem"),
        tls_cert_file=Path("/etc/bellwether/tls/meetadev-ai.combined.pem"),
        target_node=TARGET,
        fallback_nodes=FALLBACKS,
    )
    return ReadOnlyMongo(config, client_factory=cluster.factory())


def collector(**overrides: Any) -> QueryProfileCollector:
    return QueryProfileCollector(QueryProfileCollectorConfig(**{"databases": [DB], **overrides}))


def collect(cluster: FakeCluster, **overrides: Any) -> list[Signal]:
    return collector(**overrides).collect_signals(read_client(cluster))


def evidence(signal: Signal) -> dict[str, Any]:
    return {e.name: e.value for e in signal.evidence}


# --- Spec acceptance ---------------------------------------------------------------


def test_profiling_off_yields_a_level_zero_signal() -> None:
    cluster = cluster_with(level=0)

    [signal] = collect(cluster)

    ev = evidence(signal)
    assert ev["db"] == DB
    assert ev["profiler_level"] == 0
    assert ev["slow_ms"] == 100
    assert "ops" not in ev
    assert signal.signal_class is SignalClass.PERFORMANCE
    assert [c for c in cluster.calls if c.target == "system.profile"] == []  # not even read


def test_collscan_ops_yield_shapes_and_index_stats() -> None:
    [signal] = collect(cluster_with())

    ev = evidence(signal)
    assert signal.source == "query_profile"
    assert signal.node == TARGET
    assert (ev["db"], ev["collection"]) == (DB, "transactions")
    assert (ev["profiler_level"], ev["slow_ms"]) == (1, 100)
    assert (ev["collection_doc_count"], ev["avg_object_size"]) == (2_400_000, 436.0)
    assert ev["slow_ops_seen"] == 1
    assert ev["ops"] == [
        {
            "op": "query",
            "ns": f"{DB}.transactions",
            "command": "find",
            "filter": {"account_id": "eq", "posted_at": "range", "status": "eq"},
            "sort": [["posted_at", -1]],
            "millis": 1840,
            "docsExamined": 1_250_000,
            "keysExamined": 0,
            "nreturned": 50,
            "responseLength": 21450,
            "planSummary": "COLLSCAN",
            "ts": "2026-09-14T12:00:00+00:00",
        }
    ]
    by_name = {i["name"]: i for i in ev["index_stats"]}
    assert list(by_name) == ["_id_", "account_id_1", "status_1"]
    assert by_name["account_id_1"] == {
        "name": "account_id_1",
        "key": [{"field": "account_id", "direction": 1}],
        "accesses_ops": 0,
        "accesses_since": "2026-09-01T00:00:00+00:00",
        "unique": False,
        "partial": False,
        "sparse": False,
        "ttl": False,
        "hidden": False,
    }
    assert by_name["status_1"]["partial"] is True
    assert by_name["status_1"]["key"] == [{"field": "status", "direction": 1}]


def test_literal_values_never_leave_the_collector() -> None:
    docs = [find_doc(), aggregate_doc(), getmore_doc()]

    [signal] = collect(cluster_with(docs=docs))

    serialized = json.dumps([e.value for e in signal.evidence])
    for literal in UNIQUE_LITERALS:
        assert literal not in serialized, literal
    for word in WORD_LITERALS:
        assert f'"{word}"' not in serialized, word
    for record in evidence(signal)["ops"]:
        assert set(record["filter"].values()) <= OPERATOR_CLASSES  # classes, never values
        for field in ("client", "user", "allUsers", "appName", "queryHash", "lsid"):
            assert field not in record


# --- Which ops are carried -----------------------------------------------------------


def test_fast_and_well_targeted_ops_are_dropped() -> None:
    docs = [
        find_doc(millis=40),  # under slow_ms: filtered server-side
        find_doc(plan="IXSCAN { account_id: 1 }", examined=50, returned=50),  # efficient
        find_doc(plan="IXSCAN { status: 1 }", examined=25_000, returned=50),  # ratio 500: kept
        find_doc(),  # COLLSCAN: kept
    ]

    [signal] = collect(cluster_with(docs=docs))

    plans = sorted(op["planSummary"] for op in evidence(signal)["ops"])
    assert plans == ["COLLSCAN", "IXSCAN { status: 1 }"]


def test_one_signal_per_collection() -> None:
    docs = [find_doc(coll="transactions"), find_doc(coll="accounts", filter={"owner": "x"})]

    signals = collect(cluster_with(docs=docs))

    assert [evidence(s)["collection"] for s in signals] == ["accounts", "transactions"]


def test_worst_ops_are_kept_when_capped() -> None:
    docs = [find_doc(millis=100 + i) for i in range(1, 61)]

    [signal] = collect(cluster_with(docs=docs), max_ops_per_collection=50)

    ev = evidence(signal)
    assert ev["slow_ops_seen"] == 60
    millis = [op["millis"] for op in ev["ops"]]
    assert len(millis) == 50
    assert millis == sorted(millis, reverse=True)
    assert min(millis) == 111  # the ten fastest were dropped


def test_aggregate_and_getmore_are_shaped() -> None:
    [signal] = collect(cluster_with(docs=[aggregate_doc(), getmore_doc()]))

    ops = {op["op"]: op for op in evidence(signal)["ops"]}
    assert ops["command"]["command"] == "aggregate"
    assert ops["command"]["filter"] == {"account_id": "eq", "amount": "range"}
    assert ops["command"]["sort"] == [["posted_at", -1]]
    assert ops["getmore"]["command"] == "find"
    assert ops["getmore"]["filter"] == {"status": "eq"}


def test_non_query_commands_are_ignored() -> None:
    docs = [
        {
            "op": "command",
            "ns": f"{DB}.$cmd",
            "command": {"createIndexes": "transactions", "indexes": [], "$db": DB},
            "millis": 5000,
            "docsExamined": 0,
            "nreturned": 0,
            "planSummary": "",
            "ts": TS,
        },
        {"op": "command", "ns": f"{DB}.$cmd", "command": {"dbStats": 1}, "millis": 300, "ts": TS},
    ]

    assert collect(cluster_with(docs=docs)) == []


# --- Databases and profiling status ---------------------------------------------------


def test_databases_are_discovered_when_not_configured() -> None:
    cluster = cluster_with()
    cluster.reply(
        "listDatabases",
        {
            "databases": [
                {"name": "admin"},
                {"name": "config"},
                {"name": "local"},
                {"name": DB},
                {"name": "reports"},
            ],
            "ok": 1.0,
        },
    )
    cluster.reply("reports.profile", {"was": 0, "slowms": 100, "ok": 1.0})

    signals = collect(cluster, databases=[])

    assert [(evidence(s)["db"], evidence(s)["profiler_level"]) for s in signals] == [
        (DB, 1),
        ("reports", 0),
    ]
    assert len(cluster.commands_sent("listDatabases")) == 1


def test_configured_databases_skip_discovery() -> None:
    cluster = cluster_with()

    collect(cluster)

    assert cluster.commands_sent("listDatabases") == []


def test_unknown_profiling_status_still_reads_the_profiler() -> None:
    cluster = cluster_with()

    def unauthorized(node: str, doc: dict[str, Any]) -> dict[str, Any]:
        raise OperationFailure("not authorized on meetadev_ledger to execute command", code=13)

    cluster.commands[f"{DB}.profile"] = unauthorized

    [signal] = collect(cluster)

    ev = evidence(signal)
    assert ev["profiler_level"] is None
    assert "OperationFailure" in ev["profiler_status_error"]
    assert ev["ops"]


def test_missing_collection_stats_are_tolerated() -> None:
    cluster = cluster_with()

    def gone(node: str, ns: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise OperationFailure("ns does not exist", code=26)

    cluster.aggregations["$collStats"] = gone

    [signal] = collect(cluster)

    ev = evidence(signal)
    assert ev["collection_doc_count"] is None and ev["avg_object_size"] is None


# --- Read-only, bounded, JSON-native ----------------------------------------------------


def test_collector_only_reads() -> None:
    cluster = cluster_with()
    profile_commands: list[dict[str, Any]] = []

    def profile(node: str, doc: dict[str, Any]) -> dict[str, Any]:
        profile_commands.append(doc)
        return {"was": 1, "slowms": 100, "ok": 1.0}

    cluster.commands[f"{DB}.profile"] = profile

    collect(cluster)

    assert profile_commands == [{"profile": -1}]  # reads the level, never sets it
    assert {(c.op, c.target) for c in cluster.calls if c.target != "ping"} == {
        ("command", "profile"),
        ("find", "system.profile"),
        ("aggregate", "$indexStats"),
        ("aggregate", "$collStats"),
    }


def test_evidence_is_json_native() -> None:
    signals = collect(cluster_with(docs=[find_doc(), aggregate_doc(), getmore_doc()]))

    for signal in signals:
        values = [e.value for e in signal.evidence]
        assert json.loads(json.dumps(values)) == values  # no tuples, datetimes, or BSON types


def test_collect_returns_the_first_signal() -> None:
    cluster = cluster_with(docs=[find_doc(coll="transactions"), find_doc(coll="accounts")])

    signal = collector().collect(read_client(cluster))

    assert signal is not None
    assert evidence(signal)["collection"] == "accounts"


def test_it_is_a_performance_collector() -> None:
    c = collector()

    assert isinstance(c, Collector)
    assert c.name == "query_profile"
    assert c.signal_class is SignalClass.PERFORMANCE


@pytest.mark.parametrize("bad", [0, 1001])
def test_profile_entry_limit_is_bounded(bad: int) -> None:
    with pytest.raises(ValueError):
        QueryProfileCollectorConfig(max_profile_entries=bad)
