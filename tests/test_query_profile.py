"""Acceptance tests for the query profile collector — INDEX_ADVISOR_SPEC §A.

Spec acceptance (mocked mongo, real profiler-document fixtures):
  - a profiling-off DB yields a level-0 signal
  - profiling-on with COLLSCAN ops yields a signal carrying the shape-normalised
    ops and the collection's index stats
  - literal values are redacted
"""

from __future__ import annotations

import json
import logging
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
from bellwether.mongo import NoReachableNode, ReadOnlyMongo
from tests.fakes import (
    FALLBACKS,
    READ_URI,
    MEMBER_3,
    TARGET,
    MEMBER_2,
    MEMBER_1,
    FakeCluster,
)

DB = "appdb"
CERT = Path("/etc/bellwether/tls/bellwether-reader.combined.pem")
MEMBERS = [TARGET, *FALLBACKS]  # hidden backup first, then the voters
PRIMARY = MEMBER_2  # where the application's queries run in these tests
TS = datetime(2026, 9, 14, 12, 0)  # pymongo returns naive UTC datetimes
SINCE = datetime(2026, 9, 1)  # $indexStats accesses.since: metadata, may be carried
FILTER_DATE = datetime(2026, 8, 17)  # a literal inside a query filter: must never be carried

# Literals in the fixtures below that are unique strings: none may appear anywhere.
UNIQUE_LITERALS = ("ACC-9931", "2026-08-17", "192.0.2.10", "ledger-api", "c0ffee", "2F1AB33C")
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
        "client": "192.0.2.10",
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
        "client": "192.0.2.10",
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
        "host": "mongo-hidden.example.internal:27017",
        "accesses": {"ops": Int64(912), "since": SINCE},
        "spec": {"v": 2, "key": {"_id": 1}, "name": "_id_"},
    },
    {
        "name": "account_id_1",
        "key": {"account_id": 1},
        "host": "mongo-hidden.example.internal:27017",
        "accesses": {"ops": Int64(0), "since": SINCE},
        "spec": {"v": 2, "key": {"account_id": 1}, "name": "account_id_1"},
    },
    {
        "name": "status_1",
        "key": {"status": 1.0},
        "host": "mongo-hidden.example.internal:27017",
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


def read_client(cluster: FakeCluster, fallbacks: list[str] | None = None) -> ReadOnlyMongo:
    """A read client; with no fallbacks the replica set is one member (the target)."""
    config = MongoConfig(
        uri=READ_URI,
        tls_ca_file=Path("/etc/mongodb/tls/ca-chain.cert.pem"),
        tls_cert_file=CERT,
        target_node=TARGET,
        fallback_nodes=fallbacks or [],
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
        raise OperationFailure("not authorized on appdb to execute command", code=13)

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


# --- The sweep: traffic-dependent signals are read on every member ---------------------


def swept_cluster(
    *,
    docs_by_node: dict[str, list[dict[str, Any]]] | None = None,
    levels: dict[str, int] | None = None,
    index_ops: dict[str, int] | None = None,
) -> FakeCluster:
    """A replica set whose members each keep their own system.profile and index counters."""
    cluster = cluster_with(docs=[])
    for node, docs in (docs_by_node if docs_by_node is not None else {PRIMARY: [find_doc()]}).items():
        cluster.node_collections.setdefault(node, {})[f"{DB}.system.profile"] = docs
    level_of = levels or {}

    def profile(node: str, doc: dict[str, Any]) -> dict[str, Any]:
        return {"was": level_of.get(node, 1), "slowms": 100, "ok": 1.0}

    cluster.commands[f"{DB}.profile"] = profile
    if index_ops is not None:
        ops_of = index_ops

        def index_stats(node: str, ns: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
            entry = dict(INDEX_STATS[1])
            entry["accesses"] = {"ops": Int64(ops_of.get(node, 0)), "since": SINCE}
            return [entry]

        cluster.aggregations["$indexStats"] = index_stats
    return cluster


def sweep(cluster: FakeCluster, **overrides: Any) -> list[Signal]:
    return collector(**overrides).collect_signals(read_client(cluster, FALLBACKS))


def test_sweep_reads_every_member_and_tags_each_signal_with_its_node() -> None:
    signals = sweep(swept_cluster())

    assert [s.node for s in signals] == MEMBERS
    by_node = {s.node: evidence(s) for s in signals}
    assert [op["millis"] for op in by_node[PRIMARY]["ops"]] == [1840]
    for node in MEMBERS:
        ev = by_node[node]
        assert (ev["db"], ev["collection"]) == (DB, "transactions")
        assert ev["index_stats"]  # every member reports its own index use
        assert ev["members_swept"] == MEMBERS
        assert ev["members_unreachable"] == []
        if node != PRIMARY:
            assert ev["ops"] == []


def test_sweep_connections_are_direct_read_identity_and_closed() -> None:
    cluster = swept_cluster()

    sweep(cluster)

    assert cluster.attempted_nodes == MEMBERS
    for client, node in zip(cluster.clients, MEMBERS):
        assert client.uri == READ_URI.replace(TARGET, node)
        assert client.kwargs["directConnection"] is True
        assert client.kwargs["tlsCertificateKeyFile"] == str(CERT)
        assert "tlsCertificateKeyFile" not in client.uri
        assert client.closed


def test_sweep_only_ever_reads() -> None:
    cluster = swept_cluster()
    profile_commands: list[dict[str, Any]] = []

    def profile(node: str, doc: dict[str, Any]) -> dict[str, Any]:
        profile_commands.append(doc)
        return {"was": 1, "slowms": 100, "ok": 1.0}

    cluster.commands[f"{DB}.profile"] = profile

    sweep(cluster)

    assert profile_commands == [{"profile": -1}] * len(MEMBERS)
    assert {(c.op, c.target) for c in cluster.calls} <= {
        ("command", "ping"),
        ("command", "profile"),
        ("find", "system.profile"),
        ("aggregate", "$indexStats"),
        ("aggregate", "$collStats"),
    }


def test_unreachable_member_is_reported_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    cluster = swept_cluster()
    cluster.down.add(MEMBER_1)

    with caplog.at_level(logging.WARNING, logger="bellwether.collectors"):
        signals = sweep(cluster)

    assert [s.node for s in signals] == [TARGET, MEMBER_2, MEMBER_3]
    ev = evidence(signals[0])
    assert ev["members_swept"] == [TARGET, MEMBER_2, MEMBER_3]
    assert ev["members_unreachable"] == [MEMBER_1]
    assert any(getattr(r, "node", None) == MEMBER_1 for r in caplog.records)


def test_sweep_with_no_reachable_member_raises() -> None:
    cluster = swept_cluster()
    cluster.down.update(MEMBERS)

    with pytest.raises(NoReachableNode) as excinfo:
        sweep(cluster)

    for node in MEMBERS:
        assert node in str(excinfo.value)


def test_sweep_leaves_the_backup_serving_everything_else() -> None:
    cluster = swept_cluster()
    cluster.reply("listDatabases", {"databases": [{"name": DB}], "ok": 1.0})
    reader = read_client(cluster, FALLBACKS)

    collector(databases=[]).collect_signals(reader)

    [listing] = cluster.commands_sent("listDatabases")
    assert listing.node == TARGET  # not traffic-dependent: the hidden backup answers
    assert reader.served_by == TARGET


def test_each_member_reports_its_own_profiling_level() -> None:
    signals = sweep(swept_cluster(levels={TARGET: 0}))

    disabled = [s for s in signals if "collection" not in evidence(s)]
    assert [(s.node, evidence(s)["profiler_level"]) for s in disabled] == [(TARGET, 0)]
    per_collection = {s.node: evidence(s) for s in signals if "collection" in evidence(s)}
    assert set(per_collection) == set(MEMBERS)  # the backup's index use still counts
    assert (per_collection[TARGET]["profiler_level"], per_collection[TARGET]["ops"]) == (0, [])


def test_index_use_is_read_from_every_member() -> None:
    signals = sweep(swept_cluster(index_ops={PRIMARY: 5_000}))

    usage = {s.node: evidence(s)["index_stats"][0]["accesses_ops"] for s in signals}
    assert usage == {TARGET: 0, MEMBER_1: 0, MEMBER_2: 5_000, MEMBER_3: 0}


def test_slow_queries_on_any_member_bring_in_every_members_index_stats() -> None:
    docs = {PRIMARY: [find_doc(coll="transactions")], MEMBER_3: [find_doc(coll="accounts")]}

    signals = sweep(swept_cluster(docs_by_node=docs))

    pairs = [(evidence(s)["collection"], s.node) for s in signals]
    assert pairs == [("accounts", n) for n in MEMBERS] + [("transactions", n) for n in MEMBERS]


def test_duplicate_members_are_swept_once() -> None:
    cluster = swept_cluster()
    config = MongoConfig(
        uri=READ_URI,
        tls_ca_file=Path("/etc/mongodb/tls/ca-chain.cert.pem"),
        tls_cert_file=CERT,
        target_node=TARGET,
        fallback_nodes=[MEMBER_1, TARGET, MEMBER_1],
    )

    collector().collect_signals(ReadOnlyMongo(config, client_factory=cluster.factory()))

    assert cluster.attempted_nodes == [TARGET, MEMBER_1]


def test_sweep_logs_each_member_read(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="bellwether.collectors"):
        sweep(swept_cluster())

    read = [r for r in caplog.records if r.getMessage() == "query profile read"]
    assert [getattr(r, "node") for r in read] == MEMBERS


def test_no_databases_means_no_sweep() -> None:
    cluster = swept_cluster()
    cluster.reply("listDatabases", {"databases": [{"name": "admin"}, {"name": "local"}], "ok": 1.0})

    assert collector(databases=[]).collect_signals(read_client(cluster, FALLBACKS)) == []
    assert cluster.attempted_nodes == [TARGET]  # only the serving connection, for the listing


@pytest.mark.parametrize("bad", [0, 1001])
def test_profile_entry_limit_is_bounded(bad: int) -> None:
    with pytest.raises(ValueError):
        QueryProfileCollectorConfig(max_profile_entries=bad)
