"""Acceptance tests for bellwether.mongo — BUILD_SPEC §3.3.

Spec acceptance (unit, mocked):
  - a fake MongoClient proves fallback ordering
  - the client class exposes no method that issues a write
  - a monkeypatched failure on the target node causes a documented fallback
    to the next node

Plus decision 6: the cert and CA reach MongoClient as kwargs, never in the URI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from bson import Timestamp
from pydantic import SecretStr

from bellwether.config import MongoConfig
from bellwether.detectors.base import OP_NODE_EVIDENCE, OPID_EVIDENCE, killable_op_evidence
from bellwether.mongo import ForbiddenCommand, NoReachableNode, ReadOnlyMongo
from tests.fakes import FALLBACKS, READ_URI, SOUTHAFRICA, TARGET, UAE, WESTEUROPE, FakeCluster

CERT = Path("/etc/bellwether/tls/meetadev-ai.combined.pem")
CA = Path("/etc/mongodb/tls/ca-chain.cert.pem")


def mongo_config(**overrides: Any) -> MongoConfig:
    values: dict[str, Any] = {
        "uri": READ_URI,
        "tls_ca_file": CA,
        "tls_cert_file": CERT,
        "target_node": TARGET,
        "fallback_nodes": FALLBACKS,
    }
    values.update(overrides)
    return MongoConfig(**values)


@pytest.fixture
def cluster() -> FakeCluster:
    cluster = FakeCluster()
    cluster.reply("serverStatus", {"host": "whichever", "ok": 1.0})
    cluster.reply("replSetGetStatus", {"set": "rs0", "members": [], "ok": 1.0})
    return cluster


def client_for(cluster: FakeCluster, **overrides: Any) -> ReadOnlyMongo:
    return ReadOnlyMongo(mongo_config(**overrides), client_factory=cluster.factory())


# --- Rule 1: target first, then fallbacks in order --------------------------


def test_connects_to_target_first_with_direct_connection(cluster: FakeCluster) -> None:
    mongo = client_for(cluster)

    mongo.server_status()

    assert cluster.attempted_nodes == [TARGET]
    assert mongo.served_by == TARGET
    assert cluster.clients[0].kwargs["directConnection"] is True


def test_target_down_falls_back_to_first_fallback(cluster: FakeCluster) -> None:
    cluster.down.add(TARGET)
    mongo = client_for(cluster)

    mongo.server_status()

    assert cluster.attempted_nodes == [TARGET, WESTEUROPE]
    assert mongo.served_by == WESTEUROPE


def test_fallback_order_is_followed(cluster: FakeCluster) -> None:
    cluster.down.update({TARGET, WESTEUROPE})
    mongo = client_for(cluster)

    mongo.rs_status()

    assert cluster.attempted_nodes == [TARGET, WESTEUROPE, UAE]
    assert mongo.served_by == UAE


def test_unreachable_nodes_are_closed(cluster: FakeCluster) -> None:
    cluster.down.update({TARGET, WESTEUROPE})
    client_for(cluster).rs_status()

    assert [c.closed for c in cluster.clients] == [True, True, False]


def test_all_nodes_down_raises_naming_every_node(cluster: FakeCluster) -> None:
    cluster.down.update({TARGET, *FALLBACKS})
    mongo = client_for(cluster)

    with pytest.raises(NoReachableNode) as excinfo:
        mongo.server_status()

    assert cluster.attempted_nodes == [TARGET, WESTEUROPE, UAE, SOUTHAFRICA]
    for node in (TARGET, *FALLBACKS):
        assert node in str(excinfo.value)


def test_target_failure_mid_session_fails_over_and_is_logged(
    cluster: FakeCluster, caplog: pytest.LogCaptureFixture
) -> None:
    mongo = client_for(cluster)
    mongo.server_status()
    assert mongo.served_by == TARGET

    cluster.down.add(TARGET)  # target dies between reads
    with caplog.at_level(logging.INFO, logger="bellwether.mongo"):
        reply = mongo.server_status()

    assert reply["ok"] == 1.0
    assert mongo.served_by == WESTEUROPE
    fallback_logs = [r for r in caplog.records if getattr(r, "node", None) == TARGET]
    assert any(r.levelno == logging.WARNING for r in fallback_logs)
    served = [r for r in caplog.records if r.getMessage() == "read served"]
    assert served and getattr(served[-1], "node") == WESTEUROPE


def test_every_read_logs_the_serving_node(
    cluster: FakeCluster, caplog: pytest.LogCaptureFixture
) -> None:
    mongo = client_for(cluster)

    with caplog.at_level(logging.INFO, logger="bellwether.mongo"):
        mongo.rs_status()

    served = [r for r in caplog.records if r.getMessage() == "read served"]
    assert len(served) == 1
    assert getattr(served[0], "node") == TARGET
    assert getattr(served[0], "operation") == "rs_status"


def test_duplicate_nodes_are_tried_once(cluster: FakeCluster) -> None:
    cluster.down.add(TARGET)
    mongo = client_for(cluster, fallback_nodes=[TARGET, UAE, UAE])

    mongo.server_status()

    assert cluster.attempted_nodes == [TARGET, UAE]


# --- Decision 6: TLS material as kwargs, never in the URI --------------------


def test_cert_and_ca_are_kwargs_not_uri(cluster: FakeCluster) -> None:
    client_for(cluster).server_status()
    client = cluster.clients[0]

    assert client.kwargs["tlsCertificateKeyFile"] == str(CERT)
    assert client.kwargs["tlsCAFile"] == str(CA)
    assert client.kwargs["tls"] is True
    assert "tlsCertificateKeyFile" not in client.uri
    assert "tlsCAFile" not in client.uri
    assert client.uri.startswith(f"mongodb://{TARGET}/?")
    assert "authMechanism=MONGODB-X509" in client.uri


def test_hostname_verification_is_never_disabled(cluster: FakeCluster) -> None:
    client_for(cluster).server_status()
    kwargs = cluster.clients[0].kwargs

    for insecure in ("tlsInsecure", "tlsAllowInvalidHostnames", "tlsAllowInvalidCertificates"):
        assert insecure not in kwargs


def test_passphrase_passed_only_when_configured(cluster: FakeCluster) -> None:
    client_for(cluster).server_status()
    assert "tlsCertificateKeyFilePassword" not in cluster.clients[0].kwargs

    other = FakeCluster()
    client_for(other, tls_cert_passphrase=SecretStr("hunter2")).run_admin_command("ping")
    assert other.clients[0].kwargs["tlsCertificateKeyFilePassword"] == "hunter2"


def test_fallback_uri_swaps_only_the_host(cluster: FakeCluster) -> None:
    cluster.down.add(TARGET)
    client_for(cluster).server_status()

    fallback_uri = cluster.clients[1].uri
    assert fallback_uri == READ_URI.replace(TARGET, WESTEUROPE)


def test_server_selection_timeout_comes_from_config(cluster: FakeCluster) -> None:
    client_for(cluster, server_selection_timeout_ms=1234).server_status()

    assert cluster.clients[0].kwargs["serverSelectionTimeoutMS"] == 1234


# --- Rule 2: read helpers only; no write method exists ----------------------

READ_API = {
    "run_admin_command",
    "server_status",
    "rs_status",
    "oplog_stats",
    "profile_read",
    "index_stats",
    "current_op",
    "current_op_all_nodes",
    "served_by",
    "close",
}

WRITE_VERBS = (
    "insert", "update", "delete", "remove", "drop", "create", "kill", "replace",
    "write", "bulk", "save", "rename", "shutdown", "reconfig", "resize", "compact",
    "repair", "fsync", "exec", "set_", "grant", "revoke", "step",
)


def test_public_api_is_exactly_the_read_helpers() -> None:
    public = {name for name in dir(ReadOnlyMongo) if not name.startswith("_")}

    assert public == READ_API


def test_no_public_name_suggests_a_write() -> None:
    for name in dir(ReadOnlyMongo):
        if name.startswith("_"):
            continue
        assert not any(verb in name for verb in WRITE_VERBS), name


def test_underlying_client_is_not_exposed(cluster: FakeCluster) -> None:
    mongo = client_for(cluster)
    mongo.server_status()

    assert [name for name in vars(mongo) if not name.startswith("_")] == []


@pytest.mark.parametrize(
    "command",
    [
        "killOp", "shutdown", "createIndexes", "dropDatabase", "drop", "insert",
        "update", "delete", "findAndModify", "replSetReconfig", "replSetStepDown",
        "replSetResizeOplog", "setParameter", "fsync", "createUser", "compact",
        "aggregate", "eval",
    ],
)
def test_run_admin_command_refuses_non_read_commands(cluster: FakeCluster, command: str) -> None:
    mongo = client_for(cluster)

    with pytest.raises(ForbiddenCommand, match=command):
        mongo.run_admin_command(command)

    # Refused before any connection is made.
    assert cluster.clients == []


def test_run_admin_command_allows_read_commands(cluster: FakeCluster) -> None:
    cluster.reply("getLog", {"log": [], "ok": 1.0})

    reply = client_for(cluster).run_admin_command("getLog", "global")

    assert reply["ok"] == 1.0
    assert cluster.commands_sent("getLog")[0].db == "admin"


# --- The read helpers --------------------------------------------------------


def test_oplog_stats(cluster: FakeCluster) -> None:
    first, last = 1_757_000_000, 1_757_000_000 + 2400
    cluster.collections["local.oplog.rs"] = [
        {"ts": Timestamp(first, 1)},
        {"ts": Timestamp(first + 100, 7)},
        {"ts": Timestamp(last, 3)},
    ]
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"ns": ns, "storageStats": {"maxSize": 990 * 2**20, "size": 512 * 2**20}}
    ]

    stats = client_for(cluster).oplog_stats()

    assert stats.max_size_bytes == 990 * 2**20
    assert stats.used_bytes == 512 * 2**20
    assert stats.first_entry_ts == first
    assert stats.last_entry_ts == last
    assert stats.window_seconds == 2400


def test_oplog_stats_empty_oplog(cluster: FakeCluster) -> None:
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"storageStats": {"maxSize": 1024, "size": 0}}
    ]

    stats = client_for(cluster).oplog_stats()

    assert stats.window_seconds is None


def test_current_op_uses_currentop_stage_with_filter(cluster: FakeCluster) -> None:
    seen: list[list[dict[str, Any]]] = []

    def handler(node: str, ns: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen.append(pipeline)
        return [{"opid": 42, "secs_running": 900}]

    cluster.aggregations["$currentOp"] = handler

    ops = client_for(cluster).current_op({"secs_running": {"$gte": 60}})

    assert ops == [{"opid": 42, "secs_running": 900}]
    assert seen[0][0]["$currentOp"]["allUsers"] is True
    assert seen[0][1] == {"$match": {"secs_running": {"$gte": 60}}}


def test_profile_read_reads_system_profile(cluster: FakeCluster) -> None:
    cluster.collections["meetadev_ledger.system.profile"] = [
        {"ts": 1, "millis": 5},
        {"ts": 3, "millis": 900},
        {"ts": 2, "millis": 40},
    ]

    docs = client_for(cluster).profile_read("meetadev_ledger", limit=2)

    assert [d["ts"] for d in docs] == [3, 2]  # newest first, limited


def test_profile_read_rejects_unbounded_limit(cluster: FakeCluster) -> None:
    with pytest.raises(ValueError):
        client_for(cluster).profile_read("meetadev_ledger", limit=0)


def test_index_stats(cluster: FakeCluster) -> None:
    cluster.aggregations["$indexStats"] = lambda node, ns, pipeline: [
        {"name": "_id_", "ns": ns, "accesses": {"ops": 10}}
    ]

    stats = client_for(cluster).index_stats("meetadev_ledger", "transactions")

    assert stats[0]["ns"] == "meetadev_ledger.transactions"


def test_close_closes_the_client(cluster: FakeCluster) -> None:
    mongo = client_for(cluster)
    mongo.server_status()

    mongo.close()

    assert cluster.clients[0].closed
    assert mongo.served_by is None


# --- The one exception: cluster-wide currentOp ---------------------------------------

MEMBERS = [TARGET, WESTEUROPE, UAE, SOUTHAFRICA]  # backup (default target) first, then voters


def sweep_cluster(
    table: dict[str, list[dict[str, Any]]],
) -> tuple[FakeCluster, list[tuple[str, list[dict[str, Any]]]]]:
    """A cluster whose members report the ops in `table`; records each $currentOp read."""
    cluster = FakeCluster()
    seen: list[tuple[str, list[dict[str, Any]]]] = []

    def handler(node: str, ns: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen.append((node, pipeline))
        return [dict(op) for op in table.get(node, [])]

    cluster.aggregations["$currentOp"] = handler
    return cluster, seen


def test_current_op_all_nodes_reads_every_member_and_tags_the_node() -> None:
    cluster, seen = sweep_cluster(
        {
            TARGET: [{"opid": 11, "op": "query"}],
            UAE: [{"opid": 4242, "secs_running": 900}, {"opid": 4243}],
        }
    )

    result = client_for(cluster).current_op_all_nodes()

    assert [node for node, _ in seen] == MEMBERS
    assert result.nodes_read == tuple(MEMBERS)
    assert result.unreachable == {}
    assert [(op.node, op.opid) for op in result.ops] == [(TARGET, 11), (UAE, 4242), (UAE, 4243)]
    assert result.ops[1].op["secs_running"] == 900


def test_sweep_connections_are_direct_read_identity_and_closed() -> None:
    cluster, _ = sweep_cluster({})

    client_for(cluster).current_op_all_nodes()

    assert cluster.attempted_nodes == MEMBERS
    for client, node in zip(cluster.clients, MEMBERS):
        assert client.uri == READ_URI.replace(TARGET, node)
        assert client.kwargs["directConnection"] is True
        assert client.kwargs["tlsCertificateKeyFile"] == str(CERT)  # meetadev-ai, the read identity
        assert "tlsCertificateKeyFile" not in client.uri
        assert client.closed


def test_sweep_only_ever_runs_currentop() -> None:
    cluster, seen = sweep_cluster({UAE: [{"opid": 1}]})

    client_for(cluster).current_op_all_nodes({"secs_running": {"$gte": 60}})

    assert {(c.op, c.target) for c in cluster.calls} == {("aggregate", "$currentOp")}
    for _, pipeline in seen:
        assert pipeline[0]["$currentOp"]["allUsers"] is True
        assert pipeline[1] == {"$match": {"secs_running": {"$gte": 60}}}


def test_unreachable_member_is_reported_not_fatal() -> None:
    cluster, _ = sweep_cluster({WESTEUROPE: [{"opid": 7}]})
    cluster.down.add(UAE)

    result = client_for(cluster).current_op_all_nodes()

    assert set(result.unreachable) == {UAE}
    assert "ServerSelectionTimeoutError" in result.unreachable[UAE]
    assert result.nodes_read == (TARGET, WESTEUROPE, SOUTHAFRICA)
    assert [(op.node, op.opid) for op in result.ops] == [(WESTEUROPE, 7)]


def test_sweep_with_no_reachable_member_raises() -> None:
    cluster, _ = sweep_cluster({})
    cluster.down.update(MEMBERS)

    with pytest.raises(NoReachableNode) as excinfo:
        client_for(cluster).current_op_all_nodes()

    for node in MEMBERS:
        assert node in str(excinfo.value)


def test_sweep_leaves_the_single_node_default_alone() -> None:
    cluster, _ = sweep_cluster({})
    cluster.reply("serverStatus", {"ok": 1.0})
    mongo = client_for(cluster)
    mongo.server_status()
    serving = cluster.clients[0]

    mongo.current_op_all_nodes()
    mongo.server_status()

    assert mongo.served_by == TARGET
    assert not serving.closed
    assert len(cluster.clients) == 1 + len(MEMBERS)  # the second read reused the serving client
    assert [c.node for c in cluster.commands_sent("serverStatus")] == [TARGET, TARGET]


def test_sweep_logs_each_member_read(caplog: pytest.LogCaptureFixture) -> None:
    cluster, _ = sweep_cluster({})

    with caplog.at_level(logging.INFO, logger="bellwether.mongo"):
        client_for(cluster).current_op_all_nodes()

    served = [
        r
        for r in caplog.records
        if r.getMessage() == "read served" and getattr(r, "operation") == "current_op_all_nodes"
    ]
    assert [getattr(r, "node") for r in served] == MEMBERS


def test_duplicate_members_are_swept_once() -> None:
    cluster, _ = sweep_cluster({})

    client_for(cluster, fallback_nodes=[WESTEUROPE, TARGET, WESTEUROPE]).current_op_all_nodes()

    assert cluster.attempted_nodes == [TARGET, WESTEUROPE]


def test_non_integer_opid_is_not_killable() -> None:
    cluster, _ = sweep_cluster({UAE: [{"opid": "shard01:4242"}, {"desc": "no opid"}]})

    ops = client_for(cluster).current_op_all_nodes().ops

    assert [op.opid for op in ops] == [None, None]
    assert [op.node for op in ops] == [UAE, UAE]


def test_node_op_feeds_killable_op_evidence() -> None:
    cluster, _ = sweep_cluster({SOUTHAFRICA: [{"opid": 777, "secs_running": 1200}]})

    [op] = client_for(cluster).current_op_all_nodes().ops
    assert op.opid is not None
    opid_evidence, node_evidence = killable_op_evidence(op.opid, op.node)

    assert (opid_evidence.name, opid_evidence.value) == (OPID_EVIDENCE, 777)
    assert (node_evidence.name, node_evidence.value) == (OP_NODE_EVIDENCE, SOUTHAFRICA)
