"""
The read-only MongoDB client — the only way Bellwether's read side touches the
cluster.

Two rules, enforced in code (BUILD_SPEC §3.3):

1. Connect to ``target_node`` with ``directConnection=True``; if it is
   unreachable, try ``fallback_nodes`` in order. A connection failure on a
   serving node mid-session re-walks that order once. Every read logs the
   node that served it.
2. Expose read helpers only. There is no write method — not disabled, absent.
   ``run_admin_command`` accepts an allowlist of read-only commands and refuses
   everything else before a connection is even opened. The underlying
   ``MongoClient`` is private and never handed out.

One deliberate exception to the single-node default:
``current_op_all_nodes`` runs ``$currentOp`` on every configured member
(backup target and voters), each over its own short-lived direct connection.
An operation is visible only on the mongod running it — a runaway query on
the primary never appears in the backup node's currentOp — and killOp must be
sent to that same member. Finding a killable op therefore means looking where
ops actually run. The exception is narrow: it is the only helper that leaves
the serving node, it issues nothing but ``$currentOp`` (a read, allowed by
clusterMonitor's inprog privilege), it uses the same read identity and TLS
material, and it leaves the serving connection untouched.

Identity is ``meetadev-ai`` (clusterMonitor@admin, read@local,
read@meetadev_ledger). TLS material reaches ``MongoClient`` as keyword
arguments, never through the URI, and hostname verification is never disabled.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from bellwether.config import MongoConfig

logger = logging.getLogger(__name__)

Doc = dict[str, Any]
ClientFactory = Callable[..., "MongoClient[Doc]"]
T = TypeVar("T")

# Admin commands that only read. Anything else — killOp, shutdown, reconfig,
# setParameter, createIndexes, CRUD — is refused by run_admin_command.
READ_ONLY_COMMANDS = frozenset(
    {
        "buildInfo",
        "collStats",
        "connectionStatus",
        "dbStats",
        "getCmdLineOpts",
        "getLog",
        "getParameter",
        "hello",
        "hostInfo",
        "isMaster",
        "listCollections",
        "listDatabases",
        "listIndexes",
        "ping",
        "replSetGetConfig",
        "replSetGetStatus",
        "serverStatus",
        "top",
    }
)

_MAX_PROFILE_DOCS = 1000


class NoReachableNode(Exception):
    """Neither the target nor any fallback node answered."""

    def __init__(self, failures: Mapping[str, str]) -> None:
        self.failures = dict(failures)
        detail = "; ".join(f"{node} ({error})" for node, error in self.failures.items())
        super().__init__(f"no replica set member reachable, tried in order: {detail}")


class ForbiddenCommand(Exception):
    """A command outside the read-only allowlist was requested."""


@dataclass(frozen=True)
class OplogStats:
    max_size_bytes: int
    used_bytes: int
    first_entry_ts: int | None  # epoch seconds of the oldest entry
    last_entry_ts: int | None  # epoch seconds of the newest entry

    @property
    def window_seconds(self) -> int | None:
        """Seconds of history the oplog holds; None if it is empty."""
        if self.first_entry_ts is None or self.last_entry_ts is None:
            return None
        return self.last_entry_ts - self.first_entry_ts


@dataclass(frozen=True)
class NodeOp:
    """One in-progress operation, tagged with the member it is running on.

    ``node`` is the configured member name (as in config, matching the
    executor's known members), not the server's self-reported ``host`` field.
    Together with ``opid`` it is what ``detectors.base.killable_op_evidence``
    needs: killOp must be sent to this node.
    """

    node: str
    op: Doc

    @property
    def opid(self) -> int | None:
        """The integer opid, or None if absent or not an integer (not killable)."""
        value = self.op.get("opid")
        return value if type(value) is int else None


@dataclass(frozen=True)
class ClusterOps:
    ops: tuple[NodeOp, ...]
    nodes_read: tuple[str, ...]  # members that answered, in sweep order
    unreachable: dict[str, str]  # member -> error, for members that did not


class ReadOnlyMongo:
    def __init__(self, config: MongoConfig, *, client_factory: ClientFactory = MongoClient) -> None:
        self._config = config
        self._factory = client_factory
        self._client: MongoClient[Doc] | None = None
        self._node: str | None = None

    @property
    def served_by(self) -> str | None:
        """The node currently serving reads; None before the first read."""
        return self._node

    def run_admin_command(self, command: str, value: Any = 1, **arguments: Any) -> Doc:
        """Run an allowlisted read-only command against the admin database."""
        if command not in READ_ONLY_COMMANDS:
            raise ForbiddenCommand(
                f"{command!r} is not an allowlisted read-only command; the read client cannot run it"
            )
        doc = {command: value, **arguments}
        return self._read("run_admin_command", lambda c: c.admin.command(doc), command=command)

    def server_status(self) -> Doc:
        return self._read("server_status", lambda c: c.admin.command({"serverStatus": 1}))

    def rs_status(self) -> Doc:
        return self._read("rs_status", lambda c: c.admin.command({"replSetGetStatus": 1}))

    def oplog_stats(self) -> OplogStats:
        def read(client: MongoClient[Doc]) -> OplogStats:
            oplog = client["local"]["oplog.rs"]
            stats = next(iter(oplog.aggregate([{"$collStats": {"storageStats": {}}}])))
            storage = stats["storageStats"]
            first = oplog.find_one({}, sort=[("$natural", 1)], projection={"ts": 1, "_id": 0})
            last = oplog.find_one({}, sort=[("$natural", -1)], projection={"ts": 1, "_id": 0})
            return OplogStats(
                max_size_bytes=int(storage["maxSize"]),
                used_bytes=int(storage["size"]),
                first_entry_ts=_ts_seconds(first),
                last_entry_ts=_ts_seconds(last),
            )

        return self._read("oplog_stats", read)

    def profile_read(
        self, db: str, filter: Mapping[str, Any] | None = None, limit: int = 100
    ) -> list[Doc]:
        """Newest-first entries from ``<db>.system.profile``."""
        if not 1 <= limit <= _MAX_PROFILE_DOCS:
            raise ValueError(f"limit must be between 1 and {_MAX_PROFILE_DOCS}, got {limit}")
        query = dict(filter or {})
        return self._read(
            "profile_read",
            lambda c: list(c[db]["system.profile"].find(query, sort=[("ts", -1)], limit=limit)),
            db=db,
        )

    def index_stats(self, db: str, collection: str) -> list[Doc]:
        return self._read(
            "index_stats",
            lambda c: list(c[db][collection].aggregate([{"$indexStats": {}}])),
            namespace=f"{db}.{collection}",
        )

    def current_op(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        """In-progress operations on the serving node only."""
        pipeline = _current_op_pipeline(filter)
        return self._read("current_op", lambda c: list(c.admin.aggregate(pipeline)))

    def current_op_all_nodes(self, filter: Mapping[str, Any] | None = None) -> ClusterOps:
        """In-progress operations on every configured member, each tagged with its node.

        THE ONE DELIBERATE EXCEPTION to the single-node default (see the
        module docstring): currentOp only shows ops on the mongod it is run
        against, so a runaway op can only be found by asking each member.
        Strictly read-only — each member gets one ``$currentOp`` aggregation
        over its own direct connection, closed straight after; the serving
        connection used by every other helper is not touched.

        A member that cannot be reached is reported in ``unreachable`` rather
        than failing the sweep; if no member answers, NoReachableNode.
        """
        pipeline = _current_op_pipeline(filter)
        target = self._config.target_node
        ops: list[NodeOp] = []
        nodes_read: list[str] = []
        unreachable: dict[str, str] = {}
        for node in self._members():
            client = self._factory(self._uri_for(node), **self._client_kwargs())
            try:
                docs = list(client.admin.aggregate(pipeline))
            except PyMongoError as exc:
                unreachable[node] = _describe(exc)
                logger.warning(
                    "member unreachable during cluster-wide currentOp",
                    extra={"node": node, "error": unreachable[node]},
                )
                continue
            finally:
                client.close()
            nodes_read.append(node)
            ops.extend(NodeOp(node=node, op=doc) for doc in docs)
            logger.info(
                "read served",
                extra={
                    "node": node,
                    "operation": "current_op_all_nodes",
                    "is_target": node == target,
                    "ops": len(docs),
                },
            )
        if not nodes_read:
            raise NoReachableNode(unreachable)
        return ClusterOps(ops=tuple(ops), nodes_read=tuple(nodes_read), unreachable=unreachable)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._node = None

    # --- connection handling ------------------------------------------------

    def _read(self, operation: str, fn: Callable[[MongoClient[Doc]], T], **extra: Any) -> T:
        client, node = self._connected()
        try:
            result = fn(client)
        except ConnectionFailure as exc:
            logger.warning(
                "serving node failed mid-session; re-walking target and fallbacks",
                extra={"node": node, "operation": operation, "error": _describe(exc)},
            )
            self.close()
            client, node = self._connected()
            result = fn(client)
        logger.info(
            "read served",
            extra={
                "node": node,
                "operation": operation,
                "is_target": node == self._config.target_node,
                **extra,
            },
        )
        return result

    def _connected(self) -> tuple[MongoClient[Doc], str]:
        if self._client is not None and self._node is not None:
            return self._client, self._node
        target = self._config.target_node
        failures: dict[str, str] = {}
        for attempt, node in enumerate(self._members(), 1):
            client = self._factory(self._uri_for(node), **self._client_kwargs())
            try:
                client.admin.command({"ping": 1})
            except PyMongoError as exc:
                failures[node] = _describe(exc)
                client.close()
                logger.warning(
                    "node unreachable; trying next in fallback order",
                    extra={"node": node, "attempt": attempt, "error": failures[node]},
                )
                continue
            if node == target:
                logger.info("connected to target node", extra={"node": node})
            else:
                logger.warning(
                    "serving reads from fallback node",
                    extra={"node": node, "target_node": target, "attempt": attempt},
                )
            self._client, self._node = client, node
            return client, node
        raise NoReachableNode(failures)

    def _members(self) -> list[str]:
        """Configured members in order — target first, then fallbacks — once each."""
        return list(dict.fromkeys([self._config.target_node, *self._config.fallback_nodes]))

    def _uri_for(self, node: str) -> str:
        """The configured URI with its host swapped for `node`; options untouched."""
        parts = urlsplit(self._config.uri)
        userinfo = parts.netloc.rpartition("@")[0]
        return urlunsplit(parts._replace(netloc=f"{userinfo}@{node}" if userinfo else node))

    def _client_kwargs(self) -> dict[str, Any]:
        cfg = self._config
        kwargs: dict[str, Any] = {
            "directConnection": True,
            "tls": True,
            "tlsCAFile": str(cfg.tls_ca_file),
            "tlsCertificateKeyFile": str(cfg.tls_cert_file),
            "serverSelectionTimeoutMS": cfg.server_selection_timeout_ms,
            "appname": "bellwether",
        }
        if cfg.tls_cert_passphrase is not None:
            kwargs["tlsCertificateKeyFilePassword"] = cfg.tls_cert_passphrase.get_secret_value()
        return kwargs


def _current_op_pipeline(filter: Mapping[str, Any] | None) -> list[Doc]:
    return [
        {"$currentOp": {"allUsers": True, "idleConnections": False}},
        {"$match": dict(filter or {})},
    ]


def _ts_seconds(entry: Mapping[str, Any] | None) -> int | None:
    if entry is None:
        return None
    return int(entry["ts"].time)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]
