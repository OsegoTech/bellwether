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

One deliberate exception to the single-node default, in two instances. Most
diagnostics come from the hidden backup node, sparing the voters. But a signal
that depends on traffic exists only on the member that served the traffic, so
it must be read where the traffic is:

1. ``current_op_all_nodes`` runs ``$currentOp`` on every configured member.
   An operation is visible only on the mongod running it — a runaway query on
   the primary never appears in the backup node's currentOp — and killOp must
   be sent to that same member.
2. ``member_reader(node)`` pins a read client to one member for the
   query-profile sweep: each member's profiler (``system.profile`` and its
   level), ``$indexStats`` and ``$collStats``. The application's queries run
   on the primary; read only from the backup, the profiler would be empty and
   every index would look unused.

Both are narrow and read-only. They reach only configured members, each over
its own short-lived direct connection, with the same read identity and TLS
material; a member reader is this same class, so it has these read helpers and
nothing else; and the serving connection every other read uses is untouched.
Whatever is not traffic-dependent — oplog stats, replica-set status, the list
of databases — stays on the backup node.

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
class ProfilingStatus:
    level: int  # 0 off, 1 slow operations, 2 all operations
    slow_ms: int | None
    sample_rate: float | None


@dataclass(frozen=True)
class CollectionStats:
    count: int
    avg_obj_size: float | None  # absent for an empty collection
    size_bytes: int


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

    def profiling_status(self, db: str) -> ProfilingStatus:
        """The profiler level on `db`, as db.getProfilingStatus() reports it.

        Sends ``{profile: -1}``, which only reads the level. The forms that set
        it (0, 1, 2) are unreachable from here, which is also why ``profile`` is
        not in run_admin_command's allowlist.

        Even reading the level needs the ``enableProfiler`` action, which
        MongoDB grants only through ``dbAdmin`` — together with createIndex,
        dropIndex and dropDatabase. The read identity deliberately does not
        hold it, so on an enforcing cluster this raises OperationFailure
        (Unauthorized) and callers treat the level as unknown.
        """
        reply = self._read(
            "profiling_status", lambda c: c[db].command({"profile": -1}), db=db
        )
        return ProfilingStatus(
            level=int(reply["was"]),
            slow_ms=_opt_int(reply.get("slowms")),
            sample_rate=_opt_float(reply.get("sampleRate")),
        )

    def collection_stats(self, db: str, collection: str) -> CollectionStats:
        """Document count and average object size, from ``$collStats``."""
        stats = self._read(
            "collection_stats",
            lambda c: next(iter(c[db][collection].aggregate([{"$collStats": {"storageStats": {}}}]))),
            namespace=f"{db}.{collection}",
        )
        storage = stats.get("storageStats", {})
        return CollectionStats(
            count=int(storage.get("count", 0)),
            avg_obj_size=_opt_float(storage.get("avgObjSize")) or None,
            size_bytes=int(storage.get("size", 0)),
        )

    def members(self) -> list[str]:
        """Configured members in order — target first, then fallbacks — once each."""
        return self._members()

    def member_reader(self, node: str) -> ReadOnlyMongo:
        """A read client pinned to one configured member, with no fallback.

        The second instance of the deliberate exception (module docstring):
        the query-profile sweep reads each member's profiler and index use
        through these. Same class, same read identity and TLS material, so the
        same read-only helpers and nothing more; only configured members.
        """
        if node not in self._members():
            raise ValueError(f"{node!r} is not a configured member")
        pinned = self._config.model_copy(update={"target_node": node, "fallback_nodes": []})
        return ReadOnlyMongo(pinned, client_factory=self._factory)

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


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _ts_seconds(entry: Mapping[str, Any] | None) -> int | None:
    if entry is None:
        return None
    return int(entry["ts"].time)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]
