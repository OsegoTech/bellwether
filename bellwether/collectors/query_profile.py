"""Query profile collector — the Index Advisor's input (INDEX_ADVISOR_SPEC §A).

Community Edition's equivalent of the slow-query log Performance Advisor reads
is the database profiler (``system.profile``).

**Read on every member.** The profiler, its level, and ``$indexStats`` counters
are per mongod, and the application's queries run on the primary. Read only
from the hidden backup node, the profiler is empty and every index looks
unused. This is the second instance of the deliberate exception documented in
``bellwether.mongo``: most diagnostics come from the hidden backup node to
spare the voters; traffic-dependent signals must look where the traffic is.
Listing the databases is not traffic-dependent and stays on the serving node.

The sweep, read-only, each member over its own pinned client (closed after):

1. Per application database: the profiler level (``{profile: -1}``) and,
   unless profiling is off, the newest ``system.profile`` entries for find /
   aggregate / getMore at or over ``slow_ms``, kept if their plan is a
   COLLSCAN or they examine many documents per document returned. Reading
   the level needs ``enableProfiler``, which only ``dbAdmin`` grants — with
   createIndex, dropIndex and dropDatabase — so the read identity is refused
   it by design: the level is then recorded as unknown, the profiler is read
   anyway, and profiler_disabled does not fire. An accepted blind spot.
2. For every collection with slow queries on *any* member: each member's
   ``$indexStats`` and ``$collStats`` — index use must be known everywhere,
   not only where the slow queries ran.

Signals: one per (member, collection), tagged with the member as its node and
carrying that member's ops (possibly none), index statistics, and the sweep's
``members_swept`` / ``members_unreachable``; plus one ``profiler_level: 0``
signal per (member, database) with profiling off. An unreachable member is
logged and listed, and the sweep continues; if no member can be read at all,
NoReachableNode.

Ops are reduced to query shapes (detectors.query_shape.redact_op): field names
and operator classes, never literal values, clients, or users. At most
``max_ops_per_collection`` ops are carried per signal, the slowest first.
Index definitions are reduced the same way — a partial filter expression
becomes a flag, not its literals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar

from pymongo.errors import OperationFailure, PyMongoError

from bellwether.collectors.base import Collector
from bellwether.config import QueryProfileCollectorConfig
from bellwether.detectors.query_shape import iso, redact_op
from bellwether.models import Evidence, Signal, SignalClass
from bellwether.mongo import NoReachableNode, ReadOnlyMongo

logger = logging.getLogger(__name__)

SYSTEM_DATABASES = frozenset({"admin", "local", "config"})
PROFILED_OPS = ("query", "getmore", "command")

Record = dict[str, Any]
Level = tuple[int | None, str | None]  # (profiler level or None if unreadable, why)


class QueryProfileCollector(Collector):
    name: ClassVar[str] = "query_profile"

    def __init__(self, config: QueryProfileCollectorConfig | None = None) -> None:
        self._config = config or QueryProfileCollectorConfig()

    @property
    def signal_class(self) -> SignalClass:
        return SignalClass.PERFORMANCE

    def collect(self, mongo: ReadOnlyMongo) -> Signal | None:
        """The first signal only; the pipeline uses collect_signals for all of them."""
        signals = self.collect_signals(mongo)
        return signals[0] if signals else None

    def collect_signals(self, mongo: ReadOnlyMongo) -> list[Signal]:
        databases = self._databases(mongo)  # not traffic-dependent: the serving node answers
        if not databases:
            return []
        members = mongo.members()
        readers: dict[str, ReadOnlyMongo] = {}
        unreachable: dict[str, str] = {}
        levels: dict[tuple[str, str], Level] = {}
        slow: dict[tuple[str, str], dict[str, list[Record]]] = {}
        stats: dict[tuple[str, str, str], tuple[list[Record], int | None, float | None]] = {}
        try:
            for node in members:  # pass 1: every member's profiler
                reader = mongo.member_reader(node)
                try:
                    node_levels, node_records = self._read_profiles(reader, databases)
                except (PyMongoError, NoReachableNode) as exc:
                    reader.close()
                    self._mark_unreachable(unreachable, node, exc)
                    continue
                readers[node] = reader
                for db, level in node_levels.items():
                    levels[(node, db)] = level
                for key, records in node_records.items():
                    slow.setdefault(key, {})[node] = records
                logger.info(
                    "query profile read",
                    extra={
                        "node": node,
                        "databases": databases,
                        "slow_ops": sum(len(r) for r in node_records.values()),
                    },
                )
            if not readers:
                raise NoReachableNode(unreachable)

            for db, collection in sorted(slow):  # pass 2: every member's index use
                for node in [n for n in members if n in readers]:
                    try:
                        index_stats = [
                            _index_record(doc) for doc in readers[node].index_stats(db, collection)
                        ]
                        count, avg_size = self._collection_stats(readers[node], db, collection)
                    except (PyMongoError, NoReachableNode) as exc:
                        readers.pop(node).close()
                        self._mark_unreachable(unreachable, node, exc)
                        continue
                    stats[(node, db, collection)] = (index_stats, count, avg_size)
        finally:
            for reader in readers.values():
                reader.close()

        return self._signals(members, unreachable, databases, levels, slow, stats)

    # --- reading ----------------------------------------------------------------------

    def _databases(self, mongo: ReadOnlyMongo) -> list[str]:
        if self._config.databases:
            return list(self._config.databases)
        reply = mongo.run_admin_command("listDatabases", 1, nameOnly=True)
        names = (d.get("name") for d in reply.get("databases", []))
        return sorted(n for n in names if isinstance(n, str) and n not in SYSTEM_DATABASES)

    def _read_profiles(
        self, reader: ReadOnlyMongo, databases: Sequence[str]
    ) -> tuple[dict[str, Level], dict[tuple[str, str], list[Record]]]:
        levels: dict[str, Level] = {}
        records: dict[tuple[str, str], list[Record]] = {}
        for db in databases:
            levels[db] = self._profiling_level(reader, db)
            if levels[db][0] == 0:
                continue  # profiling off on this member: nothing recorded to read
            entries = reader.profile_read(
                db,
                {"op": {"$in": list(PROFILED_OPS)}, "millis": {"$gte": self._config.slow_ms}},
                limit=self._config.max_profile_entries,
            )
            for entry in entries:
                record = redact_op(entry)
                if record is not None and self._worth_carrying(record):
                    collection = record["ns"].split(".", 1)[1]
                    records.setdefault((db, collection), []).append(record)
        return levels, records

    @staticmethod
    def _profiling_level(reader: ReadOnlyMongo, db: str) -> Level:
        # Deliberate, accepted blind spot. {profile: -1} needs the enableProfiler
        # action, and MongoDB grants that only through dbAdmin — bundled with
        # createIndex, dropIndex and dropDatabase. meetadev-ai is a read identity
        # and will not be given dbAdmin to learn a profiler level. So the level is
        # often refused: it is recorded as unknown, system.profile is read anyway
        # (reading it is covered by read@<db>, as $collStats is; $indexStats by
        # clusterMonitor), and profiler_disabled simply does not fire. No role
        # change, no new privilege.
        try:
            return reader.profiling_status(db).level, None
        except OperationFailure as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning(
                "profiler level unreadable; reading system.profile anyway",
                extra={"db": db, "node": reader.served_by, "error": error},
            )
            return None, error

    def _worth_carrying(self, record: Mapping[str, Any]) -> bool:
        if "COLLSCAN" in str(record["planSummary"]):
            return True
        ratio = float(record["docsExamined"]) / max(int(record["nreturned"]), 1)
        return ratio >= self._config.examined_returned_ratio

    @staticmethod
    def _collection_stats(
        reader: ReadOnlyMongo, db: str, collection: str
    ) -> tuple[int | None, float | None]:
        try:
            stats = reader.collection_stats(db, collection)
        except OperationFailure as exc:
            logger.warning(
                "collection stats unavailable",
                extra={"namespace": f"{db}.{collection}", "error": str(exc)[:200]},
            )
            return None, None
        return stats.count, stats.avg_obj_size

    @staticmethod
    def _mark_unreachable(unreachable: dict[str, str], node: str, exc: BaseException) -> None:
        unreachable[node] = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning(
            "member unreachable during query profile sweep; continuing without it",
            extra={"node": node, "error": unreachable[node]},
        )

    # --- signals ------------------------------------------------------------------------

    def _signals(
        self,
        members: Sequence[str],
        unreachable: Mapping[str, str],
        databases: Sequence[str],
        levels: Mapping[tuple[str, str], Level],
        slow: Mapping[tuple[str, str], Mapping[str, list[Record]]],
        stats: Mapping[tuple[str, str, str], tuple[list[Record], int | None, float | None]],
    ) -> list[Signal]:
        swept = [n for n in members if n not in unreachable]
        missing = [n for n in members if n in unreachable]
        slow_ms = self._config.slow_ms
        keyed: list[tuple[tuple[str, str, int], Signal]] = []

        for node in swept:
            for db in databases:
                if levels.get((node, db), (None, None))[0] == 0:
                    signal = Signal(
                        signal_class=self.signal_class,
                        source=self.name,
                        node=node,
                        evidence=(
                            Evidence("db", db),
                            Evidence("profiler_level", 0),
                            Evidence("slow_ms", slow_ms, "ms"),
                        ),
                    )
                    keyed.append(((db, "", members.index(node)), signal))

        for db, collection in sorted(slow):
            for node in swept:
                if (node, db, collection) not in stats:
                    continue
                index_stats, count, avg_size = stats[(node, db, collection)]
                level, status_error = levels.get((node, db), (None, None))
                records = sorted(
                    slow[(db, collection)].get(node, []), key=lambda r: (-r["millis"], r["ts"] or "")
                )
                evidence = [
                    Evidence("db", db),
                    Evidence("collection", collection),
                    Evidence("profiler_level", level),
                    Evidence("slow_ms", slow_ms, "ms"),
                    Evidence("slow_ops_seen", len(records)),
                    Evidence("ops", records[: self._config.max_ops_per_collection]),
                    Evidence("index_stats", index_stats),
                    Evidence("collection_doc_count", count, "docs"),
                    Evidence("avg_object_size", avg_size, "bytes"),
                    Evidence("members_swept", swept),
                    Evidence("members_unreachable", missing),
                ]
                if status_error is not None:
                    evidence.append(Evidence("profiler_status_error", status_error))
                signal = Signal(
                    signal_class=self.signal_class,
                    source=self.name,
                    node=node,
                    evidence=tuple(evidence),
                )
                keyed.append(((db, collection, members.index(node)), signal))

        return [signal for _, signal in sorted(keyed, key=lambda item: item[0])]


def _index_record(doc: Mapping[str, Any]) -> dict[str, Any]:
    """An $indexStats entry reduced to its definition and use — no literals."""
    spec = doc.get("spec") or {}
    key = doc.get("key") or spec.get("key") or {}
    accesses = doc.get("accesses") or {}
    since = accesses.get("since")
    return {
        "name": str(doc.get("name")),
        "key": [{"field": field, "direction": _direction(d)} for field, d in key.items()],
        "accesses_ops": int(accesses.get("ops", 0)),
        "accesses_since": iso(since) if isinstance(since, datetime) else None,
        "unique": bool(spec.get("unique")),
        "partial": "partialFilterExpression" in spec,
        "sparse": bool(spec.get("sparse")),
        "ttl": "expireAfterSeconds" in spec,
        "hidden": bool(spec.get("hidden")),
    }


def _direction(value: Any) -> int | str:
    """1 / -1 for ordered keys; the index type ("text", "hashed", "2dsphere") otherwise."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if value > 0 else -1
    return str(value)
