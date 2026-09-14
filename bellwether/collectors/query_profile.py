"""Query profile collector — the Index Advisor's input (INDEX_ADVISOR_SPEC §A).

Community Edition's equivalent of the slow-query log Performance Advisor reads
is the database profiler (``system.profile``). Per application database, from
the serving node, read-only:

1. The profiler level (``{profile: -1}``). Bellwether does not assume
   profiling is on: a database with profiling off yields a signal carrying
   ``profiler_level: 0`` and no ops, which the detector turns into a
   profiler_disabled finding. If the level cannot be read (the read identity
   may lack the privilege), it is reported as unknown and the profiler is
   read anyway.
2. The newest ``system.profile`` entries for find / aggregate / getMore at or
   over ``slow_ms``, kept if their plan is a COLLSCAN or they examine many
   documents per document returned.
3. For each collection in that slow set: ``$indexStats`` (existing indexes and
   their use) and ``$collStats`` (document count, average object size).

One signal per collection with slow queries. Ops are reduced to query shapes
(detectors.query_shape.redact_op): field names and operator classes, never
literal values, clients, or users. At most ``max_ops_per_collection`` ops are
carried, the slowest first. Index definitions are reduced the same way — a
partial filter expression becomes a flag, not its literals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar

from pymongo.errors import OperationFailure

from bellwether.collectors.base import Collector
from bellwether.config import QueryProfileCollectorConfig
from bellwether.detectors.query_shape import iso, redact_op
from bellwether.models import Evidence, Signal, SignalClass
from bellwether.mongo import ReadOnlyMongo

logger = logging.getLogger(__name__)

SYSTEM_DATABASES = frozenset({"admin", "local", "config"})
PROFILED_OPS = ("query", "getmore", "command")


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
        signals: list[Signal] = []
        for db in self._databases(mongo):
            signals.extend(self._collect_database(mongo, db))
        return signals

    def _databases(self, mongo: ReadOnlyMongo) -> list[str]:
        if self._config.databases:
            return list(self._config.databases)
        reply = mongo.run_admin_command("listDatabases", 1, nameOnly=True)
        names = (d.get("name") for d in reply.get("databases", []))
        return sorted(n for n in names if isinstance(n, str) and n not in SYSTEM_DATABASES)

    def _collect_database(self, mongo: ReadOnlyMongo, db: str) -> list[Signal]:
        slow_ms = self._config.slow_ms
        level: int | None = None
        status_error: str | None = None
        try:
            level = mongo.profiling_status(db).level
        except OperationFailure as exc:
            status_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning(
                "profiler level unreadable; reading system.profile anyway",
                extra={"db": db, "error": status_error},
            )
        node = mongo.served_by or "unknown"
        if level == 0:
            logger.info("profiling is off", extra={"db": db, "node": node})
            return [
                Signal(
                    signal_class=self.signal_class,
                    source=self.name,
                    node=node,
                    evidence=(
                        Evidence("db", db),
                        Evidence("profiler_level", 0),
                        Evidence("slow_ms", slow_ms, "ms"),
                    ),
                )
            ]

        entries = mongo.profile_read(
            db,
            {"op": {"$in": list(PROFILED_OPS)}, "millis": {"$gte": slow_ms}},
            limit=self._config.max_profile_entries,
        )
        by_collection: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            record = redact_op(entry)
            if record is not None and self._worth_carrying(record):
                collection = record["ns"].split(".", 1)[1]
                by_collection.setdefault(collection, []).append(record)

        signals = []
        for collection in sorted(by_collection):
            records = sorted(by_collection[collection], key=lambda r: (-r["millis"], r["ts"] or ""))
            index_stats = [_index_record(doc) for doc in mongo.index_stats(db, collection)]
            count, avg_size = self._collection_stats(mongo, db, collection)
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
            ]
            if status_error is not None:
                evidence.append(Evidence("profiler_status_error", status_error))
            signals.append(
                Signal(
                    signal_class=self.signal_class,
                    source=self.name,
                    node=node,
                    evidence=tuple(evidence),
                )
            )
        logger.info(
            "query profile collected",
            extra={
                "db": db,
                "node": node,
                "profile_entries": len(entries),
                "collections": sorted(by_collection),
            },
        )
        return signals

    def _worth_carrying(self, record: Mapping[str, Any]) -> bool:
        if "COLLSCAN" in str(record["planSummary"]):
            return True
        ratio = float(record["docsExamined"]) / max(int(record["nreturned"]), 1)
        return ratio >= self._config.examined_returned_ratio

    @staticmethod
    def _collection_stats(
        mongo: ReadOnlyMongo, db: str, collection: str
    ) -> tuple[int | None, float | None]:
        try:
            stats = mongo.collection_stats(db, collection)
        except OperationFailure as exc:
            logger.warning(
                "collection stats unavailable",
                extra={"namespace": f"{db}.{collection}", "error": str(exc)[:200]},
            )
            return None, None
        return stats.count, stats.avg_obj_size


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
