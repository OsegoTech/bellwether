"""Index Advisor detector (INDEX_ADVISOR_SPEC §E) — Community Edition's
Performance Advisor, deterministic.

From ``query_profile`` signals it finds three failure modes:

``profiler_disabled`` (INFO)
    Profiling is off on a database on a member (the level is per mongod), so
    that member's slow queries cannot be analysed.

``missing_index_collscan`` (WARNING, or CRITICAL past the high bars)
    A query shape reads more than ``targeting_ratio_threshold`` documents per
    document returned and at least ``wasted_bytes_floor`` bytes for nothing,
    and no existing index supports it. The ESR candidate index is computed
    here and carried in the evidence, so the model reasons about it and never
    invents it. Findings are ranked by Impact (wasted bytes) and capped at
    ``max_suggestions``, as Performance Advisor caps its list at 20 shapes.

``redundant_index`` (INFO)
    An index that is a strict prefix of another index, or one with no
    accesses on any member over at least ``min_unused_index_age_seconds`` of
    statistics. Never ``_id``, a unique index (it backs a constraint), a TTL
    index (the TTL monitor's deletes are not counted as accesses), or a hidden
    one. A replica set has no shard key, so none is excluded for that.
    Dropping is always a human decision.

**Members are merged per collection.** The collector sweeps every member,
because the profiler and index counters are per mongod. A collection's ops
from all members are pooled into one finding per query shape, placed on the
member where that shape wastes most. An index is called unused only if it
has zero accesses on *every* member, every member's counters have run long
enough (the youngest decides), and no member was unreachable: an index busy
on the primary is not unused because the backup never touched it.

**Profiler levels are written into findings as text** (``profiler_level_text``):
an unreadable level — the read identity lacks enableProfiler, which only
dbAdmin grants — becomes "unknown (not readable without dbAdmin; not
necessarily off)", never a bare None that reads as "off"; a real 0 is "off".
Detection itself still uses the raw level from each signal.

An existing index **supports** a shape when its leading fields are the
candidate's equality fields (any order, any direction), then its sort fields
in order (directions as given or all reversed), then its range fields — i.e.
it is the candidate up to ESR-equivalent reordering, possibly with more fields
after. Hidden, partial, and sparse indexes are never counted as supporting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from bellwether.config import IndexAdvisorDetectorConfig
from bellwether.detectors.base import Detector
from bellwether.detectors.query_shape import (
    Candidate,
    IndexKey,
    QueryShape,
    ShapeImpact,
    build_candidate,
    equality_frequency,
    impact,
    rank_impacts,
    shape_of,
)
from bellwether.models import Evidence, Finding, Severity, Signal, SignalClass

logger = logging.getLogger(__name__)

PROFILER_DISABLED = "profiler_disabled"
MISSING_INDEX_COLLSCAN = "missing_index_collscan"
REDUNDANT_INDEX = "redundant_index"

# How a profiler level is written into findings — what the analysis stage, and
# every human reading a proposal, sees. An unreadable level (the accepted
# enableProfiler/dbAdmin blind spot) must never read as "off".
PROFILER_LEVEL_UNKNOWN = "unknown (not readable without dbAdmin; not necessarily off)"
PROFILER_LEVEL_TEXT = {
    0: "off (level 0: profiling disabled)",
    1: "on (level 1: operations slower than slowms)",
    2: "on (level 2: all operations)",
}


def profiler_level_text(level: Any) -> str:
    """A self-explaining rendering of a profiler level for evidence."""
    if level is None:
        return PROFILER_LEVEL_UNKNOWN
    if isinstance(level, int) and not isinstance(level, bool):
        return PROFILER_LEVEL_TEXT.get(level, f"level {level}")
    return f"level {level}"


@dataclass(frozen=True)
class _CollectionView:
    """One collection as every swept member reported it."""

    db: str
    collection: str
    nodes: tuple[str, ...]  # members that reported it, in sweep order
    ops: tuple[tuple[str, Mapping[str, Any]], ...]  # (member, redacted op)
    indexes: tuple[dict[str, Any], ...]  # merged across members by name
    doc_count: int | None
    avg_object_size: float | None
    profiler_levels: dict[str, Any]
    slow_ms: Any
    unreachable: tuple[str, ...]
    collected_at: datetime
    signals: tuple[Signal, ...]

    @property
    def namespace(self) -> str:
        return f"{self.db}.{self.collection}"


class IndexAdvisorDetector(Detector):
    # The detector's name; the findings it emits carry one of failure_modes.
    failure_mode: ClassVar[str] = "index_advisor"
    failure_modes: ClassVar[tuple[str, ...]] = (
        PROFILER_DISABLED,
        MISSING_INDEX_COLLSCAN,
        REDUNDANT_INDEX,
    )

    def __init__(self, config: IndexAdvisorDetectorConfig | None = None) -> None:
        self._config = config or IndexAdvisorDetectorConfig()

    def evaluate(self, signals: Sequence[Signal]) -> Finding | None:
        """The most important finding; the pipeline uses evaluate_all for all of them."""
        findings = self.evaluate_all(signals)
        return findings[0] if findings else None

    def evaluate_all(self, signals: Sequence[Signal]) -> list[Finding]:
        """Missing-index findings by Impact (capped), then profiler-disabled, then redundant."""
        disabled: list[Finding] = []
        grouped: dict[tuple[str, str], list[tuple[Signal, dict[str, Any]]]] = {}
        for signal in signals:
            if signal.source != "query_profile":
                continue
            evidence = {e.name: e.value for e in signal.evidence}
            if "collection" not in evidence:
                if evidence.get("profiler_level") == 0:
                    disabled.append(self._profiler_disabled(signal, evidence))
                continue
            key = (str(evidence.get("db")), str(evidence.get("collection")))
            grouped.setdefault(key, []).append((signal, evidence))

        missing: dict[str, Finding] = {}
        impacts: list[ShapeImpact] = []
        redundant: list[Finding] = []
        for members in grouped.values():
            view = _merge(members)
            for result, finding in self._missing_indexes(view):
                impacts.append(result)
                missing[result.shape_key] = finding
            redundant.extend(self._redundant_indexes(view))

        top = [missing[i.shape_key] for i in rank_impacts(impacts)][: self._config.max_suggestions]
        findings = [*top, *disabled, *redundant]
        logger.info(
            "index advisor evaluated",
            extra={
                MISSING_INDEX_COLLSCAN: len(top),
                PROFILER_DISABLED: len(disabled),
                REDUNDANT_INDEX: len(redundant),
                "missing_index_candidates_cut": max(len(impacts) - len(top), 0),
            },
        )
        return findings

    # --- profiler_disabled ------------------------------------------------------

    def _profiler_disabled(self, signal: Signal, evidence: Mapping[str, Any]) -> Finding:
        db = str(evidence.get("db"))
        slow_ms = evidence.get("slow_ms")
        return Finding(
            signal_class=SignalClass.PERFORMANCE,
            failure_mode=PROFILER_DISABLED,
            severity=Severity.INFO,
            node=signal.node,
            summary=(
                f"Profiling is off on {db}, so slow queries cannot be analysed; enabling it "
                f"(level 1, slowms={slow_ms}) has a small overhead."
            ),
            evidence=(
                Evidence("db", db),
                Evidence("subject", db),
                Evidence("profiler_level", profiler_level_text(0)),
                Evidence("slow_ms", slow_ms, "ms"),
            ),
            signals=(signal,),
        )

    # --- missing_index_collscan -------------------------------------------------

    def _missing_indexes(self, view: _CollectionView) -> Iterator[tuple[ShapeImpact, Finding]]:
        cfg = self._config
        by_shape: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
        for node, record in view.ops:
            by_shape.setdefault(shape_of(record).key, []).append((node, record))
        shapes = {key: shape_of(pairs[0][1]) for key, pairs in by_shape.items()}
        frequency = equality_frequency(shapes.values())
        for key, pairs in by_shape.items():
            shape = shapes[key]
            candidate = build_candidate(shape, frequency)
            if not candidate.keys:
                continue  # nothing a compound index could serve
            result = impact([r for _, r in pairs], avg_object_size=view.avg_object_size)
            if (
                result.targeting_ratio < cfg.targeting_ratio_threshold
                or result.wasted_bytes < cfg.wasted_bytes_floor
            ):
                continue
            if any(_supports(index, candidate) for index in view.indexes):
                continue
            yield result, self._missing_index_finding(view, shape, candidate, result, pairs)

    def _missing_index_finding(
        self,
        view: _CollectionView,
        shape: QueryShape,
        candidate: Candidate,
        result: ShapeImpact,
        pairs: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> Finding:
        cfg = self._config
        observed = {n: sum(1 for node, _ in pairs if node == n) for n in view.nodes}
        observed = {n: count for n, count in observed.items() if count}
        waste = {
            n: impact([r for node, r in pairs if node == n], view.avg_object_size).wasted_bytes
            for n in observed
        }
        busiest = min(observed, key=lambda n: (-waste[n], view.nodes.index(n)))
        critical = (
            result.targeting_ratio >= cfg.critical_targeting_ratio
            and result.wasted_bytes >= cfg.critical_wasted_bytes
        )
        items = [
            Evidence("db", view.db),
            Evidence("collection", view.collection),
            Evidence("namespace", view.namespace),
            Evidence("subject", f"{view.namespace}#{shape.key}"),
            Evidence("shape_key", shape.key),
            Evidence("query_shape", shape.as_evidence()),
            Evidence("op_count", result.op_count),
            Evidence("observed_on", observed),
            Evidence("targeting_ratio", round(result.targeting_ratio, 1)),
            Evidence("docs_examined", result.total_docs_examined, "docs"),
            Evidence("docs_returned", result.total_docs_returned, "docs"),
            Evidence("wasted_bytes", result.wasted_bytes, "bytes"),
            Evidence("avg_object_size", result.avg_object_size, "bytes"),
            Evidence("avg_object_size_source", result.avg_object_size_source),
            Evidence("total_millis", result.total_millis, "ms"),
            Evidence("worst_millis", result.worst_millis, "ms"),
            Evidence("candidate_index", candidate.as_evidence()),
            Evidence(
                "existing_indexes",
                [{"name": i.get("name"), "key": list(i.get("key") or [])} for i in view.indexes],
            ),
            Evidence("collection_doc_count", view.doc_count, "docs"),
            Evidence("members", list(view.nodes)),
            Evidence("members_unreachable", list(view.unreachable)),
            Evidence(
                "profiler_levels",
                {node: profiler_level_text(level) for node, level in view.profiler_levels.items()},
            ),
            Evidence("slow_ms", view.slow_ms, "ms"),
        ]
        if candidate.dropped:
            items.append(Evidence("candidate_index_dropped_fields", list(candidate.dropped)))
        summary = (
            f"Query shape {_shape_text(shape)} on {view.namespace} examined "
            f"{result.total_docs_examined:,} documents to return {result.total_docs_returned:,} "
            f"({result.targeting_ratio:,.0f} per document returned) across {result.op_count} "
            f"slow op(s), about {_bytes_text(result.wasted_bytes)} read for nothing; no existing "
            f"index supports it. ESR candidate index: {_keys_text(candidate.keys)}."
        )
        return Finding(
            signal_class=SignalClass.PERFORMANCE,
            failure_mode=MISSING_INDEX_COLLSCAN,
            severity=Severity.CRITICAL if critical else Severity.WARNING,
            node=busiest,
            summary=summary,
            evidence=tuple(items),
            signals=view.signals,
        )

    # --- redundant_index ----------------------------------------------------------

    def _redundant_indexes(self, view: _CollectionView) -> Iterator[Finding]:
        indexes = list(view.indexes)
        for index in indexes:
            if _protected(index):
                continue
            covering = _covering_index(index, indexes)
            age = _stats_age_seconds(index, view.collected_at)
            if covering is not None:
                yield self._redundant_finding(view, index, "prefix", age, covering)
            elif (
                int(index.get("accesses_ops") or 0) == 0
                and index.get("seen_on_every_member")
                and not view.unreachable
                and age is not None
                and age >= self._config.min_unused_index_age_seconds
            ):
                yield self._redundant_finding(view, index, "unused", age, None)

    def _redundant_finding(
        self,
        view: _CollectionView,
        index: Mapping[str, Any],
        reason: str,
        age: int | None,
        covering: Mapping[str, Any] | None,
    ) -> Finding:
        name = str(index.get("name"))
        key = list(index.get("key") or [])
        items = [
            Evidence("db", view.db),
            Evidence("collection", view.collection),
            Evidence("namespace", view.namespace),
            Evidence("subject", f"{view.namespace}#{name}"),
            Evidence("index_name", name),
            Evidence("index_key", key),
            Evidence("accesses_ops", int(index.get("accesses_ops") or 0)),
            Evidence("accesses_by_node", dict(index.get("accesses_by_node") or {})),
            Evidence("accesses_since", index.get("accesses_since")),
            Evidence("index_stats_age_seconds", age, "s"),
            Evidence("members", list(view.nodes)),
            Evidence("reason", reason),
        ]
        if covering is not None:
            cover_name = str(covering.get("name"))
            cover_key: list[Mapping[str, Any]] = list(covering.get("key") or [])
            items.append(Evidence("covering_index", {"name": cover_name, "key": cover_key}))
            summary = (
                f"Index {name} {_record_keys_text(key)} on {view.namespace} is a prefix of "
                f"{cover_name} {_record_keys_text(cover_key)}, which serves the same "
                "queries; it costs write throughput and disk for no read benefit."
            )
        else:
            days = (age or 0) // 86_400
            summary = (
                f"Index {name} {_record_keys_text(key)} on {view.namespace} has had no "
                f"accesses on any of the {len(view.nodes)} member(s) swept since "
                f"{index.get('accesses_since')} ({days} days of index statistics); every write "
                "still maintains it."
            )
        return Finding(
            signal_class=SignalClass.PERFORMANCE,
            failure_mode=REDUNDANT_INDEX,
            severity=Severity.INFO,
            node=view.nodes[0],
            summary=summary,
            evidence=tuple(items),
            signals=view.signals,
        )


def _merge(members: Sequence[tuple[Signal, Mapping[str, Any]]]) -> _CollectionView:
    """One collection's signals from every member, merged."""
    first = members[0][1]
    nodes: list[str] = []
    ops: list[tuple[str, Mapping[str, Any]]] = []
    merged: dict[str, dict[str, Any]] = {}
    since_by_index: dict[str, list[Any]] = {}
    counts: list[int] = []
    sizes: list[float] = []
    levels: dict[str, Any] = {}
    unreachable: list[str] = []
    for signal, evidence in members:
        node = signal.node
        if node not in nodes:
            nodes.append(node)
        ops.extend((node, r) for r in evidence.get("ops") or [] if isinstance(r, Mapping))
        for index in evidence.get("index_stats") or []:
            if not isinstance(index, Mapping):
                continue
            name = str(index.get("name"))
            entry = merged.setdefault(name, {**index, "accesses_ops": 0, "accesses_by_node": {}})
            accesses = int(index.get("accesses_ops") or 0)
            entry["accesses_ops"] += accesses
            entry["accesses_by_node"][node] = accesses
            since_by_index.setdefault(name, []).append(index.get("accesses_since"))
        count = evidence.get("collection_doc_count")
        if isinstance(count, int) and not isinstance(count, bool):
            counts.append(count)
        size = evidence.get("avg_object_size")
        if isinstance(size, (int, float)) and size:
            sizes.append(float(size))
        levels[node] = evidence.get("profiler_level")
        unreachable.extend(n for n in evidence.get("members_unreachable") or [] if n not in unreachable)
    for name, entry in merged.items():
        # The youngest counters decide how long "no accesses" has been observed.
        entry["accesses_since"] = _latest(since_by_index[name])
        entry["seen_on_every_member"] = set(entry["accesses_by_node"]) == set(nodes)
    return _CollectionView(
        db=str(first.get("db")),
        collection=str(first.get("collection")),
        nodes=tuple(nodes),
        ops=tuple(ops),
        indexes=tuple(merged.values()),
        doc_count=max(counts) if counts else None,
        avg_object_size=sizes[0] if sizes else None,
        profiler_levels=levels,
        slow_ms=first.get("slow_ms"),
        unreachable=tuple(unreachable),
        collected_at=max(signal.collected_at for signal, _ in members),
        signals=tuple(signal for signal, _ in members),
    )


def _latest(stamps: Sequence[Any]) -> str | None:
    """The most recent ISO timestamp, or None if any is missing or unreadable."""
    parsed: list[tuple[datetime, str]] = []
    for stamp in stamps:
        moment = _parse(stamp)
        if moment is None:
            return None
        parsed.append((moment, str(stamp)))
    return max(parsed)[1] if parsed else None


def _parse(stamp: Any) -> datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _supports(index: Mapping[str, Any], candidate: Candidate) -> bool:
    """Whether `index` is the candidate up to ESR-equivalent reordering."""
    if index.get("hidden") or index.get("partial") or index.get("sparse"):
        return False
    keys = [(k.get("field"), k.get("direction")) for k in index.get("key") or []]
    n_eq, n_sort, n_range = len(candidate.equality), len(candidate.sort), len(candidate.range)
    if len(keys) < n_eq + n_sort + n_range:
        return False
    if {field for field, _ in keys[:n_eq]} != set(candidate.equality):
        return False
    wanted = [(k.field, k.direction) for k in candidate.sort]
    got = keys[n_eq : n_eq + n_sort]
    if got != wanted and got != [(field, -direction) for field, direction in wanted]:
        return False
    tail = keys[n_eq + n_sort : n_eq + n_sort + n_range]
    return {field for field, _ in tail} == set(candidate.range)


def _protected(index: Mapping[str, Any]) -> bool:
    return (
        index.get("name") == "_id_"
        or bool(index.get("unique"))
        or bool(index.get("ttl"))
        or bool(index.get("hidden"))
    )


def _covering_index(
    index: Mapping[str, Any], indexes: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Another index of which `index` is a strict prefix (directions equal or all reversed)."""
    if index.get("partial") or index.get("sparse"):
        return None
    keys = _ordered_keys(index)
    if not keys:
        return None
    reversed_keys = [(field, -direction) for field, direction in keys]
    for other in indexes:
        if other is index or other.get("hidden") or other.get("partial") or other.get("sparse"):
            continue
        other_keys = _ordered_keys(other)
        if len(other_keys) > len(keys) and other_keys[: len(keys)] in (keys, reversed_keys):
            return other
    return None


def _ordered_keys(index: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The index keys if every key is ascending/descending; [] for text, hashed, geo."""
    keys: list[tuple[str, int]] = []
    for k in index.get("key") or []:
        direction = k.get("direction")
        if isinstance(direction, bool) or direction not in (1, -1):
            return []
        keys.append((str(k.get("field")), int(direction)))
    return keys


def _stats_age_seconds(index: Mapping[str, Any], now: datetime) -> int | None:
    moment = _parse(index.get("accesses_since"))
    if moment is None:
        return None
    reference = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return int((reference - moment).total_seconds())


def _shape_text(shape: QueryShape) -> str:
    text = "{" + ", ".join(f"{field}: {cls}" for field, cls in shape.filter) + "}"
    if shape.sort:
        text += " sort {" + ", ".join(f"{field}: {d}" for field, d in shape.sort) + "}"
    return text


def _keys_text(keys: Sequence[IndexKey]) -> str:
    return "{" + ", ".join(f"{k.field}: {k.direction}" for k in keys) + "}"


def _record_keys_text(keys: Sequence[Mapping[str, Any]]) -> str:
    return "{" + ", ".join(f"{k.get('field')}: {k.get('direction')}" for k in keys) + "}"


def _bytes_text(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{count} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"
