"""Index Advisor detector (INDEX_ADVISOR_SPEC §E) — Community Edition's
Performance Advisor, deterministic.

From ``query_profile`` signals it finds three failure modes:

``profiler_disabled`` (INFO)
    Profiling is off on a database, so its slow queries cannot be analysed.

``missing_index_collscan`` (WARNING, or CRITICAL past the high bars)
    A query shape reads more than ``targeting_ratio_threshold`` documents per
    document returned and at least ``wasted_bytes_floor`` bytes for nothing,
    and no existing index supports it. The ESR candidate index is computed
    here and carried in the evidence, so the model reasons about it and never
    invents it. Findings are ranked by Impact (wasted bytes) and capped at
    ``max_suggestions``, as Performance Advisor caps its list at 20 shapes.

``redundant_index`` (INFO)
    An index that is a strict prefix of another index, or one with no
    accesses over at least ``min_unused_index_age_seconds`` of statistics.
    Never ``_id``, a unique index (it backs a constraint), a TTL index (the TTL
    monitor's deletes are not counted as accesses), or a hidden one. A replica
    set has no shard key, so none is excluded for that. Dropping is always a
    human decision.

An existing index **supports** a shape when its leading fields are the
candidate's equality fields (any order, any direction), then its sort fields
in order (directions as given or all reversed), then its range fields — i.e.
it is the candidate up to ESR-equivalent reordering, possibly with more fields
after. Hidden, partial, and sparse indexes are never counted as supporting.

``$indexStats`` counters and the profiler are per mongod: both describe the
member they were read from, which each finding names as its node.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
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
    group_by_shape,
    impact,
    rank_impacts,
    shape_of,
)
from bellwether.models import Evidence, Finding, Severity, Signal, SignalClass

logger = logging.getLogger(__name__)

PROFILER_DISABLED = "profiler_disabled"
MISSING_INDEX_COLLSCAN = "missing_index_collscan"
REDUNDANT_INDEX = "redundant_index"


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
        missing: dict[str, Finding] = {}
        impacts: list[ShapeImpact] = []
        disabled: list[Finding] = []
        redundant: list[Finding] = []
        for signal in signals:
            if signal.source != "query_profile":
                continue
            evidence = {e.name: e.value for e in signal.evidence}
            if evidence.get("profiler_level") == 0:
                disabled.append(self._profiler_disabled(signal, evidence))
                continue
            for result, finding in self._missing_indexes(signal, evidence):
                impacts.append(result)
                missing[result.shape_key] = finding
            redundant.extend(self._redundant_indexes(signal, evidence))

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
                Evidence("profiler_level", 0),
                Evidence("slow_ms", slow_ms, "ms"),
            ),
            signals=(signal,),
        )

    # --- missing_index_collscan -------------------------------------------------

    def _missing_indexes(
        self, signal: Signal, evidence: Mapping[str, Any]
    ) -> Iterator[tuple[ShapeImpact, Finding]]:
        cfg = self._config
        records = [r for r in evidence.get("ops") or [] if isinstance(r, Mapping)]
        groups = group_by_shape(records)
        shapes = {key: shape_of(ops[0]) for key, ops in groups.items()}
        frequency = equality_frequency(shapes.values())
        existing = [i for i in evidence.get("index_stats") or [] if isinstance(i, Mapping)]
        for key, ops in groups.items():
            shape = shapes[key]
            candidate = build_candidate(shape, frequency)
            if not candidate.keys:
                continue  # nothing a compound index could serve
            result = impact(ops, avg_object_size=evidence.get("avg_object_size"))
            if (
                result.targeting_ratio < cfg.targeting_ratio_threshold
                or result.wasted_bytes < cfg.wasted_bytes_floor
            ):
                continue
            if any(_supports(index, candidate) for index in existing):
                continue
            yield result, self._missing_index_finding(
                signal, evidence, shape, candidate, result, existing
            )

    def _missing_index_finding(
        self,
        signal: Signal,
        evidence: Mapping[str, Any],
        shape: QueryShape,
        candidate: Candidate,
        result: ShapeImpact,
        existing: Sequence[Mapping[str, Any]],
    ) -> Finding:
        cfg = self._config
        db, collection = str(evidence.get("db")), str(evidence.get("collection"))
        namespace = f"{db}.{collection}"
        critical = (
            result.targeting_ratio >= cfg.critical_targeting_ratio
            and result.wasted_bytes >= cfg.critical_wasted_bytes
        )
        items = [
            Evidence("db", db),
            Evidence("collection", collection),
            Evidence("namespace", namespace),
            Evidence("subject", f"{namespace}#{shape.key}"),
            Evidence("shape_key", shape.key),
            Evidence("query_shape", shape.as_evidence()),
            Evidence("op_count", result.op_count),
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
                [{"name": i.get("name"), "key": list(i.get("key") or [])} for i in existing],
            ),
            Evidence("collection_doc_count", evidence.get("collection_doc_count"), "docs"),
            Evidence("profiler_level", evidence.get("profiler_level")),
            Evidence("slow_ms", evidence.get("slow_ms"), "ms"),
        ]
        if candidate.dropped:
            items.append(Evidence("candidate_index_dropped_fields", list(candidate.dropped)))
        summary = (
            f"Query shape {_shape_text(shape)} on {namespace} examined "
            f"{result.total_docs_examined:,} documents to return {result.total_docs_returned:,} "
            f"({result.targeting_ratio:,.0f} per document returned) across {result.op_count} "
            f"slow op(s), about {_bytes_text(result.wasted_bytes)} read for nothing; no existing "
            f"index supports it. ESR candidate index: {_keys_text(candidate.keys)}."
        )
        return Finding(
            signal_class=SignalClass.PERFORMANCE,
            failure_mode=MISSING_INDEX_COLLSCAN,
            severity=Severity.CRITICAL if critical else Severity.WARNING,
            node=signal.node,
            summary=summary,
            evidence=tuple(items),
            signals=(signal,),
        )

    # --- redundant_index ----------------------------------------------------------

    def _redundant_indexes(self, signal: Signal, evidence: Mapping[str, Any]) -> Iterator[Finding]:
        indexes = [i for i in evidence.get("index_stats") or [] if isinstance(i, Mapping)]
        for index in indexes:
            if _protected(index):
                continue
            covering = _covering_index(index, indexes)
            age = _stats_age_seconds(index, signal.collected_at)
            if covering is not None:
                yield self._redundant_finding(signal, evidence, index, "prefix", age, covering)
            elif (
                int(index.get("accesses_ops") or 0) == 0
                and age is not None
                and age >= self._config.min_unused_index_age_seconds
            ):
                yield self._redundant_finding(signal, evidence, index, "unused", age, None)

    def _redundant_finding(
        self,
        signal: Signal,
        evidence: Mapping[str, Any],
        index: Mapping[str, Any],
        reason: str,
        age: int | None,
        covering: Mapping[str, Any] | None,
    ) -> Finding:
        db, collection = str(evidence.get("db")), str(evidence.get("collection"))
        namespace = f"{db}.{collection}"
        name = str(index.get("name"))
        key = list(index.get("key") or [])
        items = [
            Evidence("db", db),
            Evidence("collection", collection),
            Evidence("namespace", namespace),
            Evidence("subject", f"{namespace}#{name}"),
            Evidence("index_name", name),
            Evidence("index_key", key),
            Evidence("accesses_ops", int(index.get("accesses_ops") or 0)),
            Evidence("accesses_since", index.get("accesses_since")),
            Evidence("index_stats_age_seconds", age, "s"),
            Evidence("reason", reason),
        ]
        if covering is not None:
            cover_name = str(covering.get("name"))
            cover_key: list[Mapping[str, Any]] = list(covering.get("key") or [])
            items.append(Evidence("covering_index", {"name": cover_name, "key": cover_key}))
            summary = (
                f"Index {name} {_record_keys_text(key)} on {namespace} is a prefix of "
                f"{cover_name} {_record_keys_text(cover_key)}, which serves the same "
                "queries; it costs write throughput and disk for no read benefit."
            )
        else:
            days = (age or 0) // 86_400
            summary = (
                f"Index {name} {_record_keys_text(key)} on {namespace} has had no accesses "
                f"since {index.get('accesses_since')} ({days} days of index statistics on "
                f"{signal.node}); every write still maintains it."
            )
        return Finding(
            signal_class=SignalClass.PERFORMANCE,
            failure_mode=REDUNDANT_INDEX,
            severity=Severity.INFO,
            node=signal.node,
            summary=summary,
            evidence=tuple(items),
            signals=(signal,),
        )


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
    since = index.get("accesses_since")
    if not isinstance(since, str):
        return None
    try:
        moment = datetime.fromisoformat(since)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
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
