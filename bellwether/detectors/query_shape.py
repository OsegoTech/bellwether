"""Query-shape normalisation (INDEX_ADVISOR_SPEC §B).

Profiler entries carry literal values from application documents — account
numbers, amounts, timestamps. None of that may travel past the collector. A
filter is reduced to its field names and the *class* of operator applied to
each field, which is also exactly what an index decision needs:

    eq        equality: a literal, ``$eq``, or ``$in`` of scalars
    range     ``$gt`` / ``$gte`` / ``$lt`` / ``$lte``
    regex     ``$regex`` or a regex literal
    negation  ``$ne`` / ``$nin`` / ``$not`` (and fields under ``$nor``)
    exists    ``$exists``
    or        a field that appears only inside ``$or`` branches
    other     anything else (``$elemMatch``, ``$all``, ``$type``, geo, ``$in`` of
              non-scalars, ...)

A field carrying several operators takes the strongest class, in the order
above: ``{a: {$in: [1, 2], $gt: 0}}`` is equality.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from bson.regex import Regex

EQ = "eq"
RANGE = "range"
REGEX = "regex"
NEGATION = "negation"
EXISTS = "exists"
OR = "or"
OTHER = "other"

_PRECEDENCE = (EQ, RANGE, REGEX, NEGATION, EXISTS, OR, OTHER)

_OPERATOR_CLASS = {
    "$eq": EQ,
    "$gt": RANGE,
    "$gte": RANGE,
    "$lt": RANGE,
    "$lte": RANGE,
    "$ne": NEGATION,
    "$nin": NEGATION,
    "$not": NEGATION,
    "$regex": REGEX,
    "$options": REGEX,
    "$exists": EXISTS,
}


def operator_class(condition: Any) -> str:
    """The class of the condition applied to one field."""
    if isinstance(condition, (Regex, re.Pattern)):
        return REGEX
    if not _is_operator_document(condition):
        return EQ  # a literal, or equality on an embedded document
    classes = set()
    for operator, argument in condition.items():
        if operator == "$in":
            classes.add(EQ if _all_scalars(argument) else OTHER)
        else:
            classes.add(_OPERATOR_CLASS.get(operator, OTHER))
    return _strongest(classes)


def redact_filter(filter: Mapping[str, Any]) -> dict[str, str]:
    """``{field: class}`` for every field the filter constrains — no values."""
    classes: dict[str, set[str]] = {}
    _collect(filter, classes, branch=None)
    return {field: _strongest(found) for field, found in sorted(classes.items())}


def sort_spec(sort: Any) -> list[list[Any]]:
    """``[[field, 1 | -1], ...]`` in sort order; ``$meta`` sorts are skipped."""
    if not isinstance(sort, Mapping):
        return []
    spec: list[list[Any]] = []
    for field, direction in sort.items():
        if isinstance(direction, bool) or not isinstance(direction, (int, float)) or not direction:
            continue
        spec.append([field, 1 if direction > 0 else -1])
    return spec


def leading_match_and_sort(pipeline: Sequence[Any]) -> tuple[Mapping[str, Any], Any]:
    """The filter and sort an aggregation's leading stages apply.

    Consecutive leading ``$match`` stages coalesce into one filter, as the
    server's optimizer coalesces them; the first ``$sort`` directly after them
    is the sort. This is an approximation: a ``$match`` after ``$project``,
    ``$unwind`` or ``$lookup`` is not seen, mirroring MongoDB's own caveat
    about its query shapes for pipelines.
    """
    stages = [stage for stage in pipeline if isinstance(stage, Mapping)]
    clauses: list[Any] = []
    position = 0
    while position < len(stages) and "$match" in stages[position]:
        clauses.append(stages[position]["$match"])
        position += 1
    sort = stages[position].get("$sort", {}) if position < len(stages) else {}
    if not clauses:
        return {}, sort
    return (clauses[0] if len(clauses) == 1 else {"$and": clauses}), sort


def redact_op(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """A system.profile entry reduced to what shape analysis needs, or None.

    Only find and aggregate (and getMores continuing them) are shaped; other
    commands return None. Client, user, session, and every literal value are
    dropped — the result is JSON-native and safe to carry through the pipeline.
    """
    op = entry.get("op")
    command = entry.get("originatingCommand") if op == "getmore" else entry.get("command")
    if not isinstance(command, Mapping):
        return None
    if "find" in command:
        kind, collection = "find", command.get("find")
        filter, sort = command.get("filter") or {}, command.get("sort") or {}
    elif "aggregate" in command:
        kind, collection = "aggregate", command.get("aggregate")
        filter, sort = leading_match_and_sort(command.get("pipeline") or [])
    else:
        return None
    if not isinstance(collection, str):
        return None  # e.g. a database-level aggregate({aggregate: 1})
    db = str(entry.get("ns", "")).split(".", 1)[0]
    return {
        "op": str(op),
        "ns": f"{db}.{collection}",
        "command": kind,
        "filter": redact_filter(filter if isinstance(filter, Mapping) else {}),
        "sort": sort_spec(sort),
        "millis": _count(entry.get("millis")),
        "docsExamined": _count(entry.get("docsExamined")),
        "keysExamined": _count(entry.get("keysExamined")),
        "nreturned": _count(entry.get("nreturned")),
        "responseLength": _count(entry.get("responseLength")),
        "planSummary": str(entry.get("planSummary") or ""),
        "ts": iso(entry.get("ts")),
    }


def iso(value: Any) -> str | None:
    """UTC ISO-8601; pymongo's naive datetimes are UTC."""
    if not isinstance(value, datetime):
        return None
    moment: datetime = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _collect(filter: Mapping[str, Any], classes: dict[str, set[str]], branch: str | None) -> None:
    for key, value in filter.items():
        if key == "$and" and isinstance(value, list):
            for clause in value:
                if isinstance(clause, Mapping):
                    _collect(clause, classes, branch)
        elif key in ("$or", "$nor") and isinstance(value, list):
            for clause in value:
                if isinstance(clause, Mapping):
                    _collect(clause, classes, OR if key == "$or" else NEGATION)
        elif key.startswith("$"):
            continue  # $expr, $text, $where, $comment: no indexable field
        else:
            classes.setdefault(key, set()).add(branch or operator_class(value))


def _is_operator_document(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(k, str) and k.startswith("$") for k in value)
    )


def _all_scalars(values: Any) -> bool:
    return isinstance(values, (list, tuple)) and all(
        not isinstance(v, (Mapping, list, tuple, Regex, re.Pattern)) for v in values
    )


def _strongest(classes: Iterable[str]) -> str:
    found = set(classes)
    return next(c for c in _PRECEDENCE if c in found)


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
