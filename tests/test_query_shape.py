"""Acceptance tests for query-shape normalisation — INDEX_ADVISOR_SPEC §B.

Spec acceptance:
  - two queries differing only in literal values hash to the same shape
  - a different sort or an added range field changes the shape
  - $in of scalars is equality, $gt is range
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pytest
from bson.regex import Regex

from bellwether.detectors.query_shape import (
    QueryShape,
    group_by_shape,
    operator_class,
    redact_op,
    shape_of,
    shape_of_query,
)

NS = "appdb.transactions"


# --- Spec acceptance ---------------------------------------------------------------


def test_queries_differing_only_in_literals_share_a_shape() -> None:
    a = shape_of_query(NS, {"account_id": "ACC-9931", "amount": {"$gt": 100}}, {"posted_at": -1})
    b = shape_of_query(NS, {"account_id": "ACC-0001", "amount": {"$gt": 999_999}}, {"posted_at": -1})

    assert a == b
    assert a.key == b.key


def test_a_different_sort_changes_the_shape() -> None:
    base = shape_of_query(NS, {"account_id": "x"}, {"posted_at": -1})

    assert shape_of_query(NS, {"account_id": "x"}, {"amount": -1}).key != base.key
    assert shape_of_query(NS, {"account_id": "x"}, {"posted_at": 1}).key != base.key
    assert shape_of_query(NS, {"account_id": "x"}).key != base.key


def test_an_added_range_field_changes_the_shape() -> None:
    base = shape_of_query(NS, {"account_id": "x"})

    with_range = shape_of_query(NS, {"account_id": "x", "amount": {"$gte": 10}})

    assert with_range.key != base.key


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"$in": ["a", "b", 3]}, "eq"),
        ({"$gt": 1}, "range"),
        ("literal", "eq"),
        (42, "eq"),
        ({"$eq": 5}, "eq"),
        ({"street": "Main", "no": 1}, "eq"),  # embedded-document equality
        ({"$gte": 1, "$lt": 5}, "range"),
        ({"$in": [1, 2], "$gt": 0}, "eq"),  # equality wins its position
        ({"$in": [{"a": 1}]}, "other"),  # $in of documents is not plain equality
        ({"$ne": 3}, "negation"),
        ({"$nin": [1, 2]}, "negation"),
        ({"$not": {"$gt": 5}}, "negation"),
        ({"$regex": "^ACC", "$options": "i"}, "regex"),
        (re.compile("^ACC"), "regex"),
        (Regex("^ACC"), "regex"),
        ({"$exists": True}, "exists"),
        ({"$elemMatch": {"qty": {"$gt": 1}}}, "other"),
        ({"$type": "string"}, "other"),
    ],
)
def test_operator_classes(condition: Any, expected: str) -> None:
    assert operator_class(condition) == expected


# --- Shape details ---------------------------------------------------------------


def test_field_order_in_the_filter_does_not_matter() -> None:
    a = shape_of_query(NS, {"a": 1, "b": {"$gt": 2}, "c": 3})
    b = shape_of_query(NS, {"c": 9, "a": 8, "b": {"$gt": 7}})

    assert a.key == b.key
    assert a.filter == (("a", "eq"), ("b", "range"), ("c", "eq"))


def test_logical_operators() -> None:
    shape = shape_of_query(
        NS,
        {
            "$and": [{"a": 1}, {"b": {"$lt": 5}}],
            "$or": [{"c": 1}, {"d": 2}],
            "$nor": [{"e": 1}],
            "$expr": {"$gt": ["$x", "$y"]},
            "$comment": "nightly report",
        },
    )

    assert dict(shape.filter) == {"a": "eq", "b": "range", "c": "or", "d": "or", "e": "negation"}


def test_a_different_namespace_is_a_different_shape() -> None:
    a = shape_of_query(NS, {"a": 1})
    b = shape_of_query("appdb.accounts", {"a": 1})

    assert a.key != b.key


def test_meta_sorts_are_skipped() -> None:
    shape = shape_of_query(NS, {"a": 1}, {"score": {"$meta": "textScore"}, "b": -1})

    assert shape.sort == (("b", -1),)


def test_the_key_is_stable() -> None:
    shape = shape_of_query(NS, {"a": 1, "b": {"$gt": 2}}, {"c": 1})

    assert shape.key == shape_of_query(NS, {"b": {"$lt": 9}, "a": 7}, {"c": 1}).key
    assert re.fullmatch(r"[0-9a-f]{16}", shape.key)


def test_find_and_aggregate_with_the_same_predicate_share_a_shape() -> None:
    find = redact_op(
        {
            "op": "query",
            "ns": NS,
            "command": {"find": "transactions", "filter": {"a": 1, "b": {"$gt": 2}}, "sort": {"c": 1}},
            "ts": datetime(2026, 9, 14),
        }
    )
    aggregate = redact_op(
        {
            "op": "command",
            "ns": NS,
            "command": {
                "aggregate": "transactions",
                "pipeline": [{"$match": {"a": 5}}, {"$match": {"b": {"$gt": 9}}}, {"$sort": {"c": 1}}],
            },
            "ts": datetime(2026, 9, 14),
        }
    )
    assert find is not None and aggregate is not None

    assert shape_of(find).key == shape_of(aggregate).key  # consecutive $match coalesced


def test_shape_of_a_redacted_record_matches_the_raw_query() -> None:
    record = redact_op(
        {
            "op": "query",
            "ns": NS,
            "command": {"find": "transactions", "filter": {"a": "x", "b": {"$gte": 1}}, "sort": {"c": -1}},
        }
    )
    assert record is not None

    assert shape_of(record) == shape_of_query(NS, {"a": "y", "b": {"$gte": 2}}, {"c": -1})


def test_group_by_shape() -> None:
    records: list[dict[str, Any]] = [
        {"ns": NS, "filter": {"a": "eq"}, "sort": [], "millis": 1},
        {"ns": NS, "filter": {"a": "eq", "b": "range"}, "sort": [], "millis": 2},
        {"ns": NS, "filter": {"a": "eq"}, "sort": [], "millis": 3},
    ]

    groups = group_by_shape(records)

    assert [[r["millis"] for r in group] for group in groups.values()] == [[1, 3], [2]]
    assert list(groups) == [shape_of(records[0]).key, shape_of(records[1]).key]


def test_shape_evidence_is_json_native() -> None:
    shape = shape_of_query(NS, {"a": 1, "b": {"$gt": 2}}, {"c": -1})

    evidence = shape.as_evidence()

    assert evidence == {"filter": {"a": "eq", "b": "range"}, "sort": [["c", -1]]}
    assert json.loads(json.dumps(evidence)) == evidence


def test_query_shape_is_a_value_object() -> None:
    shape = QueryShape(namespace=NS, filter=(("a", "eq"),), sort=(("b", 1),))

    assert shape == QueryShape(namespace=NS, filter=(("a", "eq"),), sort=(("b", 1),))
    assert shape.fields("eq") == ("a",)
    assert shape.fields("range") == ()
