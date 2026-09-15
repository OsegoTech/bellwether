"""Acceptance tests for ESR candidate-index construction — INDEX_ADVISOR_SPEC §C.

Spec acceptance:
  - filter {a: 5, b: {$gt: 1}} sort {c: 1} -> index [(a,1),(c,1),(b,1)] (E, S, R)
  - >16 fields truncates
  - a field that is both eq and range appears once, in the equality position

The field ORDER is computed here, deterministically. The model never chooses it.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

from bellwether.detectors.query_shape import (
    MAX_INDEX_FIELDS,
    IndexKey,
    build_candidate,
    candidate_index,
    equality_frequency,
    shape_of_query,
)

NS = "appdb.transactions"


def keys(filter: dict[str, Any], sort: Any = None, **kw: Any) -> list[tuple[str, int]]:
    return [(k.field, k.direction) for k in candidate_index(shape_of_query(NS, filter, sort), **kw)]


# --- Spec acceptance ---------------------------------------------------------------


def test_equality_then_sort_then_range() -> None:
    assert keys({"a": 5, "b": {"$gt": 1}}, {"c": 1}) == [("a", 1), ("c", 1), ("b", 1)]


def test_more_than_sixteen_fields_truncates_and_notes_it() -> None:
    filter = {f"f{i:02d}": i for i in range(20)}

    candidate = build_candidate(shape_of_query(NS, filter))

    assert MAX_INDEX_FIELDS == 16
    assert len(candidate.keys) == 16
    assert [k.field for k in candidate.keys] == [f"f{i:02d}" for i in range(16)]
    assert candidate.dropped == ("f16", "f17", "f18", "f19")


def test_a_field_both_eq_and_range_appears_once_in_the_equality_position() -> None:
    result = keys({"a": {"$in": [1, 2], "$gt": 0}, "b": {"$lt": 5}}, {"c": -1})

    assert result == [("a", 1), ("c", -1), ("b", 1)]
    assert [field for field, _ in result].count("a") == 1


# --- Overlaps between the segments ---------------------------------------------------


def test_a_field_both_eq_and_sort_keeps_the_equality_position() -> None:
    assert keys({"a": 1, "b": {"$gt": 1}}, {"a": 1, "c": -1}) == [("a", 1), ("c", -1), ("b", 1)]


def test_a_field_both_range_and_sort_takes_the_sort_position() -> None:
    # One index entry serves both the range bound and the sort.
    assert keys({"a": 1, "b": {"$gt": 1}}, {"b": -1}) == [("a", 1), ("b", -1)]


def test_sort_fields_keep_their_order_and_direction() -> None:
    assert keys({"a": 1}, {"z": -1, "m": 1, "b": -1}) == [("a", 1), ("z", -1), ("m", 1), ("b", -1)]


# --- Equality ordering: frequency, then lexical ------------------------------------------


def test_equality_fields_order_by_frequency_then_name() -> None:
    shape = shape_of_query(NS, {"b": 1, "a": 1, "c": 1})

    assert [k.field for k in candidate_index(shape)] == ["a", "b", "c"]
    assert [k.field for k in candidate_index(shape, frequency={"c": 5, "a": 2})] == ["c", "a", "b"]


def test_equality_frequency_counts_shapes() -> None:
    shapes = [
        shape_of_query(NS, {"account_id": 1, "status": 1}),
        shape_of_query(NS, {"account_id": 1, "amount": {"$gt": 1}}),
        shape_of_query(NS, {"status": 1}),
        shape_of_query(NS, {"account_id": 1}),
    ]

    assert equality_frequency(shapes) == {"account_id": 3, "status": 2}


# --- Which predicates can be indexed -----------------------------------------------------


def test_range_like_predicates_go_last_and_unindexable_ones_are_left_out() -> None:
    filter = {
        "a": 1,
        "b": {"$regex": "^x"},
        "c": {"$ne": 1},
        "d": {"$exists": True},
        "e": {"$elemMatch": {"q": 1}},
        "$or": [{"f": 1}, {"g": 2}],
    }

    assert keys(filter) == [("a", 1), ("b", 1), ("c", 1), ("d", 1)]


def test_no_indexable_field_gives_no_candidate() -> None:
    assert keys({"$or": [{"a": 1}, {"b": 2}]}) == []
    assert keys({}) == []


# --- Determinism (the model never chooses the order) --------------------------------------


def test_the_order_is_deterministic_whatever_the_input_order() -> None:
    fields = {"a": 1, "b": 2, "c": {"$gt": 1}, "d": {"$lt": 9}, "e": "x"}
    expected = keys(fields, {"s": -1})

    for permutation in itertools.permutations(fields.items()):
        assert keys(dict(permutation), {"s": -1}) == expected


def test_no_field_is_ever_emitted_twice() -> None:
    rng = random.Random(20260915)
    classes: list[Any] = [1, {"$gt": 1}, {"$in": [1, 2]}, {"$regex": "x"}, {"$ne": 1}, {"$exists": 1}]
    names = [f"f{i}" for i in range(12)]
    for _ in range(200):
        filter = {name: rng.choice(classes) for name in rng.sample(names, rng.randint(0, 12))}
        sort = {name: rng.choice([1, -1]) for name in rng.sample(names, rng.randint(0, 4))}

        fields = [field for field, _ in keys(filter, sort)]

        assert len(fields) == len(set(fields))
        assert len(fields) <= MAX_INDEX_FIELDS


def test_candidate_evidence_matches_the_executor_key_format() -> None:
    candidate = build_candidate(shape_of_query(NS, {"a": 5, "b": {"$gt": 1}}, {"c": -1}))

    evidence = candidate.as_evidence()

    assert evidence == [
        {"field": "a", "direction": 1},
        {"field": "c", "direction": -1},
        {"field": "b", "direction": 1},
    ]
    assert json.loads(json.dumps(evidence)) == evidence
    assert candidate.keys[0] == IndexKey("a", 1)
    assert (candidate.equality, candidate.sort, candidate.range) == (
        ("a",),
        (IndexKey("c", -1),),
        ("b",),
    )
