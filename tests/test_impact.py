"""Acceptance tests for Impact scoring — INDEX_ADVISOR_SPEC §D.

Spec acceptance:
  - a COLLSCAN examining 1e6 returning 10 scores high
  - an efficient shape (targeting ~1) scores ~0
  - ranking orders by wasted_bytes

Impact is Performance Advisor's "total wasted bytes read": documents examined
but not returned, times their size.
"""

from __future__ import annotations

from typing import Any

import pytest

from bellwether.detectors.query_shape import impact, rank_impacts, shape_of

NS = "meetadev_ledger.transactions"


def op(
    examined: int,
    returned: int,
    *,
    millis: int = 500,
    response_length: int = 0,
    filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ns": NS,
        "filter": filter if filter is not None else {"account_id": "eq"},
        "sort": [],
        "millis": millis,
        "docsExamined": examined,
        "nreturned": returned,
        "responseLength": response_length,
    }


# --- Spec acceptance ---------------------------------------------------------------


def test_a_collscan_examining_a_million_to_return_ten_scores_high() -> None:
    result = impact([op(1_000_000, 10)], avg_object_size=500)

    assert result.targeting_ratio == 100_000
    assert result.wasted_bytes == (1_000_000 - 10) * 500
    assert result.avg_object_size_source == "collstats"


def test_an_efficient_shape_scores_about_zero() -> None:
    result = impact([op(50, 50), op(120, 118)], avg_object_size=500)

    assert result.targeting_ratio == pytest.approx(1.0, abs=0.02)
    assert result.wasted_bytes == 2 * 500


def test_ranking_orders_by_wasted_bytes() -> None:
    small = impact([op(10_000, 10, filter={"a": "eq"})], avg_object_size=400)
    large = impact([op(2_000_000, 10, filter={"b": "eq"})], avg_object_size=400)
    none = impact([op(10, 10, filter={"c": "eq"})], avg_object_size=400)

    assert rank_impacts([small, none, large]) == [large, small, none]


# --- The inputs Performance Advisor documents -----------------------------------------


def test_totals_are_summed_across_the_shapes_ops() -> None:
    result = impact(
        [op(1_000, 10, millis=300), op(3_000, 20, millis=900), op(6_000, 0, millis=150)],
        avg_object_size=200,
    )

    assert result.op_count == 3
    assert (result.total_docs_examined, result.total_docs_returned) == (10_000, 30)
    assert result.targeting_ratio == pytest.approx(10_000 / 30)
    assert result.wasted_bytes == ((1_000 - 10) + (3_000 - 20) + 6_000) * 200
    assert (result.total_millis, result.worst_millis) == (1_350, 900)
    assert result.shape_key == shape_of(op(1, 1)).key


def test_returning_nothing_does_not_divide_by_zero() -> None:
    result = impact([op(50_000, 0)], avg_object_size=100)

    assert result.targeting_ratio == 50_000
    assert result.wasted_bytes == 5_000_000


def test_avg_object_size_is_estimated_from_the_ops_without_collstats() -> None:
    # 100 docs returned in 40,000 bytes of response: ~400 bytes a document.
    result = impact([op(10_000, 100, response_length=40_000)])

    assert result.avg_object_size == 400
    assert result.avg_object_size_source == "estimated_from_ops"
    assert result.wasted_bytes == (10_000 - 100) * 400


def test_collstats_size_beats_the_estimate() -> None:
    result = impact([op(10_000, 100, response_length=40_000)], avg_object_size=512)

    assert (result.avg_object_size, result.avg_object_size_source) == (512, "collstats")


def test_unknown_object_size_scores_zero_bytes() -> None:
    result = impact([op(10_000, 0)])

    assert result.avg_object_size is None
    assert result.avg_object_size_source == "unknown"
    assert result.wasted_bytes == 0
    assert result.targeting_ratio == 10_000  # still visible for the detector


def test_ties_break_on_total_millis_then_shape_key() -> None:
    fast = impact([op(1_000, 10, millis=100, filter={"a": "eq"})], avg_object_size=100)
    slow = impact([op(1_000, 10, millis=900, filter={"b": "eq"})], avg_object_size=100)

    assert rank_impacts([fast, slow]) == [slow, fast]


def test_impact_needs_at_least_one_op() -> None:
    with pytest.raises(ValueError):
        impact([])
