"""Tests for the data contracts in bellwether.models."""

from __future__ import annotations

from typing import Any

import pytest

from bellwether.models import Evidence, Finding, Severity, SignalClass


def finding_with_horizon(horizon_seconds: int | None) -> Finding:
    return Finding(
        signal_class=SignalClass.REPLICATION,
        failure_mode="oplog_window_below_resync",
        severity=Severity.CRITICAL,
        node="mongo-hidden.example.internal:27017",
        summary="Oplog window is 40 min.",
        evidence=(Evidence("oplog_window_seconds", 2400, "s"),),
        horizon_seconds=horizon_seconds,
    )


def test_horizon_renders_its_three_cases_distinctly() -> None:
    no_bound = finding_with_horizon(None).horizon_human()
    crossed = finding_with_horizon(0).horizon_human()
    countdown = finding_with_horizon(5400).horizon_human()

    assert no_bound == "no time bound"
    assert crossed == "already crossed"  # a present condition, not "impact in zero minutes"
    assert countdown == "~1 h"
    assert len({no_bound, crossed, countdown}) == 3


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(1, "~0 min"), (900, "~15 min"), (3599, "~59 min"), (3600, "~1 h"), (86_400, "~1 d")],
)
def test_positive_horizon_keeps_the_countdown(seconds: int, expected: str) -> None:
    assert finding_with_horizon(seconds).horizon_human() == expected


def test_only_zero_means_already_crossed() -> None:
    rendered: dict[Any, str] = {s: finding_with_horizon(s).horizon_human() for s in (None, 0, 1)}

    assert [s for s, text in rendered.items() if text == "already crossed"] == [0]
