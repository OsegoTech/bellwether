"""Oplog window detector — failure mode ``oplog_window_below_resync``.

The oplog is a capped collection: it holds a fixed number of bytes, so the
seconds of history it holds (the *window*) shrink as the write rate rises. A
secondary taken down for maintenance resumes by replaying the oplog from where
it stopped. If the entries it needs have already been truncated, it cannot
resume and needs a full initial sync — hours of copying on a large dataset,
with the replica set a member short meanwhile.

The rule compares the window against a conservative resync estimate, not a bare
threshold:

- ``resync_seconds`` = how long a secondary may be down (config
  ``maintenance_window_seconds``, default 3600).
- window < resync                  -> CRITICAL (a secondary down that long is lost)
- window < resync x safety_factor  -> WARNING  (margin is thin; default 2.0)

Horizon: when the collector sampled a live write rate ``r`` above the oplog's
mean rate ``a``, the window shrinks linearly. With the oplog full (conservative
— free space only delays it), after ``t`` seconds at rate ``r`` the window is
``W + t * (1 - r / a)``, converging on ``size / r`` once all old entries are
gone. If that steady state is below resync, the crossing is at
``t = (W - resync) / (r / a - 1)``. Otherwise there is no time bound (None).
Already below resync -> horizon 0.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar

from bellwether.collectors.oplog_window import RATE_SAMPLED
from bellwether.config import OplogWindowDetectorConfig
from bellwether.detectors.base import Detector
from bellwether.models import Evidence, Finding, Severity, Signal, SignalClass

logger = logging.getLogger(__name__)


class OplogWindowDetector(Detector):
    failure_mode: ClassVar[str] = "oplog_window_below_resync"

    def __init__(self, config: OplogWindowDetectorConfig | None = None) -> None:
        self._config = config or OplogWindowDetectorConfig()

    def evaluate(self, signals: Sequence[Signal]) -> Finding | None:
        candidates = [s for s in signals if s.source == "oplog_window"]
        if not candidates:
            return None
        signal = max(candidates, key=lambda s: s.collected_at)

        window = int(signal.get("oplog_window_seconds"))
        resync = self._config.maintenance_window_seconds
        factor = self._config.safety_factor
        warn_below = resync * factor
        if window >= warn_below:
            return None

        severity = Severity.CRITICAL if window < resync else Severity.WARNING
        projected = _projected_window(signal)
        horizon = _horizon(window, resync, signal, projected)

        evidence = [
            Evidence("oplog_window_seconds", window, "s"),
            Evidence("resync_seconds", resync, "s"),
            Evidence("safety_factor", factor),
            Evidence("warning_threshold_seconds", _whole(warn_below), "s"),
            Evidence("oplog_size_bytes", signal.get("oplog_size_bytes"), "bytes"),
            Evidence("write_rate_bytes_per_sec", signal.get("write_rate_bytes_per_sec"), "bytes/s"),
            Evidence("write_rate_source", signal.get("write_rate_source")),
        ]
        if projected is not None:
            evidence.append(Evidence("projected_window_seconds", projected, "s"))

        finding = Finding(
            signal_class=SignalClass.REPLICATION,
            failure_mode=self.failure_mode,
            severity=severity,
            node=signal.node,
            summary=_summary(signal.node, window, resync, factor, severity),
            evidence=tuple(evidence),
            horizon_seconds=horizon,
            signals=(signal,),
        )
        logger.info(
            "finding",
            extra={
                "failure_mode": self.failure_mode,
                "severity": severity.value,
                "node": signal.node,
                "oplog_window_seconds": window,
                "resync_seconds": resync,
                "horizon_seconds": horizon,
            },
        )
        return finding


def _projected_window(signal: Signal) -> int | None:
    """Steady-state window at the sampled rate; None without a live sample."""
    try:
        if signal.get("write_rate_source") != RATE_SAMPLED:
            return None
        rate = float(signal.get("write_rate_bytes_per_sec"))
        size = float(signal.get("oplog_size_bytes"))
    except KeyError:
        return None
    if rate <= 0:
        return None
    return round(size / rate)


def _horizon(window: int, resync: int, signal: Signal, projected: int | None) -> int | None:
    if window < resync:
        return 0
    if projected is None or projected >= resync:
        return None
    mean = float(signal.get("oplog_mean_rate_bytes_per_sec"))
    rate = float(signal.get("write_rate_bytes_per_sec"))
    if mean <= 0 or rate <= mean:
        return None
    return round((window - resync) / (rate / mean - 1))


def _summary(node: str, window: int, resync: int, factor: float, severity: Severity) -> str:
    observed = f"Oplog window on {node} is {_human(window)} ({window} s)"
    estimate = f"the {_human(resync)} ({resync} s) resync estimate"
    if severity is Severity.CRITICAL:
        return (
            f"{observed}, below {estimate}: a secondary down that long could not "
            "catch up and would need a full initial sync."
        )
    return f"{observed}, inside the {factor:g}x safety margin over {estimate}."


def _human(seconds: int) -> str:
    if seconds < 3 * 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _whole(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value
