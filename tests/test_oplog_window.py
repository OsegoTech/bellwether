"""Acceptance tests for the oplog-window vertical slice — BUILD_SPEC §3.4.

Spec acceptance (unit):
  - a healthy window (6 h against a 1 h resync) -> the detector returns None
  - 40 min against a 1 h resync -> CRITICAL, evidence includes the observed
    window and the resync estimate, summary states both numbers
  - 90 min -> WARNING
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from bson import Timestamp

from bellwether.collectors.base import Collector
from bellwether.collectors.oplog_window import OplogWindowCollector
from bellwether.config import MongoConfig, OplogWindowCollectorConfig, OplogWindowDetectorConfig
from bellwether.detectors.base import Detector
from bellwether.detectors.oplog_window import OplogWindowDetector
from bellwether.models import Evidence, Finding, Severity, Signal, SignalClass
from bellwether.mongo import ReadOnlyMongo
from tests.fakes import FALLBACKS, READ_URI, TARGET, WESTEUROPE, FakeCluster

MIN = 60
HOUR = 3600
OPLOG_BYTES = 990 * 2**20
T0 = 1_757_000_000


# --- helpers -----------------------------------------------------------------


def oplog_signal(
    window: int,
    *,
    rate: float | None = None,
    rate_source: str = "oplog_mean",
    size: int = OPLOG_BYTES,
    node: str = TARGET,
) -> Signal:
    mean = size / window
    return Signal(
        signal_class=SignalClass.REPLICATION,
        source="oplog_window",
        node=node,
        evidence=(
            Evidence("oplog_size_bytes", size, "bytes"),
            Evidence("oplog_used_bytes", size, "bytes"),
            Evidence("oplog_window_seconds", window, "s"),
            Evidence("oplog_mean_rate_bytes_per_sec", mean, "bytes/s"),
            Evidence("write_rate_bytes_per_sec", mean if rate is None else rate, "bytes/s"),
            Evidence("write_rate_source", rate_source),
        ),
    )


def evidence_of(finding: Finding) -> dict[str, Any]:
    return {e.name: e.value for e in finding.evidence}


def detector(**overrides: Any) -> OplogWindowDetector:
    return OplogWindowDetector(OplogWindowDetectorConfig(**overrides))


def server_status(*, secondary: bool, repl_bytes: int, uptime_ms: int) -> dict[str, Any]:
    return {
        "repl": {"setName": "rs0", "secondary": secondary, "isWritablePrimary": not secondary},
        "metrics": {"repl": {"network": {"bytes": repl_bytes}}},
        "uptimeMillis": uptime_ms,
        "ok": 1.0,
    }


def cluster_with_oplog(window: int, used: int = 512 * 2**20) -> FakeCluster:
    cluster = FakeCluster()
    cluster.collections["local.oplog.rs"] = [
        {"ts": Timestamp(T0, 1)},
        {"ts": Timestamp(T0 + window, 1)},
    ]
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"storageStats": {"maxSize": OPLOG_BYTES, "size": used}}
    ]
    return cluster


def read_client(cluster: FakeCluster) -> ReadOnlyMongo:
    config = MongoConfig(
        uri=READ_URI,
        tls_ca_file=Path("/etc/mongodb/tls/ca-chain.cert.pem"),
        tls_cert_file=Path("/etc/bellwether/tls/meetadev-ai.combined.pem"),
        target_node=TARGET,
        fallback_nodes=FALLBACKS,
    )
    return ReadOnlyMongo(config, client_factory=cluster.factory())


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# --- Detector: spec acceptance ---------------------------------------------


def test_healthy_window_returns_none() -> None:
    assert detector().evaluate([oplog_signal(6 * HOUR)]) is None


def test_forty_minutes_is_critical_with_both_numbers() -> None:
    finding = detector().evaluate([oplog_signal(40 * MIN)])

    assert finding is not None
    assert finding.severity is Severity.CRITICAL
    assert finding.failure_mode == "oplog_window_below_resync"
    assert finding.signal_class is SignalClass.REPLICATION
    evidence = evidence_of(finding)
    assert evidence["oplog_window_seconds"] == 2400
    assert evidence["resync_seconds"] == 3600
    assert "40 min" in finding.summary and "2400 s" in finding.summary
    assert "60 min" in finding.summary and "3600 s" in finding.summary


def test_ninety_minutes_is_warning() -> None:
    finding = detector().evaluate([oplog_signal(90 * MIN)])

    assert finding is not None
    assert finding.severity is Severity.WARNING
    evidence = evidence_of(finding)
    assert evidence["oplog_window_seconds"] == 5400
    assert evidence["resync_seconds"] == 3600
    assert evidence["warning_threshold_seconds"] == 7200
    assert "90 min" in finding.summary and "60 min" in finding.summary


# --- Detector: the resync comparison, not a bare threshold -----------------


def test_thresholds_follow_the_resync_estimate() -> None:
    # 90 min is WARNING against 1 h, but healthy against a 30 min maintenance window.
    assert detector(maintenance_window_seconds=30 * MIN).evaluate([oplog_signal(90 * MIN)]) is None
    # ...and CRITICAL against a 2 h maintenance window.
    finding = detector(maintenance_window_seconds=2 * HOUR).evaluate([oplog_signal(90 * MIN)])
    assert finding is not None and finding.severity is Severity.CRITICAL


def test_safety_factor_widens_the_warning_band() -> None:
    signal = oplog_signal(150 * MIN)

    assert detector().evaluate([signal]) is None  # 150 min >= 2.0 x 60 min
    finding = detector(safety_factor=3.0).evaluate([signal])  # 150 < 180
    assert finding is not None and finding.severity is Severity.WARNING


def test_boundaries() -> None:
    assert detector().evaluate([oplog_signal(2 * HOUR)]) is None  # exactly resync x factor
    at_resync = detector().evaluate([oplog_signal(HOUR)])  # exactly resync: not below it
    assert at_resync is not None and at_resync.severity is Severity.WARNING


def test_finding_carries_its_signal_and_node() -> None:
    signal = oplog_signal(40 * MIN, node=WESTEUROPE)

    finding = detector().evaluate([signal])

    assert finding is not None
    assert finding.signals == (signal,)
    assert finding.node == WESTEUROPE
    assert WESTEUROPE in finding.summary


def test_summary_is_deterministic() -> None:
    first = detector().evaluate([oplog_signal(40 * MIN)])
    second = detector().evaluate([oplog_signal(40 * MIN)])

    assert first is not None and second is not None
    assert first.summary == second.summary


def test_no_oplog_signal_returns_none() -> None:
    other = Signal(SignalClass.CAPACITY, "cache", TARGET, (Evidence("dirty", 0.1),))

    assert detector().evaluate([other]) is None
    assert detector().evaluate([]) is None


def test_latest_oplog_signal_wins() -> None:
    older = oplog_signal(6 * HOUR)
    newer = Signal(
        signal_class=older.signal_class,
        source=older.source,
        node=older.node,
        evidence=oplog_signal(40 * MIN).evidence,
        collected_at=older.collected_at.replace(year=older.collected_at.year + 1),
    )

    finding = detector().evaluate([newer, older])

    assert finding is not None and finding.severity is Severity.CRITICAL


# --- Detector: horizon ----------------------------------------------------------


def test_horizon_none_without_a_trend() -> None:
    finding = detector().evaluate([oplog_signal(90 * MIN)])  # mean rate only

    assert finding is not None
    assert finding.horizon_seconds is None


def test_horizon_projected_from_sampled_rate() -> None:
    # Window 90 min at mean rate a; current sampled rate 3a. Steady-state window
    # becomes 30 min (< 60 min resync); it shrinks at (3 - 1) s per s, so the
    # 30 min of margin is gone in 900 s.
    mean = OPLOG_BYTES / (90 * MIN)
    signal = oplog_signal(90 * MIN, rate=3 * mean, rate_source="repl_network_sample")

    finding = detector().evaluate([signal])

    assert finding is not None
    assert finding.horizon_seconds == 900
    assert evidence_of(finding)["projected_window_seconds"] == 1800


def test_horizon_none_when_steady_state_stays_above_resync() -> None:
    mean = OPLOG_BYTES / (90 * MIN)
    signal = oplog_signal(90 * MIN, rate=1.2 * mean, rate_source="repl_network_sample")

    finding = detector().evaluate([signal])

    assert finding is not None
    assert finding.horizon_seconds is None  # converges to 75 min, above 60 min


def test_horizon_zero_when_already_below_resync() -> None:
    finding = detector().evaluate([oplog_signal(40 * MIN)])

    assert finding is not None
    assert finding.horizon_seconds == 0


# --- Collector ---------------------------------------------------------------


def test_collector_emits_signal_with_spec_evidence() -> None:
    cluster = cluster_with_oplog(window=40 * MIN, used=512 * 2**20)
    cluster.reply_in_sequence(
        "serverStatus",
        [
            server_status(secondary=True, repl_bytes=1_000_000, uptime_ms=50_000),
            server_status(secondary=True, repl_bytes=1_600_000, uptime_ms=60_000),
        ],
    )
    sleeps = Sleeps()
    collector = OplogWindowCollector(OplogWindowCollectorConfig(), sleep=sleeps)

    signal = collector.collect(read_client(cluster))

    assert signal is not None
    assert signal.signal_class is SignalClass.REPLICATION
    assert signal.source == "oplog_window"
    assert signal.node == TARGET
    assert signal.get("oplog_size_bytes") == OPLOG_BYTES
    assert signal.get("oplog_used_bytes") == 512 * 2**20
    assert signal.get("oplog_window_seconds") == 2400
    assert signal.get("write_rate_bytes_per_sec") == pytest.approx(60_000)
    assert signal.get("write_rate_source") == "repl_network_sample"
    assert signal.get("oplog_mean_rate_bytes_per_sec") == pytest.approx(512 * 2**20 / 2400)
    assert sleeps.calls == [10.0]


def test_collector_falls_back_to_mean_rate_on_primary() -> None:
    cluster = cluster_with_oplog(window=2 * HOUR, used=720_000)
    cluster.reply("serverStatus", server_status(secondary=False, repl_bytes=0, uptime_ms=1))
    sleeps = Sleeps()

    signal = OplogWindowCollector(sleep=sleeps).collect(read_client(cluster))

    assert signal is not None
    assert signal.get("write_rate_source") == "oplog_mean"
    assert signal.get("write_rate_bytes_per_sec") == pytest.approx(100.0)
    assert sleeps.calls == []


def test_collector_sampling_can_be_disabled() -> None:
    cluster = cluster_with_oplog(window=2 * HOUR, used=720_000)
    config = OplogWindowCollectorConfig(sample_interval_seconds=0)

    signal = OplogWindowCollector(config, sleep=Sleeps()).collect(read_client(cluster))

    assert signal is not None
    assert signal.get("write_rate_source") == "oplog_mean"
    assert cluster.commands_sent("serverStatus") == []


def test_collector_ignores_a_counter_reset() -> None:
    cluster = cluster_with_oplog(window=2 * HOUR, used=720_000)
    cluster.reply_in_sequence(
        "serverStatus",
        [
            server_status(secondary=True, repl_bytes=9_000_000, uptime_ms=90_000),
            server_status(secondary=True, repl_bytes=10, uptime_ms=5),  # mongod restarted
        ],
    )

    signal = OplogWindowCollector(sleep=Sleeps()).collect(read_client(cluster))

    assert signal is not None
    assert signal.get("write_rate_source") == "oplog_mean"


def test_collector_returns_none_for_empty_oplog() -> None:
    cluster = FakeCluster()
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"storageStats": {"maxSize": OPLOG_BYTES, "size": 0}}
    ]

    assert OplogWindowCollector(sleep=Sleeps()).collect(read_client(cluster)) is None


def test_collector_to_detector_end_to_end() -> None:
    cluster = cluster_with_oplog(window=40 * MIN)
    config = OplogWindowCollectorConfig(sample_interval_seconds=0)

    signal = OplogWindowCollector(config).collect(read_client(cluster))
    assert signal is not None
    finding = OplogWindowDetector().evaluate([signal])

    assert finding is not None
    assert finding.severity is Severity.CRITICAL
    assert finding.node == TARGET


# --- Structure: ABCs and "deterministic before AI" --------------------------


def test_classes_implement_the_abcs() -> None:
    assert isinstance(OplogWindowCollector(), Collector)
    assert OplogWindowCollector().signal_class is SignalClass.REPLICATION
    assert isinstance(OplogWindowDetector(), Detector)
    with pytest.raises(TypeError):
        Collector()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


@pytest.mark.parametrize("package", ["collectors", "detectors"])
def test_collectors_and_detectors_never_touch_a_model(package: str) -> None:
    root = Path(__file__).resolve().parents[1] / "bellwether" / package
    for path in root.glob("*.py"):
        source = path.read_text()
        for forbidden in ("anthropic", "openai", "bellwether.analysis"):
            assert forbidden not in source, f"{path.name} references {forbidden}"
