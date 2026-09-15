"""Oplog window collector — the first vertical-slice signal.

Reads, from whichever node the read client is serving from:

- ``oplog_size_bytes`` / ``oplog_used_bytes`` — ``$collStats`` storageStats
  ``maxSize`` and ``size`` on ``local.oplog.rs`` (what rs.printReplicationInfo
  reports).
- ``oplog_window_seconds`` — newest minus oldest oplog entry timestamp.
- ``write_rate_bytes_per_sec`` — on a secondary, two serverStatus reads of
  ``metrics.repl.network.bytes`` (oplog bytes fetched from the sync source),
  timed by the server's own ``uptimeMillis``. Where that is unavailable (a
  primary, sampling disabled, a counter reset between samples, or a failover
  between reads) it falls back to the oplog's mean rate, ``used / window``.
  ``write_rate_source`` records which one was used, so a detector knows
  whether it is looking at a live trend or an average.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from pymongo.errors import OperationFailure

from bellwether.collectors.base import Collector
from bellwether.config import OplogWindowCollectorConfig
from bellwether.models import Evidence, Signal, SignalClass
from bellwether.mongo import ReadOnlyMongo

logger = logging.getLogger(__name__)

RATE_SAMPLED = "repl_network_sample"
RATE_MEAN = "oplog_mean"
NAMESPACE_NOT_FOUND = 26  # local.oplog.rs absent: a standalone, not a replica-set member


class OplogWindowCollector(Collector):
    name: ClassVar[str] = "oplog_window"

    def __init__(
        self,
        config: OplogWindowCollectorConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config or OplogWindowCollectorConfig()
        self._sleep = sleep

    @property
    def signal_class(self) -> SignalClass:
        return SignalClass.REPLICATION

    def collect(self, mongo: ReadOnlyMongo) -> Signal | None:
        try:
            stats = mongo.oplog_stats()
        except OperationFailure as exc:
            if exc.code != NAMESPACE_NOT_FOUND:
                raise
            # A standalone instance (not a replica-set member) has no oplog: there
            # is no replication window to watch, so no signal rather than an error.
            logger.info(
                "no oplog on this node (a standalone instance); no oplog signal",
                extra={"node": mongo.served_by},
            )
            return None
        node = mongo.served_by or "unknown"
        window = stats.window_seconds
        if window is None or window <= 0:
            logger.info("oplog empty or single-entry; no signal", extra={"node": node})
            return None

        mean_rate = stats.used_bytes / window
        sampled = self._sampled_rate(mongo)
        if sampled is not None and mongo.served_by != node:
            sampled = None  # failover between reads: the sample is from another member
        rate, source = (sampled, RATE_SAMPLED) if sampled is not None else (mean_rate, RATE_MEAN)

        logger.info(
            "oplog window collected",
            extra={
                "node": node,
                "oplog_window_seconds": window,
                "write_rate_bytes_per_sec": rate,
                "write_rate_source": source,
            },
        )
        return Signal(
            signal_class=self.signal_class,
            source=self.name,
            node=node,
            evidence=(
                Evidence("oplog_size_bytes", stats.max_size_bytes, "bytes"),
                Evidence("oplog_used_bytes", stats.used_bytes, "bytes"),
                Evidence("oplog_window_seconds", window, "s"),
                Evidence("oplog_mean_rate_bytes_per_sec", mean_rate, "bytes/s"),
                Evidence("write_rate_bytes_per_sec", rate, "bytes/s"),
                Evidence("write_rate_source", source),
            ),
        )

    def _sampled_rate(self, mongo: ReadOnlyMongo) -> float | None:
        interval = self._config.sample_interval_seconds
        if interval <= 0:
            return None
        first = mongo.server_status()
        if not first.get("repl", {}).get("secondary"):
            return None
        self._sleep(interval)
        second = mongo.server_status()

        bytes0, bytes1 = _repl_network_bytes(first), _repl_network_bytes(second)
        ms0, ms1 = first.get("uptimeMillis"), second.get("uptimeMillis")
        if bytes0 is None or bytes1 is None or ms0 is None or ms1 is None:
            return None
        elapsed_ms = int(ms1) - int(ms0)
        if bytes1 < bytes0 or elapsed_ms <= 0:
            return None  # mongod restarted between samples
        return (bytes1 - bytes0) / (elapsed_ms / 1000)


def _repl_network_bytes(status: Mapping[str, Any]) -> int | None:
    value = status.get("metrics", {}).get("repl", {}).get("network", {}).get("bytes")
    return None if value is None else int(value)
