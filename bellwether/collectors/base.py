"""The Collector contract: read the cluster, emit evidence, conclude nothing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from bellwether.models import Signal, SignalClass
from bellwether.mongo import ReadOnlyMongo


class Collector(ABC):
    """Deterministic reader of one family of cluster health.

    A collector only reads, through the read-only client, and returns a Signal
    carrying the raw numbers. It never judges them — that is a detector's job —
    and never calls a model. It returns None when there is nothing to observe;
    it raises nothing on a healthy cluster.
    """

    name: ClassVar[str]  # becomes Signal.source

    @property
    @abstractmethod
    def signal_class(self) -> SignalClass: ...

    @abstractmethod
    def collect(self, mongo: ReadOnlyMongo) -> Signal | None: ...
