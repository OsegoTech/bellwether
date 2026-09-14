"""The Detector contract: turn signals into a finding, deterministically."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from bellwether.models import Finding, Signal


class Detector(ABC):
    """Encodes one known MongoDB failure mode as a rule over signals.

    Pure function of its input: no cluster access, no model calls. Returns a
    Finding when the rule fires, None otherwise.
    """

    failure_mode: ClassVar[str]

    @abstractmethod
    def evaluate(self, signals: Sequence[Signal]) -> Finding | None: ...
