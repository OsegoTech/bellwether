"""The Notifier contract: one-way out.

A notifier tells humans a proposal exists. It never receives decisions —
Slack button callbacks land on the approval endpoint (``bellwether serve``),
not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bellwether.models import ApprovalRecord, Proposal


class Notifier(ABC):
    name: str  # matches the notify.channels config value

    @abstractmethod
    def notify(self, proposal: Proposal, approval_record: ApprovalRecord) -> None: ...
