"""The Detector contract: turn signals into a finding, deterministically."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from bellwether.models import Evidence, Finding, Signal

# Evidence a runaway-operation finding must carry for an approved kill_op to
# act. killOp only affects the mongod it is sent to, so the op's node travels
# with its opid; the Finding's `node` must be that same member. Both come from
# the detector, never from the model.
OPID_EVIDENCE = "opid"
OP_NODE_EVIDENCE = "op_node"


def killable_op_evidence(opid: int, node: str) -> tuple[Evidence, Evidence]:
    """The (opid, op_node) evidence pair a runaway-op detector attaches."""
    return (Evidence(OPID_EVIDENCE, opid), Evidence(OP_NODE_EVIDENCE, node))


class Detector(ABC):
    """Encodes one known MongoDB failure mode as a rule over signals.

    Pure function of its input: no cluster access, no model calls. Returns a
    Finding when the rule fires, None otherwise.
    """

    failure_mode: ClassVar[str]

    @abstractmethod
    def evaluate(self, signals: Sequence[Signal]) -> Finding | None: ...

    def evaluate_all(self, signals: Sequence[Signal]) -> list[Finding]:
        """Every finding from these signals. The pipeline calls this.

        A detector for one condition (the oplog window) finds at most one; one
        that can fire many times per pass (an index per query shape) overrides
        this.
        """
        finding = self.evaluate(signals)
        return [] if finding is None else [finding]
