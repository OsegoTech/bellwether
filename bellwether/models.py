"""
Core data contracts for Bellwether.

Three objects flow down the pipeline, one per stage boundary:

    Signal   — what a collector read from the cluster (raw evidence, deterministic)
    Finding  — what a detector concluded from signals (a trajectory toward a known
               failure mode, with a horizon and severity — still deterministic)
    Proposal — what the AI stage produced from a finding (a diagnosis and a
               remediation a human can approve — the only non-deterministic object)

The stages are decoupled through these types. A collector knows nothing about AI;
the AI stage never touches the cluster. Each object carries the evidence that
justifies it, so a Proposal shown to a human — or stored for audit — can always be
traced back to the raw numbers a collector observed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class SignalClass(str, Enum):
    """The families of cluster health a collector can observe.

    These map to MongoDB-specific failure domains, not generic host metrics.
    Each collector belongs to exactly one class.
    """

    REPLICATION = "replication"   # oplog window, lag, election stability
    PERFORMANCE = "performance"   # query shapes, index usage, slow ops
    CAPACITY = "capacity"         # cache pressure, connection saturation
    SECURITY = "security"         # authentication / authorization patterns


class Severity(str, Enum):
    """How close a finding is to becoming user-impacting.

    Ordered. INFO is a trajectory worth noting; CRITICAL is imminent.
    Only WARNING and above escalate to the AI stage by default — INFO is
    recorded but does not spend tokens.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass(frozen=True)
class Evidence:
    """A single named measurement behind a signal or finding.

    Frozen: evidence is a fact that was observed at a moment. It is never
    edited after the fact — a later observation is a new Evidence, so an
    audit trail can show exactly what was true when a conclusion was drawn.
    """

    name: str
    value: Any
    unit: str | None = None
    observed_at: datetime = field(default_factory=_utcnow)

    def render(self) -> str:
        v = self.value
        if self.unit:
            return f"{self.name}: {v} {self.unit}"
        return f"{self.name}: {v}"


@dataclass(frozen=True)
class Signal:
    """A deterministic observation from one collector at one point in time.

    A Signal is raw material, not a conclusion. "The oplog holds 42 minutes of
    writes at the current rate" is a Signal. Whether 42 minutes is a problem is
    a detector's job, not a collector's.
    """

    signal_class: SignalClass
    source: str                       # which collector produced it, e.g. "oplog_window"
    node: str                         # which member it was read from
    evidence: tuple[Evidence, ...]
    collected_at: datetime = field(default_factory=_utcnow)
    signal_id: str = field(default_factory=_new_id)

    def get(self, name: str) -> Any:
        for e in self.evidence:
            if e.name == name:
                return e.value
        raise KeyError(f"no evidence named {name!r} in signal {self.source}")


@dataclass(frozen=True)
class Finding:
    """A deterministic conclusion: a trajectory toward a known failure mode.

    A Finding is what a detector emits when the numbers in one or more Signals
    cross a rule the detector encodes. It carries a horizon — roughly how long
    until impact — and the evidence that justifies it, so the AI stage (and any
    human) can see the reasoning without re-querying the cluster.

    The `failure_mode` is a stable identifier (e.g. "oplog_window_below_resync")
    used for deduplication, routing, and correlating repeat findings over time.
    """

    signal_class: SignalClass
    failure_mode: str
    severity: Severity
    node: str
    summary: str                      # one deterministic sentence, no AI
    evidence: tuple[Evidence, ...]
    horizon_seconds: int | None = None  # est. time to impact; None if not time-bound
    signals: tuple[Signal, ...] = ()    # the signals this finding was derived from
    detected_at: datetime = field(default_factory=_utcnow)
    finding_id: str = field(default_factory=_new_id)

    @property
    def escalates(self) -> bool:
        """Whether this finding is severe enough to spend AI tokens on."""
        return self.severity.rank >= Severity.WARNING.rank

    def horizon_human(self) -> str:
        if self.horizon_seconds is None:
            return "no time bound"
        s = self.horizon_seconds
        if s < 3600:
            return f"~{s // 60} min"
        if s < 86400:
            return f"~{s // 3600} h"
        return f"~{s // 86400} d"


class ActionKind(str, Enum):
    """What a proposal asks for.

    PROPOSE_ONLY  — the system displays a command for a human to run by hand.
                    The default, and the only option for anything touching
                    replica-set config, dropping data, or storage parameters.
    EXECUTABLE    — a bounded, reversible, whitelisted action the executor may
                    run *after* human approval. Only two action types ever
                    qualify (see executor whitelist): killing an identified
                    runaway op, and creating a small proposed index.
    """

    PROPOSE_ONLY = "propose_only"
    EXECUTABLE = "executable"


@dataclass(frozen=True)
class RemediationAction:
    """The concrete thing a proposal recommends.

    `command` is always populated — it is what a human runs if the action is
    propose-only, and what the audit log records regardless. For EXECUTABLE
    actions, `executor_op` and `executor_args` name the whitelisted operation
    the executor will perform; the raw command string is never shell-executed.
    """

    kind: ActionKind
    title: str
    command: str                      # human-runnable form; audit record
    rationale: str
    reversible: bool
    executor_op: str | None = None    # whitelist key, only for EXECUTABLE
    executor_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """The AI stage's output: a diagnosis plus a proposed remediation.

    This is the only object in the system produced by a model rather than by
    deterministic code. It is schema-validated on the way out of the analysis
    stage — a malformed model response never becomes a Proposal. It carries a
    reference back to the Finding (and thus the Signals and Evidence) so the
    whole chain is auditable, and it names which provider produced it.

    A Proposal is never self-executing. Reaching the executor requires a
    separate, human approval event recorded against this proposal's id.
    """

    finding_id: str
    failure_mode: str
    node: str
    diagnosis: str                    # what is happening and why
    mechanism: str                    # the MongoDB internal that explains it
    impact_if_ignored: str            # what happens if nothing changes
    action: RemediationAction
    confidence: float                 # 0.0–1.0, model's own stated confidence
    provider: str                     # which model produced this
    evidence_refs: tuple[Evidence, ...] = ()
    created_at: datetime = field(default_factory=_utcnow)
    proposal_id: str = field(default_factory=_new_id)


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class ApprovalRecord:
    """The audit trail entry for a proposal's journey past the guarantee boundary.

    Mutable by design — it accumulates state transitions (pending → approved →
    executed) with who and when at each step. This is the record a regulator or
    an incident review reads to answer "who let the machine touch production".
    """

    proposal_id: str
    state: ApprovalState = ApprovalState.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    execution_result: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    record_id: str = field(default_factory=_new_id)
