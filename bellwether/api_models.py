"""The HTTP API's documented schemas: what every endpoint returns or accepts.

These mirror the store's records (``bellwether.models`` and
``bellwether.store.sqlite``) field for field, so the OpenAPI document at
``/openapi.json`` — rendered at ``/docs`` and ``/redoc`` — is the reference a
frontend or an integration is built against. Every field carries a
description. Evidence values are passed through exactly as measured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bellwether.models import ApprovalRecord, Evidence, Proposal, RemediationAction
from bellwether.store.sqlite import AuditEvent, FindingRow, RunRecord, Transition

StateName = Literal["pending", "approved", "rejected", "expired", "executed", "failed"]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True)


class EvidenceOut(_Schema):
    name: str = Field(description="What was measured, e.g. `oplog_window_seconds` or `candidate_index`.")
    value: Any = Field(description="The measured value exactly as observed: a number, string, list or object.")
    unit: str | None = Field(default=None, description="Unit of the value (`s`, `ms`, `bytes`, `docs`), if any.")
    observed_at: datetime = Field(description="When the value was observed (UTC).")

    @classmethod
    def of(cls, evidence: Evidence) -> EvidenceOut:
        return cls(
            name=evidence.name,
            value=evidence.value,
            unit=evidence.unit,
            observed_at=evidence.observed_at,
        )


class RemediationActionOut(_Schema):
    kind: Literal["propose_only", "executable"] = Field(
        description="`propose_only`: a human runs `command`. `executable`: one of the two "
        "whitelisted operations, which the executor may run after approval."
    )
    title: str = Field(description="One-line description of the remediation.")
    command: str = Field(
        description="The exact command a human would run. Display and audit text only: "
        "Bellwether never executes this string."
    )
    rationale: str = Field(description="Why this remediation, weighed against its costs.")
    reversible: bool = Field(description="Whether the action can be undone.")
    executor_op: str | None = Field(
        default=None,
        description="For executable actions: `kill_op` or `create_small_index`. Null otherwise.",
    )
    executor_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for `executor_op`, taken from the finding's evidence. Empty for "
        "propose-only actions.",
    )

    @classmethod
    def of(cls, action: RemediationAction) -> RemediationActionOut:
        return cls(
            kind=action.kind.value,
            title=action.title,
            command=action.command,
            rationale=action.rationale,
            reversible=action.reversible,
            executor_op=action.executor_op,
            executor_args=dict(action.executor_args),
        )


class ProposalOut(_Schema):
    proposal_id: str = Field(description="Unique id of the proposal.")
    finding_id: str = Field(description="The finding this proposal answers.")
    failure_mode: str = Field(
        description="Stable failure-mode code, e.g. `oplog_window_below_resync`, "
        "`missing_index_collscan`."
    )
    node: str = Field(description="The replica-set member the finding concerns (`host:port`).")
    diagnosis: str = Field(description="What is happening and why, grounded in the evidence.")
    mechanism: str = Field(description="The MongoDB internal that explains it.")
    impact_if_ignored: str = Field(description="What happens on this deployment if nothing changes.")
    action: RemediationActionOut = Field(description="The one proposed remediation.")
    confidence: float = Field(description="The model's stated confidence in the diagnosis, 0.0–1.0.")
    provider: str = Field(description="Which model provider produced the proposal (`claude`, `openai`).")
    evidence_refs: list[EvidenceOut] = Field(
        description="The measured evidence the proposal rests on, as the detector recorded it."
    )
    created_at: datetime = Field(description="When the proposal was produced (UTC).")

    @classmethod
    def of(cls, proposal: Proposal) -> ProposalOut:
        return cls(
            proposal_id=proposal.proposal_id,
            finding_id=proposal.finding_id,
            failure_mode=proposal.failure_mode,
            node=proposal.node,
            diagnosis=proposal.diagnosis,
            mechanism=proposal.mechanism,
            impact_if_ignored=proposal.impact_if_ignored,
            action=RemediationActionOut.of(proposal.action),
            confidence=proposal.confidence,
            provider=proposal.provider,
            evidence_refs=[EvidenceOut.of(e) for e in proposal.evidence_refs],
            created_at=proposal.created_at,
        )


class ApprovalRecordOut(_Schema):
    proposal_id: str = Field(description="The proposal this record belongs to.")
    record_id: str = Field(description="Id of the approval record.")
    state: StateName = Field(description="Current state, reconstructed from the latest transition.")
    decided_by: str | None = Field(
        default=None,
        description="Who approved or rejected it, e.g. `slack:alice (U0123ABCD)` or `api:U0123ABCD`.",
    )
    decided_at: datetime | None = Field(default=None, description="When it was approved or rejected.")
    executed_at: datetime | None = Field(
        default=None, description="When the executor finished (executed or failed)."
    )
    execution_result: str | None = Field(
        default=None, description="The executor's result or error, if it ran."
    )
    created_at: datetime = Field(description="When the proposal entered the approval flow.")

    @classmethod
    def of(cls, record: ApprovalRecord) -> ApprovalRecordOut:
        return cls(
            proposal_id=record.proposal_id,
            record_id=record.record_id,
            state=record.state.value,
            decided_by=record.decided_by,
            decided_at=record.decided_at,
            executed_at=record.executed_at,
            execution_result=record.execution_result,
            created_at=record.created_at,
        )


class TransitionOut(_Schema):
    state: StateName = Field(description="The state the proposal moved into.")
    actor: str | None = Field(default=None, description="Who caused the transition, if a person.")
    at: datetime = Field(description="When the transition was recorded (UTC).")
    result: str | None = Field(default=None, description="The execution result, on executed/failed.")

    @classmethod
    def of(cls, transition: Transition) -> TransitionOut:
        return cls(
            state=transition.state.value,
            actor=transition.actor,
            at=transition.at,
            result=transition.result,
        )


class AuditEventOut(_Schema):
    kind: str = Field(
        description="Event kind, e.g. `unauthorized_decision` for a refused approve/reject."
    )
    proposal_id: str | None = Field(default=None, description="The proposal the event concerns, if any.")
    actor: str | None = Field(default=None, description="Who attempted the action.")
    detail: str = Field(description="What happened, in words.")
    at: datetime = Field(description="When the event was recorded (UTC).")

    @classmethod
    def of(cls, event: AuditEvent) -> AuditEventOut:
        return cls(
            kind=event.kind,
            proposal_id=event.proposal_id,
            actor=event.actor,
            detail=event.detail,
            at=event.at,
        )


class FindingOut(_Schema):
    finding_id: str = Field(description="Unique id of the finding.")
    run_id: str = Field(description="The pipeline run that produced it.")
    failure_mode: str = Field(description="Stable failure-mode code.")
    severity: Literal["info", "warning", "critical"] = Field(description="How close it is to impact.")
    node: str = Field(description="The replica-set member it concerns (`host:port`).")
    summary: str = Field(description="One deterministic sentence from the detector, with the real numbers.")
    horizon_seconds: int | None = Field(
        default=None,
        description="Estimated seconds to impact; 0 means the threshold is already crossed; null "
        "means not time-bound.",
    )
    state: Literal["escalated", "noted"] = Field(
        description="`escalated`: sent to analysis. `noted`: stored only (below the escalation "
        "threshold, e.g. INFO)."
    )
    proposal_id: str | None = Field(default=None, description="The proposal made for it, if any.")
    detected_at: datetime = Field(description="When the detector produced it (UTC).")

    @classmethod
    def of(cls, row: FindingRow, proposal_id: str | None) -> FindingOut:
        return cls(
            finding_id=row.finding_id,
            run_id=row.run_id,
            failure_mode=row.failure_mode,
            severity=row.severity.value,
            node=row.node,
            summary=row.summary,
            horizon_seconds=row.horizon_seconds,
            state="escalated" if row.escalated else "noted",
            proposal_id=proposal_id,
            detected_at=row.detected_at,
        )


class RunOut(_Schema):
    run_id: str = Field(description="Unique id of the pipeline run.")
    started_at: datetime = Field(description="When the run started (UTC).")
    finished_at: datetime = Field(description="When the run finished (UTC).")
    findings: int = Field(description="Findings the run produced.")
    proposals: int = Field(description="Proposals the run produced.")
    errors: list[str] = Field(description="Errors recorded during the run (collector, analysis, notify).")

    @classmethod
    def of(cls, run: RunRecord) -> RunOut:
        return cls(
            run_id=run.run_id,
            started_at=run.started_at,
            finished_at=run.finished_at,
            findings=run.findings,
            proposals=run.proposals,
            errors=list(run.errors),
        )


class ProposalListItem(_Schema):
    proposal: ProposalOut = Field(description="The proposal.")
    state: StateName = Field(description="Its current approval state.")
    created_at: datetime = Field(description="When it was produced (UTC).")


class ProposalDetailOut(_Schema):
    proposal: ProposalOut = Field(description="The proposal in full.")
    state: ApprovalRecordOut = Field(description="Its current approval record.")
    history: list[TransitionOut] = Field(description="Every state transition, oldest first.")
    audit_events: list[AuditEventOut] = Field(
        description="Refused decision attempts and other audit events, oldest first."
    )


class AuditTrailOut(_Schema):
    proposal_id: str = Field(description="The proposal.")
    state: StateName = Field(description="Its current approval state.")
    transitions: list[TransitionOut] = Field(description="Every state transition, oldest first.")
    audit_events: list[AuditEventOut] = Field(
        description="Refused decision attempts (`unauthorized_decision`) and other events, oldest first."
    )
    execution_result: str | None = Field(
        default=None, description="The executor's result or error, if it ran."
    )


class DecisionRequest(BaseModel):
    """Body of ``POST /api/proposals/{proposal_id}/decision``."""

    model_config = ConfigDict(strict=True, extra="forbid")

    decision: Literal["approve", "reject"] = Field(description="Approve or reject the proposal.")
    approver_id: str = Field(
        min_length=1,
        description="The deciding approver's id; must be in `approval.approver_ids`. Recorded in "
        "the audit trail as `api:<id>`.",
    )
    acknowledged: bool = Field(
        default=False,
        description="Required `true` to approve: the caller acknowledges that approving may let "
        "Bellwether change the production database.",
    )


class DecisionResponse(_Schema):
    proposal_id: str = Field(description="The proposal decided.")
    state: StateName = Field(
        description="Its state after the decision. An approved executable proposal is executed "
        "after the response; read it back to see `executed` or `failed`."
    )
    message: str = Field(description="What was done, in words.")


class ErrorOut(_Schema):
    error: str = Field(
        description="Machine-readable code: `malformed_request`, `acknowledgement_required`, "
        "`no_approval_path`, `approval_not_permitted`, `not_authorized`, `not_found`, "
        "`already_decided`."
    )
    detail: str = Field(description="Human-readable explanation.")
