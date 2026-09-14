"""JSON encoding of the data contracts for storage.

Proposals round-trip exactly (the audit record must show what the human saw).
Findings are stored as an audit copy: their evidence and summary, with the
signals they came from referenced by id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from bellwether.models import ActionKind, Evidence, Finding, Proposal, RemediationAction


def dt_to_str(value: datetime) -> str:
    """UTC ISO-8601 with microseconds: sorts lexically in time order."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def dt_from_str(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dumps(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, default=str)


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "name": evidence.name,
        "value": evidence.value,
        "unit": evidence.unit,
        "observed_at": dt_to_str(evidence.observed_at),
    }


def evidence_from_dict(doc: dict[str, Any]) -> Evidence:
    return Evidence(
        name=doc["name"],
        value=doc["value"],
        unit=doc["unit"],
        observed_at=dt_from_str(doc["observed_at"]),
    )


def proposal_to_json(proposal: Proposal) -> str:
    action = proposal.action
    return _dumps(
        {
            "proposal_id": proposal.proposal_id,
            "finding_id": proposal.finding_id,
            "failure_mode": proposal.failure_mode,
            "node": proposal.node,
            "diagnosis": proposal.diagnosis,
            "mechanism": proposal.mechanism,
            "impact_if_ignored": proposal.impact_if_ignored,
            "action": {
                "kind": action.kind.value,
                "title": action.title,
                "command": action.command,
                "rationale": action.rationale,
                "reversible": action.reversible,
                "executor_op": action.executor_op,
                "executor_args": action.executor_args,
            },
            "confidence": proposal.confidence,
            "provider": proposal.provider,
            "evidence_refs": [evidence_to_dict(e) for e in proposal.evidence_refs],
            "created_at": dt_to_str(proposal.created_at),
        }
    )


def proposal_from_json(text: str) -> Proposal:
    doc = json.loads(text)
    action = doc["action"]
    return Proposal(
        proposal_id=doc["proposal_id"],
        finding_id=doc["finding_id"],
        failure_mode=doc["failure_mode"],
        node=doc["node"],
        diagnosis=doc["diagnosis"],
        mechanism=doc["mechanism"],
        impact_if_ignored=doc["impact_if_ignored"],
        action=RemediationAction(
            kind=ActionKind(action["kind"]),
            title=action["title"],
            command=action["command"],
            rationale=action["rationale"],
            reversible=action["reversible"],
            executor_op=action["executor_op"],
            executor_args=action["executor_args"],
        ),
        confidence=doc["confidence"],
        provider=doc["provider"],
        evidence_refs=tuple(evidence_from_dict(e) for e in doc["evidence_refs"]),
        created_at=dt_from_str(doc["created_at"]),
    )


def finding_to_json(finding: Finding) -> str:
    return _dumps(
        {
            "finding_id": finding.finding_id,
            "signal_class": finding.signal_class.value,
            "failure_mode": finding.failure_mode,
            "severity": finding.severity.value,
            "node": finding.node,
            "summary": finding.summary,
            "horizon_seconds": finding.horizon_seconds,
            "evidence": [evidence_to_dict(e) for e in finding.evidence],
            "signal_ids": [s.signal_id for s in finding.signals],
            "detected_at": dt_to_str(finding.detected_at),
        }
    )
