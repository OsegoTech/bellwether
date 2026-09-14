"""Plain-text rendering of a proposal, for the CLI and development."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from bellwether.models import ApprovalRecord, Proposal
from bellwether.notify.base import Notifier


def render_text(proposal: Proposal, record: ApprovalRecord) -> str:
    action = proposal.action
    lines = [
        f"Bellwether proposal {proposal.proposal_id}",
        f"  state:        {record.state.value}",
        f"  failure mode: {proposal.failure_mode}",
        f"  node:         {proposal.node}",
        f"  finding:      {proposal.finding_id}",
        f"  provider:     {proposal.provider} (confidence {proposal.confidence:.2f})",
        f"  created:      {proposal.created_at.isoformat()}",
        "",
        "Diagnosis:",
        f"  {proposal.diagnosis}",
        "Mechanism:",
        f"  {proposal.mechanism}",
        "Impact if ignored:",
        f"  {proposal.impact_if_ignored}",
        "",
        f"Action ({action.kind.value}): {action.title}",
        f"  command:    {action.command}",
        f"  rationale:  {action.rationale}",
        f"  reversible: {'yes' if action.reversible else 'no'}",
    ]
    if action.executor_op:
        lines += [
            f"  executor op:   {action.executor_op}",
            f"  executor args: {json.dumps(action.executor_args, sort_keys=True)}",
        ]
    lines += ["", "Evidence:"]
    lines += [f"  - {e.render()}" for e in proposal.evidence_refs] or ["  (none)"]
    if record.decided_by or record.decided_at:
        decided_at = record.decided_at.isoformat() if record.decided_at else "?"
        lines.append(f"Decision: {record.decided_by or 'system'} at {decided_at}")
    if record.executed_at or record.execution_result:
        executed_at = record.executed_at.isoformat() if record.executed_at else "?"
        lines.append(f"Outcome: {record.execution_result or '(no result)'} at {executed_at}")
    return "\n".join(lines)


class StdoutNotifier(Notifier):
    name = "stdout"

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def notify(self, proposal: Proposal, approval_record: ApprovalRecord) -> None:
        stream = self._stream or sys.stdout
        stream.write(render_text(proposal, approval_record) + "\n")
        stream.flush()
