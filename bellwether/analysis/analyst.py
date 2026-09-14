"""Finding -> Proposal: the only AI stage.

The analyst assembles a bounded context for one Finding — the finding's own
evidence, a topology summary from config, and the known remediation patterns
for its failure mode; never raw dumps — calls the ProviderChain, validates the
answer against the proposal schema (``schema.py``), and returns a Proposal.
Invalid output is rejected and retried by the chain; never coerced.

The analyst never touches the cluster. It only ever sees a Finding that a
deterministic detector already produced.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from bellwether.analysis.provider import ProviderChain
from bellwether.analysis.schema import ProposalPayload, validate_payload
from bellwether.config import MongoConfig
from bellwether.models import ActionKind, Finding, Proposal, RemediationAction

logger = logging.getLogger(__name__)

# TODO(design): prompt wording set in design review
# SYSTEM_PROMPT and PROMPT_TEMPLATE below are a minimal working version so the
# pipeline and tests run end to end. Their wording is owned by the design
# review, not by code changes.
SYSTEM_PROMPT = (
    "You are a MongoDB reliability engineer reviewing a finding from a deterministic "
    "detector on a self-hosted replica set. Answer only with a JSON object matching "
    "the provided schema."
)

PROMPT_TEMPLATE = """\
Finding: {failure_mode} ({severity}) on {node}
Time to impact: {horizon}
Summary: {summary}

Evidence:
{evidence}

Topology:
{topology}

Known remediation patterns for this failure mode:
{remediations}

Diagnose the finding and propose one remediation. Use kind "executable" only for
kill_op or create_small_index; everything else is "propose_only", with the exact
command a human would run.
"""

# Remediation patterns per failure mode: MongoDB facts handed to the model as
# context, not instructions it must follow.
KNOWN_REMEDIATIONS: dict[str, tuple[str, ...]] = {
    "oplog_window_below_resync": (
        "Grow the oplog online with replSetResizeOplog on each member, secondaries first "
        "(propose-only: storage parameter).",
        "Set a minimum retention with replSetResizeOplog minRetentionHours "
        "(storage.oplogMinRetentionHours, MongoDB 4.4+) so the window cannot fall below "
        "the maintenance window (propose-only).",
        "Find and reduce the write burst driving the rate: bulk loads, large multi-document "
        "updates, TTL deletes.",
        "Until the window is grown, keep any secondary's maintenance shorter than the "
        "current window.",
    ),
}


def render_prompt(finding: Finding, context: Mapping[str, Any]) -> str:
    """The user prompt, shared verbatim by every provider."""
    evidence = "\n".join(f"- {e.render()}" for e in finding.evidence) or "- (none)"
    remediations = "\n".join(f"- {hint}" for hint in context.get("remediations", ())) or (
        "- (none on file for this failure mode)"
    )
    return PROMPT_TEMPLATE.format(
        failure_mode=finding.failure_mode,
        severity=finding.severity.value,
        node=finding.node,
        horizon=finding.horizon_human(),
        summary=finding.summary,
        evidence=evidence,
        topology=context.get("topology") or "(not provided)",
        remediations=remediations,
    )


def topology_summary(mongo: MongoConfig) -> str:
    fallbacks = ", ".join(mongo.fallback_nodes) or "none"
    return (
        f"Diagnostic reads are served by {mongo.target_node}; "
        f"fallback order: {fallbacks}."
    )


class Analyst:
    def __init__(self, chain: ProviderChain, *, topology: str = "") -> None:
        self._chain = chain
        self._topology = topology

    def build_context(self, finding: Finding) -> dict[str, Any]:
        return {
            "topology": self._topology,
            "remediations": list(KNOWN_REMEDIATIONS.get(finding.failure_mode, ())),
        }

    def analyze(self, finding: Finding) -> Proposal:
        """Raises AnalysisUnavailable if no provider produces a valid proposal."""
        provider, payload = self._chain.run(finding, self.build_context(finding), validate_payload)
        proposal = to_proposal(payload, finding, provider)
        logger.info(
            "proposal produced",
            extra={
                "proposal_id": proposal.proposal_id,
                "finding_id": finding.finding_id,
                "provider": provider,
                "action_kind": proposal.action.kind.value,
                "executor_op": proposal.action.executor_op,
            },
        )
        return proposal


def to_proposal(payload: ProposalPayload, finding: Finding, provider: str) -> Proposal:
    action = payload.action
    return Proposal(
        finding_id=finding.finding_id,
        failure_mode=finding.failure_mode,
        node=finding.node,
        diagnosis=payload.diagnosis,
        mechanism=payload.mechanism,
        impact_if_ignored=payload.impact_if_ignored,
        action=RemediationAction(
            kind=ActionKind(action.kind),
            title=action.title,
            command=action.command,
            rationale=action.rationale,
            reversible=action.reversible,
            executor_op=action.executor_op,
            executor_args=action.executor_args.model_dump() if action.executor_args else {},
        ),
        confidence=payload.confidence,
        provider=provider,
        evidence_refs=finding.evidence,
    )
