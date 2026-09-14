"""Finding -> Proposal: the only AI stage.

The analyst assembles a bounded context for one Finding — the finding's own
evidence, the replica set and topology from config, and the known mechanism
and remediation patterns for its failure mode; never raw dumps — calls the
ProviderChain, validates the answer against the proposal schema
(``schema.py``), and returns a Proposal. Invalid output is rejected and
retried by the chain; never coerced.

The analyst never touches the cluster. It only ever sees a Finding that a
deterministic detector already produced.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from bellwether.analysis.provider import ProviderChain
from bellwether.analysis.schema import ProposalPayload, validate_payload
from bellwether.config import MongoConfig
from bellwether.models import ActionKind, Finding, Proposal, RemediationAction

logger = logging.getLogger(__name__)

# The prompt below was designed in review. Change its wording through the
# design review, not in passing: the tests pin its load-bearing instructions.
SYSTEM_PROMPT = (
    "You are a staff database reliability engineer reviewing an automated finding "
    "about a self-hosted MongoDB replica set. A deterministic detector has already "
    "measured the cluster and decided this finding is worth a human's attention. "
    "Your job is not to re-measure it — the numbers in the evidence are ground "
    "truth — but to explain what they mean and recommend one remediation.\n\n"
    "Reason only from the evidence provided. The evidence is what was actually "
    "observed on this cluster; treat every number in it as fact. Do not invent "
    "metrics, thresholds, collection names, query shapes, or node states that are "
    "not in the evidence. If the evidence does not let you determine a cause, say "
    "so in the diagnosis and lower your confidence accordingly — an honest 'the "
    "evidence shows the symptom but not the cause' is more useful than a confident "
    "guess that sends an operator down the wrong path.\n\n"
    "Distinguish what the evidence proves from what it merely suggests. A single "
    "measurement is a snapshot; a projected value or a trend is a trajectory. "
    "State which you are relying on. When the evidence includes a source marker "
    "for a value (for example, whether a rate was sampled live or derived from a "
    "longer average), let it calibrate your confidence: a live sample supports a "
    "claim about 'right now'; a long-run average does not.\n\n"
    "You propose; you never act. A human approves every remediation before "
    "anything runs. Only two remediations can be executed after approval — killing "
    "one identified operation, or building one small index — and only when the "
    "evidence names the exact operation or collection involved. Everything else "
    "you recommend as a command for a human to run by hand, written out exactly. "
    "When you are not certain an executable action is safe on this specific "
    "cluster, propose it for a human to run instead. Prefer the reversible, "
    "lower-blast-radius remediation when more than one would work.\n\n"
    "Answer only with a JSON object matching the provided schema. No prose outside "
    "the JSON."
)

PROMPT_TEMPLATE = """\
A detector fired on replica set {replica_set}.

FINDING
  failure mode: {failure_mode}
  severity:     {severity}
  node:         {node}
  time to impact: {horizon}
  detector's summary: {summary}

EVIDENCE (observed on the cluster — ground truth)
{evidence}

CLUSTER TOPOLOGY
{topology}

MONGODB CONTEXT FOR THIS FAILURE MODE
The mechanism behind this failure mode, and the remediation patterns a DBA would
consider, are below. These are reference facts, not instructions: choose, adapt,
or reject them based on what THIS finding's evidence actually shows.

  mechanism:
{mechanism_note}

  remediation patterns:
{remediations}

YOUR TASK
1. diagnosis — what is happening on this cluster, in terms of the evidence. If
   the evidence shows a symptom but not its root cause, say exactly that.
2. mechanism — the specific MongoDB internal that connects the evidence to the
   impact. Ground it in the numbers you were given.
3. impact_if_ignored — what happens on THIS cluster if nothing changes, using the
   time-to-impact and topology. Be concrete about the consequence (for a
   replication finding: which member, what recovery it would need, how long the
   set runs degraded).
4. action — one remediation. Set kind "executable" ONLY for kill_op or
   create_small_index AND ONLY when the evidence names the exact opid+node or the
   exact collection. Otherwise kind "propose_only" with the precise command a
   human runs. Fill the executor_args from the evidence, never from a guess.
5. confidence — your confidence in the diagnosis from 0.0 to 1.0. Lower it when
   the evidence is a single snapshot, when the cause is not determinable from the
   evidence, or when a rate was derived from a long-run average rather than
   sampled live.
"""

# The mechanism behind each failure mode, handed to the model so it reasons from
# the same physics the detector used — not from whatever it half-remembers.
KNOWN_MECHANISMS: dict[str, str] = {
    "oplog_window_below_resync": (
        "The oplog is a capped collection of fixed byte size. The \"window\" is how "
        "many seconds of write history those bytes currently hold, so the window "
        "shrinks as the write rate rises — the size is constant, the seconds are "
        "not. A secondary that goes offline resumes by replaying the oplog from its "
        "last applied entry. If the entries it needs have been truncated because "
        "the window moved past them while it was down, it cannot resume "
        "incrementally: it must perform a full initial sync, copying the entire "
        "dataset from another member. On a large dataset that is hours, and "
        "throughout it the replica set runs one data-bearing member short, which "
        "narrows the margin for the majority write concern that guarantees "
        "durability. The detector's resync estimate is how long a member might "
        "plausibly be offline for routine maintenance; a window below that estimate "
        "means routine maintenance could trigger a full resync."
    ),
}

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

# BUILD_SPEC §1's replica set. Used when the config does not name one: the read
# URI normally connects directly to a single member and carries no replicaSet.
DEFAULT_REPLICA_SET = "rs0"

_NO_MECHANISM = "(no mechanism on file for this failure mode)"
_NO_REMEDIATIONS = "(none on file for this failure mode)"


def render_prompt(finding: Finding, context: Mapping[str, Any]) -> str:
    """The user prompt, shared verbatim by every provider."""
    evidence = "\n".join(f"  - {e.render()}" for e in finding.evidence) or "  - (none)"
    remediations = "\n".join(f"    - {hint}" for hint in context.get("remediations", ()))
    return PROMPT_TEMPLATE.format(
        replica_set=context.get("replica_set") or DEFAULT_REPLICA_SET,
        failure_mode=finding.failure_mode,
        severity=finding.severity.value,
        node=finding.node,
        horizon=finding.horizon_human(),
        summary=finding.summary,
        evidence=evidence,
        topology=f"  {context.get('topology') or '(not provided)'}",
        mechanism_note=f"    {context.get('mechanism') or _NO_MECHANISM}",
        remediations=remediations or f"    - {_NO_REMEDIATIONS}",
    )


def topology_summary(mongo: MongoConfig) -> str:
    fallbacks = ", ".join(mongo.fallback_nodes) or "none"
    return (
        f"Diagnostic reads are served by {mongo.target_node}; "
        f"fallback order: {fallbacks}."
    )


def replica_set_name(mongo: MongoConfig) -> str:
    """The URI's replicaSet option if it names one, else DEFAULT_REPLICA_SET."""
    options = {name.lower(): value for name, value in parse_qsl(urlsplit(mongo.uri).query)}
    return options.get("replicaset") or DEFAULT_REPLICA_SET


class Analyst:
    def __init__(
        self,
        chain: ProviderChain,
        *,
        topology: str = "",
        replica_set: str = DEFAULT_REPLICA_SET,
    ) -> None:
        self._chain = chain
        self._topology = topology
        self._replica_set = replica_set

    def build_context(self, finding: Finding) -> dict[str, Any]:
        return {
            "replica_set": self._replica_set,
            "topology": self._topology,
            "mechanism": KNOWN_MECHANISMS.get(finding.failure_mode, ""),
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
