"""Acceptance tests for the analysis stage — BUILD_SPEC §3.5.

Spec acceptance (unit, mocked providers):
  - a mocked primary returning valid JSON yields a Proposal referencing the finding_id
  - a mocked primary returning garbage twice, with a fallback returning valid
    JSON, yields a Proposal with provider == "openai"
  - both failing raises AnalysisUnavailable
  - a provider returning JSON that violates the schema is rejected
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import jsonschema
import openai
import pytest

from bellwether.analysis import analyst as analyst_module
from bellwether.analysis.analyst import (
    DEFAULT_REPLICA_SET,
    KNOWN_MECHANISMS,
    KNOWN_REMEDIATIONS,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    Analyst,
    render_prompt,
    replica_set_name,
)
from bellwether.analysis.claude import ClaudeProvider
from bellwether.analysis.openai import OpenAIProvider
from bellwether.analysis.provider import (
    AnalysisUnavailable,
    InvalidResponse,
    Provider,
    ProviderChain,
    ProviderError,
    parse_json_object,
)
from bellwether.analysis.schema import PROPOSAL_SCHEMA, ProposalPayload, validate_payload, wire_schema
from bellwether.config import MongoConfig
from bellwether.detectors.oplog_window import OplogWindowDetector
from bellwether.models import (
    ActionKind,
    Evidence,
    Finding,
    Severity,
    Signal,
    SignalClass,
)

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "config" / "proposal.schema.json"
TARGET = "node-backup.mongo.internal:27017"

VALID: dict[str, Any] = {
    "diagnosis": "The oplog holds 40 minutes of history, less than a 60 minute maintenance window.",
    "mechanism": "local.oplog.rs is capped by size; a higher write rate truncates older entries sooner.",
    "impact_if_ignored": "A secondary down for maintenance longer than 40 minutes needs a full initial sync.",
    "action": {
        "kind": "propose_only",
        "title": "Grow the oplog on every member",
        "command": "db.adminCommand({ replSetResizeOplog: 1, size: 51200 })",
        "rationale": "A larger oplog restores a window above the resync estimate.",
        "reversible": True,
        "executor_op": None,
        "executor_args": None,
    },
    "confidence": 0.82,
}

VALID_KILL_OP: dict[str, Any] = {
    **VALID,
    "action": {
        "kind": "executable",
        "title": "Kill the runaway collection scan",
        "command": "db.killOp(4242)",
        "rationale": "Op 4242 has held the lock for 900 s.",
        "reversible": True,
        "executor_op": "kill_op",
        "executor_args": {"opid": 4242},
    },
}

VALID_INDEX: dict[str, Any] = {
    **VALID,
    "action": {
        "kind": "executable",
        "title": "Index transactions.account_id",
        "command": "db.transactions.createIndex({ account_id: 1 })",
        "rationale": "Every slow query filters on account_id.",
        "reversible": True,
        "executor_op": "create_small_index",
        "executor_args": {
            "db": "meetadev_ledger",
            "collection": "transactions",
            "keys": [{"field": "account_id", "direction": 1}],
            "estimated_docs": 40_000,
        },
    },
}


def mutated(path: str, value: Any = ..., base: dict[str, Any] = VALID) -> dict[str, Any]:
    """A copy of `base` with dotted `path` set to `value` (or deleted if omitted)."""
    doc = copy.deepcopy(base)
    *parents, leaf = path.split(".")
    node = doc
    for key in parents:
        node = node[key]
    if value is ...:
        del node[leaf]
    else:
        node[leaf] = value
    return doc


@pytest.fixture
def finding() -> Finding:
    signal = Signal(
        signal_class=SignalClass.REPLICATION,
        source="oplog_window",
        node=TARGET,
        evidence=(Evidence("oplog_window_seconds", 2400, "s"),),
    )
    return Finding(
        signal_class=SignalClass.REPLICATION,
        failure_mode="oplog_window_below_resync",
        severity=Severity.CRITICAL,
        node=TARGET,
        summary=(
            f"Oplog window on {TARGET} is 40 min (2400 s), below the 60 min (3600 s) "
            "resync estimate."
        ),
        evidence=(
            Evidence("oplog_window_seconds", 2400, "s"),
            Evidence("resync_seconds", 3600, "s"),
        ),
        horizon_seconds=0,
        signals=(signal,),
    )


class ScriptedProvider(Provider):
    """Replays a script: dicts are returned, strings parsed as raw model text,
    exceptions raised. The last entry repeats."""

    def __init__(self, name: str, script: list[object]) -> None:
        self.name = name
        self.script = list(script)
        self.calls = 0
        self.contexts: list[dict[str, Any]] = []

    def analyze(self, finding: Finding, context: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.contexts.append(dict(context))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return parse_json_object(item)
        return copy.deepcopy(cast(dict[str, Any], item))


def analyst_with(primary: list[object], fallback: list[object]) -> tuple[Analyst, ScriptedProvider, ScriptedProvider]:
    claude = ScriptedProvider("claude", primary)
    gpt = ScriptedProvider("openai", fallback)
    chain = ProviderChain([claude, gpt], max_retries=1)
    return Analyst(chain, topology="target node-backup (hidden); fallbacks westeurope, uae"), claude, gpt


# --- Spec acceptance ---------------------------------------------------------


def test_valid_primary_yields_proposal_for_the_finding(finding: Finding) -> None:
    analyst, claude, gpt = analyst_with([VALID], [VALID])

    proposal = analyst.analyze(finding)

    assert proposal.finding_id == finding.finding_id
    assert proposal.provider == "claude"
    assert proposal.failure_mode == finding.failure_mode
    assert proposal.node == finding.node
    assert proposal.evidence_refs == finding.evidence
    assert proposal.diagnosis == VALID["diagnosis"]
    assert proposal.confidence == pytest.approx(0.82)
    assert proposal.action.kind is ActionKind.PROPOSE_ONLY
    assert proposal.action.executor_op is None
    assert proposal.action.executor_args == {}
    assert (claude.calls, gpt.calls) == (1, 0)


def test_garbage_twice_then_fallback_yields_openai_proposal(finding: Finding) -> None:
    analyst, claude, gpt = analyst_with(["not json at all", "{still: not json"], [VALID])

    proposal = analyst.analyze(finding)

    assert proposal.provider == "openai"
    assert proposal.finding_id == finding.finding_id
    assert (claude.calls, gpt.calls) == (2, 1)


def test_both_failing_raises_analysis_unavailable(finding: Finding) -> None:
    analyst, claude, gpt = analyst_with(["garbage"], [ProviderError("timed out after 60 s")])

    with pytest.raises(AnalysisUnavailable) as excinfo:
        analyst.analyze(finding)

    assert (claude.calls, gpt.calls) == (2, 2)
    message = str(excinfo.value)
    assert "claude" in message and "openai" in message


SCHEMA_VIOLATIONS = {
    "missing diagnosis": mutated("diagnosis"),
    "extra top-level field": {**VALID, "provider": "claude"},
    "model tries to set finding_id": {**VALID, "finding_id": "forged"},
    "confidence above 1": mutated("confidence", 1.5),
    "confidence below 0": mutated("confidence", -0.1),
    "confidence as string": mutated("confidence", "0.9"),
    "empty diagnosis": mutated("diagnosis", ""),
    "unknown action kind": mutated("action.kind", "maybe"),
    "reversible as string": mutated("action.reversible", "true"),
    "missing command": mutated("action.command"),
    "extra action field": mutated("action.shell", "rm -rf /"),
    "executable without op": mutated("action.kind", "executable"),
    "propose_only with op": mutated("action.executor_op", "kill_op", base=VALID),
    "op outside whitelist": mutated("action.executor_op", "drop_database", base=VALID_KILL_OP),
    "kill_op with index args": mutated(
        "action.executor_args", VALID_INDEX["action"]["executor_args"], base=VALID_KILL_OP
    ),
    "opid as string": mutated("action.executor_args", {"opid": "4242"}, base=VALID_KILL_OP),
    "opid as bool": mutated("action.executor_args", {"opid": True}, base=VALID_KILL_OP),
    "index direction 2": mutated(
        "action.executor_args.keys", [{"field": "a", "direction": 2}], base=VALID_INDEX
    ),
    "index with no keys": mutated("action.executor_args.keys", [], base=VALID_INDEX),
    "executable marked irreversible": mutated("action.reversible", False, base=VALID_KILL_OP),
}


@pytest.mark.parametrize("payload", SCHEMA_VIOLATIONS.values(), ids=SCHEMA_VIOLATIONS.keys())
def test_schema_violations_are_rejected(finding: Finding, payload: dict[str, Any]) -> None:
    analyst, claude, gpt = analyst_with([payload], [payload])

    with pytest.raises(AnalysisUnavailable):
        analyst.analyze(finding)

    assert (claude.calls, gpt.calls) == (2, 2)  # retried, then failed over, never coerced


def test_schema_violating_primary_falls_over_to_valid_fallback(finding: Finding) -> None:
    analyst, _, _ = analyst_with([SCHEMA_VIOLATIONS["confidence as string"]], [VALID])

    assert analyst.analyze(finding).provider == "openai"


# --- ProviderChain -----------------------------------------------------------


def test_retry_within_primary_before_failing_over(finding: Finding) -> None:
    analyst, claude, gpt = analyst_with(["garbage", VALID], [VALID])

    proposal = analyst.analyze(finding)

    assert proposal.provider == "claude"
    assert (claude.calls, gpt.calls) == (2, 0)


def test_max_retries_zero_means_one_attempt(finding: Finding) -> None:
    claude = ScriptedProvider("claude", ["garbage"])
    gpt = ScriptedProvider("openai", [VALID])
    analyst = Analyst(ProviderChain([claude, gpt], max_retries=0))

    assert analyst.analyze(finding).provider == "openai"
    assert claude.calls == 1


def test_provider_exceptions_fail_over(finding: Finding) -> None:
    analyst, _, _ = analyst_with([ProviderError("HTTP 529 overloaded")], [VALID])

    assert analyst.analyze(finding).provider == "openai"


def test_health_tracks_consecutive_failures(finding: Finding) -> None:
    claude = ScriptedProvider("claude", ["garbage", "garbage", VALID])
    gpt = ScriptedProvider("openai", [VALID])
    chain = ProviderChain([claude, gpt], max_retries=1)
    analyst = Analyst(chain)

    analyst.analyze(finding)
    assert chain.health["claude"].consecutive_failures == 2
    assert chain.health["claude"].total_failures == 2
    assert chain.health["openai"].consecutive_failures == 0
    assert chain.health["openai"].last_success_at is not None

    analyst.analyze(finding)  # claude now answers
    assert chain.health["claude"].consecutive_failures == 0
    assert chain.health["claude"].total_failures == 2


def test_chain_logs_the_provider_used(finding: Finding, caplog: pytest.LogCaptureFixture) -> None:
    analyst, _, _ = analyst_with(["garbage", "garbage"], [VALID])

    with caplog.at_level(logging.INFO, logger="bellwether.analysis"):
        analyst.analyze(finding)

    served = [r for r in caplog.records if r.getMessage() == "analysis served"]
    assert [getattr(r, "provider") for r in served] == ["openai"]
    failed = [r for r in caplog.records if getattr(r, "provider", None) == "claude"]
    assert len([r for r in failed if r.levelno == logging.WARNING]) >= 2


def test_both_providers_receive_the_same_context(finding: Finding) -> None:
    analyst, claude, gpt = analyst_with(["garbage"], [VALID])

    analyst.analyze(finding)

    assert claude.contexts[0] == gpt.contexts[0]


def test_chain_requires_a_provider() -> None:
    with pytest.raises(ValueError):
        ProviderChain([], max_retries=1)


def test_parse_json_object() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(InvalidResponse):
        parse_json_object("[1, 2]")  # JSON, but not an object
    with pytest.raises(InvalidResponse):
        parse_json_object("")


# --- Payload -> Proposal -----------------------------------------------------


def test_executable_index_proposal_maps_args() -> None:
    # create_small_index must be grounded in an index finding (see §F tests below).
    grounded = index_finding(candidate=[{"field": "account_id", "direction": 1}], count=40_000)
    analyst, _, _ = analyst_with([VALID_INDEX], [VALID])

    proposal = analyst.analyze(grounded)

    assert proposal.action.kind is ActionKind.EXECUTABLE
    assert proposal.action.executor_op == "create_small_index"
    assert proposal.action.executor_args == VALID_INDEX["action"]["executor_args"]
    assert proposal.action.command == VALID_INDEX["action"]["command"]


def test_executable_kill_op_proposal_maps_args(finding: Finding) -> None:
    analyst, _, _ = analyst_with([VALID_KILL_OP], [VALID])

    proposal = analyst.analyze(finding)

    assert proposal.action.executor_op == "kill_op"
    assert proposal.action.executor_args == {"opid": 4242}


# --- Schema file and model agree; schema fits structured outputs -------------


def test_schema_file_matches_the_code() -> None:
    assert json.loads(SCHEMA_FILE.read_text()) == PROPOSAL_SCHEMA


def test_schema_is_valid_json_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(PROPOSAL_SCHEMA)


def _objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_objects(value))
    return found


def test_schema_fits_structured_output_rules() -> None:
    """Both providers constrain decoding with this schema: every object closed,
    every property required (nullables via anyOf null), no numeric bounds."""
    objects = _objects(PROPOSAL_SCHEMA)
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    text = json.dumps(PROPOSAL_SCHEMA)
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        assert f'"{keyword}"' not in text
    wire = wire_schema()
    assert "$schema" not in wire and "$id" not in wire and "title" not in wire


@pytest.mark.parametrize("payload", [VALID, VALID_KILL_OP, VALID_INDEX])
def test_valid_payloads_pass_schema_and_model(payload: dict[str, Any]) -> None:
    jsonschema.validate(payload, PROPOSAL_SCHEMA)
    assert isinstance(validate_payload(payload), ProposalPayload)


STRUCTURAL = [
    "missing diagnosis",
    "extra top-level field",
    "unknown action kind",
    "reversible as string",
    "op outside whitelist",
    "opid as string",
    "index direction 2",
]


@pytest.mark.parametrize("case", STRUCTURAL)
def test_structural_violations_fail_schema_and_model(case: str) -> None:
    payload = SCHEMA_VIOLATIONS[case]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, PROPOSAL_SCHEMA)
    with pytest.raises(ValueError):
        validate_payload(payload)


# --- Bounded context, prompt template ----------------------------------------


def test_prompt_carries_finding_topology_and_remediations(finding: Finding) -> None:
    analyst, claude, _ = analyst_with([VALID], [VALID])
    analyst.analyze(finding)
    context = claude.contexts[0]

    prompt = render_prompt(finding, context)

    assert finding.summary in prompt
    assert "critical" in prompt
    for evidence in finding.evidence:
        assert evidence.render() in prompt
    assert "node-backup (hidden)" in prompt
    for hint in KNOWN_REMEDIATIONS["oplog_window_below_resync"]:
        assert hint in prompt
    # Bounded: no raw signal dumps or ids.
    assert finding.signals[0].signal_id not in prompt


def test_context_is_bounded(finding: Finding) -> None:
    analyst, claude, _ = analyst_with([VALID], [VALID])
    analyst.analyze(finding)

    assert set(claude.contexts[0]) == {"topology", "remediations", "mechanism", "replica_set"}


def test_unknown_failure_mode_still_renders(finding: Finding) -> None:
    other = Finding(
        signal_class=finding.signal_class,
        failure_mode="something_new",
        severity=Severity.WARNING,
        node=TARGET,
        summary="Something new.",
        evidence=(),
    )

    prompt = render_prompt(other, {"topology": "", "remediations": []})

    assert "Something new." in prompt


def test_prompt_is_the_designed_version_not_the_stub() -> None:
    source = Path(analyst_module.__file__).read_text()

    assert "TODO(design)" not in source
    assert PROMPT_TEMPLATE.startswith("A detector fired on replica set {replica_set}.")


# --- The designed prompt: evidence-constrained, mechanism-grounded -----------------

TOPOLOGY = (
    "Diagnostic reads are served by node-backup.mongo.internal:27017; fallback order: "
    "node-westeurope.mongo.internal:27017, node-uae.mongo.internal:27017, "
    "node-southafrica.mongo.internal:27017."
)


@pytest.fixture
def oplog_finding() -> Finding:
    """A real Finding, produced by the oplog-window detector from a 40-minute window."""
    size, window = 990 * 2**20, 2400
    signal = Signal(
        signal_class=SignalClass.REPLICATION,
        source="oplog_window",
        node=TARGET,
        evidence=(
            Evidence("oplog_size_bytes", size, "bytes"),
            Evidence("oplog_used_bytes", size, "bytes"),
            Evidence("oplog_window_seconds", window, "s"),
            Evidence("oplog_mean_rate_bytes_per_sec", size / window, "bytes/s"),
            Evidence("write_rate_bytes_per_sec", size / window, "bytes/s"),
            Evidence("write_rate_source", "oplog_mean"),
        ),
    )
    finding = OplogWindowDetector().evaluate([signal])
    assert finding is not None
    return finding


def designed_analyst() -> Analyst:
    chain = ProviderChain([ScriptedProvider("claude", [VALID])])
    return Analyst(chain, topology=TOPOLOGY, replica_set="rs0")


def test_designed_prompt_carries_every_part_of_the_finding(oplog_finding: Finding) -> None:
    context = designed_analyst().build_context(oplog_finding)

    prompt = render_prompt(oplog_finding, context)

    assert "A detector fired on replica set rs0." in prompt
    # 40 min against a 60 min resync is CRITICAL: the threshold is already crossed.
    assert "time to impact: already crossed" in prompt
    for part in (
        oplog_finding.failure_mode,
        oplog_finding.severity.value,
        oplog_finding.node,
        oplog_finding.horizon_human(),
        oplog_finding.summary,
        TOPOLOGY,
    ):
        assert part in prompt, part
    for evidence in oplog_finding.evidence:
        assert evidence.render() in prompt, evidence.name
    assert KNOWN_MECHANISMS["oplog_window_below_resync"] in prompt
    for hint in KNOWN_REMEDIATIONS["oplog_window_below_resync"]:
        assert hint in prompt


@pytest.mark.parametrize(
    "context",
    [
        {"topology": "", "remediations": []},
        {"topology": "", "remediations": [], "mechanism": "", "replica_set": "rs0"},
    ],
    ids=["keys absent", "keys empty"],
)
def test_missing_mechanism_renders_an_explicit_line(context: dict[str, Any]) -> None:
    other = Finding(
        signal_class=SignalClass.CAPACITY,
        failure_mode="not_a_known_failure_mode",
        severity=Severity.WARNING,
        node=TARGET,
        summary="Something new.",
        evidence=(Evidence("cache_dirty_ratio", 0.21),),
    )

    prompt = render_prompt(other, context)

    assert "(no mechanism on file for this failure mode)" in prompt
    assert "(none on file for this failure mode)" in prompt
    assert "  mechanism:\n\n" not in prompt  # never an empty block


def test_system_prompt_carries_the_load_bearing_instructions() -> None:
    assert "the numbers in the evidence are ground truth" in SYSTEM_PROMPT
    assert "You propose; you never act" in SYSTEM_PROMPT
    assert "lower your confidence" in SYSTEM_PROMPT


def test_build_context_adds_mechanism_and_replica_set(oplog_finding: Finding) -> None:
    context = designed_analyst().build_context(oplog_finding)

    assert context["mechanism"] == KNOWN_MECHANISMS["oplog_window_below_resync"]
    assert context["mechanism"].strip()
    assert context["replica_set"] == "rs0"


def test_every_remediation_mode_has_a_mechanism() -> None:
    assert set(KNOWN_REMEDIATIONS) <= set(KNOWN_MECHANISMS)


def _mongo_config(uri: str) -> MongoConfig:
    return MongoConfig(
        uri=uri,
        tls_ca_file=Path("/etc/mongodb/tls/ca-chain.cert.pem"),
        tls_cert_file=Path("/etc/bellwether/tls/meetadev-ai.combined.pem"),
        target_node=TARGET,
    )


def test_replica_set_name_comes_from_the_uri_else_the_documented_default() -> None:
    named = "mongodb://node-backup.mongo.internal:27017/?replicaSet=ledger0&tls=true"
    unnamed = "mongodb://node-backup.mongo.internal:27017/?tls=true&directConnection=true"

    assert replica_set_name(_mongo_config(named)) == "ledger0"
    assert replica_set_name(_mongo_config(unnamed)) == DEFAULT_REPLICA_SET == "rs0"


# --- Index Advisor grounding (INDEX_ADVISOR_SPEC §F) -------------------------------

INDEX_CANDIDATE: list[dict[str, Any]] = [
    {"field": "account_id", "direction": 1},
    {"field": "status", "direction": 1},
    {"field": "posted_at", "direction": -1},
]
INDEX_MODES = ("missing_index_collscan", "profiler_disabled", "redundant_index")


def index_finding(
    *,
    candidate: list[dict[str, Any]] | None = None,
    count: int | None = 40_000,
    collection: str = "transactions",
) -> Finding:
    """A missing_index_collscan finding shaped like IndexAdvisorDetector's."""
    namespace = f"meetadev_ledger.{collection}"
    return Finding(
        signal_class=SignalClass.PERFORMANCE,
        failure_mode="missing_index_collscan",
        severity=Severity.WARNING,
        node=TARGET,
        summary=(
            f"Query shape {{account_id: eq, posted_at: range, status: eq}} sort {{posted_at: -1}} "
            f"on {namespace} examined 1,000,000 documents to return 1,000."
        ),
        evidence=(
            Evidence("db", "meetadev_ledger"),
            Evidence("collection", collection),
            Evidence("namespace", namespace),
            Evidence("subject", f"{namespace}#0123456789abcdef"),
            Evidence("query_shape", {"filter": {"account_id": "eq", "posted_at": "range", "status": "eq"}, "sort": [["posted_at", -1]]}),
            Evidence("targeting_ratio", 1000.0),
            Evidence("wasted_bytes", 435_564_000, "bytes"),
            Evidence("candidate_index", INDEX_CANDIDATE if candidate is None else candidate),
            Evidence("existing_indexes", [{"name": "_id_", "key": [{"field": "_id", "direction": 1}]}]),
            Evidence("collection_doc_count", count, "docs"),
        ),
    )


def index_payload(
    keys: list[dict[str, Any]] | None = None,
    *,
    db: str = "meetadev_ledger",
    collection: str = "transactions",
    estimated_docs: int = 40_000,
) -> dict[str, Any]:
    return {
        **VALID,
        "action": {
            "kind": "executable",
            "title": "Index the hot transactions shape",
            "command": "db.transactions.createIndex({ account_id: 1, status: 1, posted_at: -1 })",
            "rationale": "The ESR candidate from the evidence.",
            "reversible": True,
            "executor_op": "create_small_index",
            "executor_args": {
                "db": db,
                "collection": collection,
                "keys": INDEX_CANDIDATE if keys is None else keys,
                "estimated_docs": estimated_docs,
            },
        },
    }


def test_index_failure_modes_have_mechanisms_and_remediations() -> None:
    for mode in INDEX_MODES:
        assert KNOWN_MECHANISMS[mode].strip(), mode
        assert KNOWN_REMEDIATIONS[mode], mode
    assert "ESR" in KNOWN_MECHANISMS["missing_index_collscan"]
    assert "write" in KNOWN_MECHANISMS["missing_index_collscan"]
    assert "system.profile" in KNOWN_MECHANISMS["profiler_disabled"]
    assert "propose-only" in KNOWN_MECHANISMS["redundant_index"]


def test_prompt_carries_the_candidate_and_existing_indexes() -> None:
    finding = index_finding()
    analyst = Analyst(ProviderChain([ScriptedProvider("claude", [VALID])]), topology=TOPOLOGY)

    prompt = render_prompt(finding, analyst.build_context(finding))

    evidence = {e.name: e for e in finding.evidence}
    assert evidence["candidate_index"].render() in prompt
    assert evidence["existing_indexes"].render() in prompt
    assert KNOWN_MECHANISMS["missing_index_collscan"] in prompt


def test_grounded_create_small_index_is_accepted() -> None:
    analyst, claude, _ = analyst_with([index_payload()], [VALID])

    proposal = analyst.analyze(index_finding())

    assert proposal.provider == "claude"
    assert proposal.action.kind is ActionKind.EXECUTABLE
    assert proposal.action.executor_args["keys"] == INDEX_CANDIDATE
    assert proposal.action.executor_args["estimated_docs"] == 40_000
    assert claude.calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        index_payload([INDEX_CANDIDATE[1], INDEX_CANDIDATE[0], INDEX_CANDIDATE[2]]),
        index_payload([INDEX_CANDIDATE[0], INDEX_CANDIDATE[1], {"field": "posted_at", "direction": 1}]),
        index_payload(INDEX_CANDIDATE[:2]),
        index_payload([*INDEX_CANDIDATE, {"field": "amount", "direction": 1}]),
    ],
    ids=["field order changed", "direction changed", "field dropped", "field added"],
)
def test_the_model_can_never_choose_the_index_keys(payload: dict[str, Any]) -> None:
    analyst, claude, gpt = analyst_with([payload], [payload])

    with pytest.raises(AnalysisUnavailable) as excinfo:
        analyst.analyze(index_finding())

    assert (claude.calls, gpt.calls) == (2, 2)  # rejected and retried, never coerced
    assert "candidate_index" in str(excinfo.value)


def test_ungrounded_keys_fall_over_to_a_grounded_fallback() -> None:
    reordered = index_payload([INDEX_CANDIDATE[1], INDEX_CANDIDATE[0], INDEX_CANDIDATE[2]])
    analyst, _, _ = analyst_with([reordered], [index_payload()])

    assert analyst.analyze(index_finding()).provider == "openai"


def test_create_small_index_over_the_executor_threshold_is_rejected() -> None:
    big = index_finding(count=5_000_000)
    analyst, _, _ = analyst_with([index_payload(estimated_docs=5_000_000)], [VALID])

    proposal = analyst.analyze(big)

    assert proposal.provider == "openai"  # the primary's executable proposal was refused
    assert proposal.action.kind is ActionKind.PROPOSE_ONLY  # a human runs createIndex


def test_the_threshold_is_the_analysts() -> None:
    chain = ProviderChain([ScriptedProvider("claude", [index_payload()])], max_retries=0)

    with pytest.raises(AnalysisUnavailable, match="threshold"):
        Analyst(chain, document_threshold=10_000).analyze(index_finding(count=40_000))


@pytest.mark.parametrize(
    ("finding_kwargs", "payload"),
    [
        ({}, index_payload(collection="accounts")),
        ({}, index_payload(db="billing")),
        ({}, index_payload(estimated_docs=39_000)),
        ({"count": None}, index_payload()),
    ],
    ids=["wrong collection", "wrong db", "estimate not from evidence", "doc count unknown"],
)
def test_create_small_index_args_must_come_from_the_evidence(
    finding_kwargs: dict[str, Any], payload: dict[str, Any]
) -> None:
    analyst, _, _ = analyst_with([payload], [payload])

    with pytest.raises(AnalysisUnavailable):
        analyst.analyze(index_finding(**finding_kwargs))


def test_create_small_index_needs_an_index_finding(finding: Finding) -> None:
    # The oplog finding carries no candidate index: nothing to build.
    analyst, _, _ = analyst_with([index_payload()], [index_payload()])

    with pytest.raises(AnalysisUnavailable, match="candidate_index"):
        analyst.analyze(finding)


def test_propose_only_is_accepted_for_an_index_finding() -> None:
    analyst, _, _ = analyst_with([VALID], [VALID])

    assert analyst.analyze(index_finding()).action.kind is ActionKind.PROPOSE_ONLY


# --- Claude and OpenAI wrappers (SDK clients faked, no network) -------------


class FakeCalls:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def claude_reply(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )


def openai_reply(
    text: str | None, finish_reason: str = "stop", refusal: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1100, completion_tokens=280),
    )


def claude_with(reply: object) -> tuple[ClaudeProvider, FakeCalls]:
    calls = FakeCalls(reply)
    client = cast(anthropic.Anthropic, SimpleNamespace(messages=calls))
    return ClaudeProvider(api_key="k", model="claude-opus-5", timeout_seconds=60, client=client), calls


def openai_with(reply: object) -> tuple[OpenAIProvider, FakeCalls]:
    calls = FakeCalls(reply)
    client = cast(openai.OpenAI, SimpleNamespace(chat=SimpleNamespace(completions=calls)))
    return OpenAIProvider(api_key="k", model="gpt-5", timeout_seconds=60, client=client), calls


CONTEXT: dict[str, Any] = {"topology": "target node-backup", "remediations": ["grow the oplog"]}
REQUEST = httpx2.Request("POST", "https://api.example.invalid/v1")


def test_provider_names_match_config() -> None:
    assert ClaudeProvider.name == "claude"
    assert OpenAIProvider.name == "openai"


def test_claude_requests_schema_constrained_json(finding: Finding) -> None:
    provider, calls = claude_with(claude_reply(json.dumps(VALID)))

    assert provider.analyze(finding, CONTEXT) == VALID

    sent = calls.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": wire_schema()}}
    assert sent["messages"] == [{"role": "user", "content": render_prompt(finding, CONTEXT)}]
    assert sent["system"]


def test_openai_requests_schema_constrained_json(finding: Finding) -> None:
    provider, calls = openai_with(openai_reply(json.dumps(VALID)))

    assert provider.analyze(finding, CONTEXT) == VALID

    sent = calls.calls[0]
    assert sent["model"] == "gpt-5"
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] == wire_schema()
    assert sent["messages"][-1] == {"role": "user", "content": render_prompt(finding, CONTEXT)}


def test_providers_are_interchangeable(finding: Finding) -> None:
    claude, claude_calls = claude_with(claude_reply(json.dumps(VALID)))
    gpt, gpt_calls = openai_with(openai_reply(json.dumps(VALID)))

    claude.analyze(finding, CONTEXT)
    gpt.analyze(finding, CONTEXT)

    claude_sent, gpt_sent = claude_calls.calls[0], gpt_calls.calls[0]
    assert claude_sent["system"] == gpt_sent["messages"][0]["content"]
    assert claude_sent["messages"][0]["content"] == gpt_sent["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (claude_reply("I'd rather not.", stop_reason="refusal"), ProviderError),
        (claude_reply('{"diagnosis": "trunc', stop_reason="max_tokens"), InvalidResponse),
        (claude_reply("not json"), InvalidResponse),
        (anthropic.APITimeoutError(request=REQUEST), ProviderError),
        (anthropic.APIConnectionError(request=REQUEST), ProviderError),
    ],
)
def test_claude_failures_become_provider_errors(
    finding: Finding, reply: object, error: type[Exception]
) -> None:
    provider, _ = claude_with(reply)

    with pytest.raises(error):
        provider.analyze(finding, CONTEXT)


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (openai_reply(None, refusal="I can't help with that."), ProviderError),
        (openai_reply('{"diagnosis": "trunc', finish_reason="length"), InvalidResponse),
        (openai_reply("not json"), InvalidResponse),
        (openai.APITimeoutError(request=REQUEST), ProviderError),
    ],
)
def test_openai_failures_become_provider_errors(
    finding: Finding, reply: object, error: type[Exception]
) -> None:
    provider, _ = openai_with(reply)

    with pytest.raises(error):
        provider.analyze(finding, CONTEXT)


def test_providers_log_token_usage(finding: Finding, caplog: pytest.LogCaptureFixture) -> None:
    claude, _ = claude_with(claude_reply(json.dumps(VALID)))
    gpt, _ = openai_with(openai_reply(json.dumps(VALID)))

    with caplog.at_level(logging.INFO, logger="bellwether.analysis"):
        claude.analyze(finding, CONTEXT)
        gpt.analyze(finding, CONTEXT)

    usage = {
        getattr(r, "provider"): (getattr(r, "input_tokens"), getattr(r, "output_tokens"))
        for r in caplog.records
        if r.getMessage() == "provider response"
    }
    assert usage == {"claude": (1200, 300), "openai": (1100, 280)}


def test_sdk_clients_leave_retries_to_the_chain() -> None:
    claude = ClaudeProvider(api_key="k", model="claude-opus-5", timeout_seconds=12)
    gpt = OpenAIProvider(api_key="k", model="gpt-5", timeout_seconds=12)

    assert claude._client.max_retries == 0
    assert gpt._client.max_retries == 0
    assert claude._client.timeout == 12
    assert gpt._client.timeout == 12
