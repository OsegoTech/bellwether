"""Acceptance tests for pipeline.run_once — BUILD_SPEC §3.10.

Spec acceptance: a run against a mocked cluster with an induced oplog-window
finding produces a stored proposal and a stdout notification, and records a run.
Plus the token gate: non-escalating findings are stored but not analyzed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from bellwether import pipeline
from bellwether.analysis.analyst import Analyst
from bellwether.analysis.claude import ClaudeProvider
from bellwether.analysis.openai import OpenAIProvider
from bellwether.analysis.provider import Provider, ProviderChain, ProviderError
from bellwether.config import BellwetherConfig, load_config
from bellwether.models import (
    ApprovalRecord,
    ApprovalState,
    Finding,
    Proposal,
    Severity,
    SignalClass,
)
from bellwether.mongo import ReadOnlyMongo
from bellwether.notify.base import Notifier
from bellwether.notify.slack import SlackNotifier
from bellwether.notify.stdout import StdoutNotifier
from bellwether.pipeline import run_once
from bellwether.store.sqlite import SqliteStore
from tests.fakes import (
    FALLBACKS,
    PROPOSAL_PAYLOAD,
    PROVIDER_KEYS,
    TARGET,
    FakeCluster,
    StaticProvider,
    oplog_cluster,
    write_config,
)

MIN = 60


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("BELLWETHER_"):
            monkeypatch.delenv(name)
    for name, value in PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)


def config_in(tmp_path: Path, **sections: dict[str, object]) -> BellwetherConfig:
    return load_config(write_config(tmp_path, **sections))


def mongo_for(config: BellwetherConfig, cluster: FakeCluster) -> ReadOnlyMongo:
    return ReadOnlyMongo(config.mongo, client_factory=cluster.factory())


def analyst_of(*providers: Provider) -> Analyst:
    return Analyst(ProviderChain(list(providers), max_retries=1))


class BrokenNotifier(Notifier):
    name = "broken"

    def notify(self, proposal: Proposal, approval_record: ApprovalRecord) -> None:
        raise OSError("webhook unreachable")


# --- Spec acceptance ---------------------------------------------------------------


def test_run_stores_proposal_notifies_and_records_run(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)
    stdout = io.StringIO()

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(40 * MIN)),
        analyst=analyst_of(claude),
        notifiers=[StdoutNotifier(stdout)],
    )

    store = SqliteStore(config.store.sqlite_path)
    assert len(summary.proposals) == 1
    proposal = summary.proposals[0]
    assert store.get_proposal(proposal.proposal_id) == proposal
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING
    assert proposal.failure_mode == "oplog_window_below_resync"
    assert proposal.node == TARGET
    assert proposal.provider == "claude"

    assert proposal.proposal_id in stdout.getvalue()
    assert PROPOSAL_PAYLOAD["diagnosis"] in stdout.getvalue()

    [run] = store.list_runs()
    assert run.run_id == summary.run_id
    assert (run.findings, run.proposals, run.errors) == (1, 1, ())
    assert run.finished_at >= run.started_at
    [finding] = store.list_findings(summary.run_id)
    assert finding.severity is Severity.CRITICAL
    assert finding.escalated is True


# --- Token gate -------------------------------------------------------------------------


def test_non_escalating_finding_is_stored_not_analyzed(tmp_path: Path) -> None:
    config = config_in(tmp_path, analysis={"escalate_min_severity": "critical"})
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)
    stdout = io.StringIO()

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(90 * MIN)),  # WARNING
        analyst=analyst_of(claude),
        notifiers=[StdoutNotifier(stdout)],
    )

    assert claude.calls == 0
    assert summary.proposals == []
    assert stdout.getvalue() == ""
    store = SqliteStore(config.store.sqlite_path)
    [finding] = store.list_findings()
    assert finding.severity is Severity.WARNING
    assert finding.escalated is False
    [run] = store.list_runs()
    assert (run.findings, run.proposals) == (1, 0)


def test_default_gate_escalates_warning(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(90 * MIN)),
        analyst=analyst_of(claude),
        notifiers=[],
    )

    assert claude.calls == 1
    assert len(summary.proposals) == 1


def test_healthy_cluster_records_an_empty_run(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(6 * 60 * MIN)),
        analyst=analyst_of(claude),
        notifiers=[],
    )

    assert summary.findings == [] and summary.proposals == []
    assert claude.calls == 0
    [run] = SqliteStore(config.store.sqlite_path).list_runs()
    assert (run.findings, run.proposals, run.errors) == (0, 0, ())


# --- Failure handling -------------------------------------------------------------------


def test_analysis_unavailable_records_finding_unanalyzed(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", ProviderError("HTTP 529"))
    gpt = StaticProvider("openai", ProviderError("timeout"))

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(40 * MIN)),
        analyst=analyst_of(claude, gpt),
        notifiers=[],
    )

    assert summary.proposals == []
    assert any("analysis unavailable" in e for e in summary.errors)
    store = SqliteStore(config.store.sqlite_path)
    assert len(store.list_findings()) == 1
    assert store.list_proposals() == []
    [run] = store.list_runs()
    assert run.errors == tuple(summary.errors)


def test_unreachable_cluster_is_an_error_not_a_crash(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    cluster = oplog_cluster(40 * MIN)
    cluster.down.update({TARGET, *FALLBACKS})

    summary = run_once(
        config,
        mongo=mongo_for(config, cluster),
        analyst=analyst_of(StaticProvider("claude", PROPOSAL_PAYLOAD)),
        notifiers=[],
    )

    assert summary.findings == []
    assert any("collector oplog_window" in e for e in summary.errors)
    [run] = SqliteStore(config.store.sqlite_path).list_runs()
    assert run.errors


def test_notifier_failure_keeps_the_proposal(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    stdout = io.StringIO()

    summary = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(40 * MIN)),
        analyst=analyst_of(StaticProvider("claude", PROPOSAL_PAYLOAD)),
        notifiers=[BrokenNotifier(), StdoutNotifier(stdout)],
    )

    assert len(summary.proposals) == 1
    assert any("notify broken" in e for e in summary.errors)
    assert summary.proposals[0].proposal_id in stdout.getvalue()  # other channels still fire


def test_pending_proposal_is_not_re_analyzed(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)

    for _ in range(3):
        run_once(
            config,
            mongo=mongo_for(config, oplog_cluster(40 * MIN)),
            analyst=analyst_of(claude),
            notifiers=[],
        )

    store = SqliteStore(config.store.sqlite_path)
    assert claude.calls == 1
    assert len(store.list_proposals()) == 1
    assert len(store.list_findings()) == 3  # every run's finding is still recorded
    assert len(store.list_runs()) == 3


def test_decided_proposal_allows_a_fresh_one(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    claude = StaticProvider("claude", PROPOSAL_PAYLOAD)
    first = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(40 * MIN)),
        analyst=analyst_of(claude),
        notifiers=[],
    )
    store = SqliteStore(config.store.sqlite_path)
    store.record_approval_transition(
        first.proposals[0].proposal_id, ApprovalState.REJECTED, by="x"
    )

    second = run_once(
        config,
        mongo=mongo_for(config, oplog_cluster(40 * MIN)),
        analyst=analyst_of(claude),
        notifiers=[],
    )

    assert len(second.proposals) == 1
    assert claude.calls == 2


# --- Wiring from config ---------------------------------------------------------------


def test_build_providers_follows_config(tmp_path: Path) -> None:
    providers = pipeline.build_providers(config_in(tmp_path))

    assert [type(p) for p in providers] == [ClaudeProvider, OpenAIProvider]


def test_build_analyst_grounds_the_prompt_in_config(tmp_path: Path) -> None:
    finding = Finding(
        signal_class=SignalClass.REPLICATION,
        failure_mode="oplog_window_below_resync",
        severity=Severity.CRITICAL,
        node=TARGET,
        summary="Oplog window is 40 min.",
        evidence=(),
    )

    context = pipeline.build_analyst(config_in(tmp_path)).build_context(finding)

    assert context["replica_set"] == "rs0"  # read URI names no set: documented default
    assert TARGET in context["topology"]
    assert context["mechanism"]


def test_build_analyst_uses_the_executor_document_threshold(tmp_path: Path) -> None:
    config = config_in(tmp_path, executor={"document_threshold": 12_345})

    assert pipeline.build_analyst(config).document_threshold == 12_345


def test_build_providers_single_provider(tmp_path: Path) -> None:
    config = config_in(tmp_path, analysis={"fallback_provider": None})

    assert [p.name for p in pipeline.build_providers(config)] == ["claude"]


def test_build_notifiers_follows_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_WEBHOOK_URL", "https://hooks.slack.com/x")
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET", "s")
    config = config_in(tmp_path, notify={"channels": ["stdout", "slack"]})

    notifiers = pipeline.build_notifiers(config)

    assert [type(n) for n in notifiers] == [StdoutNotifier, SlackNotifier]


def test_build_executor_is_none_unless_enabled(tmp_path: Path) -> None:
    config = config_in(tmp_path)

    assert pipeline.build_executor(config, SqliteStore(config.store.sqlite_path)) is None


def test_build_executor_knows_every_configured_member(tmp_path: Path) -> None:
    config = config_in(
        tmp_path,
        executor={
            "enabled": True,
            "mongo_uri": f"mongodb://{','.join(FALLBACKS)}/?replicaSet=rs0"
            "&authMechanism=MONGODB-X509&authSource=%24external&tls=true",
            "tls_cert_file": "/etc/bellwether/tls/meetadev-ai-exec.combined.pem",
            "tls_ca_file": "/etc/mongodb/tls/ca-chain.cert.pem",
            "allowed_actions": ["kill_op"],
        },
    )

    executor = pipeline.build_executor(config, SqliteStore(config.store.sqlite_path))

    assert executor is not None
    # Replica-set seeds plus the read side's target (the hidden backup) and fallbacks.
    assert {TARGET, *FALLBACKS} <= executor.known_nodes
