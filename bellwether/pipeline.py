"""One pipeline pass (BUILD_SPEC §3.10).

    collectors -> signals -> detectors -> findings
        -> (escalating only) analyst -> proposal -> store -> notify

Every finding is stored. Only findings at or above
``analysis.escalate_min_severity`` reach the analyst (the token gate), and a
finding already covered by a PENDING proposal for the same failure mode and
node is not re-analyzed, so a persistent condition does not spawn a proposal
per timer tick. If analysis is unavailable the finding stays recorded
un-analyzed; nothing is invented. Each run appends one ``runs`` row with its
counts and errors, even when it aborts.

Nothing here executes anything: execution happens only past human approval,
in ``bellwether serve`` or ``bellwether approve``.

The ``build_*`` functions wire real components from config; tests replace
them to run against a mocked cluster and mocked providers.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import SecretStr

from bellwether.analysis.analyst import Analyst, replica_set_name, topology_summary
from bellwether.analysis.claude import ClaudeProvider
from bellwether.analysis.openai import OpenAIProvider
from bellwether.analysis.provider import AnalysisUnavailable, Provider, ProviderChain
from bellwether.collectors.base import Collector
from bellwether.collectors.oplog_window import OplogWindowCollector
from bellwether.config import BellwetherConfig
from bellwether.detectors.base import Detector
from bellwether.detectors.oplog_window import OplogWindowDetector
from bellwether.executor.executor import Executor
from bellwether.models import Finding, Proposal, Signal
from bellwether.mongo import ReadOnlyMongo
from bellwether.notify.base import Notifier
from bellwether.notify.slack import SlackNotifier
from bellwether.notify.stdout import StdoutNotifier
from bellwether.store.sqlite import RunRecord, SqliteStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    started_at: datetime
    finished_at: datetime
    findings: list[Finding]
    proposals: list[Proposal]
    errors: list[str]


# --- wiring from config -------------------------------------------------------------


def build_mongo(config: BellwetherConfig) -> ReadOnlyMongo:
    return ReadOnlyMongo(config.mongo)


def build_collectors(config: BellwetherConfig) -> list[Collector]:
    return [OplogWindowCollector(config.collectors.oplog_window)]


def build_detectors(config: BellwetherConfig) -> list[Detector]:
    return [OplogWindowDetector(config.detectors.oplog_window)]


def build_providers(config: BellwetherConfig) -> list[Provider]:
    analysis = config.analysis
    providers: list[Provider] = []
    for name in analysis.providers:
        if name == "claude":
            providers.append(
                ClaudeProvider(
                    api_key=_secret(analysis.anthropic_api_key, "anthropic_api_key"),
                    model=_required(analysis.claude_model, "claude_model"),
                    timeout_seconds=analysis.timeout_seconds,
                )
            )
        else:
            providers.append(
                OpenAIProvider(
                    api_key=_secret(analysis.openai_api_key, "openai_api_key"),
                    model=_required(analysis.openai_model, "openai_model"),
                    timeout_seconds=analysis.timeout_seconds,
                )
            )
    return providers


def build_analyst(config: BellwetherConfig) -> Analyst:
    chain = ProviderChain(build_providers(config), max_retries=config.analysis.max_retries)
    return Analyst(
        chain,
        topology=topology_summary(config.mongo),
        replica_set=replica_set_name(config.mongo),
        document_threshold=config.executor.document_threshold,
    )


def build_store(config: BellwetherConfig) -> SqliteStore:
    return SqliteStore(config.store.sqlite_path)


def build_notifiers(config: BellwetherConfig) -> list[Notifier]:
    notifiers: list[Notifier] = []
    for channel in config.notify.channels:
        if channel == "stdout":
            notifiers.append(StdoutNotifier())
        else:
            webhook = _secret(config.notify.slack_webhook_url, "slack_webhook_url")
            notifiers.append(SlackNotifier(webhook))
    return notifiers


def build_executor(config: BellwetherConfig, store: SqliteStore) -> Executor | None:
    if not config.executor.enabled:
        return None
    # kill_op may target any configured member, including the hidden backup
    # node that is not a replica-set seed.
    known = [config.mongo.target_node, *config.mongo.fallback_nodes]
    return Executor(config.executor, store, known_nodes=known)


# --- one pass ---------------------------------------------------------------------------


def run_once(
    config: BellwetherConfig,
    *,
    mongo: ReadOnlyMongo | None = None,
    collectors: Sequence[Collector] | None = None,
    detectors: Sequence[Detector] | None = None,
    analyst: Analyst | None = None,
    store: SqliteStore | None = None,
    notifiers: Sequence[Notifier] | None = None,
) -> RunSummary:
    run_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc)
    store = store or build_store(config)
    findings: list[Finding] = []
    proposals: list[Proposal] = []
    errors: list[str] = []
    logger.info("run started", extra={"run_id": run_id})
    try:
        signals = _collect(config, mongo, collectors, errors)
        findings = _detect(config, detectors, signals, errors)
        channels = list(notifiers) if notifiers is not None else build_notifiers(config)
        gate = config.analysis.escalate_min_severity
        for finding in findings:
            escalated = finding.severity.rank >= gate.rank
            store.record_finding(finding, run_id=run_id, escalated=escalated)
            context = {"run_id": run_id, "finding_id": finding.finding_id, "severity": finding.severity.value}
            if not escalated:
                logger.info("finding below escalation threshold; stored, not analyzed", extra=context)
                continue
            if store.has_pending(finding.failure_mode, finding.node):
                logger.info("a pending proposal already covers this finding; not re-analyzed", extra=context)
                continue
            analyst = analyst or build_analyst(config)
            logger.info("analysis started", extra=context)
            try:
                proposal = analyst.analyze(finding)
            except AnalysisUnavailable as exc:
                errors.append(
                    f"analysis unavailable for {finding.failure_mode} on {finding.node}: {exc}"
                )
                logger.error("analysis unavailable; finding recorded un-analyzed", extra=context)
                continue
            record = store.record_proposal(proposal)
            proposals.append(proposal)
            for notifier in channels:
                try:
                    notifier.notify(proposal, record)
                except Exception as exc:
                    errors.append(f"notify {notifier.name}: {_describe(exc)}")
                    logger.exception("notification failed", extra={"channel": notifier.name})
    except Exception as exc:
        errors.append(f"run aborted: {_describe(exc)}")
        logger.exception("run aborted", extra={"run_id": run_id})
        raise
    finally:
        finished = datetime.now(timezone.utc)
        store.record_run(
            RunRecord(
                run_id=run_id,
                started_at=started,
                finished_at=finished,
                findings=len(findings),
                proposals=len(proposals),
                errors=tuple(errors),
            )
        )
        logger.info(
            "run finished",
            extra={
                "run_id": run_id,
                "findings": len(findings),
                "proposals": len(proposals),
                "errors": len(errors),
                "elapsed_seconds": round((finished - started).total_seconds(), 3),
            },
        )
    return RunSummary(run_id, started, finished, findings, proposals, errors)


def _collect(
    config: BellwetherConfig,
    mongo: ReadOnlyMongo | None,
    collectors: Sequence[Collector] | None,
    errors: list[str],
) -> list[Signal]:
    reader = mongo or build_mongo(config)
    signals: list[Signal] = []
    try:
        for collector in collectors if collectors is not None else build_collectors(config):
            logger.info("collector started", extra={"collector": collector.name})
            try:
                signal = collector.collect(reader)
            except Exception as exc:
                errors.append(f"collector {collector.name}: {_describe(exc)}")
                logger.exception("collector failed", extra={"collector": collector.name})
                continue
            logger.info(
                "collector finished",
                extra={
                    "collector": collector.name,
                    "node": reader.served_by,
                    "signal": signal is not None,
                },
            )
            if signal is not None:
                signals.append(signal)
    finally:
        if mongo is None:
            reader.close()
    return signals


def _detect(
    config: BellwetherConfig,
    detectors: Sequence[Detector] | None,
    signals: list[Signal],
    errors: list[str],
) -> list[Finding]:
    findings: list[Finding] = []
    for detector in detectors if detectors is not None else build_detectors(config):
        try:
            finding = detector.evaluate(signals)
        except Exception as exc:
            errors.append(f"detector {detector.failure_mode}: {_describe(exc)}")
            logger.exception("detector failed", extra={"detector": detector.failure_mode})
            continue
        logger.info(
            "detector finished",
            extra={"detector": detector.failure_mode, "finding": finding is not None},
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _secret(value: SecretStr | None, field: str) -> str:
    # load_config guarantees presence for enabled features; this guards direct use.
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"missing secret {field}")
    return value.get_secret_value()


def _required(value: str | None, field: str) -> str:
    if not value:
        raise RuntimeError(f"missing config value {field}")
    return value


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
