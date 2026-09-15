"""Acceptance tests for the CLI — BUILD_SPEC §3.10.

Spec acceptance: ``bellwether run`` against a mocked cluster with an induced
oplog-window finding produces a stored proposal and a stdout notification, and
records a run. Plus ``list``, ``show``, ``approve --by``, and ``serve``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bellwether import cli, pipeline
from bellwether.analysis.analyst import Analyst
from bellwether.analysis.provider import ProviderChain
from bellwether.executor.executor import ExecutionRefused
from bellwether.logs import JsonFormatter
from bellwether.models import (
    ActionKind,
    ApprovalRecord,
    ApprovalState,
    Proposal,
    RemediationAction,
)
from bellwether.mongo import ReadOnlyMongo
from bellwether.store.sqlite import SqliteStore
from tests.fakes import (
    FALLBACKS,
    PROPOSAL_PAYLOAD,
    PROVIDER_KEYS,
    TARGET,
    StaticProvider,
    oplog_cluster,
    write_config,
)

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(os.environ):
        if name.upper().startswith("BELLWETHER_"):
            monkeypatch.delenv(name)
    for name, value in PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):  # main() installs a JSON handler on stderr
        if isinstance(handler.formatter, JsonFormatter):
            root.removeHandler(handler)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path)


@pytest.fixture
def store(tmp_path: Path, config_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bellwether.db")


def make_proposal(kind: ActionKind = ActionKind.PROPOSE_ONLY) -> Proposal:
    if kind is ActionKind.EXECUTABLE:
        action = RemediationAction(
            kind=kind,
            title="Index account_id",
            command="db.transactions.createIndex({account_id: 1})",
            rationale="Hot filter.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "appdb",
                "collection": "transactions",
                "keys": [{"field": "account_id", "direction": 1}],
                "estimated_docs": 100,
            },
        )
    else:
        action = RemediationAction(
            kind=kind,
            title="Grow the oplog",
            command="db.adminCommand({replSetResizeOplog: 1, size: 51200})",
            rationale="Restore the window.",
            reversible=True,
        )
    return Proposal(
        finding_id="f" * 32,
        failure_mode="oplog_window_below_resync",
        node="mongo-hidden.example.internal:27017",
        diagnosis="Window 40 min < resync 60 min.",
        mechanism="Capped oplog.",
        impact_if_ignored="Initial sync after maintenance.",
        action=action,
        confidence=0.8,
        provider="claude",
        created_at=T0,
    )


class FakeExecutor:
    def __init__(self, store: SqliteStore, error: Exception | None = None) -> None:
        self.store = store
        self.error = error
        self.calls = 0

    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.store.record_approval_transition(
            proposal.proposal_id, ApprovalState.EXECUTED, result="created index account_id_1"
        )


# --- run (spec acceptance) --------------------------------------------------------------


def test_run_against_mocked_cluster(
    tmp_path: Path,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = oplog_cluster(40 * 60)
    monkeypatch.setattr(
        pipeline,
        "build_mongo",
        lambda config: ReadOnlyMongo(config.mongo, client_factory=cluster.factory()),
    )
    monkeypatch.setattr(
        pipeline,
        "build_analyst",
        lambda config: Analyst(ProviderChain([StaticProvider("claude", PROPOSAL_PAYLOAD)])),
    )

    code = cli.main(["--config", str(config_path), "run"])

    assert code == 0
    out = capsys.readouterr().out
    store = SqliteStore(tmp_path / "bellwether.db")
    [(proposal, record)] = store.list_proposals()
    assert record.state is ApprovalState.PENDING
    assert f"Bellwether proposal {proposal.proposal_id}" in out  # the stdout notification
    assert PROPOSAL_PAYLOAD["diagnosis"] in out
    [run] = store.list_runs()
    assert (run.findings, run.proposals) == (1, 1)


def test_run_exits_nonzero_when_the_run_had_errors(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = oplog_cluster(40 * 60)
    cluster.down.update({TARGET, *FALLBACKS})  # every member unreachable
    monkeypatch.setattr(
        pipeline,
        "build_mongo",
        lambda config: ReadOnlyMongo(config.mongo, client_factory=cluster.factory()),
    )

    assert cli.main(["--config", str(config_path), "run"]) == 1


# --- list / show ----------------------------------------------------------------------------


def test_list_shows_proposals_and_state(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    pending = make_proposal()
    rejected = make_proposal()
    store.record_proposal(pending)
    store.record_proposal(rejected)
    store.record_approval_transition(rejected.proposal_id, ApprovalState.REJECTED, by="x")

    assert cli.main(["--config", str(config_path), "list"]) == 0

    out = capsys.readouterr().out
    assert pending.proposal_id in out and rejected.proposal_id in out
    assert "pending" in out and "rejected" in out


def test_list_pending_only(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    pending = make_proposal()
    rejected = make_proposal()
    store.record_proposal(pending)
    store.record_proposal(rejected)
    store.record_approval_transition(rejected.proposal_id, ApprovalState.REJECTED, by="x")

    assert cli.main(["--config", str(config_path), "list", "--pending"]) == 0

    out = capsys.readouterr().out
    assert pending.proposal_id in out and rejected.proposal_id not in out


def test_show_renders_the_proposal_and_audit_trail(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    assert cli.main(["--config", str(config_path), "show", proposal.proposal_id]) == 0

    out = capsys.readouterr().out
    assert proposal.diagnosis in out
    assert proposal.action.command in out
    assert "pending" in out


def test_show_includes_refused_decision_attempts(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    store.record_audit_event(
        "unauthorized_decision",
        proposal_id=proposal.proposal_id,
        actor="slack:mallory (U0MALLORY)",
        detail="approve refused: U0MALLORY is not in approval.approver_ids",
    )

    assert cli.main(["--config", str(config_path), "show", proposal.proposal_id]) == 0

    out = capsys.readouterr().out
    assert "unauthorized_decision" in out and "U0MALLORY" in out


def test_show_unknown_proposal(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(config_path), "show", "0" * 32]) == 1
    assert "no proposal" in capsys.readouterr().err


# --- approve --------------------------------------------------------------------------------


def test_approve_propose_only(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    code = cli.main(["--config", str(config_path), "approve", proposal.proposal_id, "--by", "osego"])

    assert code == 0
    record = store.current_state(proposal.proposal_id)
    assert record.state is ApprovalState.APPROVED
    assert record.decided_by == "cli:osego"
    assert proposal.action.command in capsys.readouterr().out


def test_approve_executable_with_executor_disabled(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)

    code = cli.main(["--config", str(config_path), "approve", proposal.proposal_id, "--by", "osego"])

    assert code == 0
    assert store.current_state(proposal.proposal_id).state is ApprovalState.APPROVED
    assert "executor is disabled" in capsys.readouterr().out


def test_approve_executable_runs_the_executor(
    config_path: Path,
    store: SqliteStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeExecutor(store)
    monkeypatch.setattr(pipeline, "build_executor", lambda config, s: fake)
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)

    code = cli.main(["--config", str(config_path), "approve", proposal.proposal_id, "--by", "osego"])

    assert code == 0
    assert fake.calls == 1
    assert store.current_state(proposal.proposal_id).state is ApprovalState.EXECUTED
    assert "created index account_id_1" in capsys.readouterr().out


def test_approve_executable_reports_execution_failure(
    config_path: Path,
    store: SqliteStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeExecutor(store, error=ExecutionRefused("store records proposal as failed"))
    monkeypatch.setattr(pipeline, "build_executor", lambda config, s: fake)
    proposal = make_proposal(ActionKind.EXECUTABLE)
    store.record_proposal(proposal)

    code = cli.main(["--config", str(config_path), "approve", proposal.proposal_id, "--by", "osego"])

    assert code == 1
    assert "execution" in capsys.readouterr().err


def test_approve_twice_is_refused(
    config_path: Path, store: SqliteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)
    args = ["--config", str(config_path), "approve", proposal.proposal_id, "--by", "osego"]
    assert cli.main(args) == 0

    assert cli.main(args) == 1
    assert "not allowed" in capsys.readouterr().err


def test_approve_requires_by(config_path: Path, store: SqliteStore) -> None:
    proposal = make_proposal()
    store.record_proposal(proposal)

    with pytest.raises(SystemExit):
        cli.main(["--config", str(config_path), "approve", proposal.proposal_id])


def test_approve_unknown_proposal(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(config_path), "approve", "0" * 32, "--by", "osego"]) == 1
    assert "no proposal" in capsys.readouterr().err


# --- serve ----------------------------------------------------------------------------------


def capture_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    started: dict[str, Any] = {}

    def fake_run(app: FastAPI, **kwargs: Any) -> None:
        started["app"] = app
        started.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return started


def slack_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/x")
    return write_config(tmp_path, notify={"channels": ["stdout", "slack"]})


def test_serve_starts_without_slack_when_slack_is_not_a_channel(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No Slack secret anywhere; the fixture's notify.channels is [stdout].
    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO"]')
    started = capture_uvicorn(monkeypatch)

    assert cli.main(["--config", str(config_path), "serve"]) == 0
    app = started["app"]
    assert app.state.slack_enabled is False
    assert app.state.approval_paths == ("api",)
    assert "/slack/actions" not in {getattr(route, "path", None) for route in app.routes}


def test_serve_runs_a_read_only_api_when_no_approval_path_is_enabled(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No Slack, decision endpoint off, no allowlist: still starts.
    monkeypatch.setenv("BELLWETHER_APPROVAL__UI_APPROVAL_ENABLED", "false")
    started = capture_uvicorn(monkeypatch)

    assert cli.main(["--config", str(config_path), "serve"]) == 0
    client = TestClient(started["app"], client=("127.0.0.1", 50000))
    assert client.get("/api/proposals").json() == []
    response = client.post(
        f"/api/proposals/{'0' * 32}/decision", json={"decision": "reject", "approver_id": "U0OSEGO"}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "no_approval_path"


def test_serve_requires_an_allowlist_when_the_decision_endpoint_is_reachable(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--config", str(config_path), "serve"]) == 2
    err = capsys.readouterr().err
    assert "BELLWETHER_APPROVAL__APPROVER_IDS" in err
    assert "decision endpoint" in err


def test_serve_with_slack_requires_the_signing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO"]')
    path = slack_config(tmp_path, monkeypatch)

    assert cli.main(["--config", str(path), "serve"]) == 2
    assert "BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET" in capsys.readouterr().err


def test_serve_with_slack_requires_an_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET", "s3cret")
    path = slack_config(tmp_path, monkeypatch)

    # Public bind: the decision endpoint is off, so Slack is the only approval path.
    assert cli.main(["--config", str(path), "serve", "--host", "0.0.0.0"]) == 2
    err = capsys.readouterr().err
    assert "BELLWETHER_APPROVAL__APPROVER_IDS" in err
    assert "Slack" in err


def test_serve_with_slack_mounts_the_signed_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET", "s3cret")
    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO"]')
    started = capture_uvicorn(monkeypatch)
    path = slack_config(tmp_path, monkeypatch)

    assert cli.main(["--config", str(path), "serve"]) == 0
    assert started["app"].state.slack_enabled is True
    unsigned = TestClient(started["app"]).post(
        "/slack/actions",
        content=b"payload=%7B%7D",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert unsigned.status_code == 401


def test_serve_starts_uvicorn_on_localhost(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET", "s3cret")
    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO"]')
    started: dict[str, Any] = {}

    def fake_run(app: FastAPI, **kwargs: Any) -> None:
        started["app"] = app
        started.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert cli.main(["--config", str(config_path), "serve", "--port", "8099"]) == 0
    assert isinstance(started["app"], FastAPI)
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 8099
    assert started["app"].state.ui_approval_permitted is True  # loopback-bound
    assert started["proxy_headers"] is False  # X-Forwarded-For never rewrites the caller


def test_serve_on_a_public_interface_disables_in_ui_approval(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET", "s3cret")
    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO"]')
    started: dict[str, Any] = {}

    def fake_run(app: FastAPI, **kwargs: Any) -> None:
        started["app"] = app

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert cli.main(["--config", str(config_path), "serve", "--host", "0.0.0.0"]) == 0
    assert started["app"].state.ui_approval_permitted is False


# --- configuration errors ---------------------------------------------------------------------


def test_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(tmp_path / "missing.yaml"), "list"]) == 2
    assert "missing.yaml" in capsys.readouterr().err


def test_config_path_from_env(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BELLWETHER_CONFIG", str(config_path))

    assert cli.main(["list"]) == 0
