"""Acceptance tests for notifiers — BUILD_SPEC §3.7.

Spec acceptance:
  - the stdout notifier renders all proposal fields
  - the slack notifier builds valid Block Kit JSON (validated structurally)
    without network in tests
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from bellwether.models import (
    ActionKind,
    ApprovalRecord,
    ApprovalState,
    Evidence,
    Proposal,
    RemediationAction,
)
from bellwether.notify.base import Notifier
from bellwether.notify.slack import (
    APPROVE_ACTION_ID,
    REJECT_ACTION_ID,
    SlackNotifier,
    build_payload,
)
from bellwether.notify.stdout import StdoutNotifier, render_text

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_proposal(**overrides: Any) -> Proposal:
    values: dict[str, Any] = {
        "finding_id": "a" * 32,
        "failure_mode": "oplog_window_below_resync",
        "node": "mongo-hidden.example.internal:27017",
        "diagnosis": "The oplog holds 40 min against a 60 min resync estimate.",
        "mechanism": "local.oplog.rs is capped; a burst of writes truncated history.",
        "impact_if_ignored": "A secondary down for maintenance needs a full initial sync.",
        "action": RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Index transactions.account_id",
            command="db.transactions.createIndex({ account_id: 1 })",
            rationale="Slow scans filter on account_id.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={
                "db": "appdb",
                "collection": "transactions",
                "keys": [{"field": "account_id", "direction": 1}],
                "estimated_docs": 40000,
            },
        ),
        "confidence": 0.82,
        "provider": "openai",
        "evidence_refs": (
            Evidence("oplog_window_seconds", 2400, "s", observed_at=T0),
            Evidence("resync_seconds", 3600, "s", observed_at=T0),
        ),
        "created_at": T0,
    }
    values.update(overrides)
    return Proposal(**values)


def pending(proposal: Proposal) -> ApprovalRecord:
    return ApprovalRecord(proposal_id=proposal.proposal_id, created_at=T0)


PROPOSE_ONLY = RemediationAction(
    kind=ActionKind.PROPOSE_ONLY,
    title="Grow the oplog",
    command="db.adminCommand({ replSetResizeOplog: 1, size: 51200 })",
    rationale="Restore a window above the resync estimate.",
    reversible=True,
)


# --- stdout ----------------------------------------------------------------------


def test_stdout_renders_all_proposal_fields() -> None:
    proposal = make_proposal()
    record = pending(proposal)

    text = render_text(proposal, record)

    expected = [
        proposal.proposal_id,
        proposal.finding_id,
        proposal.failure_mode,
        proposal.node,
        proposal.diagnosis,
        proposal.mechanism,
        proposal.impact_if_ignored,
        proposal.action.kind.value,
        proposal.action.title,
        proposal.action.command,
        proposal.action.rationale,
        "reversible: yes",
        "create_small_index",
        json.dumps(proposal.action.executor_args, sort_keys=True),
        "0.82",
        proposal.provider,
        proposal.created_at.isoformat(),
        record.state.value,
    ]
    for value in expected:
        assert value in text, value
    for evidence in proposal.evidence_refs:
        assert evidence.render() in text


def test_stdout_renders_decision_and_outcome() -> None:
    proposal = make_proposal()
    record = ApprovalRecord(
        proposal_id=proposal.proposal_id,
        state=ApprovalState.EXECUTED,
        decided_by="cli:osego",
        decided_at=T0,
        executed_at=T0,
        execution_result="index account_id_1 built",
    )

    text = render_text(proposal, record)

    assert "executed" in text
    assert "cli:osego" in text
    assert "index account_id_1 built" in text


def test_stdout_notifier_writes_to_its_stream() -> None:
    stream = io.StringIO()
    proposal = make_proposal()

    StdoutNotifier(stream).notify(proposal, pending(proposal))

    assert proposal.proposal_id in stream.getvalue()


def test_stdout_notifier_defaults_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    proposal = make_proposal()

    StdoutNotifier().notify(proposal, pending(proposal))

    assert proposal.diagnosis in capsys.readouterr().out


# --- Block Kit structural validator ---------------------------------------------

TEXT_TYPES = {"plain_text", "mrkdwn"}


def check_text(obj: dict[str, Any], limit: int, *, plain_only: bool = False) -> None:
    assert obj["type"] in ({"plain_text"} if plain_only else TEXT_TYPES)
    assert isinstance(obj["text"], str) and 0 < len(obj["text"]) <= limit


def check_button(button: dict[str, Any]) -> None:
    assert button["type"] == "button"
    check_text(button["text"], 75, plain_only=True)
    assert 0 < len(button["action_id"]) <= 255
    assert 0 < len(button["value"]) <= 2000
    assert button.get("style") in (None, "primary", "danger")
    if "confirm" in button:
        confirm = button["confirm"]
        check_text(confirm["title"], 100, plain_only=True)
        check_text(confirm["text"], 300)
        check_text(confirm["confirm"], 30, plain_only=True)
        check_text(confirm["deny"], 30, plain_only=True)


def validate_block_kit(payload: dict[str, Any]) -> None:
    json.dumps(payload)  # serialisable
    assert isinstance(payload["text"], str) and payload["text"]
    blocks = payload["blocks"]
    assert 0 < len(blocks) <= 50
    block_ids = [b["block_id"] for b in blocks if "block_id" in b]
    assert len(block_ids) == len(set(block_ids))
    for block in blocks:
        kind = block["type"]
        assert kind in {"header", "section", "context", "actions", "divider"}
        if "block_id" in block:
            assert 0 < len(block["block_id"]) <= 255
        if kind == "header":
            check_text(block["text"], 150, plain_only=True)
        elif kind == "section":
            assert "text" in block or "fields" in block
            if "text" in block:
                check_text(block["text"], 3000)
            for field in block.get("fields", []):
                check_text(field, 2000)
            assert len(block.get("fields", [])) <= 10
        elif kind == "context":
            assert 0 < len(block["elements"]) <= 10
            for element in block["elements"]:
                check_text(element, 3000)
        elif kind == "actions":
            assert 0 < len(block["elements"]) <= 25
            action_ids = [e["action_id"] for e in block["elements"]]
            assert len(action_ids) == len(set(action_ids))
            for element in block["elements"]:
                check_button(element)


def buttons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for b in payload["blocks"] if b["type"] == "actions" for e in b["elements"]]


# --- slack -------------------------------------------------------------------------


def test_slack_payload_is_valid_block_kit() -> None:
    proposal = make_proposal()

    validate_block_kit(build_payload(proposal, pending(proposal)))


def test_slack_buttons_reference_the_proposal() -> None:
    proposal = make_proposal()

    found = {b["action_id"]: b for b in buttons(build_payload(proposal, pending(proposal)))}

    assert set(found) == {APPROVE_ACTION_ID, REJECT_ACTION_ID}
    assert found[APPROVE_ACTION_ID]["value"] == proposal.proposal_id
    assert found[REJECT_ACTION_ID]["value"] == proposal.proposal_id
    assert found[APPROVE_ACTION_ID]["style"] == "primary"
    assert found[REJECT_ACTION_ID]["style"] == "danger"


def test_executable_approve_asks_for_confirmation() -> None:
    proposal = make_proposal()

    approve = {b["action_id"]: b for b in buttons(build_payload(proposal, pending(proposal)))}[
        APPROVE_ACTION_ID
    ]

    assert "confirm" in approve
    assert "create_small_index" in approve["confirm"]["text"]["text"]


def test_propose_only_approve_has_no_executor_confirmation() -> None:
    proposal = make_proposal(action=PROPOSE_ONLY)
    payload = build_payload(proposal, pending(proposal))

    validate_block_kit(payload)
    approve = {b["action_id"]: b for b in buttons(payload)}[APPROVE_ACTION_ID]
    assert "confirm" not in approve


def test_decided_proposal_has_no_buttons() -> None:
    proposal = make_proposal()
    record = replace(pending(proposal), state=ApprovalState.APPROVED, decided_by="x")

    payload = build_payload(proposal, record)

    validate_block_kit(payload)
    assert buttons(payload) == []


def test_slack_payload_carries_the_substance() -> None:
    proposal = make_proposal()

    rendered = json.dumps(build_payload(proposal, pending(proposal)))

    for value in (
        proposal.failure_mode,
        proposal.node,
        proposal.action.title,
        proposal.action.command,
        proposal.provider,
        proposal.proposal_id,
        "oplog_window_seconds: 2400 s",
    ):
        assert value in rendered, value


def test_long_text_is_clipped_to_block_kit_limits() -> None:
    proposal = make_proposal(
        diagnosis="d" * 10_000,
        mechanism="m" * 10_000,
        impact_if_ignored="i" * 10_000,
        failure_mode="f" * 400,
        evidence_refs=tuple(Evidence(f"e{i}", "v" * 300, observed_at=T0) for i in range(50)),
        action=replace(PROPOSE_ONLY, title="t" * 5000, command="c" * 10_000),
    )

    validate_block_kit(build_payload(proposal, pending(proposal)))


def test_mrkdwn_control_characters_are_escaped() -> None:
    proposal = make_proposal(diagnosis="a < b && c > d <!channel>")

    rendered = json.dumps(build_payload(proposal, pending(proposal)))

    assert "<!channel>" not in rendered
    assert "&lt;!channel&gt;" in rendered


def test_slack_notifier_posts_json_to_webhook_without_network() -> None:
    sent: list[tuple[str, bytes]] = []
    url = "https://hooks.slack.com/services/T000/B000/XXXX"
    notifier = SlackNotifier(url, transport=lambda u, body: sent.append((u, body)))
    proposal = make_proposal()

    notifier.notify(proposal, pending(proposal))

    assert len(sent) == 1
    assert sent[0][0] == url
    validate_block_kit(json.loads(sent[0][1]))


def test_slack_webhook_must_be_https() -> None:
    with pytest.raises(ValueError):
        SlackNotifier("http://hooks.slack.com/services/T000/B000/XXXX")


def test_transport_errors_propagate() -> None:
    def broken(url: str, body: bytes) -> None:
        raise OSError("connection refused")

    notifier = SlackNotifier("https://hooks.slack.com/services/x", transport=broken)
    proposal = make_proposal()

    with pytest.raises(OSError):
        notifier.notify(proposal, pending(proposal))


def test_notifier_is_abstract() -> None:
    with pytest.raises(TypeError):
        Notifier()  # type: ignore[abstract]
    assert StdoutNotifier.name == "stdout"
    assert SlackNotifier.name == "slack"
