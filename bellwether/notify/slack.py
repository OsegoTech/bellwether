"""Slack notifier: a Block Kit message through an incoming webhook (one-way).

The Approve / Reject buttons carry the proposal_id as their value. Slack sends
the click to the app's interactivity URL — the approval endpoint, which
verifies Slack's signature before recording anything. This module never sees
a decision.

All model- and cluster-derived text is escaped for mrkdwn (``&``, ``<``, ``>``)
so a diagnosis cannot ping ``<!channel>`` or forge a link, and every field is
clipped to Block Kit's limits.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable
from typing import Any

from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.notify.base import Notifier

logger = logging.getLogger(__name__)

APPROVE_ACTION_ID = "bellwether_approve"
REJECT_ACTION_ID = "bellwether_reject"

Transport = Callable[[str, bytes], None]  # (webhook url, JSON body)


class NotificationError(Exception):
    pass


def build_payload(proposal: Proposal, record: ApprovalRecord) -> dict[str, Any]:
    action = proposal.action
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": _plain(f"Bellwether: {proposal.failure_mode} on {proposal.node}", 150),
        },
        _section(f"*{_esc(action.title)}*\n{_esc(proposal.diagnosis)}"),
        {
            "type": "section",
            "fields": [
                _mrkdwn(f"*Action*\n{action.kind.value}", 2000),
                _mrkdwn(f"*Executor op*\n{action.executor_op or '-'}", 2000),
                _mrkdwn(f"*Reversible*\n{'yes' if action.reversible else 'no'}", 2000),
                _mrkdwn(f"*Provider*\n{_esc(proposal.provider)}", 2000),
                _mrkdwn(f"*Confidence*\n{proposal.confidence:.2f}", 2000),
                _mrkdwn(f"*State*\n{record.state.value}", 2000),
            ],
        },
        _section(f"*Mechanism*\n{_esc(proposal.mechanism)}"),
        _section(f"*Impact if ignored*\n{_esc(proposal.impact_if_ignored)}"),
        _section(f"*Command*\n```{_esc(_clip(action.command, 2800))}```"),
        _section(
            "*Evidence*\n"
            + ("\n".join(f"• {_esc(e.render())}" for e in proposal.evidence_refs) or "(none)")
        ),
        {
            "type": "context",
            "elements": [
                _mrkdwn(
                    f"proposal `{proposal.proposal_id}` · finding `{proposal.finding_id}` · "
                    f"created {proposal.created_at.isoformat()}",
                    3000,
                )
            ],
        },
    ]
    if record.state is ApprovalState.PENDING:
        approve: dict[str, Any] = {
            "type": "button",
            "action_id": APPROVE_ACTION_ID,
            "text": _plain("Approve", 75),
            "style": "primary",
            "value": proposal.proposal_id,
        }
        if action.kind is ActionKind.EXECUTABLE:
            approve["confirm"] = {
                "title": _plain("Run on the cluster?", 100),
                "text": _plain(
                    f"Approving runs {action.executor_op} against the cluster with the "
                    "meetadev-ai-exec identity.",
                    300,
                ),
                "confirm": _plain("Approve and run", 30),
                "deny": _plain("Cancel", 30),
            }
        blocks.append(
            {
                "type": "actions",
                "block_id": "bellwether_decision",
                "elements": [
                    approve,
                    {
                        "type": "button",
                        "action_id": REJECT_ACTION_ID,
                        "text": _plain("Reject", 75),
                        "style": "danger",
                        "value": proposal.proposal_id,
                    },
                ],
            }
        )
    return {
        "text": _clip(
            f"Bellwether proposal: {_esc(action.title)} "
            f"({_esc(proposal.failure_mode)} on {_esc(proposal.node)})",
            3000,
        ),
        "blocks": blocks,
    }


class SlackNotifier(Notifier):
    name = "slack"

    def __init__(
        self,
        webhook_url: str,
        *,
        transport: Transport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not webhook_url.startswith("https://"):
            raise ValueError("Slack webhook URL must be https")
        self._url = webhook_url
        self._transport = transport or (lambda url, body: _post_json(url, body, timeout_seconds))

    def notify(self, proposal: Proposal, approval_record: ApprovalRecord) -> None:
        body = json.dumps(build_payload(proposal, approval_record)).encode()
        self._transport(self._url, body)  # the URL is a credential: never logged
        logger.info(
            "notification sent", extra={"channel": self.name, "proposal_id": proposal.proposal_id}
        )


def _post_json(url: str, body: bytes, timeout: float) -> None:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # https enforced above
        if response.status != 200:
            raise NotificationError(f"Slack webhook answered HTTP {response.status}")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain(text: str, limit: int) -> dict[str, Any]:
    return {"type": "plain_text", "text": _clip(text, limit)}


def _mrkdwn(text: str, limit: int) -> dict[str, Any]:
    return {"type": "mrkdwn", "text": _clip(text, limit)}


def _section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": _mrkdwn(text, 3000)}
