"""Server-rendered HTML for the operations UI (``bellwether serve``).

Three views — the overview, a proposal's detail, and the same detail once
decided — in one dark, self-contained page frame: no JavaScript framework, no
build step, one embedded stylesheet. The only script is an inline
``confirm()`` on the Approve button.

Everything shown comes from the store's real records: proposals and their
approval history, findings (for severity and the one-line summary), runs (for
freshness), and audit events. Every value that came from the cluster or a model
is HTML-escaped. Technical values — nodes, commands, evidence, identifiers —
are set in monospace; colour carries meaning only (severity and state).
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Evidence, Proposal
from bellwether.store.sqlite import AuditEvent, Transition

TYPE_LABELS: dict[str, str] = {
    "missing_index_collscan": "Missing index",
    "oplog_window_below_resync": "Oplog window shrinking",
    "profiler_disabled": "Profiling disabled",
    "redundant_index": "Redundant index",
}

STEP_LABELS: dict[ApprovalState, str] = {
    ApprovalState.PENDING: "Proposed",
    ApprovalState.APPROVED: "Approved",
    ApprovalState.REJECTED: "Rejected",
    ApprovalState.EXPIRED: "Expired",
    ApprovalState.EXECUTED: "Executed",
    ApprovalState.FAILED: "Execution failed",
}

GROUNDING_TEXT = (
    "The index fields and order were computed by deterministic code, not chosen by the AI. "
    "Bellwether verifies the AI's proposal matches the computed index exactly before it can "
    "run; a mismatch is rejected."
)


@dataclass(frozen=True)
class OverviewRow:
    href: str | None  # proposals link to their detail page; noted findings have none
    severity: str | None  # critical | warning | info
    failure_mode: str
    node: str
    summary: str
    state: str  # pending | approved | rejected | executed | failed | expired | noted
    at: datetime


# --- small helpers -------------------------------------------------------------------------


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def type_label(failure_mode: str) -> str:
    return TYPE_LABELS.get(failure_mode, failure_mode.replace("_", " ").capitalize())


def model_label(model: str) -> str:
    """A friendly model name: "claude-opus-5" -> "Claude Opus 5", "gpt-5.6-sol" -> "GPT-5.6 Sol"."""
    parts = [p for p in model.split("-") if p]
    if len(parts) >= 2 and parts[0].lower() == "gpt":
        return " ".join([f"GPT-{parts[1]}", *(p.capitalize() for p in parts[2:])])
    return " ".join(p.capitalize() for p in parts) or model


def relative(at: datetime, now: datetime) -> str:
    moment = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    seconds = int((now - moment).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86_400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86_400} d ago"


def stamp(at: datetime) -> str:
    moment = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def human_bytes(count: float) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{int(value):,}"


def _index_keys(value: Any) -> str | None:
    """``{field: direction, ...}`` for a list of index keys, else None."""
    if (
        isinstance(value, list)
        and value
        and all(isinstance(k, Mapping) and set(k) == {"field", "direction"} for k in value)
    ):
        return "{" + ", ".join(f"{k['field']}: {k['direction']}" for k in value) + "}"
    return None


def evidence_value(evidence: Evidence) -> str:
    """One evidence value as display text (not yet escaped)."""
    value, unit = evidence.value, evidence.unit
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        text = _number(value)
        if unit:
            text += f" {unit}"
        if unit == "bytes" and value >= 1024:
            text += f" ({human_bytes(value)})"
        return text
    if value is None:
        return "—"
    keys = _index_keys(value)
    if keys is not None:
        return keys
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(", ", ": "))
    return f"{value} {unit}" if unit else str(value)


# --- the frame ------------------------------------------------------------------------------


def frame(title: str, body: str, *, cluster_label: str, updated_note: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)} · Bellwether</title><style>{STYLESHEET}</style></head><body>"
        '<header class="topbar">'
        '<a class="wordmark" href="/">Bellwether</a>'
        f'<span class="cluster mono">{e(cluster_label)}</span>'
        f'<span class="updated">{e(updated_note)}</span>'
        f"</header><main>{body}</main></body></html>"
    )


# --- View 1: the overview ------------------------------------------------------------------


def overview_page(
    rows: Sequence[OverviewRow], *, now: datetime, cluster_label: str, updated_note: str
) -> str:
    pending = [r for r in rows if r.state == "pending"]
    rest = [r for r in rows if r.state != "pending"]
    ordered = sorted(pending, key=lambda r: r.at, reverse=True) + sorted(
        rest, key=lambda r: r.at, reverse=True
    )
    noted = sum(1 for r in rows if r.state == "noted")
    if pending:
        verb = "needs" if len(pending) == 1 else "need"
        status = f"{len(pending)} {verb} review · {noted} noted"
        status_class = "status attention"
    elif noted:
        plural = "signal" if noted == 1 else "signals"
        status = f"All clear — {noted} {plural} noted, none need action"
        status_class = "status"
    elif rows:
        status = "All clear — nothing needs action"
        status_class = "status"
    else:
        status = "All clear — no signals noted yet"
        status_class = "status"

    body_rows = "".join(_overview_row(r, now) for r in ordered)
    if not body_rows:
        body_rows = '<tr><td colspan="6" class="empty">Nothing recorded yet. Signals appear here after the first run.</td></tr>'
    body = (
        f'<p class="{status_class}"><span class="pulse"></span>{e(status)}</p>'
        '<div class="table-wrap"><table class="signals"><thead><tr>'
        "<th>Severity</th><th>Type</th><th>Node</th><th>Summary</th><th>State</th><th>Age</th>"
        f"</tr></thead><tbody>{body_rows}</tbody></table></div>"
    )
    return frame("Overview", body, cluster_label=cluster_label, updated_note=updated_note)


def _overview_row(row: OverviewRow, now: datetime) -> str:
    classes = ["row"]
    if row.state == "pending":
        classes.append("row-pending")
    elif row.state == "noted":
        classes.append("row-muted")
    summary = e(row.summary)
    if row.href:
        classes.append("row-link")
        summary = f'<a href="{e(row.href)}">{summary}</a>'
    return (
        f'<tr class="{" ".join(classes)}">'
        f"<td>{_severity(row.severity)}</td>"
        f"<td>{e(type_label(row.failure_mode))}</td>"
        f'<td class="mono node">{e(row.node)}</td>'
        f'<td class="summary-cell" title="{e(row.summary)}">{summary}</td>'
        f'<td><span class="badge state-{e(row.state)}">{e(row.state)}</span></td>'
        f'<td class="mono muted age">{e(relative(row.at, now))}</td>'
        "</tr>"
    )


def _severity(severity: str | None) -> str:
    if not severity:
        return '<span class="sev sev-unknown">—</span>'
    return f'<span class="sev sev-{e(severity)}">{e(severity)}</span>'


# --- Views 2 and 3: proposal detail ---------------------------------------------------------


def detail_page(
    proposal: Proposal,
    record: ApprovalRecord,
    *,
    severity: str | None,
    summary: str | None,
    history: Sequence[Transition],
    events: Sequence[AuditEvent],
    provider_label: str,
    ui_approval: str,  # "permitted" | "not_loopback" | "disabled"
    now: datetime,
    cluster_label: str,
    updated_note: str,
) -> str:
    action = proposal.action
    confidence = f"{round(proposal.confidence * 100)}%"
    header = (
        '<a class="back" href="/">&larr; Overview</a>'
        '<header class="detail-head">'
        f'<div class="title-row">{_severity(severity)}'
        f"<h1>{e(type_label(proposal.failure_mode))}</h1>"
        f'<span class="badge state-{e(record.state.value)}">{e(record.state.value)}</span></div>'
        '<div class="meta">'
        f'<span class="mono">{e(proposal.node)}</span><span class="dot">·</span>'
        f"<span>{e(provider_label)}</span><span class=\"dot\">·</span>"
        f"<span>{e(confidence)} confidence</span><span class=\"dot\">·</span>"
        f"<span>proposed {e(relative(proposal.created_at, now))}</span><span class=\"dot\">·</span>"
        f'<span class="mono faint">{e(proposal.proposal_id)}</span></div>'
        + (f'<p class="finding-summary">{e(summary)}</p>' if summary else "")
        + "</header>"
    )
    prose = (
        _section("Diagnosis", f'<p class="prose">{e(proposal.diagnosis)}</p>')
        + _section(
            "Mechanism", f'<p class="prose">{e(proposal.mechanism)}</p>', extra_class="mechanism"
        )
        + _section("Impact if ignored", f'<p class="prose">{e(proposal.impact_if_ignored)}</p>')
    )
    evidence = _section("Evidence", _evidence_block(proposal.evidence_refs), extra_class="evidence-section")
    action_card = _section("Proposed action", _action_card(proposal))
    grounding = _grounding(proposal)
    if record.state is ApprovalState.PENDING:
        decision = _decision_block(proposal, ui_approval)
    else:
        decision = _section("Audit trail", _timeline(history, provider_label))
    refused = _refused_attempts(events)
    body = header + prose + evidence + action_card + grounding + decision + refused
    return frame(
        f"{type_label(proposal.failure_mode)} on {proposal.node}",
        f'<article class="detail">{body}</article>',
        cluster_label=cluster_label,
        updated_note=updated_note,
    )


def _section(title: str, content: str, *, extra_class: str = "") -> str:
    cls = f"section {extra_class}".strip()
    return f'<section class="{cls}"><h2>{e(title)}</h2>{content}</section>'


def _evidence_block(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return '<p class="muted">No evidence was attached to this proposal.</p>'
    items = "".join(
        f'<div class="ev-row"><dt>{e(item.name.replace("_", " "))}</dt>'
        f'<dd class="mono">{e(evidence_value(item))}</dd></div>'
        for item in evidence
    )
    return f'<dl class="evidence">{items}</dl>'


def _action_card(proposal: Proposal) -> str:
    action = proposal.action
    if action.kind is ActionKind.EXECUTABLE:
        kind = '<span class="badge kind-executable">Executable</span>'
        hint = "Bellwether can run this after approval, through the executor's whitelist."
    else:
        kind = '<span class="badge kind-manual">Manual</span>'
        hint = "A human runs this command; approving records the decision."
    reversible = '<span class="tag">Reversible</span>' if action.reversible else ""
    op = (
        f'<div class="op mono">executor op: {e(action.executor_op)}</div>'
        if action.executor_op
        else ""
    )
    return (
        '<div class="card action-card">'
        f'<div class="action-head"><h3>{e(action.title)}</h3><div class="chips">{kind}{reversible}</div></div>'
        f'<pre class="command mono" title="Click to select the whole command">{e(action.command)}</pre>'
        '<div class="hint">Click the command to select all of it, then copy.</div>'
        f"{op}"
        f'<p class="prose rationale">{e(action.rationale)}</p>'
        f'<p class="hint">{e(hint)}</p>'
        "</div>"
    )


SHIELD = (
    '<svg class="shield" viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">'
    '<path d="M8 1.2 2.5 3.2v4.1c0 3.4 2.3 6.2 5.5 7.5 3.2-1.3 5.5-4.1 5.5-7.5V3.2L8 1.2Z" '
    'fill="none" stroke="currentColor" stroke-width="1.3"/>'
    '<path d="m5.4 8.1 1.8 1.8 3.5-3.6" fill="none" stroke="currentColor" stroke-width="1.4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _grounding(proposal: Proposal) -> str:
    action = proposal.action
    if action.kind is not ActionKind.EXECUTABLE or action.executor_op != "create_small_index":
        return ""
    computed = next(
        (_index_keys(item.value) for item in proposal.evidence_refs if item.name == "candidate_index"),
        None,
    )
    proposed = _index_keys(action.executor_args.get("keys"))
    rows = ""
    if computed or proposed:
        match = computed is not None and computed == proposed
        rows = (
            '<dl class="grounding-keys">'
            f'<div><dt>computed index</dt><dd class="mono">{e(computed or "—")}</dd></div>'
            f'<div><dt>proposed index</dt><dd class="mono">{e(proposed or "—")}</dd></div>'
            f'<div><dt>check</dt><dd class="mono {"ok" if match else "bad"}">'
            f'{"exact match" if match else "mismatch — would be rejected"}</dd></div></dl>'
        )
    return (
        '<aside class="grounded">'
        f'<div class="grounded-head">{SHIELD}<strong>Deterministically grounded</strong></div>'
        f"<p>{e(GROUNDING_TEXT)}</p>{rows}</aside>"
    )


def _decision_block(proposal: Proposal, ui_approval: str) -> str:
    if ui_approval != "permitted":
        why = (
            "In-UI approval is disabled on this server (approval.ui_approval_enabled is false)."
            if ui_approval == "disabled"
            else "In-UI approval only works on a server bound to a loopback address; this one is not."
        )
        return _section(
            "Decision",
            f'<div class="card notice"><p>{e(why)}</p><p class="muted">Approve or reject from '
            "Slack, or reach a loopback-bound <span class=\"mono\">bellwether serve</span> over an "
            "SSH tunnel.</p></div>",
        )
    action = proposal.action
    if action.kind is ActionKind.EXECUTABLE:
        ack = (
            "I have reviewed the evidence and understand that approving lets Bellwether run "
            f"{action.executor_op} against the production cluster."
        )
        confirm = f"Approve and run {action.executor_op} against the production cluster?"
    else:
        ack = "I have reviewed the evidence. Approving records the decision; a human runs the command."
        confirm = "Approve this proposal? The decision is recorded in the audit trail."
    return _section(
        "Decision",
        f'<form class="card decision" method="post" action="/proposals/{e(proposal.proposal_id)}/decision">'
        '<label class="field-label" for="approver_id">Approver ID</label>'
        '<input id="approver_id" name="approver_id" class="mono" required autocomplete="off" '
        'spellcheck="false" placeholder="Slack user ID, e.g. U0123ABCD">'
        '<p class="hint">Must be on approval.approver_ids. Recorded in the audit trail as '
        '<span class="mono">ui:&lt;id&gt;</span>.</p>'
        f'<label class="ack"><input type="checkbox" name="acknowledge" value="yes" required> {e(ack)}</label>'
        '<div class="buttons">'
        '<button type="submit" name="decision" value="approve" class="btn btn-approve" '
        f'onclick="return confirm({e(json.dumps(confirm))})">Approve</button>'
        '<button type="submit" name="decision" value="reject" class="btn btn-reject" '
        "formnovalidate>Reject</button>"
        "</div></form>",
    )


def _timeline(history: Sequence[Transition], provider_label: str) -> str:
    steps = []
    for transition in history:
        label = STEP_LABELS.get(transition.state, transition.state.value)
        if transition.state is ApprovalState.PENDING:
            who = f"by Bellwether ({provider_label})"
        elif transition.actor:
            who = f"by {transition.actor}"
        else:
            who = "by the executor"
        result = (
            f'<div class="step-result mono">{e(transition.result)}</div>' if transition.result else ""
        )
        steps.append(
            f'<li class="step step-{e(transition.state.value)}"><span class="step-dot"></span>'
            f'<div class="step-body"><div class="step-title">{e(label)} '
            f'<span class="step-who">{e(who)}</span></div>'
            f'<div class="step-time mono">{e(stamp(transition.at))}</div>{result}</div></li>'
        )
    return f'<ol class="timeline">{"".join(steps)}</ol>'


def _refused_attempts(events: Sequence[AuditEvent]) -> str:
    if not events:
        return ""
    items = "".join(
        f'<li><span class="mono">{e(stamp(event.at))}</span>'
        f'<span class="mono kind">{e(event.kind)}</span>'
        f'<span class="mono">{e(event.actor or "")}</span>'
        f"<span>{e(event.detail)}</span></li>"
        for event in events
    )
    return _section("Refused decision attempts", f'<ul class="audit-events">{items}</ul>', extra_class="refused")


# --- errors ------------------------------------------------------------------------------------


def error_page(
    title: str, message: str, *, status: int, back_href: str, cluster_label: str, updated_note: str
) -> str:
    body = (
        f'<div class="card error"><div class="error-code mono">{status}</div>'
        f"<h1>{e(title)}</h1><p>{e(message)}</p>"
        f'<a class="back" href="{e(back_href)}">&larr; Back</a></div>'
    )
    return frame(title, body, cluster_label=cluster_label, updated_note=updated_note)


# --- the stylesheet ----------------------------------------------------------------------------

STYLESHEET = """
:root{--bg:#0d1117;--panel:#161b22;--panel-2:#1c2128;--border:#30363d;--border-soft:#21262d;
--text:#e6edf3;--muted:#8b949e;--faint:#6e7681;--crit:#f85149;--warn:#d29922;--info:#768390;
--ok:#3fb950;--focus:#58a6ff;
--mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono);font-size:12.5px;letter-spacing:0}
.muted{color:var(--muted)}.faint{color:var(--faint)}
.topbar{display:flex;align-items:center;gap:14px;height:48px;padding:0 24px;background:var(--panel);border-bottom:1px solid var(--border)}
.wordmark{font-weight:600;font-size:15px;letter-spacing:.01em}
.cluster{color:var(--muted);padding:1px 9px;border:1px solid var(--border);border-radius:999px}
.updated{margin-left:auto;color:var(--faint);font-size:12px}
main{max-width:1180px;margin:0 auto;padding:28px 24px 72px}
.status{display:flex;align-items:center;gap:10px;margin:4px 0 20px;font-size:15px;color:var(--text)}
.pulse{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(63,185,80,.15)}
.status.attention .pulse{background:var(--warn);box-shadow:0 0 0 3px rgba(210,153,34,.18)}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:var(--panel)}
table.signals{width:100%;border-collapse:collapse}
.signals th{text-align:left;font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);padding:10px 14px;border-bottom:1px solid var(--border);white-space:nowrap}
.signals td{padding:11px 14px;border-bottom:1px solid var(--border-soft);vertical-align:middle}
.signals tr:last-child td{border-bottom:0}
.row-link:hover td{background:var(--panel-2)}
.row-link a{display:block}
.row-pending td{background:rgba(210,153,34,.05)}
.row-pending td:first-child{box-shadow:inset 3px 0 0 var(--warn)}
.row-muted td{color:var(--muted)}
.summary-cell{max-width:480px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.node,.age{white-space:nowrap}
.empty{color:var(--muted);text-align:center;padding:28px}
.sev{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);text-transform:lowercase;white-space:nowrap}
.sev::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--info)}
.sev-critical{color:var(--crit)}.sev-critical::before{background:var(--crit)}
.sev-warning{color:var(--warn)}.sev-warning::before{background:var(--warn)}
.sev-unknown::before{background:transparent;border:1px solid var(--faint)}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;line-height:1.6;border:1px solid var(--border);color:var(--muted);white-space:nowrap}
.state-pending{color:var(--warn);border-color:rgba(210,153,34,.45);background:rgba(210,153,34,.1)}
.state-approved,.state-executed{color:var(--ok);border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.1)}
.state-rejected,.state-expired{color:var(--muted);background:var(--panel-2)}
.state-failed{color:var(--crit);border-color:rgba(248,81,73,.4);background:rgba(248,81,73,.08)}
.state-noted{color:var(--faint);border-style:dashed}
.detail{max-width:900px}
.back{display:inline-block;color:var(--muted);font-size:13px;margin-bottom:18px}.back:hover{color:var(--text)}
.detail-head{padding-bottom:20px;border-bottom:1px solid var(--border);margin-bottom:8px}
.title-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.title-row h1{font-size:22px;font-weight:600;margin:0}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:8px;color:var(--muted);font-size:13px}
.dot{color:var(--faint)}
.finding-summary{margin:14px 0 0;color:var(--muted);max-width:72ch}
.section{padding:22px 0;border-bottom:1px solid var(--border-soft)}
.section h2{font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin:0 0 10px}
.prose{margin:0;max-width:74ch;font-size:15px;line-height:1.65}
.mechanism .prose{padding-left:14px;border-left:2px solid var(--border);color:#c9d1d9}
.evidence{margin:0;border:1px solid var(--border);border-radius:8px;background:var(--panel);overflow:hidden}
.ev-row{display:grid;grid-template-columns:minmax(160px,30%) 1fr;gap:16px;padding:8px 14px;border-bottom:1px solid var(--border-soft)}
.ev-row:last-child{border-bottom:0}
.ev-row:nth-child(even){background:rgba(255,255,255,.015)}
.ev-row dt{color:var(--muted);font-size:13px}
.ev-row dd{margin:0;word-break:break-word;color:var(--text)}
.card{border:1px solid var(--border);border-radius:8px;background:var(--panel);padding:18px}
.action-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
.action-head h3{margin:0;font-size:15px;font-weight:600}
.chips{display:flex;gap:6px;flex-shrink:0}
.kind-executable{color:var(--warn);border-color:rgba(210,153,34,.5)}
.kind-manual{color:var(--muted)}
.tag{display:inline-block;padding:1px 9px;border-radius:4px;font-size:12px;color:var(--ok);border:1px solid rgba(63,185,80,.35)}
.command{margin:14px 0 6px;padding:14px 16px;background:var(--bg);border:1px solid var(--border);border-radius:6px;white-space:pre-wrap;word-break:break-word;user-select:all;-webkit-user-select:all;cursor:text;color:#d2d8de}
.op{color:var(--muted);margin:6px 0}
.rationale{margin-top:12px;font-size:14px;color:#c9d1d9}
.hint{color:var(--faint);font-size:12px;margin:4px 0 0}
.grounded{margin:22px 0 0;padding:16px 18px;border:1px solid rgba(63,185,80,.35);border-radius:8px;background:rgba(63,185,80,.05)}
.grounded-head{display:flex;align-items:center;gap:8px;color:var(--ok);margin-bottom:6px}
.grounded p{margin:0;color:#c9d1d9;max-width:74ch}
.grounding-keys{display:grid;gap:4px;margin:12px 0 0}
.grounding-keys div{display:grid;grid-template-columns:150px 1fr;gap:12px}
.grounding-keys dt{color:var(--muted);font-size:13px}.grounding-keys dd{margin:0}
.grounding-keys .ok{color:var(--ok)}.grounding-keys .bad{color:var(--crit)}
.decision{display:grid;gap:8px;max-width:620px}
.field-label{font-size:13px;color:var(--muted)}
.decision input[type=text],.decision input:not([type]){background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:9px 11px;font-size:13px}
.decision input:focus{outline:2px solid var(--focus);outline-offset:1px;border-color:transparent}
.ack{display:flex;gap:10px;align-items:flex-start;margin-top:8px;font-size:13px;color:#c9d1d9}
.ack input{margin-top:3px}
.buttons{display:flex;gap:10px;margin-top:10px}
.btn{font:inherit;font-size:13px;font-weight:600;border-radius:6px;padding:8px 18px;cursor:pointer;border:1px solid var(--border)}
.btn-approve{background:#238636;border-color:rgba(240,246,252,.1);color:#fff}.btn-approve:hover{background:#2ea043}
.btn-reject{background:var(--panel-2);color:var(--text)}.btn-reject:hover{border-color:var(--muted)}
.notice p{margin:0 0 6px}
.timeline{list-style:none;margin:0;padding:0}
.step{position:relative;display:grid;grid-template-columns:18px 1fr;gap:12px;padding:0 0 18px}
.step:not(:last-child)::after{content:"";position:absolute;left:5px;top:16px;bottom:0;width:1px;background:var(--border)}
.step-dot{width:11px;height:11px;border-radius:50%;margin-top:5px;border:2px solid var(--muted);background:var(--bg)}
.step-approved .step-dot,.step-executed .step-dot{border-color:var(--ok);background:var(--ok)}
.step-rejected .step-dot,.step-expired .step-dot{border-color:var(--muted);background:var(--muted)}
.step-failed .step-dot{border-color:var(--crit);background:var(--crit)}
.step-title{font-weight:600}
.step-who{font-weight:400;color:var(--muted);font-family:var(--mono);font-size:12.5px}
.step-time{color:var(--faint);margin-top:2px}
.step-result{margin-top:6px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--panel);display:inline-block}
.audit-events{list-style:none;margin:0;padding:0;display:grid;gap:6px}
.audit-events li{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:13px}
.audit-events .kind{color:var(--crit)}
.error{max-width:620px;margin:40px auto}
.error-code{color:var(--faint)}
.error h1{font-size:20px;margin:6px 0 8px}
@media (max-width:720px){.ev-row,.grounding-keys div{grid-template-columns:1fr;gap:2px}.updated{display:none}}
"""
