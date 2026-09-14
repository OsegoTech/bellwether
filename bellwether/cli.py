"""Command-line entrypoint (BUILD_SPEC §3.10).

    bellwether run                      one pipeline pass (for the systemd timer)
    bellwether serve                    approval endpoint + read-only UI
    bellwether list [--pending]         proposals and their state
    bellwether show <id>                one proposal with its audit trail
    bellwether approve <id> --by NAME   manual approval, independent of Slack

The config file comes from ``--config`` or ``BELLWETHER_CONFIG``. Structured
JSON logs go to stderr; human output goes to stdout. Exit codes: 0 success,
1 the command ran but failed (run errors, refused transition, failed
execution), 2 configuration problem.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence

import uvicorn

from bellwether import pipeline
from bellwether.app import create_app
from bellwether.config import BellwetherConfig, ConfigError, env_var_for, load_config
from bellwether.logs import configure_logging
from bellwether.models import ActionKind, ApprovalState
from bellwether.notify.stdout import render_text
from bellwether.store.sqlite import InvalidTransition

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "/etc/bellwether/bellwether.yaml"

Handler = Callable[[BellwetherConfig, argparse.Namespace], int]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"bellwether: {exc}", file=sys.stderr)
        return 2
    handler: Handler = args.handler
    return handler(config, args)


def _run(config: BellwetherConfig, args: argparse.Namespace) -> int:
    summary = pipeline.run_once(config)
    print(
        f"run {summary.run_id}: {len(summary.findings)} finding(s), "
        f"{len(summary.proposals)} proposal(s), {len(summary.errors)} error(s)"
    )
    for error in summary.errors:
        print(f"bellwether: {error}", file=sys.stderr)
    return 1 if summary.errors else 0


def _serve(config: BellwetherConfig, args: argparse.Namespace) -> int:
    secret = config.notify.slack_signing_secret
    if secret is None or not secret.get_secret_value():
        print(
            "bellwether: serve verifies Slack callbacks and requires "
            f"{env_var_for('notify', 'slack_signing_secret')}",
            file=sys.stderr,
        )
        return 2
    store = pipeline.build_store(config)
    executor = pipeline.build_executor(config, store)
    app = create_app(store, signing_secret=secret.get_secret_value(), executor=executor)
    logger.info(
        "serving approval endpoint",
        extra={"host": args.host, "port": args.port, "executor_enabled": executor is not None},
    )
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


def _list(config: BellwetherConfig, args: argparse.Namespace) -> int:
    rows = pipeline.build_store(config).list_proposals()
    if args.pending:
        rows = [(p, r) for p, r in rows if r.state is ApprovalState.PENDING]
    if not rows:
        print("no proposals")
        return 0
    print(f"{'PROPOSAL':32}  {'STATE':9}  {'ACTION':12}  {'CREATED (UTC)':16}  FAILURE MODE / NODE")
    for proposal, record in rows:
        print(
            f"{proposal.proposal_id:32}  {record.state.value:9}  "
            f"{proposal.action.kind.value:12}  {proposal.created_at:%Y-%m-%d %H:%M}  "
            f"{proposal.failure_mode} on {proposal.node}"
        )
    return 0


def _show(config: BellwetherConfig, args: argparse.Namespace) -> int:
    store = pipeline.build_store(config)
    proposal = store.get_proposal(args.proposal_id)
    if proposal is None:
        print(f"bellwether: no proposal {args.proposal_id}", file=sys.stderr)
        return 1
    print(render_text(proposal, store.current_state(args.proposal_id)))
    print("\nAudit trail:")
    for t in store.history(args.proposal_id):
        detail = " ".join(part for part in (t.actor, t.result) if part)
        print(f"  {t.at.isoformat()}  {t.state.value:9}  {detail}".rstrip())
    return 0


def _approve(config: BellwetherConfig, args: argparse.Namespace) -> int:
    store = pipeline.build_store(config)
    proposal = store.get_proposal(args.proposal_id)
    if proposal is None:
        print(f"bellwether: no proposal {args.proposal_id}", file=sys.stderr)
        return 1
    actor = f"cli:{args.by}"
    try:
        record = store.record_approval_transition(
            proposal.proposal_id, ApprovalState.APPROVED, by=actor
        )
    except InvalidTransition as exc:
        print(f"bellwether: {exc}", file=sys.stderr)
        return 1
    print(f"approved {proposal.proposal_id} by {actor}")

    command = proposal.action.command
    if proposal.action.kind is ActionKind.PROPOSE_ONLY:
        print(f"propose-only; run this by hand:\n  {command}")
        return 0
    executor = pipeline.build_executor(config, store)
    if executor is None:
        print(f"executor is disabled (executor.enabled is false); run this by hand:\n  {command}")
        return 0
    try:
        outcome = executor.execute(proposal, record)
    except Exception as exc:
        print(f"bellwether: execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{outcome.state.value}: {outcome.execution_result}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bellwether",
        description="Predictive reliability for self-hosted MongoDB replica sets.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("BELLWETHER_CONFIG", DEFAULT_CONFIG),
        help="config YAML (default: $BELLWETHER_CONFIG or %(default)s)",
    )
    parser.add_argument("--log-level", default="INFO", help="log level for stderr JSON logs")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="one pipeline pass")
    run.set_defaults(handler=_run)

    serve = commands.add_parser("serve", help="approval endpoint and read-only UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(handler=_serve)

    listing = commands.add_parser("list", help="proposals and their state")
    listing.add_argument("--pending", action="store_true", help="only undecided proposals")
    listing.set_defaults(handler=_list)

    show = commands.add_parser("show", help="one proposal with its audit trail")
    show.add_argument("proposal_id")
    show.set_defaults(handler=_show)

    approve = commands.add_parser("approve", help="approve a proposal (CLI path, no Slack)")
    approve.add_argument("proposal_id")
    approve.add_argument("--by", required=True, help="who is approving (recorded in the audit)")
    approve.set_defaults(handler=_approve)

    return parser
