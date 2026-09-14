"""The executor — the only write-capable component (BUILD_SPEC §3.8).

Holds the write identity ``meetadev-ai-exec`` (its own cert, from the
``executor`` config section), used nowhere else and reached only past
approval. ``execute`` refuses unless:

- the executor is enabled;
- the approval record is APPROVED, belongs to this proposal, and the store
  agrees — a replayed or forged record cannot run anything;
- the proposal is exactly the stored, approved one;
- the action is EXECUTABLE.

It then dispatches to the whitelist operation named by ``executor_op`` and
records EXECUTED or FAILED back through the store.

The action's ``command`` string is display and audit text only. Nothing in
this package interprets or runs it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pymongo import MongoClient

from bellwether.config import ExecutorConfig
from bellwether.executor.whitelist import (
    WHITELIST,
    ActionNotWhitelisted,
    InvalidActionArgs,
    create_small_index,
    kill_op,
)
from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.mongo import ClientFactory
from bellwether.store.sqlite import SqliteStore

logger = logging.getLogger(__name__)

Doc = dict[str, Any]

WRITE_IDENTITY = "meetadev-ai-exec"


class ExecutionRefused(Exception):
    """A gate in front of the executor said no; nothing touched the cluster."""


class Executor:
    def __init__(
        self,
        config: ExecutorConfig,
        store: SqliteStore,
        *,
        client_factory: ClientFactory = MongoClient,
    ) -> None:
        self._config = config
        self._store = store
        self._factory = client_factory
        self._mongo: MongoClient[Doc] | None = None

    def execute(self, proposal: Proposal, approval_record: ApprovalRecord) -> ApprovalRecord:
        self._check_gates(proposal, approval_record)
        proposal_id = proposal.proposal_id
        op = proposal.action.executor_op
        logger.info(
            "executing approved proposal",
            extra={"proposal_id": proposal_id, "executor_op": op, "identity": WRITE_IDENTITY},
        )
        try:
            result = self._dispatch(proposal)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            self._store.record_approval_transition(proposal_id, ApprovalState.FAILED, result=error)
            logger.error(
                "execution failed",
                extra={"proposal_id": proposal_id, "executor_op": op, "error": error},
            )
            raise
        record = self._store.record_approval_transition(
            proposal_id, ApprovalState.EXECUTED, result=result
        )
        logger.info(
            "execution succeeded",
            extra={"proposal_id": proposal_id, "executor_op": op, "result": result},
        )
        return record

    def close(self) -> None:
        if self._mongo is not None:
            self._mongo.close()
            self._mongo = None

    def _check_gates(self, proposal: Proposal, record: ApprovalRecord) -> None:
        proposal_id = proposal.proposal_id
        if not self._config.enabled:
            raise ExecutionRefused("executor is disabled (executor.enabled is false)")
        if record.proposal_id != proposal_id:
            raise ExecutionRefused("approval record belongs to another proposal")
        if record.state is not ApprovalState.APPROVED:
            raise ExecutionRefused(f"proposal {proposal_id} is {record.state.value}, not approved")
        current = self._store.current_state(proposal_id)
        if current.state is not ApprovalState.APPROVED:
            raise ExecutionRefused(
                f"store records proposal {proposal_id} as {current.state.value}; "
                "only an approved proposal executes"
            )
        if self._store.get_proposal(proposal_id) != proposal:
            raise ExecutionRefused(f"proposal {proposal_id} differs from the approved, stored version")
        if proposal.action.kind is not ActionKind.EXECUTABLE:
            raise ExecutionRefused(
                f"proposal {proposal_id} is propose-only; a human runs its command"
            )

    def _dispatch(self, proposal: Proposal) -> str:
        op = proposal.action.executor_op
        if op not in WHITELIST:
            raise ActionNotWhitelisted(f"{op!r} is not a whitelisted executor operation")
        if op not in self._config.allowed_actions:
            raise ActionNotWhitelisted(
                f"{op!r} is whitelisted but not enabled in executor.allowed_actions"
            )
        args = proposal.action.executor_args
        if op == "kill_op":
            _expect_keys(args, {"opid"})
            return kill_op(self._client(), args["opid"], identified_opids=_identified_opids(proposal))
        _expect_keys(args, {"db", "collection", "keys", "estimated_docs"})
        return create_small_index(
            self._client(),
            db=args["db"],
            collection=args["collection"],
            keys=args["keys"],
            estimated_docs=args["estimated_docs"],
            document_threshold=self._config.document_threshold,
        )

    def _client(self) -> MongoClient[Doc]:
        if self._mongo is None:
            cfg = self._config
            if cfg.mongo_uri is None or cfg.tls_cert_file is None or cfg.tls_ca_file is None:
                raise ExecutionRefused("executor connection is not configured")
            kwargs: dict[str, Any] = {
                "tls": True,
                "tlsCAFile": str(cfg.tls_ca_file),
                "tlsCertificateKeyFile": str(cfg.tls_cert_file),
                "serverSelectionTimeoutMS": 10_000,
                "appname": "bellwether-executor",
            }
            if cfg.tls_cert_passphrase is not None:
                kwargs["tlsCertificateKeyFilePassword"] = cfg.tls_cert_passphrase.get_secret_value()
            logger.info(
                "opening write connection",
                extra={"identity": WRITE_IDENTITY, "cert": str(cfg.tls_cert_file)},
            )
            self._mongo = self._factory(cfg.mongo_uri, **kwargs)
        return self._mongo


def _expect_keys(args: Mapping[str, Any], expected: set[str]) -> None:
    if set(args) != expected:
        raise InvalidActionArgs(f"expected arguments {sorted(expected)}, got {sorted(args)}")


def _identified_opids(proposal: Proposal) -> set[int]:
    """opids from the finding's evidence — detector output, not model output."""
    return {
        e.value for e in proposal.evidence_refs if e.name == "opid" and type(e.value) is int
    }
