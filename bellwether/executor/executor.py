"""The executor — the only write-capable component (BUILD_SPEC §3.8).

Holds the write identity ``bellwether-exec`` (its own cert, from the
``executor`` config section), used nowhere else and reached only past
approval. ``execute`` refuses unless:

- the executor is enabled;
- the approval record is APPROVED, belongs to this proposal, and the store
  agrees — a replayed or forged record cannot run anything;
- the proposal is exactly the stored, approved one;
- the action is EXECUTABLE.

It then dispatches to the whitelist operation named by ``executor_op`` and
records EXECUTED or FAILED back through the store.

Connection topology is per action:

- ``create_small_index`` uses the replica-set URI (all members,
  ``replicaSet=rs0``, no directConnection) so the driver sends the build to
  the primary, wherever it is.
- ``kill_op`` connects directly to the member running the op — killOp only
  affects the mongod it is sent to. That member is the finding's ``op_node``
  evidence (detector output), which must match the finding's node and be a
  known member of the replica set.

The action's ``command`` string is display and audit text only. Nothing in
this package interprets or runs it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pymongo import MongoClient

from bellwether.config import ExecutorConfig
from bellwether.detectors.base import OP_NODE_EVIDENCE, OPID_EVIDENCE
from bellwether.executor.whitelist import (
    WHITELIST,
    ActionNotWhitelisted,
    ActionRefused,
    InvalidActionArgs,
    create_small_index,
    kill_op,
)
from bellwether.models import ActionKind, ApprovalRecord, ApprovalState, Proposal
from bellwether.mongo import ClientFactory
from bellwether.store.sqlite import SqliteStore

logger = logging.getLogger(__name__)

Doc = dict[str, Any]

WRITE_IDENTITY = "bellwether-exec"


class ExecutionRefused(Exception):
    """A gate in front of the executor said no; nothing touched the cluster."""


class Executor:
    def __init__(
        self,
        config: ExecutorConfig,
        store: SqliteStore,
        *,
        client_factory: ClientFactory = MongoClient,
        known_nodes: Iterable[str] = (),
    ) -> None:
        self._config = config
        self._store = store
        self._factory = client_factory
        self._clients: dict[str, MongoClient[Doc]] = {}
        seeds = _uri_hosts(config.mongo_uri) if config.mongo_uri else []
        self._known_nodes = frozenset([*seeds, *known_nodes])

    @property
    def known_nodes(self) -> frozenset[str]:
        """Members kill_op may connect to: replica-set seeds plus configured read nodes."""
        return self._known_nodes

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
        for client in self._clients.values():
            client.close()
        self._clients.clear()

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
            node = self._op_node(proposal)
            return kill_op(
                self._direct_client(node),
                args["opid"],
                identified_opids=_identified_opids(proposal),
            )
        _expect_keys(args, {"db", "collection", "keys", "estimated_docs"})
        return create_small_index(
            self._replica_set_client(),
            db=args["db"],
            collection=args["collection"],
            keys=args["keys"],
            estimated_docs=args["estimated_docs"],
            document_threshold=self._config.document_threshold,
        )

    def _op_node(self, proposal: Proposal) -> str:
        """The member running the op, from the finding's evidence."""
        values = [e.value for e in proposal.evidence_refs if e.name == OP_NODE_EVIDENCE]
        if len(values) != 1 or not isinstance(values[0], str):
            raise ActionRefused(
                f"kill_op needs exactly one {OP_NODE_EVIDENCE} in the finding's evidence "
                f"(killOp only acts on the mongod running the op); found {len(values)}"
            )
        node = values[0]
        if node not in self._known_nodes:
            raise ActionRefused(f"op_node {node!r} is not a known replica set member")
        if node != proposal.node:
            raise ActionRefused(f"op_node {node} disagrees with the finding's node {proposal.node}")
        return node

    def _replica_set_client(self) -> MongoClient[Doc]:
        uri, kwargs = self._settings()
        # No directConnection: the driver discovers the set and writes to the primary.
        return self._client_for("replica_set", uri, {**kwargs, "directConnection": False})

    def _direct_client(self, node: str) -> MongoClient[Doc]:
        uri, kwargs = self._settings()
        return self._client_for(
            f"node:{node}", _direct_uri(uri, node), {**kwargs, "directConnection": True}
        )

    def _client_for(self, key: str, uri: str, kwargs: dict[str, Any]) -> MongoClient[Doc]:
        if key not in self._clients:
            logger.info(
                "opening write connection",
                extra={
                    "identity": WRITE_IDENTITY,
                    "target": key,
                    "cert": kwargs["tlsCertificateKeyFile"],
                },
            )
            self._clients[key] = self._factory(uri, **kwargs)
        return self._clients[key]

    def _settings(self) -> tuple[str, dict[str, Any]]:
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
        return cfg.mongo_uri, kwargs


def _uri_hosts(uri: str) -> list[str]:
    return urlsplit(uri).netloc.rpartition("@")[2].split(",")


def _direct_uri(uri: str, node: str) -> str:
    """The replica-set URI narrowed to one host: same auth options, no replicaSet."""
    parts = urlsplit(uri)
    userinfo = parts.netloc.rpartition("@")[0]
    query = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name.lower() not in ("replicaset", "directconnection")
    ]
    netloc = f"{userinfo}@{node}" if userinfo else node
    return urlunsplit(parts._replace(netloc=netloc, query=urlencode(query)))


def _expect_keys(args: Mapping[str, Any], expected: set[str]) -> None:
    if set(args) != expected:
        raise InvalidActionArgs(f"expected arguments {sorted(expected)}, got {sorted(args)}")


def _identified_opids(proposal: Proposal) -> set[int]:
    """opids from the finding's evidence — detector output, not model output."""
    return {
        e.value
        for e in proposal.evidence_refs
        if e.name == OPID_EVIDENCE and type(e.value) is int
    }
