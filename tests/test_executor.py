"""Acceptance tests for the executor — BUILD_SPEC §3.8.

Spec acceptance:
  - executing an un-approved proposal raises
  - a propose-only proposal raises
  - an approved EXECUTABLE proposal with a whitelisted op calls the right
    whitelist function with validated args (mocked mongo)
  - a create-index over the doc threshold refuses
"""

from __future__ import annotations

import inspect
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from pymongo.errors import OperationFailure

from bellwether.config import ExecutorConfig
from bellwether.executor import whitelist
from bellwether.executor.executor import ExecutionRefused, Executor
from bellwether.executor.whitelist import ActionNotWhitelisted, ActionRefused, InvalidActionArgs
from bellwether.models import (
    ActionKind,
    ApprovalRecord,
    ApprovalState,
    Evidence,
    Proposal,
    RemediationAction,
)
from bellwether.mongo import ClientFactory
from bellwether.store.sqlite import SqliteStore

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EXEC_URI = (
    "mongodb://node-westeurope.mongo.internal:27017/"
    "?authMechanism=MONGODB-X509&authSource=%24external&tls=true&directConnection=true"
)
EXEC_CERT = Path("/etc/bellwether/tls/meetadev-ai-exec.combined.pem")
CA = Path("/etc/mongodb/tls/ca-chain.cert.pem")

INDEX_ARGS: dict[str, Any] = {
    "db": "meetadev_ledger",
    "collection": "transactions",
    "keys": [{"field": "account_id", "direction": 1}, {"field": "posted_at", "direction": -1}],
    "estimated_docs": 40_000,
}


# --- fakes --------------------------------------------------------------------------


class FakeWriteClient:
    def __init__(self, world: FakeWorld, uri: str, kwargs: dict[str, Any]) -> None:
        self.world = world
        self.uri = uri
        self.kwargs = kwargs

    @property
    def admin(self) -> FakeWriteDb:
        return FakeWriteDb(self.world, "admin")

    def __getitem__(self, name: str) -> FakeWriteDb:
        return FakeWriteDb(self.world, name)

    def close(self) -> None:
        self.world.closed = True


class FakeWriteDb:
    def __init__(self, world: FakeWorld, name: str) -> None:
        self.world = world
        self.name = name

    def command(self, doc: dict[str, Any]) -> dict[str, Any]:
        self.world.commands.append((self.name, doc))
        return {"info": "attempting to kill op", "ok": 1.0}

    def __getitem__(self, name: str) -> FakeWriteCollection:
        return FakeWriteCollection(self.world, self.name, name)


class FakeWriteCollection:
    def __init__(self, world: FakeWorld, db: str, name: str) -> None:
        self.world = world
        self.namespace = f"{db}.{name}"

    def estimated_document_count(self) -> int:
        return self.world.doc_count

    def create_index(self, keys: list[tuple[str, int]], **kwargs: Any) -> str:
        if self.world.create_index_error is not None:
            raise self.world.create_index_error
        self.world.indexes.append((self.namespace, keys, kwargs))
        return "_".join(f"{field}_{direction}" for field, direction in keys)


class FakeWorld:
    def __init__(self, doc_count: int = 40_000) -> None:
        self.doc_count = doc_count
        self.clients: list[FakeWriteClient] = []
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.indexes: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = []
        self.create_index_error: Exception | None = None
        self.closed = False

    def factory(self) -> ClientFactory:
        def build(uri: str, **kwargs: Any) -> FakeWriteClient:
            client = FakeWriteClient(self, uri, kwargs)
            self.clients.append(client)
            return client

        return cast(ClientFactory, build)

    @property
    def touched(self) -> bool:
        return bool(self.commands or self.indexes)


# --- helpers ------------------------------------------------------------------------


def executor_config(**overrides: Any) -> ExecutorConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "mongo_uri": EXEC_URI,
        "tls_cert_file": EXEC_CERT,
        "tls_ca_file": CA,
        "allowed_actions": ["kill_op", "create_small_index"],
        "document_threshold": 100_000,
    }
    values.update(overrides)
    return ExecutorConfig(**values)


def index_proposal(**arg_overrides: Any) -> Proposal:
    return Proposal(
        finding_id="f" * 32,
        failure_mode="unindexed_hot_query",
        node="node-backup.mongo.internal:27017",
        diagnosis="Collection scans on transactions.account_id.",
        mechanism="No index covers the filter.",
        impact_if_ignored="Latency grows with the collection.",
        action=RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Index account_id",
            command="rm -rf / ; db.transactions.createIndex({account_id: 1})",
            rationale="Covers the hot filter.",
            reversible=True,
            executor_op="create_small_index",
            executor_args={**INDEX_ARGS, **arg_overrides},
        ),
        confidence=0.9,
        provider="claude",
        created_at=T0,
    )


def kill_proposal(opid: object = 4242, identified: tuple[int, ...] = (4242,)) -> Proposal:
    return Proposal(
        finding_id="f" * 32,
        failure_mode="runaway_operation",
        node="node-uae.mongo.internal:27017",
        diagnosis="Op 4242 has scanned for 900 s.",
        mechanism="Unbounded collection scan holding a ticket.",
        impact_if_ignored="Ticket exhaustion.",
        action=RemediationAction(
            kind=ActionKind.EXECUTABLE,
            title="Kill op 4242",
            command="db.killOp(4242)",
            rationale="The op was not meant to run.",
            reversible=True,
            executor_op="kill_op",
            executor_args={"opid": opid},
        ),
        confidence=0.9,
        provider="claude",
        evidence_refs=tuple(Evidence("opid", o, observed_at=T0) for o in identified),
        created_at=T0,
    )


def propose_only_proposal() -> Proposal:
    return replace(
        index_proposal(),
        action=RemediationAction(
            kind=ActionKind.PROPOSE_ONLY,
            title="Grow the oplog",
            command="db.adminCommand({replSetResizeOplog: 1, size: 51200})",
            rationale="Restore the window.",
            reversible=True,
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bellwether.db")


@pytest.fixture
def world() -> FakeWorld:
    return FakeWorld()


def make_executor(store: SqliteStore, world: FakeWorld, **config: Any) -> Executor:
    return Executor(executor_config(**config), store, client_factory=world.factory())


def approved(store: SqliteStore, proposal: Proposal) -> ApprovalRecord:
    store.record_proposal(proposal)
    return store.record_approval_transition(
        proposal.proposal_id, ApprovalState.APPROVED, by="cli:osego"
    )


# --- Spec acceptance ------------------------------------------------------------------


def test_unapproved_proposal_raises(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    record = store.record_proposal(proposal)  # PENDING

    with pytest.raises(ExecutionRefused, match="pending"):
        make_executor(store, world).execute(proposal, record)

    assert world.clients == []
    assert store.current_state(proposal.proposal_id).state is ApprovalState.PENDING


def test_rejected_proposal_raises(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    store.record_proposal(proposal)
    record = store.record_approval_transition(
        proposal.proposal_id, ApprovalState.REJECTED, by="x"
    )

    with pytest.raises(ExecutionRefused):
        make_executor(store, world).execute(proposal, record)

    assert world.clients == []


def test_propose_only_proposal_raises(store: SqliteStore, world: FakeWorld) -> None:
    proposal = propose_only_proposal()
    record = approved(store, proposal)

    with pytest.raises(ExecutionRefused, match="propose"):
        make_executor(store, world).execute(proposal, record)

    assert world.clients == []
    # Stays APPROVED: a human runs the displayed command.
    assert store.current_state(proposal.proposal_id).state is ApprovalState.APPROVED


def test_approved_index_calls_create_small_index_with_validated_args(
    store: SqliteStore, world: FakeWorld
) -> None:
    proposal = index_proposal()
    record = approved(store, proposal)

    result = make_executor(store, world).execute(proposal, record)

    assert world.indexes == [
        (
            "meetadev_ledger.transactions",
            [("account_id", 1), ("posted_at", -1)],
            {"background": True},
        )
    ]
    assert world.commands == []
    assert result.state is ApprovalState.EXECUTED
    assert result.execution_result is not None
    assert "account_id_1_posted_at_-1" in result.execution_result
    assert store.current_state(proposal.proposal_id).state is ApprovalState.EXECUTED


def test_approved_kill_op_calls_kill_op(store: SqliteStore, world: FakeWorld) -> None:
    proposal = kill_proposal()
    record = approved(store, proposal)

    result = make_executor(store, world).execute(proposal, record)

    assert world.commands == [("admin", {"killOp": 1, "op": 4242})]
    assert world.indexes == []
    assert result.state is ApprovalState.EXECUTED


def test_create_index_over_threshold_refuses(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal(estimated_docs=150_000)
    record = approved(store, proposal)

    with pytest.raises(ActionRefused, match="100000"):
        make_executor(store, world).execute(proposal, record)

    assert world.indexes == []
    state = store.current_state(proposal.proposal_id)
    assert state.state is ApprovalState.FAILED
    assert state.execution_result is not None and "ActionRefused" in state.execution_result


# --- The threshold is real, not the model's word ----------------------------------


def test_live_count_over_threshold_refuses(store: SqliteStore) -> None:
    world = FakeWorld(doc_count=2_500_000)  # the model under-estimated
    proposal = index_proposal(estimated_docs=40_000)
    record = approved(store, proposal)

    with pytest.raises(ActionRefused, match="2500000"):
        make_executor(store, world).execute(proposal, record)

    assert world.indexes == []


def test_threshold_comes_from_config(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal(estimated_docs=40_000)
    record = approved(store, proposal)

    with pytest.raises(ActionRefused):
        make_executor(store, world, document_threshold=5_000).execute(proposal, record)


# --- Whitelist ----------------------------------------------------------------------


def test_whitelist_module_has_exactly_two_operations() -> None:
    functions = {
        name
        for name, obj in inspect.getmembers(whitelist, inspect.isfunction)
        if obj.__module__ == whitelist.__name__ and not name.startswith("_")
    }

    assert functions == {"kill_op", "create_small_index"}
    assert whitelist.WHITELIST == frozenset({"kill_op", "create_small_index"})


def test_op_outside_whitelist_raises(store: SqliteStore, world: FakeWorld) -> None:
    base = index_proposal()
    proposal = replace(base, action=replace(base.action, executor_op="drop_database"))
    record = approved(store, proposal)

    with pytest.raises(ActionNotWhitelisted, match="drop_database"):
        make_executor(store, world).execute(proposal, record)

    assert not world.touched
    assert store.current_state(proposal.proposal_id).state is ApprovalState.FAILED


def test_op_not_enabled_in_config_raises(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    record = approved(store, proposal)

    with pytest.raises(ActionNotWhitelisted, match="allowed_actions"):
        make_executor(store, world, allowed_actions=["kill_op"]).execute(proposal, record)

    assert not world.touched


def test_kill_op_refuses_an_opid_no_detector_identified(
    store: SqliteStore, world: FakeWorld
) -> None:
    proposal = kill_proposal(opid=9999, identified=(4242,))
    record = approved(store, proposal)

    with pytest.raises(ActionRefused, match="9999"):
        make_executor(store, world).execute(proposal, record)

    assert world.commands == []


BAD_INDEX_ARGS = {
    "system db": {"db": "admin"},
    "local db": {"db": "local"},
    "system collection": {"collection": "system.users"},
    "dollar in collection": {"collection": "tx$"},
    "empty keys": {"keys": []},
    "direction as string": {"keys": [{"field": "a", "direction": "1"}]},
    "direction 2": {"keys": [{"field": "a", "direction": 2}]},
    "direction as bool": {"keys": [{"field": "a", "direction": True}]},
    "text index": {"keys": [{"field": "a", "direction": "text"}]},
    "operator field": {"keys": [{"field": "$where", "direction": 1}]},
    "duplicate field": {"keys": [{"field": "a", "direction": 1}, {"field": "a", "direction": -1}]},
    "extra key attribute": {"keys": [{"field": "a", "direction": 1, "unique": True}]},
    "estimate as string": {"estimated_docs": "40000"},
    "negative estimate": {"estimated_docs": -1},
    "unexpected arg": {"unique": True},
}


@pytest.mark.parametrize("override", BAD_INDEX_ARGS.values(), ids=BAD_INDEX_ARGS.keys())
def test_create_small_index_args_are_strict(
    store: SqliteStore, world: FakeWorld, override: dict[str, Any]
) -> None:
    proposal = index_proposal(**override)
    record = approved(store, proposal)

    with pytest.raises(InvalidActionArgs):
        make_executor(store, world).execute(proposal, record)

    assert world.indexes == []
    assert store.current_state(proposal.proposal_id).state is ApprovalState.FAILED


@pytest.mark.parametrize("opid", ["4242", True, 42.0, None])
def test_kill_op_opid_must_be_an_integer(
    store: SqliteStore, world: FakeWorld, opid: object
) -> None:
    proposal = kill_proposal(opid=opid)
    record = approved(store, proposal)

    with pytest.raises(InvalidActionArgs):
        make_executor(store, world).execute(proposal, record)

    assert world.commands == []


def test_kill_op_rejects_extra_args(store: SqliteStore, world: FakeWorld) -> None:
    base = kill_proposal()
    proposal = replace(
        base, action=replace(base.action, executor_args={"opid": 4242, "comment": "x"})
    )
    record = approved(store, proposal)

    with pytest.raises(InvalidActionArgs):
        make_executor(store, world).execute(proposal, record)


# --- Gating beyond the record handed in ------------------------------------------------


def test_disabled_executor_refuses(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    record = approved(store, proposal)

    with pytest.raises(ExecutionRefused, match="disabled"):
        make_executor(store, world, enabled=False).execute(proposal, record)

    assert world.clients == []


def test_stale_approval_record_cannot_execute_twice(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    record = approved(store, proposal)
    executor = make_executor(store, world)
    executor.execute(proposal, record)

    with pytest.raises(ExecutionRefused, match="executed"):
        executor.execute(proposal, record)  # same APPROVED record replayed

    assert len(world.indexes) == 1


def test_forged_approval_record_is_refused(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    store.record_proposal(proposal)  # store says PENDING
    forged = ApprovalRecord(proposal_id=proposal.proposal_id, state=ApprovalState.APPROVED)

    with pytest.raises(ExecutionRefused):
        make_executor(store, world).execute(proposal, forged)

    assert world.clients == []


def test_record_for_another_proposal_is_refused(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    other = index_proposal()
    approved(store, proposal)
    other_record = approved(store, other)

    with pytest.raises(ExecutionRefused, match="another proposal"):
        make_executor(store, world).execute(proposal, other_record)


def test_tampered_proposal_is_refused(store: SqliteStore, world: FakeWorld) -> None:
    proposal = index_proposal()
    record = approved(store, proposal)
    tampered = replace(
        proposal, action=replace(proposal.action, executor_args={**INDEX_ARGS, "db": "billing"})
    )

    with pytest.raises(ExecutionRefused, match="differs"):
        make_executor(store, world).execute(tampered, record)

    assert world.indexes == []


def test_mongo_failure_is_recorded_as_failed(store: SqliteStore, world: FakeWorld) -> None:
    world.create_index_error = OperationFailure("not primary", code=10107)
    proposal = index_proposal()
    record = approved(store, proposal)

    with pytest.raises(OperationFailure):
        make_executor(store, world).execute(proposal, record)

    state = store.current_state(proposal.proposal_id)
    assert state.state is ApprovalState.FAILED
    assert state.execution_result is not None and "not primary" in state.execution_result


# --- Identity and the command string ----------------------------------------------------


def test_write_identity_is_separate_and_passed_as_kwargs(
    store: SqliteStore, world: FakeWorld
) -> None:
    proposal = index_proposal()
    make_executor(store, world).execute(proposal, approved(store, proposal))

    client = world.clients[0]
    assert client.uri == EXEC_URI
    assert client.kwargs["tlsCertificateKeyFile"] == str(EXEC_CERT)
    assert client.kwargs["tlsCAFile"] == str(CA)
    assert client.kwargs["tls"] is True
    assert "tlsCertificateKeyFile" not in client.uri
    for insecure in ("tlsInsecure", "tlsAllowInvalidHostnames", "tlsAllowInvalidCertificates"):
        assert insecure not in client.kwargs


def test_command_string_is_never_shell_executed(
    store: SqliteStore, world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the executor tried to run a shell command")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "popen", forbidden)
    proposal = index_proposal()  # command starts with "rm -rf /"

    result = make_executor(store, world).execute(proposal, approved(store, proposal))

    assert result.state is ApprovalState.EXECUTED


def test_executor_package_has_no_shell_or_eval() -> None:
    root = Path(__file__).resolve().parents[1] / "bellwether" / "executor"
    for path in root.glob("*.py"):
        source = path.read_text()
        for forbidden in ("subprocess", "os.system", "os.popen", "eval(", "exec(", "shlex"):
            assert forbidden not in source, f"{path.name} contains {forbidden}"
