"""In-memory pymongo stand-ins shared by the test suite.

FakeCluster plays rs0: every node serves the same data, any node can be marked
down, and every client it hands out records the URI and kwargs it was built
with — so tests can assert on connection order and on how TLS was configured.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import yaml
from bson import Timestamp
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from bellwether.analysis.provider import Provider
from bellwether.models import Finding
from bellwether.mongo import ClientFactory

Doc = dict[str, Any]
CommandHandler = Callable[[str, Doc], Doc]  # (node, command doc) -> reply
AggregateHandler = Callable[[str, str, list[Doc]], list[Doc]]  # (node, ns, pipeline) -> docs

TARGET = "node-backup.mongo.internal:27017"
WESTEUROPE = "node-westeurope.mongo.internal:27017"
UAE = "node-uae.mongo.internal:27017"
SOUTHAFRICA = "node-southafrica.mongo.internal:27017"
FALLBACKS = [WESTEUROPE, UAE, SOUTHAFRICA]

READ_URI = (
    "mongodb://node-backup.mongo.internal:27017/"
    "?authMechanism=MONGODB-X509&authSource=%24external&tls=true&directConnection=true"
)


@dataclass(frozen=True)
class Call:
    node: str
    db: str
    op: str  # "command" | "find" | "find_one" | "aggregate"
    target: str  # command name, collection name, or first pipeline stage


class FakeCluster:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.clients: list[FakeClient] = []
        self.calls: list[Call] = []
        self.commands: dict[str, CommandHandler] = {"ping": lambda node, doc: {"ok": 1.0}}
        self.collections: dict[str, list[Doc]] = {}
        self.aggregations: dict[str, AggregateHandler] = {}

    def factory(self) -> ClientFactory:
        def build(uri: str, **kwargs: Any) -> FakeClient:
            client = FakeClient(self, uri, kwargs)
            self.clients.append(client)
            return client

        return cast(ClientFactory, build)

    def reply(self, name: str, response: Doc) -> None:
        """Answer command `name` with a fixed document."""
        self.commands[name] = lambda node, doc: response

    def reply_in_sequence(self, name: str, responses: Sequence[Doc]) -> None:
        """Answer successive `name` commands with successive documents."""
        queue = list(responses)
        self.commands[name] = lambda node, doc: queue.pop(0)

    @property
    def attempted_nodes(self) -> list[str]:
        return [client.node for client in self.clients]

    def commands_sent(self, name: str) -> list[Call]:
        return [c for c in self.calls if c.op == "command" and c.target == name]

    def _record(self, node: str, db: str, op: str, target: str) -> None:
        if node in self.down:
            raise ServerSelectionTimeoutError(f"{node}: [Errno 111] Connection refused")
        self.calls.append(Call(node, db, op, target))


class FakeClient:
    def __init__(self, cluster: FakeCluster, uri: str, kwargs: dict[str, Any]) -> None:
        self.cluster = cluster
        self.uri = uri
        self.kwargs = kwargs
        self.node = urlsplit(uri).netloc.rpartition("@")[2]
        self.closed = False

    def __getitem__(self, name: str) -> FakeDatabase:
        return FakeDatabase(self, name)

    @property
    def admin(self) -> FakeDatabase:
        return self["admin"]

    def close(self) -> None:
        self.closed = True


class FakeDatabase:
    def __init__(self, client: FakeClient, name: str) -> None:
        self.client = client
        self.name = name

    def __getitem__(self, name: str) -> FakeCollection:
        return FakeCollection(self, name)

    def command(self, command: str | Mapping[str, Any], **kwargs: Any) -> Doc:
        doc: Doc = {command: 1, **kwargs} if isinstance(command, str) else dict(command)
        name = next(iter(doc))
        cluster = self.client.cluster
        cluster._record(self.client.node, self.name, "command", name)
        # A "<db>.<command>" handler answers for one database; "<command>" for all.
        handler = cluster.commands.get(f"{self.name}.{name}") or cluster.commands.get(name)
        if handler is None:
            raise OperationFailure(f"no such command: '{name}'", code=59)
        return handler(self.client.node, doc)

    def aggregate(self, pipeline: list[Doc]) -> list[Doc]:
        return _aggregate(self.client, self.name, "", pipeline)


class FakeCollection:
    def __init__(self, database: FakeDatabase, name: str) -> None:
        self.database = database
        self.name = name

    @property
    def _docs(self) -> list[Doc]:
        return self.database.client.cluster.collections.get(f"{self.database.name}.{self.name}", [])

    def find_one(
        self,
        filter: Mapping[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        projection: Mapping[str, Any] | None = None,
    ) -> Doc | None:
        self.database.client.cluster._record(
            self.database.client.node, self.database.name, "find_one", self.name
        )
        docs = self._docs
        if not docs:
            return None
        if sort and sort[0] == ("$natural", -1):
            return docs[-1]
        return docs[0]

    def find(
        self,
        filter: Mapping[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
    ) -> list[Doc]:
        self.database.client.cluster._record(
            self.database.client.node, self.database.name, "find", self.name
        )
        docs = [d for d in self._docs if _matches(d, filter or {})]
        for key, direction in reversed(sort or []):
            docs.sort(key=lambda d: d[key], reverse=direction < 0)
        return docs[:limit] if limit else docs

    def aggregate(self, pipeline: list[Doc]) -> list[Doc]:
        return _aggregate(self.database.client, self.database.name, self.name, pipeline)


def oplog_cluster(window_seconds: int) -> FakeCluster:
    """A cluster whose oplog holds `window_seconds` of history (read from a primary,
    so the collector uses the mean write rate and needs no sampling)."""
    cluster = FakeCluster()
    start = 1_757_000_000
    cluster.collections["local.oplog.rs"] = [
        {"ts": Timestamp(start, 1)},
        {"ts": Timestamp(start + window_seconds, 1)},
    ]
    cluster.aggregations["$collStats"] = lambda node, ns, pipeline: [
        {"storageStats": {"maxSize": 990 * 2**20, "size": 512 * 2**20}}
    ]
    cluster.reply("serverStatus", {"repl": {"setName": "rs0", "secondary": False}, "ok": 1.0})
    # No application databases: the query profile collector finds nothing to read.
    cluster.reply(
        "listDatabases",
        {"databases": [{"name": "admin"}, {"name": "config"}, {"name": "local"}], "ok": 1.0},
    )
    return cluster


PROPOSAL_PAYLOAD: Doc = {
    "diagnosis": "The oplog holds 40 minutes of history against a 60 minute resync estimate.",
    "mechanism": "local.oplog.rs is capped by size; the write rate truncates older entries.",
    "impact_if_ignored": "A secondary down for maintenance past the window needs an initial sync.",
    "action": {
        "kind": "propose_only",
        "title": "Grow the oplog on every member",
        "command": "db.adminCommand({ replSetResizeOplog: 1, size: 51200 })",
        "rationale": "A larger oplog restores a window above the resync estimate.",
        "reversible": True,
        "executor_op": None,
        "executor_args": None,
    },
    "confidence": 0.8,
}


class StaticProvider(Provider):
    """Answers every call with the same payload (or raises the same error)."""

    def __init__(self, name: str, reply: Doc | Exception) -> None:
        self.name = name
        self.reply = reply
        self.calls = 0

    def analyze(self, finding: Finding, context: dict[str, Any]) -> Doc:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return copy.deepcopy(self.reply)


PROVIDER_KEYS = {
    "BELLWETHER_ANALYSIS__ANTHROPIC_API_KEY": "sk-ant-test-0000",
    "BELLWETHER_ANALYSIS__OPENAI_API_KEY": "sk-openai-test-0000",
}


def write_config(directory: Path, **sections: Doc) -> Path:
    """A config YAML for rs0 with the store under `directory`; sections merge in."""
    data: Doc = {
        "mongo": {
            "uri": READ_URI,
            "tls_ca_file": "/etc/mongodb/tls/ca-chain.cert.pem",
            "tls_cert_file": "/etc/bellwether/tls/meetadev-ai.combined.pem",
            "target_node": TARGET,
            "fallback_nodes": FALLBACKS,
        },
        "analysis": {
            "primary_provider": "claude",
            "fallback_provider": "openai",
            "claude_model": "claude-opus-5",
            "openai_model": "gpt-5",
        },
        "notify": {"channels": ["stdout"]},
        "store": {"sqlite_path": str(directory / "bellwether.db")},
        "collectors": {"oplog_window": {"sample_interval_seconds": 0}},
    }
    for name, values in sections.items():
        data.setdefault(name, {}).update(values)
    path = directory / "bellwether.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


_QUERY_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda value, arg: value == arg,
    "$ne": lambda value, arg: value != arg,
    "$in": lambda value, arg: value in arg,
    "$nin": lambda value, arg: value not in arg,
    "$gt": lambda value, arg: value is not None and value > arg,
    "$gte": lambda value, arg: value is not None and value >= arg,
    "$lt": lambda value, arg: value is not None and value < arg,
    "$lte": lambda value, arg: value is not None and value <= arg,
}


def _matches(doc: Doc, filter: Mapping[str, Any]) -> bool:
    """Top-level equality and comparison operators — enough for the helpers' filters."""
    for field, condition in filter.items():
        value = doc.get(field)
        if isinstance(condition, Mapping) and condition and all(k.startswith("$") for k in condition):
            if not all(_QUERY_OPERATORS[op](value, arg) for op, arg in condition.items()):
                return False
        elif value != condition:
            return False
    return True


def _aggregate(client: FakeClient, db: str, collection: str, pipeline: list[Doc]) -> list[Doc]:
    stage = next(iter(pipeline[0]))
    client.cluster._record(client.node, db, "aggregate", stage)
    handler = client.cluster.aggregations.get(stage)
    if handler is None:
        raise OperationFailure(f"unrecognized pipeline stage: '{stage}'", code=40324)
    namespace = f"{db}.{collection}" if collection else db
    return handler(client.node, namespace, pipeline)
