"""In-memory pymongo stand-ins shared by the test suite.

FakeCluster plays rs0: every node serves the same data, any node can be marked
down, and every client it hands out records the URI and kwargs it was built
with — so tests can assert on connection order and on how TLS was configured.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

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
        handler = cluster.commands.get(name)
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
        docs = [d for d in self._docs if all(d.get(k) == v for k, v in (filter or {}).items())]
        for key, direction in reversed(sort or []):
            docs.sort(key=lambda d: d[key], reverse=direction < 0)
        return docs[:limit] if limit else docs

    def aggregate(self, pipeline: list[Doc]) -> list[Doc]:
        return _aggregate(self.database.client, self.database.name, self.name, pipeline)


def _aggregate(client: FakeClient, db: str, collection: str, pipeline: list[Doc]) -> list[Doc]:
    stage = next(iter(pipeline[0]))
    client.cluster._record(client.node, db, "aggregate", stage)
    handler = client.cluster.aggregations.get(stage)
    if handler is None:
        raise OperationFailure(f"unrecognized pipeline stage: '{stage}'", code=40324)
    namespace = f"{db}.{collection}" if collection else db
    return handler(client.node, namespace, pipeline)
