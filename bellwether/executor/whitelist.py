"""The executor whitelist — exactly two operations (BUILD_SPEC §3.8).

- ``kill_op``: kill an operation a detector identified. Reversible in the
  operational sense: the op was not meant to run.
- ``create_small_index``: build an index on a collection under the document
  threshold. Reversible: the index can be dropped.

Each validates its arguments strictly before touching the cluster. Malformed
arguments raise ``InvalidActionArgs``; a well-formed request that a safety
limit refuses raises ``ActionRefused``. Any other operation name is
``ActionNotWhitelisted`` (raised by the executor's dispatch). A third operation
is a design decision, not a code change (BUILD_SPEC §5).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from pymongo import MongoClient

Doc = dict[str, Any]

WHITELIST = frozenset({"kill_op", "create_small_index"})

_SYSTEM_DATABASES = frozenset({"admin", "local", "config"})
_DB_NAME_FORBIDDEN = frozenset('/\\. "$\x00')


class ActionNotWhitelisted(Exception):
    """The requested operation is not one of the two whitelisted operations."""


class InvalidActionArgs(ValueError):
    """The operation's arguments failed strict validation."""


class ActionRefused(Exception):
    """Valid arguments, but a safety limit refuses the operation."""


def kill_op(client: MongoClient[Doc], opid: object, *, identified_opids: Collection[int]) -> str:
    """Kill `opid`, which must be an integer a detector identified for this proposal."""
    if type(opid) is not int:
        raise InvalidActionArgs(f"opid must be an integer, got {type(opid).__name__}")
    if opid not in identified_opids:
        raise ActionRefused(f"opid {opid} was not identified by a detector for this proposal")
    reply = client.admin.command({"killOp": 1, "op": opid})
    return f"killOp {opid}: {reply.get('info', 'ok')}"


def create_small_index(
    client: MongoClient[Doc],
    *,
    db: object,
    collection: object,
    keys: object,
    estimated_docs: object,
    document_threshold: int,
) -> str:
    """Build an ascending/descending index on a collection under the threshold.

    The threshold is checked against both the proposal's estimate and the
    collection's live estimated count, so an under-estimate cannot slip a
    large build through.
    """
    db_name = _database_name(db)
    collection_name = _collection_name(collection)
    index_keys = _index_keys(keys)
    if type(estimated_docs) is not int or estimated_docs < 0:
        raise InvalidActionArgs("estimated_docs must be a non-negative integer")
    if estimated_docs > document_threshold:
        raise ActionRefused(
            f"estimated {estimated_docs} documents exceeds the {document_threshold} document threshold"
        )
    target = client[db_name][collection_name]
    live = target.estimated_document_count()
    if live > document_threshold:
        raise ActionRefused(
            f"{db_name}.{collection_name} holds ~{live} documents, over the "
            f"{document_threshold} document threshold"
        )
    # background=True is ignored by MongoDB 4.2+, whose optimized build takes
    # exclusive locks only briefly at the start and end; older servers honour it.
    name = target.create_index(index_keys, background=True)
    return f"created index {name} on {db_name}.{collection_name} (~{live} documents)"


def _database_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 63:
        raise InvalidActionArgs("db must be a non-empty database name")
    if set(value) & _DB_NAME_FORBIDDEN:
        raise InvalidActionArgs(f"db {value!r} contains a forbidden character")
    if value in _SYSTEM_DATABASES:
        raise InvalidActionArgs(f"db {value!r} is a system database")
    return value


def _collection_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidActionArgs("collection must be a non-empty collection name")
    if "$" in value or "\x00" in value:
        raise InvalidActionArgs(f"collection {value!r} contains a forbidden character")
    if value.startswith("system."):
        raise InvalidActionArgs(f"collection {value!r} is a system collection")
    return value


def _index_keys(value: object) -> list[tuple[str, int]]:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise InvalidActionArgs("keys must be a non-empty list of {field, direction}")
    keys: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"field", "direction"}:
            raise InvalidActionArgs("each key must be exactly {field, direction}")
        field, direction = item["field"], item["direction"]
        if not isinstance(field, str) or not field or field.startswith("$") or "\x00" in field:
            raise InvalidActionArgs(f"invalid index field {field!r}")
        if type(direction) is not int or direction not in (1, -1):
            raise InvalidActionArgs(f"direction for {field!r} must be 1 or -1")
        keys.append((field, direction))
    if len({field for field, _ in keys}) != len(keys):
        raise InvalidActionArgs("index fields must be unique")
    return keys
