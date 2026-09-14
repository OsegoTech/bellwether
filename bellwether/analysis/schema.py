"""The proposal schema: what a model must return, and the model that enforces it.

``PROPOSAL_SCHEMA`` is the published contract (exported verbatim to
``config/proposal.schema.json``; a test keeps them identical). It is written to
fit both providers' constrained-decoding rules — every object closed, every
property required (nullable ones via ``anyOf`` null), no numeric bounds — so
Claude (``output_config.format``) and OpenAI (``response_format`` strict) are
held to the same shape.

``ProposalPayload`` mirrors the schema in strict Pydantic and adds what JSON
Schema here cannot say: confidence within 0..1, non-empty text, and the
action's internal consistency (an EXECUTABLE action names one of the two
whitelisted ops with that op's arguments; PROPOSE_ONLY carries neither).
Validation runs in strict JSON mode: ``"0.9"`` is not a number, ``"true"`` is
not a boolean. Nothing is coerced.

The model fills only the diagnosis and the action. Which finding, node, and
failure mode a proposal belongs to, and which provider wrote it, are set by
the analyst from the Finding — a model cannot forge them.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_NULL = {"type": "null"}

PROPOSAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Bellwether proposal",
    "description": "A diagnosis of one finding and one remediation for a human to approve.",
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "mechanism", "impact_if_ignored", "action", "confidence"],
    "properties": {
        "diagnosis": {"type": "string", "description": "What is happening and why."},
        "mechanism": {"type": "string", "description": "The MongoDB internal that explains it."},
        "impact_if_ignored": {"type": "string", "description": "What happens if nothing changes."},
        "action": {"$ref": "#/$defs/action"},
        "confidence": {"type": "number", "description": "Confidence in the diagnosis, 0.0 to 1.0."},
    },
    "$defs": {
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kind",
                "title",
                "command",
                "rationale",
                "reversible",
                "executor_op",
                "executor_args",
            ],
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["propose_only", "executable"],
                    "description": (
                        "propose_only: a human runs `command` by hand. executable: only "
                        "kill_op or create_small_index, run after human approval."
                    ),
                },
                "title": {"type": "string"},
                "command": {"type": "string", "description": "Human-runnable form; audit record."},
                "rationale": {"type": "string"},
                "reversible": {"type": "boolean"},
                "executor_op": {
                    "anyOf": [
                        {"type": "string", "enum": ["kill_op", "create_small_index"]},
                        _NULL,
                    ]
                },
                "executor_args": {
                    "anyOf": [
                        {"$ref": "#/$defs/kill_op_args"},
                        {"$ref": "#/$defs/create_small_index_args"},
                        _NULL,
                    ]
                },
            },
        },
        "kill_op_args": {
            "type": "object",
            "additionalProperties": False,
            "required": ["opid"],
            "properties": {"opid": {"type": "integer"}},
        },
        "create_small_index_args": {
            "type": "object",
            "additionalProperties": False,
            "required": ["db", "collection", "keys", "estimated_docs"],
            "properties": {
                "db": {"type": "string"},
                "collection": {"type": "string"},
                "keys": {"type": "array", "items": {"$ref": "#/$defs/index_key"}},
                "estimated_docs": {"type": "integer"},
            },
        },
        "index_key": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "direction"],
            "properties": {
                "field": {"type": "string"},
                "direction": {"type": "integer", "enum": [1, -1]},
            },
        },
    },
}


def wire_schema() -> dict[str, Any]:
    """The schema as sent to a provider: document metadata stripped."""
    schema = copy.deepcopy(PROPOSAL_SCHEMA)
    for key in ("$schema", "$id", "title"):
        schema.pop(key, None)
    return schema


NonEmpty = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class IndexKey(_Strict):
    field: NonEmpty
    direction: Literal[1, -1]


class KillOpArgs(_Strict):
    opid: int


class CreateSmallIndexArgs(_Strict):
    db: NonEmpty
    collection: NonEmpty
    keys: list[IndexKey] = Field(min_length=1)
    estimated_docs: int = Field(ge=0)


_ARGS_FOR_OP: dict[str, type[_Strict]] = {
    "kill_op": KillOpArgs,
    "create_small_index": CreateSmallIndexArgs,
}


class ActionPayload(_Strict):
    kind: Literal["propose_only", "executable"]
    title: NonEmpty
    command: NonEmpty
    rationale: NonEmpty
    reversible: bool
    executor_op: Literal["kill_op", "create_small_index"] | None
    executor_args: KillOpArgs | CreateSmallIndexArgs | None

    @model_validator(mode="after")
    def _consistent(self) -> ActionPayload:
        if self.kind == "propose_only":
            if self.executor_op is not None or self.executor_args is not None:
                raise ValueError("propose_only actions carry no executor_op or executor_args")
            return self
        if self.executor_op is None:
            raise ValueError("executable actions must name an executor_op")
        if not isinstance(self.executor_args, _ARGS_FOR_OP[self.executor_op]):
            raise ValueError(f"executor_args do not match executor_op {self.executor_op!r}")
        if not self.reversible:
            raise ValueError("executable actions must be reversible")
        return self


class ProposalPayload(_Strict):
    diagnosis: NonEmpty
    mechanism: NonEmpty
    impact_if_ignored: NonEmpty
    action: ActionPayload
    confidence: float = Field(ge=0.0, le=1.0)


def validate_payload(raw: Mapping[str, Any]) -> ProposalPayload:
    """Validate a model's raw JSON object. Raises pydantic.ValidationError."""
    return ProposalPayload.model_validate_json(json.dumps(raw))
