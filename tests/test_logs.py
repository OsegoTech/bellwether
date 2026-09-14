"""Structured JSON logging (BUILD_SPEC §4)."""

from __future__ import annotations

import io
import json
import logging

from bellwether.logs import JsonFormatter, configure_logging


def make_record(msg: str = "read served", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("bellwether.mongo", logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_emits_json_with_extras() -> None:
    line = JsonFormatter().format(make_record(node="node-backup.mongo.internal:27017"))

    out = json.loads(line)
    assert out["msg"] == "read served"
    assert out["level"] == "INFO"
    assert out["logger"] == "bellwether.mongo"
    assert out["node"] == "node-backup.mongo.internal:27017"
    assert "ts" in out


def test_non_serialisable_extra_is_stringified() -> None:
    out = json.loads(JsonFormatter().format(make_record(thing=object())))

    assert out["thing"].startswith("<object object")


def test_exception_is_included() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "bellwether", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )

    out = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in out["exc"]


def test_configure_logging_writes_json_lines() -> None:
    stream = io.StringIO()
    handler = configure_logging("INFO", stream=stream)
    try:
        logging.getLogger("bellwether.test").info("hello", extra={"provider": "claude"})
    finally:
        logging.getLogger().removeHandler(handler)

    out = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert out == {**out, "msg": "hello", "provider": "claude"}
