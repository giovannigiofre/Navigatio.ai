import json
import logging

from app.logging_config import JsonFormatter, request_id_var


def format_record(**extra: object) -> dict:
    record = logging.makeLogRecord(
        {"name": "app.test", "levelname": "INFO", "msg": "node complete", **extra}
    )
    return json.loads(JsonFormatter().format(record))


def test_extra_fields_become_json_keys():
    line = format_record(node="Supervisor", latency_ms=12.5)

    assert line["msg"] == "node complete"
    assert line["node"] == "Supervisor"
    assert line["latency_ms"] == 12.5
    assert "request_id" not in line


def test_request_id_from_context_is_included():
    token = request_id_var.set("req-123")
    try:
        line = format_record()
    finally:
        request_id_var.reset(token)

    assert line["request_id"] == "req-123"
