import json
from pathlib import Path

from fisch_macro.logging_util import JsonlLogger


def test_jsonl_logger_writes_event(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = JsonlLogger(path)
    log.log("state_change", from_state="IDLE", to_state="PRE_CAST")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "state_change"
    assert row["to_state"] == "PRE_CAST"
    assert "ts" in row
