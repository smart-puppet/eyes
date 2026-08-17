"""Read/set live play speeds saved next to brain config (follow/seek/forward)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPEED_MIN = 20
SPEED_MAX = 200
SPEEDS_FILE = "play.speeds"
DEFAULT_SPEEDS = {
  "follow_turn": 125,
  "seek_turn": 125,
  "forward": 105,
}


def clamp_speed(value: Any, default: int) -> int:
  try:
    speed = int(value)
  except (TypeError, ValueError):
    return int(default)
  return max(SPEED_MIN, min(SPEED_MAX, speed))


def normalize_speeds(data: Any) -> dict[str, int]:
  src = data if isinstance(data, dict) else {}
  return {
    "follow_turn": clamp_speed(src.get("follow_turn"), DEFAULT_SPEEDS["follow_turn"]),
    "seek_turn": clamp_speed(src.get("seek_turn"), DEFAULT_SPEEDS["seek_turn"]),
    "forward": clamp_speed(src.get("forward"), DEFAULT_SPEEDS["forward"]),
  }


def read_play_speeds(config_dir: Path) -> tuple[dict[str, int], bool]:
  path = config_dir / SPEEDS_FILE
  if not path.is_file():
    return dict(DEFAULT_SPEEDS), False
  try:
    raw = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return dict(DEFAULT_SPEEDS), True
  return normalize_speeds(raw), True


def write_play_speeds(config_dir: Path, speeds: dict[str, Any]) -> tuple[Path, dict[str, int]]:
  config_dir.mkdir(parents=True, exist_ok=True)
  normalized = normalize_speeds(speeds)
  path = config_dir / SPEEDS_FILE
  tmp = path.with_name(path.name + ".tmp")
  tmp.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
  tmp.replace(path)
  return path, normalized
