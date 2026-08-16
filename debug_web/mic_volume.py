"""Read/set PulseAudio default microphone volume (live, no brain restart)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

_VOLUME_PCT_RE = re.compile(r"/\s*(\d+)%")
_MUTE_RE = re.compile(r"Mute:\s*(yes|no)", re.IGNORECASE)
MIC_MIN_PERCENT = 0
MIC_MAX_PERCENT = 150


def parse_source_volume_percent(text: str) -> Optional[int]:
  match = _VOLUME_PCT_RE.search(text or "")
  if not match:
    return None
  return int(match.group(1))


def parse_source_mute(text: str) -> bool:
  match = _MUTE_RE.search(text or "")
  return bool(match and match.group(1).lower() == "yes")


def source_label(name: str) -> str:
  raw = (name or "").strip()
  if not raw:
    return "default microphone"
  lower = raw.lower()
  if "respeaker" in lower or "xvf3800" in lower:
    return "reSpeaker XVF3800"
  if raw.startswith("alsa_input."):
    raw = raw.split(".", 1)[-1]
  return raw.replace("_", " ")[:56]


def _pulse_env() -> dict[str, str]:
  env = os.environ.copy()
  uid = os.getuid()
  runtime = Path(f"/run/user/{uid}")
  pulse_dir = runtime / "pulse"
  pulse_sock = pulse_dir / "native"
  env.setdefault("HOME", str(Path.home()))
  env.setdefault("XDG_RUNTIME_DIR", str(runtime))
  if pulse_sock.is_socket():
    env.setdefault("PULSE_RUNTIME_PATH", str(pulse_dir))
    env.setdefault("PULSE_SERVER", f"unix:{pulse_sock}")
  return env


def _run_pactl(*args: str) -> str:
  try:
    result = subprocess.run(
      ["pactl", *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=2.5,
      env=_pulse_env(),
    )
  except FileNotFoundError as exc:
    raise RuntimeError("pactl not installed") from exc
  except subprocess.TimeoutExpired as exc:
    raise RuntimeError("pactl timed out") from exc
  if result.returncode != 0:
    err = (result.stderr or result.stdout or "pactl failed").strip()
    raise RuntimeError(err)
  return result.stdout


def get_default_mic() -> dict[str, Any]:
  source = _run_pactl("get-default-source").strip()
  volume_out = _run_pactl("get-source-volume", "@DEFAULT_SOURCE@")
  mute_out = _run_pactl("get-source-mute", "@DEFAULT_SOURCE@")
  percent = parse_source_volume_percent(volume_out)
  if percent is None:
    raise RuntimeError(f"could not parse mic volume: {volume_out.strip()[:120]}")
  percent = max(MIC_MIN_PERCENT, min(MIC_MAX_PERCENT, percent))
  return {
    "ok": True,
    "percent": percent,
    "muted": parse_source_mute(mute_out),
    "source": source,
    "label": source_label(source),
    "applies": "live",
    "note": "PulseAudio default input — applies immediately",
  }


def set_default_mic(percent: int) -> dict[str, Any]:
  value = int(percent)
  if value < MIC_MIN_PERCENT or value > MIC_MAX_PERCENT:
    raise ValueError(f"percent must be {MIC_MIN_PERCENT}-{MIC_MAX_PERCENT}")
  _run_pactl("set-source-volume", "@DEFAULT_SOURCE@", f"{value}%")
  if value > 0:
    _run_pactl("set-source-mute", "@DEFAULT_SOURCE@", "0")
  return get_default_mic()
