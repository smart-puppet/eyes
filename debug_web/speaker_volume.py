"""Read/set PulseAudio default speaker volume (live, no brain restart)."""

from __future__ import annotations

from typing import Any

from mic_volume import (
  MIC_MAX_PERCENT,
  MIC_MIN_PERCENT,
  _run_pactl,
  parse_source_mute,
  parse_source_volume_percent,
)

SPEAKER_MIN_PERCENT = MIC_MIN_PERCENT
SPEAKER_MAX_PERCENT = MIC_MAX_PERCENT


def sink_label(name: str) -> str:
  raw = (name or "").strip()
  if not raw:
    return "default speaker"
  lower = raw.lower()
  if "respeaker" in lower or "xvf3800" in lower:
    return "reSpeaker XVF3800"
  if raw.startswith("alsa_output."):
    raw = raw.split(".", 1)[-1]
  return raw.replace("_", " ")[:56]


def get_default_speaker() -> dict[str, Any]:
  sink = _run_pactl("get-default-sink").strip()
  volume_out = _run_pactl("get-sink-volume", "@DEFAULT_SINK@")
  mute_out = _run_pactl("get-sink-mute", "@DEFAULT_SINK@")
  percent = parse_source_volume_percent(volume_out)
  if percent is None:
    raise RuntimeError(f"could not parse speaker volume: {volume_out.strip()[:120]}")
  percent = max(SPEAKER_MIN_PERCENT, min(SPEAKER_MAX_PERCENT, percent))
  return {
    "ok": True,
    "percent": percent,
    "muted": parse_source_mute(mute_out),
    "sink": sink,
    "label": sink_label(sink),
    "applies": "live",
    "note": "PulseAudio default output — applies immediately",
  }


def set_default_speaker(percent: int) -> dict[str, Any]:
  value = int(percent)
  if value < SPEAKER_MIN_PERCENT or value > SPEAKER_MAX_PERCENT:
    raise ValueError(f"percent must be {SPEAKER_MIN_PERCENT}-{SPEAKER_MAX_PERCENT}")
  _run_pactl("set-sink-volume", "@DEFAULT_SINK@", f"{value}%")
  if value > 0:
    _run_pactl("set-sink-mute", "@DEFAULT_SINK@", "0")
  return get_default_speaker()
