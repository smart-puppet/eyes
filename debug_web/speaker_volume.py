"""Read/set PulseAudio default speaker volume (live, no brain restart)."""

from __future__ import annotations

from typing import Any

from mic_volume import (
  MIC_MAX_PERCENT,
  MIC_MIN_PERCENT,
  _run_pactl,
  parse_source_mute,
  parse_source_volume_percent,
  pick_respeaker_pulse_name,
  pulse_name_is_respeaker,
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


def _respeaker_or_default_sink() -> str:
  chosen = pick_respeaker_pulse_name(_run_pactl("list", "sinks", "short"))
  if chosen:
    return chosen
  return _run_pactl("get-default-sink").strip()


def get_default_speaker() -> dict[str, Any]:
  sink = _respeaker_or_default_sink()
  volume_out = _run_pactl("get-sink-volume", sink)
  mute_out = _run_pactl("get-sink-mute", sink)
  percent = parse_source_volume_percent(volume_out)
  if percent is None:
    raise RuntimeError(f"could not parse speaker volume: {volume_out.strip()[:120]}")
  percent = max(SPEAKER_MIN_PERCENT, min(SPEAKER_MAX_PERCENT, percent))
  note = "reSpeaker XVF3800 speaker — live (AEC far-end)"
  if not pulse_name_is_respeaker(sink):
    note = "Pulse default output (not reSpeaker) — AEC will not cancel TTS"
  return {
    "ok": True,
    "percent": percent,
    "muted": parse_source_mute(mute_out),
    "sink": sink,
    "label": sink_label(sink),
    "applies": "live",
    "note": note,
  }


def set_default_speaker(percent: int) -> dict[str, Any]:
  value = int(percent)
  if value < SPEAKER_MIN_PERCENT or value > SPEAKER_MAX_PERCENT:
    raise ValueError(f"percent must be {SPEAKER_MIN_PERCENT}-{SPEAKER_MAX_PERCENT}")
  sink = _respeaker_or_default_sink()
  _run_pactl("set-sink-volume", sink, f"{value}%")
  if value > 0:
    _run_pactl("set-sink-mute", sink, "0")
  return get_default_speaker()
