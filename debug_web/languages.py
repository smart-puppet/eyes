"""Numbered language profiles for Eye (stock *_1 plus local overlays)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

_LANG_ID_RE = re.compile(r"^(en|fr|de)(?:_([1-9]\d*))?$")
_LOCALE_ORDER = {"en": 0, "fr": 1, "de": 2}


def parse_language_id(text: str) -> Optional[str]:
  raw = (text or "").strip().lower().strip("'\"")
  m = _LANG_ID_RE.match(raw)
  if not m:
    return None
  return f"{m.group(1)}_{m.group(2) or 1}"


def _split_stem(stem: str) -> tuple[str, Optional[int]] | None:
  m = _LANG_ID_RE.match((stem or "").strip().lower())
  if not m:
    return None
  num = int(m.group(2)) if m.group(2) else None
  return m.group(1), num


def _next_overlay_id(locale: str, used: set[str]) -> str:
  n = 2
  while f"{locale}_{n}" in used:
    n += 1
  return f"{locale}_{n}"


def _overlay_dir(config_dir: Path) -> Path:
  return (config_dir / ".." / "brain").resolve()


def _label_from_file(path: Path) -> str:
  try:
    text = path.read_text(encoding="utf-8")
  except OSError:
    return path.stem
  for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("label:"):
      label = stripped.split(":", 1)[1].strip().strip("'\"")
      if label:
        return label
  return path.stem


def list_language_profiles(config_dir: Path) -> list[dict[str, Any]]:
  found: dict[str, tuple[Path, bool]] = {}

  def _take(lang_dir: Path, overlay: bool) -> None:
    if not lang_dir.is_dir():
      return
    numbered: list[tuple[str, Path]] = []
    unnumbered: list[tuple[str, Path]] = []
    for path in sorted(lang_dir.glob("*.yaml")):
      parsed = _split_stem(path.stem)
      if parsed is None:
        continue
      locale, num = parsed
      if num is None:
        unnumbered.append((locale, path))
      else:
        numbered.append((f"{locale}_{num}", path))
    for pid, path in numbered:
      found[pid] = (path, overlay)
    for locale, path in unnumbered:
      pid = _next_overlay_id(locale, set(found)) if overlay else f"{locale}_1"
      if pid not in found:
        found[pid] = (path, overlay)

  _take(config_dir / "language", overlay=False)
  _take(_overlay_dir(config_dir), overlay=True)
  out: list[dict[str, Any]] = []
  for pid, (path, overlay) in found.items():
    loc, _, num = pid.partition("_")
    index = int(num or 1)
    out.append(
      {
        "id": pid,
        "label": _label_from_file(path),
        "locale": loc,
        "index": index,
        "overlay": overlay,
        "file": str(path),
        "button": f"{loc.upper()} {index}",
      }
    )
  out.sort(key=lambda item: (_LOCALE_ORDER.get(item["locale"], 9), item["index"]))
  return out
