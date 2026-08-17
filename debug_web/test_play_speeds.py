from play_speeds import clamp_speed, normalize_speeds, read_play_speeds, write_play_speeds


def test_clamp_speed() -> None:
  assert clamp_speed(100, 120) == 100
  assert clamp_speed(0, 120) == 20
  assert clamp_speed(999, 120) == 200
  assert clamp_speed("nope", 120) == 120


def test_normalize_fills_defaults() -> None:
  speeds = normalize_speeds({"follow_turn": 90})
  assert speeds["follow_turn"] == 90
  assert speeds["seek_turn"] == 125
  assert speeds["forward"] == 105


def test_write_and_read_roundtrip(tmp_path) -> None:
  path, written = write_play_speeds(tmp_path, {"follow_turn": 80, "seek_turn": 140, "forward": 70})
  assert path.name == "play.speeds"
  loaded, exists = read_play_speeds(tmp_path)
  assert exists is True
  assert loaded == written
  assert loaded == {"follow_turn": 80, "seek_turn": 140, "forward": 70}
