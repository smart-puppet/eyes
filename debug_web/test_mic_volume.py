from mic_volume import parse_source_mute, parse_source_volume_percent, source_label


def test_parse_source_volume_percent() -> None:
  text = (
    "Volume: front-left: 32768 /  50% / -18.06 dB,   "
    "front-right: 32768 /  50% / -18.06 dB\n        balance 0.00\n"
  )
  assert parse_source_volume_percent(text) == 50
  assert parse_source_volume_percent("Volume: 65536 / 100%") == 100
  assert parse_source_volume_percent("") is None


def test_parse_source_mute() -> None:
  assert parse_source_mute("Mute: no") is False
  assert parse_source_mute("Mute: yes") is True


def test_source_label() -> None:
  name = "alsa_input.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_114993701261700006-00.analog-stereo"
  assert source_label(name) == "reSpeaker XVF3800"
  assert source_label("alsa_input.usb-Generic_Mic") == "usb-Generic Mic"
