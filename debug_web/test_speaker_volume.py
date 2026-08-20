from speaker_volume import sink_label


def test_sink_label() -> None:
  name = "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_114993701261700006-00.analog-stereo"
  assert sink_label(name) == "reSpeaker XVF3800"
  assert sink_label("alsa_output.usb-Generic_Speaker") == "usb-Generic Speaker"
  assert sink_label("") == "default speaker"
