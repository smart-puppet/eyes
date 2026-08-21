from speaker_volume import sink_label
from mic_volume import pick_respeaker_pulse_name, pulse_name_is_respeaker


def test_sink_label() -> None:
  name = "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_114993701261700006-00.analog-stereo"
  assert sink_label(name) == "reSpeaker XVF3800"
  assert sink_label("alsa_output.usb-Generic_Speaker") == "usb-Generic Speaker"
  assert sink_label("") == "default speaker"


def test_pick_respeaker_sink_over_jetson() -> None:
  short = (
    "1\talsa_output.platform-sound.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
    "7\talsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_x-00.analog-stereo\t"
    "module-alsa-card.c\ts16le 2ch 16000Hz\tRUNNING\n"
  )
  picked = pick_respeaker_pulse_name(short)
  assert picked is not None
  assert "reSpeaker" in picked
  assert not pulse_name_is_respeaker(
    "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800.analog-stereo.monitor"
  )
