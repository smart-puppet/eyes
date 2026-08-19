# eyes

Perception module for Puppet: camera, YOLO, metric depth → MQTT scene.

Stack overview: [architecture](https://github.com/smart-puppet/docs/blob/main/architecture.md). Topics: [mqtt](https://github.com/smart-puppet/docs/blob/main/mqtt.md).

Inference is **on demand** (not continuous):

- Eye **Capture** button, or
- MQTT `robot/nav/capture` (brain) → publishes `robot/nav/scene`

## Quick start

```bash
# Preview + capture API / MQTT listener
bash scripts/run_debug_web.sh
# open http://127.0.0.1:8091 — Eye UI (camera, drive, play speeds)
# systemd (port 80 + mDNS): http://puppet.local  — see docs systemd.md
# Language buttons (EN 1, DE 2, …) are saved to brain/config/language.active (applies on next brain start)
# Follow / seek / forward sliders apply live (brain/config/play.speeds + robot/play/speeds)
# Mic slider sets PulseAudio default input volume immediately (no brain restart)
# Clear logs empties the brain/drive panes
# Brain and drive log panes appear once those processes publish robot/log/*
# or: mosquitto_pub -t robot/nav/capture -m '{"req_id":"test","view":"traverse"}'
```

Boot service: `deploy/systemd/puppet-eyes.service` — stack install in [systemd](https://github.com/smart-puppet/docs/blob/main/systemd.md) (drive bridge starts first).

Module-local notes: [`docs/`](docs/) (DeepStream, depth, precision).
