# eyes

Perception module for Puppet: camera, YOLO, metric depth, floor segmentation → MQTT scene.

Stack overview: [architecture](https://github.com/smart-puppet/docs/blob/main/architecture.md). Topics: [mqtt](https://github.com/smart-puppet/docs/blob/main/mqtt.md).

Inference is **on demand** (not continuous):

- Debug web **Capture** button, or
- MQTT `robot/nav/capture` (brain / mcp) → publishes `robot/nav/scene`

## Quick start

```bash
# Preview + capture API / MQTT listener
bash scripts/run_debug_web.sh
# open http://127.0.0.1:8091 — pick Traversability, press Capture
# Brain and drive log panes appear once those processes publish robot/log/*
# or: mosquitto_pub -t robot/nav/capture -m '{"req_id":"test","view":"traverse"}'
```

Module-local notes: [`docs/`](docs/) (DeepStream, depth, precision).
