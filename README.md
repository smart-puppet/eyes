# eyes

Perception module for Puppet: camera, YOLO, metric depth, floor segmentation → MQTT scene.

Stack overview: [architecture](https://github.com/smart-puppet/docs/blob/main/architecture.md). Topics: [mqtt](https://github.com/smart-puppet/docs/blob/main/mqtt.md).

## Quick start

```bash
# Traversability + scene publish (debug web)
bash scripts/run_debug_web.sh --view traverse
# open http://127.0.0.1:8091
```

Module-local notes: [`docs/`](docs/) (DeepStream, depth, precision).
