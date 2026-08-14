"""MQTT for eyes: publish robot/nav/scene; subscribe robot/nav/capture."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

import paho.mqtt.client as mqtt

CaptureHandler = Callable[[Dict[str, Any]], None]

logger = logging.getLogger("eyes.mqtt")


class ScenePublisher:
    """Publish scenes and optionally dispatch capture requests to a callback."""

    def __init__(
        self,
        *,
        broker: str = "127.0.0.1",
        port: int = 1883,
        topic: str = "robot/nav/scene",
        capture_topic: str = "robot/nav/capture",
        username: Optional[str] = None,
        password: Optional[str] = None,
        on_capture: Optional[CaptureHandler] = None,
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.capture_topic = capture_topic
        self.username = username
        self.password = password
        self._on_capture = on_capture
        self._client: Optional[mqtt.Client] = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._ok = False

    def set_capture_handler(self, handler: Optional[CaptureHandler]) -> None:
        self._on_capture = handler

    def start(self) -> None:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"eyes_nav_scene_{os.getpid()}",
        )
        if self.username:
            client.username_pw_set(self.username, self.password)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        try:
            client.connect(self.broker, self.port, keepalive=30)
            client.loop_start()
            self._client = client
            self._ok = True
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._error = f"mqtt connect failed: {exc}"
            self._ok = False
            self._client = None
            logger.warning("%s", self._error)

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._ok = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        rc = getattr(reason_code, "value", reason_code)
        logger.info("MQTT connected rc=%s; subscribe %s", rc, self.capture_topic)
        if self.capture_topic:
            client.subscribe(self.capture_topic, qos=1)

    def _on_message(self, client, userdata, msg):
        if msg.topic != self.capture_topic or self._on_capture is None:
            return
        raw = msg.payload.decode("utf-8", errors="replace") if msg.payload else "{}"
        preview = raw if len(raw) <= 240 else raw[:237] + "..."
        logger.info("MQTT recv %s %s", msg.topic, preview)
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        # Do not block the MQTT loop — capture waits on the streamer thread.
        threading.Thread(
            target=self._dispatch_capture,
            args=(payload,),
            name="eyes-mqtt-capture",
            daemon=True,
        ).start()

    def _dispatch_capture(self, payload: Dict[str, Any]) -> None:
        handler = self._on_capture
        if handler is None:
            return
        try:
            handler(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("capture request failed: %s", exc)

    def publish(self, payload: Dict[str, Any]) -> bool:
        with self._lock:
            if not self._client:
                return False
            try:
                self._client.publish(self.topic, json.dumps(payload), qos=0)
                return True
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "mqtt_ok": self._ok,
            "mqtt_error": self._error,
            "topic": self.topic,
            "capture_topic": self.capture_topic,
            "broker": f"{self.broker}:{self.port}",
        }
