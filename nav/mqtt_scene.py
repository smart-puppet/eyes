"""Publish robot/nav/scene JSON over MQTT."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt


class ScenePublisher:
    def __init__(
        self,
        *,
        broker: str = "127.0.0.1",
        port: int = 1883,
        topic: str = "robot/nav/scene",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self._client: Optional[mqtt.Client] = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._ok = False

    def start(self) -> None:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"eyes_nav_scene_{os.getpid()}",
        )
        if self.username:
            client.username_pw_set(self.username, self.password)
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
            print(self._error, flush=True)

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._ok = False

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
            "broker": f"{self.broker}:{self.port}",
        }
