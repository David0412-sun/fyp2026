import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

#import requests


class CloudUploader:
    def __init__(self) -> None:
        self.enabled = os.environ.get("CLOUD_UPLOAD_ENABLED", "1") == "1"
        self.url = os.environ.get("CLOUD_API_INGEST_URL", "http://127.0.0.1:9000/api/ingest")
        self.token = os.environ.get("CLOUD_API_INGEST_TOKEN", "change-me")
        self.source = os.environ.get("CLOUD_SOURCE_NAME", "raspberry-pi")
        self.timeout = float(os.environ.get("CLOUD_UPLOAD_TIMEOUT", "2.0"))
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def enqueue_result(self,
                       angle_deg: float | None,
                       distance_m: float | None,
                       loudness_db: float | None,
                       snr_db: float | None = None,
                       coherence: float | None = None,
                       valid: bool = True) -> None:
        if not self.enabled:
            return
        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "unix_ts": time.time(),
            "angle_deg": angle_deg,
            "distance_m": distance_m,
            "loudness_db": loudness_db,
            "snr_db": snr_db,
            "coherence": coherence,
            "valid": bool(valid),
            "source": self.source,
        }
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def _worker(self) -> None:
        headers = {"X-Ingest-Token": self.token}
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            except Exception:
                pass
