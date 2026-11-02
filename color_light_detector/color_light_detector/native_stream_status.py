"""Measured native input and completed work rates, independent of ROS clocks."""
from collections import deque
import threading
import time


class NativeStreamStatus:
    def __init__(self, *, window_seconds=2.0, clock=time.monotonic):
        self.clock = clock
        self.window_seconds = float(window_seconds)
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.started_at = clock()
        self.lock = threading.Lock()
        self.events = {name: deque() for name in ("received", "processed", "yolo_completed")}
        self.counts = {name: 0 for name in self.events}
        self.last_received_at = None
        self.metadata = dict(native_width=None, native_height=None, encoding=None,
                             frame_id=None, stamp=None)

    def received(self, width, height, encoding, header, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self._record("received", now)
            self.last_received_at = now
            self.metadata = dict(native_width=int(width), native_height=int(height),
                encoding=str(encoding), frame_id=str(header.frame_id),
                stamp=dict(sec=int(header.stamp.sec), nanosec=int(header.stamp.nanosec)))

    def processed(self, *, now=None):
        self._completed("processed", now)

    def yolo_completed(self, *, now=None):
        self._completed("yolo_completed", now)

    def _completed(self, name, now):
        with self.lock:
            self._record(name, self.clock() if now is None else now)

    def _record(self, name, now):
        self.events[name].append(now)
        self.counts[name] += 1
        self._prune(name, now)

    def _prune(self, name, now):
        while self.events[name] and self.events[name][0] <= now - self.window_seconds:
            self.events[name].popleft()

    def snapshot(self, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            age = None if self.last_received_at is None else max(0.0, now-self.last_received_at)
            status = "WAITING" if age is None else ("STALE" if age >= self.window_seconds else "ACTIVE")
            duration = max(1.0, min(self.window_seconds, now-self.started_at))
            payload = dict(self.metadata, status=status,
                last_frame_age_seconds=None if age is None else round(age, 3),
                window_seconds=self.window_seconds, measurement_clock="host_monotonic")
            for name in self.events:
                self._prune(name, now)
                payload[name + "_fps"] = round(len(self.events[name])/duration, 2) if status == "ACTIVE" else 0.0
                payload[name + "_frames"] = self.counts[name]
            return payload


def status_text(status):
    width, height = status["native_width"], status["native_height"]
    size = f"{width}x{height}" if width is not None else "--x--"
    return (f"NATIVE {size} {status['status']} | RX {status['received_fps']:.1f} FPS"
            f" | PROC {status['processed_fps']:.1f} FPS | YOLO {status['yolo_completed_fps']:.1f} FPS")
