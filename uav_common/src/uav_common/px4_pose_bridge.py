"""Timestamped PX4 NED/FRD pose assembly without ROS or receipt-time stamping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


DEFAULTS = {
    "timestamp_offset_seconds": 0.0,
    "pose_sync_tolerance_seconds": 0.05,
    "max_clock_skew_seconds": 0.5,
    "buffer_seconds": 2.0,
    "max_buffer_samples": 512,
}


def _finite(values):
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite vector")
    return result


def quaternion_xyzw(q_wxyz):
    """Normalize PX4 body-FRD-to-NED q=[w,x,y,z], preserving its rotation."""
    w, x, y, z = _finite(q_wxyz)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid quaternion norm")
    return (x/norm, y/norm, z/norm, w/norm)


def _slerp(first, second, alpha):
    dot = sum(a*b for a, b in zip(first, second))
    if dot < 0:
        second, dot = tuple(-v for v in second), -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = tuple(a + alpha*(b-a) for a, b in zip(first, second))
    else:
        angle = math.acos(dot)
        a = math.sin((1-alpha)*angle) / math.sin(angle)
        b = math.sin(alpha*angle) / math.sin(angle)
        q = tuple(a*x + b*y for x, y in zip(first, second))
    norm = math.sqrt(sum(v*v for v in q))
    return tuple(v/norm for v in q)


def camera_transform(transform):
    """Convert T_body_camera to body-to-camera TF translation and xyzw."""
    rows = [_finite(row) for row in transform]
    r = [row[:3] for row in rows[:3]]
    trace = r[0][0]+r[1][1]+r[2][2]
    if trace > 0:
        scale = math.sqrt(trace+1)*2
        w = scale/4
        x, y, z = (r[2][1]-r[1][2])/scale, (r[0][2]-r[2][0])/scale, (r[1][0]-r[0][1])/scale
    elif r[0][0] > max(r[1][1], r[2][2]):
        scale = math.sqrt(1+r[0][0]-r[1][1]-r[2][2])*2
        x = scale/4
        w, y, z = (r[2][1]-r[1][2])/scale, (r[0][1]+r[1][0])/scale, (r[0][2]+r[2][0])/scale
    elif r[1][1] > r[2][2]:
        scale = math.sqrt(1+r[1][1]-r[0][0]-r[2][2])*2
        y = scale/4
        w, x, z = (r[0][2]-r[2][0])/scale, (r[0][1]+r[1][0])/scale, (r[1][2]+r[2][1])/scale
    else:
        scale = math.sqrt(1+r[2][2]-r[0][0]-r[1][1])*2
        z = scale/4
        w, x, y = (r[1][0]-r[0][1])/scale, (r[0][2]+r[2][0])/scale, (r[1][2]+r[2][1])/scale
    return tuple(row[3] for row in rows[:3]), quaternion_xyzw((w, x, y, z))


@dataclass(frozen=True)
class PoseSample:
    stamp_seconds: float
    position_ned: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    position_timestamp_us: int
    attitude_timestamps_us: tuple[int, ...]
    attitude_method: str


class Px4PoseBuffer:
    """Match actual PX4 samples, clearing interpolation history on a reset."""

    def __init__(self, parameters=None):
        self.p = {**DEFAULTS, **(parameters or {})}
        self.positions = deque(maxlen=int(self.p["max_buffer_samples"]))
        self.attitudes = deque(maxlen=int(self.p["max_buffer_samples"]))
        self.previous = {}
        self.last_published_stamp = None
        self.reset_count = 0

    def _clear(self):
        self.positions.clear()
        self.attitudes.clear()

    def _sample_stamp(self, source, timestamp, counters, now):
        if timestamp <= 0:
            return None
        previous = self.previous.get(source)
        if previous is not None:
            old_timestamp, old_counters = previous
            if timestamp < old_timestamp or counters != old_counters:
                self._clear()
                self.previous.clear()
                self.last_published_stamp = None
                self.reset_count += 1
            elif timestamp == old_timestamp:
                return None
        self.previous[source] = (timestamp, counters)
        stamp = timestamp*1e-6 + self.p["timestamp_offset_seconds"]
        if abs(now-stamp) > self.p["max_clock_skew_seconds"]:
            self._clear()
            return None
        return stamp

    def add_position(self, message, now):
        stamp = self._sample_stamp("position", message.timestamp_sample,
                                   (message.xy_reset_counter, message.z_reset_counter), now)
        if stamp is None:
            return False
        if not message.xy_valid or not message.z_valid:
            self._clear()
            return False
        try:
            point = _finite((message.x, message.y, message.z))
        except ValueError:
            self._clear()
            return False
        self.positions.append((stamp, message.timestamp_sample, point))
        self._trim(stamp)
        return True

    def add_attitude(self, message, now):
        stamp = self._sample_stamp("attitude", message.timestamp_sample,
                                   (message.quat_reset_counter,), now)
        if stamp is None:
            return False
        try:
            quaternion = quaternion_xyzw(message.q)
        except ValueError:
            self._clear()
            return False
        self.attitudes.append((stamp, message.timestamp_sample, quaternion))
        self._trim(stamp)
        return True

    def _trim(self, stamp):
        cutoff = stamp-self.p["buffer_seconds"]
        for buffer in (self.positions, self.attitudes):
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

    def _attitude_at(self, stamp):
        before = after = None
        for sample in self.attitudes:
            if sample[0] <= stamp:
                before = sample
            if sample[0] >= stamp:
                after = sample
                break
        tolerance = self.p["pose_sync_tolerance_seconds"]
        if before is not None and after is not None:
            if before[0] == after[0]:
                return before[2], (before[1],), "exact"
            if max(stamp-before[0], after[0]-stamp) <= tolerance + 1e-9:
                alpha = (stamp-before[0])/(after[0]-before[0])
                return _slerp(before[2], after[2], alpha), (before[1], after[1]), "interpolated"
        candidates = [sample for sample in (before, after) if sample is not None]
        if candidates:
            nearest = min(candidates, key=lambda sample: abs(sample[0]-stamp))
            if abs(nearest[0]-stamp) <= tolerance + 1e-9:
                return nearest[2], (nearest[1],), "nearest"
        return None

    def take_ready(self, now):
        ready, pending = [], []
        for stamp, timestamp, point in self.positions:
            if self.last_published_stamp is not None and stamp <= self.last_published_stamp:
                continue
            if abs(now-stamp) > self.p["max_clock_skew_seconds"]:
                continue
            attitude = self._attitude_at(stamp)
            if attitude is None:
                pending.append((stamp, timestamp, point))
                continue
            quaternion, attitude_times, method = attitude
            ready.append(PoseSample(stamp, point, quaternion, timestamp, attitude_times, method))
            self.last_published_stamp = stamp
        self.positions.clear()
        self.positions.extend(pending)
        return ready
