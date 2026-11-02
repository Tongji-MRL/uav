"""Conservative task-specific fallback; this is not an electrical OFF sensor."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SteadyNonBlueOff:
    """Require continuous, stable RED or GREEN measurements over two cycles.

    Missing, ambiguous, blue, saturated or widely spaced samples invalidate
    evidence. Raw luminance is checked even if the categorical color is stuck.
    The caller resets this object when the association/identity changes.
    """

    enabled: bool = False
    seconds: float = 4.0
    min_samples: int = 16
    max_gap_seconds: float = 0.65
    min_color_confidence: float = 0.55
    absolute_signal_range: float = 0.025
    relative_signal_range: float = 0.12
    max_neutral_clip_ratio: float = 0.50
    blink_cooldown_seconds: float = 2.2
    max_center_motion_ratio: float = 0.25
    active: bool = False
    reason: str = "disabled"
    span: float = 0.0
    signal_range: float = 0.0
    label: str = "UNKNOWN"
    _samples: deque = field(default_factory=deque, repr=False)
    _last_blink: Optional[float] = field(default=None, repr=False)
    _anchor: Optional[tuple] = field(default=None, repr=False)

    def __post_init__(self):
        if self.seconds < 3.0 or self.min_samples < 8:
            raise ValueError("steady OFF requires >=3 seconds and >=8 samples")
        if not 0 < self.max_gap_seconds < 1.0:
            raise ValueError("steady OFF max gap must be below one dark phase")
        for name in ("min_color_confidence", "relative_signal_range",
                     "max_neutral_clip_ratio", "absolute_signal_range",
                     "max_center_motion_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.blink_cooldown_seconds < 0:
            raise ValueError("blink cooldown must be nonnegative")

    def reset(self):
        self._samples.clear()
        self._last_blink = None
        self._anchor = None
        self.active = False
        self.span = self.signal_range = 0.0
        self.label = "UNKNOWN"
        self.reason = "warmup" if self.enabled else "disabled"

    def update(self, timestamp, observation, temporal, bbox=None):
        self.active = False
        if not self.enabled:
            return False
        if observation is None:
            self.reset()
            self.reason = "not_measured"
            return False
        if self._samples and (
            timestamp <= self._samples[-1][0]
            or timestamp - self._samples[-1][0] > self.max_gap_seconds
        ):
            self.reset()
        if temporal.state.startswith("BLINK_"):
            self._last_blink = timestamp
        if (
            observation.label not in ("RED", "GREEN")
            or observation.confidence < self.min_color_confidence
            or not np.isfinite(observation.luminance_signal)
            or observation.neutral_clip_ratio > self.max_neutral_clip_ratio
        ):
            last_blink = self._last_blink
            self.reset()
            self._last_blink = last_blink
            self.reason = "not_reliable_red_green"
            return False
        if observation.label != self.label:
            self._samples.clear()
            self._anchor = None
        if bbox is not None and self._anchor is not None:
            a = self._anchor
            scale = max(1, min(a[2] - a[0], a[3] - a[1],
                               bbox[2] - bbox[0], bbox[3] - bbox[1]))
            motion = np.hypot(bbox[0] + bbox[2] - a[0] - a[2],
                              bbox[1] + bbox[3] - a[1] - a[3]) / (2 * scale)
            if motion > self.max_center_motion_ratio:
                self._samples.clear()
                self._anchor = None
        if self._anchor is None:
            self._anchor = bbox
        self.label = observation.label
        self._samples.append((timestamp, observation.luminance_signal))
        # Keep a sample at/before the left edge, for arbitrary frame rates.
        while len(self._samples) > 1 and self._samples[1][0] <= timestamp - self.seconds:
            self._samples.popleft()
        self.span = timestamp - self._samples[0][0]
        self.reason = "warming_up"
        if self.span + 1e-6 < self.seconds or len(self._samples) < self.min_samples:
            return False
        values = np.array([item[1] for item in self._samples])
        # Full range intentionally rejects even brief ON/OFF changes.
        self.signal_range = float(np.max(values) - np.min(values))
        limit = min(self.absolute_signal_range,
                    max(0.005, abs(float(np.median(values))) * self.relative_signal_range))
        if self.signal_range > limit:
            self.reason = "brightness_is_changing"
            return False
        if self._last_blink is not None and timestamp - self._last_blink < self.blink_cooldown_seconds:
            self.reason = "recent_blink"
            return False
        self.active = True
        self.reason = "steady_nonblue_task_rule"
        return True
