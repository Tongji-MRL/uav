"""ROS-independent tracking, color and temporal logic for RobotX Task 1."""

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import hypot
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .steady_off import SteadyNonBlueOff
from .aperture_photometry import AperturePhotometer
from .emitter_fusion import EmitterFusion
from .photometric_signal import PhotometricSignalDecoder


VALID_STATES = (
    "OFF",
    "BLINK_RED",
    "BLINK_GREEN",
    "BLINK_BLUE",
    "SOLID_BLUE",
    "UNKNOWN",
)
LIT_COLORS = ("RED", "GREEN", "BLUE")
DARK_OBSERVATIONS = ("DARK", "OFF", "MISSING")
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int = 0


@dataclass
class InstantColor:
    label: str
    confidence: float
    ratios: Dict[str, float]
    value_threshold: int
    raw_ratios: Dict[str, float] = field(default_factory=dict)
    background_ratios: Dict[str, float] = field(default_factory=dict)
    brightness_contrast: Dict[str, float] = field(default_factory=dict)
    channel_shares: Dict[str, float] = field(default_factory=dict)
    center_value_p90: float = 0.0
    center_saturation_p75: float = 0.0
    emission_strength_p75: float = 0.0
    background_emission_p75: float = 0.0
    emission_contrast: float = 0.0
    neutral_clip_ratio: float = 0.0
    color_value_threshold: float = 0.0
    decision_reason: str = "uninitialized"
    luminance_signal: float = 0.0
    emitter_center_x: float = 0.5
    emitter_center_y: float = 0.5
    emitter_core_value: float = 0.0
    emitter_background_value: float = 0.0
    emitter_chroma_fraction: float = 0.0
    hue_shares: Dict[str, float] = field(default_factory=dict)
    required_signal: float = 0.0
    photometric_phase: str = "LEGACY"
    photometry_mode: str = "LEGACY"
    flux_signed: float = 0.0
    flux_snr: float = 0.0
    spatial_support: float = 0.0
    peak_excess: float = 0.0
    color_mass: float = 0.0
    color_energy: Dict[str, float] = field(default_factory=dict)
    measurement_quality: float = 0.0
    aperture_radius_px: float = 0.0
    emitter_center_source: str = "legacy"
    emission_diagnostics: Dict[str, object] = field(default_factory=dict)



@dataclass
class TemporalResult:
    state: str = "UNKNOWN"
    confidence: float = 0.0
    dominant_color: str = "UNKNOWN"
    on_ratio: float = 0.0
    transitions: int = 0
    blink_hz: float = 0.0
    history_span: float = 0.0
    method: str = "transitions"
    template_agreement: float = 0.0


@dataclass
class BlinkPhotometricReference:
    """Per-track ON/OFF reference learned only from a trusted blink."""

    enabled: bool = False
    min_samples: int = 12
    min_agreement: float = 0.82
    min_signal_separation: float = 0.045
    low_percentile: float = 20.0
    high_percentile: float = 82.0
    ema_alpha: float = 0.20
    dark_enter_ratio: float = 0.38
    lit_exit_ratio: float = 0.62

    ready: bool = False
    off_level: float = 0.0
    on_level: float = 0.0
    updates: int = 0
    normalized_signal: float = float("nan")
    dark_latched: bool = False
    source_state: str = "UNKNOWN"
    levels_by_color: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _gated_color: str = field(default="UNKNOWN", repr=False)

    def __post_init__(self) -> None:
        self.min_samples = max(6, int(self.min_samples))
        for name in (
            "min_agreement", "ema_alpha", "dark_enter_ratio",
            "lit_exit_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.dark_enter_ratio >= self.lit_exit_ratio:
            raise ValueError(
                "dark_enter_ratio must be below lit_exit_ratio"
            )
        if self.min_signal_separation <= 0.0:
            raise ValueError("min_signal_separation must be positive")

    def reset(self) -> None:
        self.ready = False
        self.off_level = 0.0
        self.on_level = 0.0
        self.updates = 0
        self.normalized_signal = float("nan")
        self.dark_latched = False
        self.source_state = "UNKNOWN"
        self.levels_by_color.clear()
        self._gated_color = "UNKNOWN"

    def learn(
        self,
        analyzer: "TemporalAnalyzer",
        temporal: TemporalResult,
    ) -> bool:
        """Update robust levels only after a high-agreement 0.5 Hz blink."""
        if (
            not self.enabled
            or not temporal.state.startswith("BLINK_")
            or temporal.template_agreement < self.min_agreement
        ):
            return False
        values = np.asarray([
            signal
            for _, signal, color, _ in analyzer.photometric_samples
            if color != "MISSING" and np.isfinite(signal)
        ], dtype=np.float64)
        if values.size < self.min_samples:
            return False
        low = float(np.percentile(values, self.low_percentile))
        high = float(np.percentile(values, self.high_percentile))
        if high - low < self.min_signal_separation:
            return False
        color = temporal.state[len("BLINK_"):]
        previous_levels = self.levels_by_color.get(color)
        if previous_levels is not None:
            alpha = self.ema_alpha
            self.off_level = (1.0 - alpha) * previous_levels[0] + alpha * low
            self.on_level = (1.0 - alpha) * previous_levels[1] + alpha * high
        else:
            self.off_level = low
            self.on_level = high
        self.ready = True
        self.updates += 1
        self.source_state = temporal.state
        self.levels_by_color[color] = (self.off_level, self.on_level)
        return True

    def gate(self, observation: InstantColor) -> InstantColor:
        """Convert a chromatic dark phase to DARK using learned contrast."""
        signal = float(observation.luminance_signal)
        if not self.enabled or not self.ready or not np.isfinite(signal):
            self.normalized_signal = float("nan")
            return observation
        color = observation.label
        reference_color = self.source_state.removeprefix("BLINK_")
        if color in LIT_COLORS:
            levels = self.levels_by_color.get(color)
            if levels is not None:
                off_level, on_level = levels
            elif reference_color in LIT_COLORS and color != reference_color:
                # A real blue LED can be dimmer than the red OFF reference.
                self.normalized_signal = float("nan")
                self.dark_latched = False
                return observation
            else:
                off_level, on_level = self.off_level, self.on_level
        else:
            off_level, on_level = self.off_level, self.on_level
        if color != self._gated_color:
            self.dark_latched = False
            self._gated_color = color
        separation = max(
            on_level - off_level,
            self.min_signal_separation,
        )
        normalized = (signal - off_level) / separation
        self.normalized_signal = float(normalized)
        if self.dark_latched:
            if normalized >= self.lit_exit_ratio:
                self.dark_latched = False
        elif normalized <= self.dark_enter_ratio:
            self.dark_latched = True
        if not self.dark_latched:
            return observation
        darkness = float(np.clip(
            1.0 - normalized / max(self.lit_exit_ratio, 1e-6),
            0.0,
            1.0,
        ))
        return replace(
            observation,
            label="DARK",
            confidence=max(observation.confidence, darkness),
            decision_reason=(
                "blink_reference_dark_from_"
                f"{observation.label.lower()}"
            ),
        )


@dataclass
class StateDecision:
    """Fast candidate plus conservative committed Task 1 state."""

    enabled: bool = False
    candidate_min_frames: int = 2
    candidate_min_seconds: float = 0.12
    candidate_hold_seconds: float = 1.20
    commit_min_frames: int = 2
    commit_confidence_min: float = 0.55
    change_pending_seconds: float = 0.30
    change_clear_seconds: float = 0.40
    blink_phase_timeout_seconds: float = 1.30

    candidate_state: str = "UNKNOWN"
    candidate_confidence: float = 0.0
    candidate_since: Optional[float] = None
    candidate_frames: int = 0
    candidate_last_evidence_at: Optional[float] = None
    committed_state: str = "UNKNOWN"
    committed_confidence: float = 0.0
    committed_since: Optional[float] = None
    previous_committed_state: str = "UNKNOWN"
    change_pending: bool = False
    pending_since: Optional[float] = None
    change_detected_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    decision_latency_seconds: float = 0.0
    state_revision: int = 0
    state_changed: bool = False
    source: str = "disabled"
    change_reason: str = ""

    _proposal_state: str = field(default="UNKNOWN", repr=False)
    _proposal_since: Optional[float] = field(default=None, repr=False)
    _proposal_frames: int = field(default=0, repr=False)
    _commit_state: str = field(default="UNKNOWN", repr=False)
    _commit_frames: int = field(default=0, repr=False)
    _observation_label: str = field(default="UNKNOWN", repr=False)
    _observation_since: Optional[float] = field(default=None, repr=False)
    _anomaly_since: Optional[float] = field(default=None, repr=False)
    _clear_since: Optional[float] = field(default=None, repr=False)
    _last_timestamp: Optional[float] = field(default=None, repr=False)
    _history_reset_requested: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.candidate_min_frames = max(1, int(self.candidate_min_frames))
        self.commit_min_frames = max(1, int(self.commit_min_frames))
        for name in (
            "candidate_min_seconds",
            "candidate_hold_seconds",
            "change_pending_seconds",
            "change_clear_seconds",
            "blink_phase_timeout_seconds",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.commit_confidence_min <= 1.0:
            raise ValueError("commit_confidence_min must be in [0, 1]")
        if self.enabled:
            self.source = "warmup"

    def reset(
        self,
        timestamp: Optional[float] = None,
        identity_reset: bool = False,
    ) -> None:
        """Clear semantic evidence without reusing a stale committed state."""
        old_state = self.committed_state
        old_revision = self.state_revision
        self.candidate_state = "UNKNOWN"
        self.candidate_confidence = 0.0
        self.candidate_since = None
        self.candidate_frames = 0
        self.candidate_last_evidence_at = None
        self.committed_state = "UNKNOWN"
        self.committed_confidence = 0.0
        self.committed_since = None
        self.previous_committed_state = old_state
        self.change_pending = False
        self.pending_since = None
        self.change_detected_at = None
        self.confirmed_at = timestamp
        self.decision_latency_seconds = 0.0
        self.state_changed = False
        self._proposal_state = "UNKNOWN"
        self._proposal_since = None
        self._proposal_frames = 0
        self._commit_state = "UNKNOWN"
        self._commit_frames = 0
        self._observation_label = "UNKNOWN"
        self._observation_since = None
        self._anomaly_since = None
        self._clear_since = None
        self._last_timestamp = timestamp
        self._history_reset_requested = False
        if identity_reset and old_state != "UNKNOWN":
            # A retained image track may now refer to another physical box.
            # Preserve monotonicity while explicitly invalidating its state.
            self.state_revision = old_revision + 1
            self.source = "identity_reset"
        else:
            self.state_revision = old_revision
            self.source = "warmup" if self.enabled else "disabled"

    def update(
        self,
        timestamp: float,
        observation_label: str,
        observation_confidence: float,
        temporal: TemporalResult,
        visible: bool,
    ) -> None:
        """Update candidate, change warning and committed state."""
        self.state_changed = False
        self._history_reset_requested = False
        if not self.enabled:
            return
        if (
            self._last_timestamp is not None
            and timestamp + 1e-6 < self._last_timestamp
        ):
            self.reset(timestamp=timestamp, identity_reset=True)
        self._last_timestamp = timestamp
        self._update_observation_run(timestamp, observation_label, visible)

        proposal, confidence, proposal_source = self._proposal(
            observation_label,
            observation_confidence,
            temporal,
            visible,
        )
        self._update_candidate(
            timestamp, proposal, confidence, proposal_source
        )
        anomaly, reason, anomaly_since = self._change_anomaly(
            timestamp, observation_label, visible
        )
        self._update_change_pending(
            timestamp, anomaly, reason, anomaly_since
        )
        self._update_commit(timestamp, temporal, observation_label, visible)

    def consume_history_reset_request(self) -> bool:
        requested = self._history_reset_requested
        self._history_reset_requested = False
        return requested

    def _update_observation_run(
        self, timestamp: float, label: str, visible: bool
    ) -> None:
        if not visible or label in ("UNKNOWN", "MISSING"):
            return
        phase = "DARK" if label in ("DARK", "OFF") else label
        if phase != self._observation_label:
            self._observation_label = phase
            self._observation_since = timestamp

    def _proposal(
        self,
        label: str,
        confidence: float,
        temporal: TemporalResult,
        visible: bool,
    ) -> Tuple[str, float, str]:
        direct = {
            "RED": "BLINK_RED",
            "GREEN": "BLINK_GREEN",
        }.get(label)
        if visible and direct is not None:
            if direct != self.committed_state:
                return direct, confidence, "instant_color"
        if temporal.state in VALID_STATES and temporal.state != "UNKNOWN":
            return temporal.state, temporal.confidence, "temporal"
        if visible and direct is not None:
            return direct, confidence, "instant_color"
        if visible and label == "BLUE":
            if self.committed_state in ("BLINK_BLUE", "SOLID_BLUE"):
                return self.committed_state, confidence, "blue_hold"
            return "BLUE_PENDING", confidence, "blue_ambiguous"
        return "UNKNOWN", 0.0, "no_reliable_candidate"

    def _update_candidate(
        self,
        timestamp: float,
        proposal: str,
        confidence: float,
        source: str,
    ) -> None:
        if proposal != "UNKNOWN":
            if proposal == self._proposal_state:
                self._proposal_frames += 1
            else:
                self._proposal_state = proposal
                self._proposal_since = timestamp
                self._proposal_frames = 1
            elapsed = timestamp - (
                self._proposal_since
                if self._proposal_since is not None else timestamp
            )
            if (
                self._proposal_frames >= self.candidate_min_frames
                and elapsed + 1e-9 >= self.candidate_min_seconds
            ):
                if self.candidate_state != proposal:
                    self.candidate_since = self._proposal_since
                    self.candidate_frames = self._proposal_frames
                else:
                    self.candidate_frames += 1
                self.candidate_state = proposal
                self.candidate_confidence = float(np.clip(
                    confidence, 0.0, 1.0
                ))
                self.candidate_last_evidence_at = timestamp
                self.source = source
            return

        self._proposal_state = "UNKNOWN"
        self._proposal_since = None
        self._proposal_frames = 0
        if (
            self.candidate_last_evidence_at is not None
            and timestamp - self.candidate_last_evidence_at
            > self.candidate_hold_seconds
        ):
            self.candidate_state = "UNKNOWN"
            self.candidate_confidence = 0.0
            self.candidate_since = None
            self.candidate_frames = 0

    @staticmethod
    def _candidate_compatible(candidate: str, committed: str) -> bool:
        if candidate == committed:
            return True
        return (
            candidate == "BLUE_PENDING"
            and committed in ("BLINK_BLUE", "SOLID_BLUE")
        )

    def _change_anomaly(
        self, timestamp: float, label: str, visible: bool
    ) -> Tuple[bool, str, Optional[float]]:
        if self.committed_state == "UNKNOWN" or not visible:
            return False, "", None
        if (
            self.candidate_state != "UNKNOWN"
            and not self._candidate_compatible(
                self.candidate_state, self.committed_state
            )
        ):
            return True, "candidate_differs", self.candidate_since

        phase_since = self._observation_since
        phase_seconds = (
            timestamp - phase_since if phase_since is not None else 0.0
        )
        if self.committed_state == "OFF" and label in LIT_COLORS:
            return True, "off_became_lit", phase_since
        if self.committed_state == "SOLID_BLUE" and label in (
            "DARK", "OFF", "RED", "GREEN"
        ):
            return True, "solid_blue_changed", phase_since
        if self.committed_state.startswith("BLINK_"):
            expected_color = self.committed_state[len("BLINK_"):]
            if label in LIT_COLORS and label != expected_color:
                return True, "blink_color_changed", phase_since
            if (
                label in (expected_color, "DARK", "OFF")
                and phase_seconds > self.blink_phase_timeout_seconds
            ):
                return True, "blink_phase_timeout", phase_since
        return False, "", None

    def _update_change_pending(
        self,
        timestamp: float,
        anomaly: bool,
        reason: str,
        anomaly_since: Optional[float],
    ) -> None:
        if anomaly:
            self._clear_since = None
            if self._anomaly_since is None:
                self._anomaly_since = (
                    anomaly_since if anomaly_since is not None else timestamp
                )
            if (
                timestamp - self._anomaly_since
                + 1e-9 >= self.change_pending_seconds
            ):
                if not self.change_pending:
                    self.pending_since = self._anomaly_since
                    self.change_detected_at = timestamp
                    self._history_reset_requested = True
                self.change_pending = True
                self.change_reason = reason
            return

        self._anomaly_since = None
        if not self.change_pending:
            self.change_reason = ""
            return
        if self._clear_since is None:
            self._clear_since = timestamp
        if timestamp - self._clear_since >= self.change_clear_seconds:
            self.change_pending = False
            self.pending_since = None
            self.change_detected_at = None
            self.change_reason = ""
            self._clear_since = None

    def _update_commit(
        self,
        timestamp: float,
        temporal: TemporalResult,
        observation_label: str,
        visible: bool,
    ) -> None:
        state = temporal.state
        valid = (
            state in VALID_STATES
            and state != "UNKNOWN"
            and temporal.confidence >= self.commit_confidence_min
            and not (
                state == "OFF"
                and (not visible or observation_label == "MISSING")
            )
        )
        if not valid:
            self._commit_state = "UNKNOWN"
            self._commit_frames = 0
            return
        if state == self._commit_state:
            self._commit_frames += 1
        else:
            self._commit_state = state
            self._commit_frames = 1
        if self._commit_frames < self.commit_min_frames:
            return

        if state == self.committed_state:
            self.committed_confidence = max(
                self.committed_confidence, temporal.confidence
            )
            return
        old_state = self.committed_state
        self.previous_committed_state = old_state
        self.committed_state = state
        self.committed_confidence = temporal.confidence
        self.committed_since = timestamp
        self.confirmed_at = timestamp
        self.state_revision += 1
        self.state_changed = old_state != "UNKNOWN"
        self.decision_latency_seconds = (
            timestamp - self.pending_since
            if self.pending_since is not None else 0.0
        )
        self.candidate_state = state
        self.candidate_confidence = temporal.confidence
        self.candidate_since = timestamp
        self.candidate_frames = self._commit_frames
        self.candidate_last_evidence_at = timestamp
        self.change_pending = False
        self.pending_since = None
        self.change_detected_at = None
        self.change_reason = ""
        self._anomaly_since = None
        self._clear_since = None
        self.source = (
            "initial_commit" if old_state == "UNKNOWN"
            else "temporal_commit"
        )


class HsvColorClassifier:
    """Measure local colored emission against the surrounding background."""

    def __init__(self, **parameters) -> None:
        self.mode = str(parameters.get("mode", "hsv"))
        self.photometer = AperturePhotometer(parameters) if self.mode == "aperture_v7_4" else None
        self.emitter_fusion = EmitterFusion(parameters) if self.mode == "emitter_fusion_v7_5" else None
        self.saturation_min = int(parameters["saturation_min"])
        self.value_min = int(parameters["value_min"])
        self.value_percentile = float(parameters["value_percentile"])
        self.min_color_ratio = float(parameters["min_color_ratio"])
        self.min_color_margin = float(parameters["min_color_margin"])
        self.background_border_ratio = float(
            parameters["background_border_ratio"]
        )
        self.background_subtract_weight = float(
            parameters["background_subtract_weight"]
        )
        self.min_component_ratio = float(
            parameters["min_component_ratio"]
        )
        self.min_brightness_contrast = float(
            parameters["min_brightness_contrast"]
        )
        self.red_low_max = int(parameters["red_low_max"])
        self.red_high_min = int(parameters["red_high_min"])
        self.green_min = int(parameters["green_min"])
        self.green_max = int(parameters["green_max"])
        self.blue_min = int(parameters["blue_min"])
        self.blue_max = int(parameters["blue_max"])
        self.channel_dominance_min = float(
            parameters.get("channel_dominance_min", 0.03)
        )
        self.chromaticity_spread_min = float(
            parameters.get("chromaticity_spread_min", 0.06)
        )
        self.center_region_ratio = float(
            parameters.get("center_region_ratio", 0.56)
        )
        self.center_saturation_percentile = float(
            parameters.get("center_saturation_percentile", 75.0)
        )
        self.center_value_percentile = float(
            parameters.get("center_value_percentile", 90.0)
        )
        self.min_center_saturation = float(
            parameters.get("min_center_saturation", 80.0)
        )
        self.min_emission_strength = float(
            parameters.get("min_emission_strength", 20.0)
        )
        self.min_center_color_fraction = float(
            parameters.get("min_center_color_fraction", 0.10)
        )
        self.strict_hue_share_min = float(
            parameters.get("strict_hue_share_min", 0.80)
        )
        self.strict_hue_margin_min = float(
            parameters.get("strict_hue_margin_min", 0.65)
        )
        self.strict_channel_share_min = float(
            parameters.get("strict_channel_share_min", 0.45)
        )
        self.strict_channel_margin_min = float(
            parameters.get("strict_channel_margin_min", 0.08)
        )
        self.color_brightness_percentile = float(
            parameters.get("color_brightness_percentile", 0.0)
        )
        self.radial_weight_strength = float(
            parameters.get("radial_weight_strength", 0.0)
        )
        self.blue_min_center_saturation = float(
            parameters.get("blue_min_center_saturation", 0.0)
        )
        self.blue_min_emission_strength = float(
            parameters.get("blue_min_emission_strength", 0.0)
        )
        self.blue_strict_hue_share_min = float(
            parameters.get(
                "blue_strict_hue_share_min", self.strict_hue_share_min
            )
        )
        self.blue_strict_channel_share_min = float(
            parameters.get(
                "blue_strict_channel_share_min",
                self.strict_channel_share_min,
            )
        )
        self.max_neutral_clip_ratio_for_blue = float(
            parameters.get("max_neutral_clip_ratio_for_blue", 1.0)
        )
        self.weak_blue_as_dark = bool(
            parameters.get("weak_blue_as_dark", False)
        )
        self.emitter_search_ratio = float(
            parameters.get("emitter_search_ratio", 0.76)
        )
        self.emitter_core_radius_ratio = float(
            parameters.get("emitter_core_radius_ratio", 0.075)
        )
        self.emitter_hint_radius_ratio = float(
            parameters.get("emitter_hint_radius_ratio", 0.18)
        )
        self.emitter_ring_inner_ratio = float(
            parameters.get("emitter_ring_inner_ratio", 0.055)
        )
        self.emitter_ring_outer_ratio = float(
            parameters.get("emitter_ring_outer_ratio", 0.26)
        )
        self.emitter_color_percentile = float(
            parameters.get("emitter_color_percentile", 58.0)
        )
        self.emitter_min_signal = float(
            parameters.get("emitter_min_signal", 0.055)
        )
        self.emitter_min_chroma_fraction = float(
            parameters.get("emitter_min_chroma_fraction", 0.012)
        )
        self.emitter_hue_share_min = float(
            parameters.get("emitter_hue_share_min", 0.62)
        )
        self.emitter_hue_margin_min = float(
            parameters.get("emitter_hue_margin_min", 0.24)
        )
        self.emitter_channel_share_min = float(
            parameters.get("emitter_channel_share_min", 0.43)
        )
        self.emitter_channel_margin_min = float(
            parameters.get("emitter_channel_margin_min", 0.06)
        )
        self.emitter_blue_share_min = float(
            parameters.get("emitter_blue_share_min", 0.78)
        )
        self.emitter_blue_signal_min = float(
            parameters.get("emitter_blue_signal_min", 0.115)
        )
        self.emitter_red_signal_min = float(parameters.get(
            "emitter_red_signal_min", self.emitter_min_signal
        ))
        self.emitter_green_signal_min = float(parameters.get(
            "emitter_green_signal_min", self.emitter_min_signal
        ))
        # The legacy hue thresholds gated 55% HSV + 45% RGB, not pure hue.
        self.emitter_color_share_min = float(parameters.get(
            "emitter_color_share_min", -1.0
        ))
        self.emitter_color_margin_min = float(parameters.get(
            "emitter_color_margin_min", -1.0
        ))
        if self.emitter_color_share_min < 0:
            self.emitter_color_share_min = self.emitter_hue_share_min
        if self.emitter_color_margin_min < 0:
            self.emitter_color_margin_min = self.emitter_hue_margin_min
        self.emitter_blue_chroma_min = float(
            parameters.get("emitter_blue_chroma_min", 0.18)
        )
        self.emitter_blue_core_contrast_min = float(
            parameters.get("emitter_blue_core_contrast_min", 12.0)
        )
        self.emitter_fragmented_signal_min = float(
            parameters.get("emitter_fragmented_signal_min", 0.20)
        )
        self.emitter_blue_core_bypass_signal = float(
            parameters.get("emitter_blue_core_bypass_signal", 0.30)
        )
        if self.mode not in (
            "hsv", "adaptive_fusion", "center_led_strict",
            "emitter_ring_v7", "aperture_v7_4", "emitter_fusion_v7_5",
        ):
            raise ValueError(
                "color mode must be hsv, adaptive_fusion, "
                "center_led_strict, emitter_ring_v7, aperture_v7_4 or emitter_fusion_v7_5"
            )
        if not 0.0 < self.background_border_ratio < 0.4:
            raise ValueError(
                "background_border_ratio must be in the interval (0, 0.4)"
            )
        if self.background_subtract_weight < 0.0:
            raise ValueError("background_subtract_weight must be nonnegative")
        if self.min_component_ratio <= 0.0:
            raise ValueError("min_component_ratio must be positive")
        if self.min_brightness_contrast < 0.0:
            raise ValueError("min_brightness_contrast must be nonnegative")
        if not 0.2 <= self.center_region_ratio <= 0.9:
            raise ValueError("center_region_ratio must be in [0.2, 0.9]")
        for name, value in (
            ("min_center_color_fraction", self.min_center_color_fraction),
            ("strict_hue_share_min", self.strict_hue_share_min),
            ("strict_hue_margin_min", self.strict_hue_margin_min),
            ("strict_channel_share_min", self.strict_channel_share_min),
            ("strict_channel_margin_min", self.strict_channel_margin_min),
            ("blue_strict_hue_share_min", self.blue_strict_hue_share_min),
            (
                "blue_strict_channel_share_min",
                self.blue_strict_channel_share_min,
            ),
            (
                "max_neutral_clip_ratio_for_blue",
                self.max_neutral_clip_ratio_for_blue,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.color_brightness_percentile <= 95.0:
            raise ValueError(
                "color_brightness_percentile must be in [0, 95]"
            )
        if self.radial_weight_strength < 0.0:
            raise ValueError("radial_weight_strength must be nonnegative")
        if not 0.3 <= self.emitter_search_ratio <= 0.95:
            raise ValueError("emitter_search_ratio must be in [0.3, 0.95]")
        if not (
            0.0 < self.emitter_ring_inner_ratio
            < self.emitter_ring_outer_ratio < 0.5
        ):
            raise ValueError("emitter ring radii are inconsistent")
        if not 0.0 < self.emitter_core_radius_ratio < 0.3:
            raise ValueError(
                "emitter_core_radius_ratio must be in (0, 0.3)"
            )
        if not 0.05 <= self.emitter_hint_radius_ratio <= 0.4:
            raise ValueError(
                "emitter_hint_radius_ratio must be in [0.05, 0.4]"
            )
        if not 0.0 <= self.emitter_color_percentile <= 95.0:
            raise ValueError(
                "emitter_color_percentile must be in [0, 95]"
            )

    def classify(
        self,
        roi: np.ndarray,
        center_hint: Optional[Tuple[float, float]] = None,
        lock_emitter_center: bool = False,
    ) -> InstantColor:
        if roi is None or roi.size == 0:
            return self._empty()
        if self.photometer is not None:
            return InstantColor(**self.photometer.measure(roi, center_hint))
        scale = 1.0
        if self.emitter_fusion is not None:
            roi, scale = self.emitter_fusion.bound(roi)
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        if self.mode in ("emitter_ring_v7", "emitter_fusion_v7_5"):
            base = self._classify_emitter_ring(blurred, hsv, center_hint=center_hint,
                                               lock_emitter_center=lock_emitter_center)
            return self.emitter_fusion.refine(blurred, hsv, base, scale) if self.emitter_fusion else base
        if self.mode == "center_led_strict":
            return self._classify_center_led_strict(blurred, hsv)
        hue, saturation, value = cv2.split(hsv)
        inner, background = self._region_masks(value.shape)
        inner_values = value[inner]
        background_values = value[background]
        if inner_values.size == 0 or background_values.size == 0:
            return self._empty()

        background_level = float(np.percentile(
            background_values, self.value_percentile
        ))
        threshold = min(
            255,
            max(
                self.value_min,
                int(np.percentile(
                    inner_values, self.value_percentile
                )),
                int(round(
                    background_level + self.min_brightness_contrast
                )),
            ),
        )
        valid = (saturation >= self.saturation_min) & (value >= threshold)
        hsv_masks = {
            "RED": valid & (
                (hue <= self.red_low_max) | (hue >= self.red_high_min)
            ),
            "GREEN": (
                valid & (hue >= self.green_min) & (hue <= self.green_max)
            ),
            "BLUE": (
                valid & (hue >= self.blue_min) & (hue <= self.blue_max)
            ),
        }
        masks = hsv_masks
        if self.mode == "adaptive_fusion":
            masks = self._adaptive_color_masks(
                blurred, value >= threshold, hsv_masks
            )

        inner_area = float(np.count_nonzero(inner))
        background_area = float(np.count_nonzero(background))
        raw_ratios = {}
        background_ratios = {}
        contrasts = {}
        scores = {}
        for name, mask in masks.items():
            foreground_mask = mask & inner
            background_mask = mask & background
            foreground_ratio = (
                float(np.count_nonzero(foreground_mask)) / inner_area
            )
            background_ratio = (
                float(np.count_nonzero(background_mask)) / background_area
            )
            component_ratio = self._largest_component_ratio(
                foreground_mask, inner_area
            )
            foreground_values = value[foreground_mask]
            foreground_level = (
                float(np.percentile(foreground_values, 75.0))
                if foreground_values.size
                else 0.0
            )
            contrast = max(0.0, foreground_level - background_level)
            excess_ratio = max(
                0.0,
                foreground_ratio
                - self.background_subtract_weight * background_ratio,
            )
            component_support = min(
                1.0, component_ratio / self.min_component_ratio
            )
            contrast_support = 0.75 + 0.25 * min(
                1.0,
                contrast / max(self.min_brightness_contrast, 1.0),
            )
            raw_ratios[name] = foreground_ratio
            background_ratios[name] = background_ratio
            contrasts[name] = contrast
            scores[name] = (
                excess_ratio * component_support * contrast_support
            )

        ordered = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        candidate, best = ordered[0]
        margin = best - ordered[1][1]
        if best < self.min_color_ratio:
            label = "DARK"
            confidence = float(np.clip(
                1.0 - best / max(self.min_color_ratio, 1e-6),
                0.0,
                1.0,
            ))
        elif margin < self.min_color_margin:
            label = "UNKNOWN"
            confidence = float(np.clip(
                margin / max(self.min_color_margin, 1e-6),
                0.0,
                1.0,
            ))
        else:
            label = candidate
            confidence = float(np.clip(
                best / max(self.min_color_ratio * 4.0, 1e-6),
                0.0,
                1.0,
            ))
        return InstantColor(
            label=label,
            confidence=confidence,
            ratios=scores,
            value_threshold=threshold,
            raw_ratios=raw_ratios,
            background_ratios=background_ratios,
            brightness_contrast=contrasts,
            decision_reason=(
                "accepted" if label in LIT_COLORS else label.lower()
            ),
        )

    def _classify_center_led_strict(
        self, bgr: np.ndarray, hsv: np.ndarray
    ) -> InstantColor:
        """Classify only a chromatically luminous central LED region."""
        hue, saturation, value = cv2.split(hsv)
        center, background = self._center_led_masks(value.shape)
        center_area = float(np.count_nonzero(center))
        if center_area <= 0.0:
            return self._empty()

        center_saturation = float(np.percentile(
            saturation[center], self.center_saturation_percentile
        ))
        center_value = float(np.percentile(
            value[center], self.center_value_percentile
        ))
        emission = (
            saturation.astype(np.float32)
            * value.astype(np.float32)
            / 255.0
        )
        emission_strength = float(np.percentile(
            emission[center], self.center_saturation_percentile
        ))
        background_emission = (
            float(np.percentile(
                emission[background], self.center_saturation_percentile
            ))
            if np.any(background)
            else 0.0
        )
        emission_contrast = emission_strength - background_emission
        neutral_clip_ratio = float(np.count_nonzero(
            center & (value >= 250) & (saturation <= 55)
        )) / center_area
        color_value_threshold = float(self.value_min)
        if self.color_brightness_percentile > 0.0:
            color_value_threshold = max(
                color_value_threshold,
                float(np.percentile(
                    value[center], self.color_brightness_percentile
                )),
            )
        zeros = {name: 0.0 for name in LIT_COLORS}

        lit_gate = (
            center_saturation >= self.min_center_saturation
            and center_value >= self.value_min
            and emission_strength >= self.min_emission_strength
        )
        if not lit_gate:
            confidence = float(np.clip(
                1.0
                - center_saturation
                / max(self.min_center_saturation, 1.0),
                0.0,
                1.0,
            ))
            return InstantColor(
                label="DARK",
                confidence=confidence,
                ratios=zeros.copy(),
                value_threshold=self.value_min,
                raw_ratios=zeros.copy(),
                background_ratios=zeros.copy(),
                brightness_contrast={
                    name: emission_contrast for name in LIT_COLORS
                },
                channel_shares=zeros.copy(),
                center_value_p90=center_value,
                center_saturation_p75=center_saturation,
                emission_strength_p75=emission_strength,
                background_emission_p75=background_emission,
                emission_contrast=emission_contrast,
                neutral_clip_ratio=neutral_clip_ratio,
                color_value_threshold=color_value_threshold,
                decision_reason="no_central_chromatic_emission",
            )

        valid = (
            center
            & (saturation >= self.saturation_min)
            & (value >= color_value_threshold)
        )
        masks = {
            "RED": valid & (
                (hue <= self.red_low_max) | (hue >= self.red_high_min)
            ),
            "GREEN": (
                valid & (hue >= self.green_min) & (hue <= self.green_max)
            ),
            "BLUE": (
                valid & (hue >= self.blue_min) & (hue <= self.blue_max)
            ),
        }
        if (
            self.color_brightness_percentile > 0.0
            or self.radial_weight_strength > 0.0
        ):
            height, width = value.shape
            yy, xx = np.ogrid[:height, :width]
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
            radius_x = max(
                width * self.center_region_ratio / 2.0, 1.0
            )
            radius_y = max(
                height * self.center_region_ratio / 2.0, 1.0
            )
            radius_squared = (
                ((xx - center_x) / radius_x) ** 2
                + ((yy - center_y) / radius_y) ** 2
            )
            radial_weight = np.exp(
                -self.radial_weight_strength * radius_squared
            )
            brightness_weight = (
                np.maximum(
                    value.astype(np.float64) - color_value_threshold,
                    0.0,
                )
                + 8.0
            )
            hue_weight = (
                saturation.astype(np.float64)
                * brightness_weight
                * radial_weight
            )
            channel_pixel_weight = brightness_weight * radial_weight
        else:
            hue_weight = emission.astype(np.float64)
            channel_pixel_weight = np.ones(
                value.shape, dtype=np.float64
            )
        weighted_scores = {
            name: float(np.sum(hue_weight[mask]))
            for name, mask in masks.items()
        }
        total_score = sum(weighted_scores.values())
        hue_shares = {
            name: score / max(total_score, 1e-9)
            for name, score in weighted_scores.items()
        }
        raw_ratios = {
            name: float(np.count_nonzero(mask)) / center_area
            for name, mask in masks.items()
        }
        ordered = sorted(
            hue_shares.items(), key=lambda item: item[1], reverse=True
        )
        candidate, best_share = ordered[0]
        hue_margin = best_share - ordered[1][1]
        candidate_component = self._largest_component_ratio(
            masks[candidate], center_area
        )

        colored_mask = np.logical_or.reduce(list(masks.values()))
        color_fraction = float(np.count_nonzero(colored_mask)) / center_area
        channel_signal = bgr[colored_mask].astype(np.float64)
        if channel_signal.size:
            channel_signal -= np.min(channel_signal, axis=1, keepdims=True)
            channel_signal *= channel_pixel_weight[
                colored_mask
            ][:, np.newaxis]
            channel_sums = np.sum(channel_signal, axis=0)
        else:
            channel_sums = np.zeros(3, dtype=np.float64)
        channel_total = float(np.sum(channel_sums))
        channel_shares = {
            "RED": float(channel_sums[2]) / max(channel_total, 1e-9),
            "GREEN": float(channel_sums[1]) / max(channel_total, 1e-9),
            "BLUE": float(channel_sums[0]) / max(channel_total, 1e-9),
        }
        channel_ordered = sorted(
            channel_shares.items(), key=lambda item: item[1], reverse=True
        )
        channel_candidate, best_channel_share = channel_ordered[0]
        channel_margin = best_channel_share - channel_ordered[1][1]

        reason = "accepted"
        label = candidate
        required_hue_share = (
            self.blue_strict_hue_share_min
            if candidate == "BLUE"
            else self.strict_hue_share_min
        )
        required_channel_share = (
            self.blue_strict_channel_share_min
            if candidate == "BLUE"
            else self.strict_channel_share_min
        )
        blue_too_weak = (
            candidate == "BLUE"
            and (
                center_saturation < self.blue_min_center_saturation
                or emission_strength < self.blue_min_emission_strength
            )
        )
        if blue_too_weak:
            label = "DARK" if self.weak_blue_as_dark else "UNKNOWN"
            reason = "weak_blue_emission"
        elif (
            candidate == "BLUE"
            and neutral_clip_ratio
            > self.max_neutral_clip_ratio_for_blue
        ):
            label = "UNKNOWN"
            reason = "overexposed_blue_ambiguous"
        elif color_fraction < self.min_center_color_fraction:
            label = "UNKNOWN"
            reason = "insufficient_center_color_pixels"
        elif candidate_component < self.min_component_ratio:
            label = "UNKNOWN"
            reason = "fragmented_center_color"
        elif (
            best_share < required_hue_share
            or hue_margin < self.strict_hue_margin_min
        ):
            label = "UNKNOWN"
            reason = "ambiguous_strict_hue"
        elif (
            channel_candidate != candidate
            or best_channel_share < required_channel_share
            or channel_margin < self.strict_channel_margin_min
        ):
            label = "UNKNOWN"
            reason = "hsv_channel_disagreement"

        if label in LIT_COLORS:
            confidence = float(np.clip(min(
                best_share,
                best_channel_share,
                color_fraction
                / max(self.min_center_color_fraction * 2.0, 1e-9),
            ), 0.0, 1.0))
        else:
            confidence = 0.0
        background_ratios = self._background_color_ratios(
            hue, saturation, value, background
        )
        return InstantColor(
            label=label,
            confidence=confidence,
            ratios=hue_shares,
            value_threshold=int(round(color_value_threshold)),
            raw_ratios=raw_ratios,
            background_ratios=background_ratios,
            brightness_contrast={
                name: emission_contrast for name in LIT_COLORS
            },
            channel_shares=channel_shares,
            center_value_p90=center_value,
            center_saturation_p75=center_saturation,
            emission_strength_p75=emission_strength,
            background_emission_p75=background_emission,
            emission_contrast=emission_contrast,
            neutral_clip_ratio=neutral_clip_ratio,
            color_value_threshold=color_value_threshold,
            decision_reason=reason,
        )

    def _classify_emitter_ring(
        self,
        bgr: np.ndarray,
        hsv: np.ndarray,
        center_hint: Optional[Tuple[float, float]] = None,
        lock_emitter_center: bool = False,
    ) -> InstantColor:
        """Locate a luminous core, then read color from its halo/annulus."""
        hue, saturation, value = cv2.split(hsv)
        height, width = value.shape
        minimum_size = float(min(height, width))
        yy, xx = np.ogrid[:height, :width]
        nominal_x = (width - 1) / 2.0
        nominal_y = (height - 1) / 2.0
        search_x = max(width * self.emitter_search_ratio / 2.0, 1.0)
        search_y = max(height * self.emitter_search_ratio / 2.0, 1.0)
        search = (
            ((xx - nominal_x) / search_x) ** 2
            + ((yy - nominal_y) / search_y) ** 2
        ) <= 1.0
        if center_hint is not None:
            hint_x = float(np.clip(
                center_hint[0], 0.0, 1.0
            )) * max(width - 1, 1)
            hint_y = float(np.clip(
                center_hint[1], 0.0, 1.0
            )) * max(height - 1, 1)
            hint_radius = max(
                2.0, minimum_size * self.emitter_hint_radius_ratio
            )
            hint_search = (
                (xx - hint_x) ** 2 + (yy - hint_y) ** 2
            ) <= hint_radius ** 2
            constrained = search & hint_search
            if np.count_nonzero(constrained) >= 4:
                search = constrained

        chromatic_emission = saturation.astype(np.float32) * value.astype(np.float32) / 255.0
        if lock_emitter_center and center_hint is not None:
            center_x = int(round(np.clip(center_hint[0], 0, 1) * (width - 1)))
            center_y = int(round(np.clip(center_hint[1], 0, 1) * (height - 1)))
        else:
            kernel = max(3, int(round(minimum_size * 0.075)) | 1)
            smooth_value = cv2.GaussianBlur(value.astype(np.float32), (kernel, kernel), 0)
            locator = smooth_value + 0.12 * cv2.GaussianBlur(chromatic_emission, (kernel, kernel), 0)
            locator[~search] = -1.0
            center_y, center_x = np.unravel_index(int(np.argmax(locator)), locator.shape)

        distance = np.sqrt(
            (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
        ) / max(minimum_size, 1.0)
        core = distance <= self.emitter_core_radius_ratio
        ring = (
            (distance >= self.emitter_ring_inner_ratio)
            & (distance <= self.emitter_ring_outer_ratio)
        )
        sample_region = core | ring
        local_background = (
            (distance >= self.emitter_ring_outer_ratio * 1.12)
            & (distance <= min(0.48, self.emitter_ring_outer_ratio * 1.65))
        )
        if np.count_nonzero(local_background) < 12:
            _, local_background = self._region_masks(value.shape)

        core_values = value[core]
        background_values = value[local_background]
        if core_values.size == 0 or background_values.size == 0:
            return self._empty()
        core_value = float(np.percentile(core_values, 90.0))
        background_value = float(np.percentile(background_values, 60.0))
        ring_value = (
            float(np.percentile(value[ring], 75.0))
            if np.any(ring) else background_value
        )
        core_mean = float(np.mean(core_values))
        background_mean = float(np.mean(background_values))
        core_emission = float(np.percentile(
            chromatic_emission[core], 75.0
        ))
        background_emission = float(np.percentile(
            chromatic_emission[local_background], 60.0
        ))
        luminance_signal = float(np.clip(
            0.65 * max(0.0, core_value - background_value) / 255.0
            + 0.25 * max(0.0, core_mean - background_mean) / 255.0
            + 0.10
            * max(0.0, core_emission - background_emission)
            / 255.0,
            0.0,
            1.0,
        ))

        region_values = value[sample_region]
        color_threshold = max(
            float(self.value_min),
            float(np.percentile(
                region_values, self.emitter_color_percentile
            )),
            background_value + 4.0,
        )
        valid = (
            sample_region
            & (saturation >= self.saturation_min)
            & (value >= color_threshold)
        )
        masks = {
            "RED": valid & (
                (hue <= self.red_low_max) | (hue >= self.red_high_min)
            ),
            "GREEN": (
                valid & (hue >= self.green_min) & (hue <= self.green_max)
            ),
            "BLUE": (
                valid & (hue >= self.blue_min) & (hue <= self.blue_max)
            ),
        }
        radial_weight = np.exp(
            -1.15
            * np.square(
                distance / max(self.emitter_ring_outer_ratio, 1e-6)
            )
        )
        brightness_delta = np.maximum(
            value.astype(np.float64) - background_value, 0.0
        )
        pixel_weight = (
            saturation.astype(np.float64)
            * np.power(brightness_delta + 6.0, 1.35)
            * radial_weight
        )
        hue_weights = {
            name: float(np.sum(pixel_weight[mask]))
            for name, mask in masks.items()
        }
        hue_total = sum(hue_weights.values())
        hue_shares = {
            name: hue_weights[name] / max(hue_total, 1e-9)
            for name in LIT_COLORS
        }

        colored = np.logical_or.reduce(list(masks.values()))
        chroma_fraction = (
            float(np.count_nonzero(colored))
            / max(float(np.count_nonzero(sample_region)), 1.0)
        )
        pixels = bgr[colored].astype(np.float64)
        if pixels.size:
            pixels -= np.min(pixels, axis=1, keepdims=True)
            pixels *= (
                np.power(brightness_delta[colored] + 6.0, 1.35)
                * radial_weight[colored]
            )[:, np.newaxis]
            channel_sums = np.sum(pixels, axis=0)
        else:
            channel_sums = np.zeros(3, dtype=np.float64)
        channel_total = float(np.sum(channel_sums))
        channel_shares = {
            "RED": float(channel_sums[2]) / max(channel_total, 1e-9),
            "GREEN": float(channel_sums[1]) / max(channel_total, 1e-9),
            "BLUE": float(channel_sums[0]) / max(channel_total, 1e-9),
        }
        combined = {
            name: 0.55 * hue_shares[name]
            + 0.45 * channel_shares[name]
            for name in LIT_COLORS
        }
        ordered = sorted(
            combined.items(), key=lambda item: item[1], reverse=True
        )
        candidate, best_share = ordered[0]
        share_margin = best_share - ordered[1][1]
        sample_area = max(float(np.count_nonzero(sample_region)), 1.0)
        candidate_component = self._largest_component_ratio(
            masks[candidate], sample_area
        )
        channel_ordered = sorted(
            channel_shares.items(), key=lambda item: item[1], reverse=True
        )
        channel_candidate, best_channel = channel_ordered[0]
        channel_margin = best_channel - channel_ordered[1][1]

        label = candidate
        reason = "accepted_emitter_ring"
        required_share = (
            self.emitter_blue_share_min
            if candidate == "BLUE"
            else self.emitter_color_share_min
        )
        required_signal = max(self.emitter_min_signal, {
            "RED": self.emitter_red_signal_min,
            "GREEN": self.emitter_green_signal_min,
            "BLUE": self.emitter_blue_signal_min,
        }[candidate])
        if luminance_signal < required_signal:
            label = "DARK"
            reason = "no_luminance_change"
        elif chroma_fraction < self.emitter_min_chroma_fraction:
            label = "UNKNOWN"
            reason = "bright_but_achromatic"
        elif (
            candidate_component < self.min_component_ratio
            and luminance_signal < self.emitter_fragmented_signal_min
        ):
            # Weak isolated dots are OFF reflections. A strongly luminous
            # LED matrix is also fragmented by design and must remain valid.
            label = "DARK"
            reason = "weak_fragmented_emitter_reflection"
        elif (
            best_share < required_share
            or share_margin < self.emitter_color_margin_min
        ):
            label = "UNKNOWN"
            reason = "ambiguous_ring_color"
        elif (
            channel_candidate != candidate
            or best_channel < self.emitter_channel_share_min
            or channel_margin < self.emitter_channel_margin_min
        ):
            label = "UNKNOWN"
            reason = "ring_channel_disagreement"
        elif (
            candidate == "BLUE"
            and (
                luminance_signal < self.emitter_blue_signal_min
                or chroma_fraction < self.emitter_blue_chroma_min
                or (
                    core_value - ring_value
                    < self.emitter_blue_core_contrast_min
                    and luminance_signal
                    < self.emitter_blue_core_bypass_signal
                )
            )
        ):
            label = "DARK" if self.weak_blue_as_dark else "UNKNOWN"
            reason = "static_blue_lens"
        confidence = (
            float(np.clip(min(
                best_share,
                best_channel,
                luminance_signal
                / max(required_signal * 2.0, 1e-9),
            ), 0.0, 1.0))
            if label in LIT_COLORS
            else 0.0
        )
        neutral_clip_ratio = (
            float(np.count_nonzero(
                core & (value >= 250) & (saturation <= 55)
            ))
            / max(float(np.count_nonzero(core)), 1.0)
        )
        return InstantColor(
            label=label,
            confidence=confidence,
            ratios=combined,
            value_threshold=int(round(color_threshold)),
            raw_ratios={
                name: float(np.count_nonzero(mask))
                / max(float(np.count_nonzero(sample_region)), 1.0)
                for name, mask in masks.items()
            },
            background_ratios=self._background_color_ratios(
                hue, saturation, value, local_background
            ),
            brightness_contrast={
                name: max(0.0, core_value - background_value)
                for name in LIT_COLORS
            },
            channel_shares=channel_shares,
            center_value_p90=core_value,
            center_saturation_p75=float(np.percentile(
                saturation[sample_region], 75.0
            )),
            emission_strength_p75=float(np.percentile(
                chromatic_emission[sample_region], 75.0
            )),
            background_emission_p75=background_emission,
            emission_contrast=core_emission - background_emission,
            neutral_clip_ratio=neutral_clip_ratio,
            color_value_threshold=color_threshold,
            decision_reason=reason,
            luminance_signal=luminance_signal,
            emitter_center_x=float(center_x) / max(width - 1, 1),
            emitter_center_y=float(center_y) / max(height - 1, 1),
            emitter_core_value=core_value,
            emitter_background_value=background_value,
            emitter_chroma_fraction=chroma_fraction,
            hue_shares=hue_shares,
            required_signal=required_signal,
        )

    def _center_led_masks(self, shape):
        height, width = shape
        yy, xx = np.ogrid[:height, :width]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius_x = max(width * self.center_region_ratio / 2.0, 1.0)
        radius_y = max(height * self.center_region_ratio / 2.0, 1.0)
        center = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0
        _, background = self._region_masks(shape)
        return center, background

    def _background_color_ratios(
        self, hue, saturation, value, background
    ) -> Dict[str, float]:
        area = float(np.count_nonzero(background))
        if area <= 0.0:
            return {name: 0.0 for name in LIT_COLORS}
        valid = (
            background
            & (saturation >= self.saturation_min)
            & (value >= self.value_min)
        )
        return {
            "RED": float(np.count_nonzero(
                valid & (
                    (hue <= self.red_low_max)
                    | (hue >= self.red_high_min)
                )
            )) / area,
            "GREEN": float(np.count_nonzero(
                valid & (hue >= self.green_min) & (hue <= self.green_max)
            )) / area,
            "BLUE": float(np.count_nonzero(
                valid & (hue >= self.blue_min) & (hue <= self.blue_max)
            )) / area,
        }

    def _adaptive_color_masks(
        self, bgr: np.ndarray, value_valid, hsv_masks
    ):
        """Fuse hue with brightness-normalized opponent-channel evidence."""
        normalized = bgr.astype(np.float32) / 255.0
        blue, green, red = cv2.split(normalized)
        total = blue + green + red + 1e-6
        shares = {
            "RED": red / total,
            "GREEN": green / total,
            "BLUE": blue / total,
        }
        opponent_masks = {}
        for name, share in shares.items():
            competitors = [
                other_share
                for other_name, other_share in shares.items()
                if other_name != name
            ]
            strongest_competitor = np.maximum(
                competitors[0], competitors[1]
            )
            dominance = share - strongest_competitor
            spread = share - np.minimum(
                competitors[0], competitors[1]
            )
            opponent_masks[name] = (
                value_valid
                & (dominance >= self.channel_dominance_min)
                & (spread >= self.chromaticity_spread_min)
            )
        opponent_any = np.logical_or.reduce(
            list(opponent_masks.values())
        )
        return {
            name: opponent_masks[name] | (mask & ~opponent_any)
            for name, mask in hsv_masks.items()
        }

    def _region_masks(self, shape):
        height, width = shape
        minimum_size = min(height, width)
        maximum_border = max(1, (minimum_size - 1) // 3)
        border = max(
            1,
            min(
                maximum_border,
                int(round(minimum_size * self.background_border_ratio)),
            ),
        )
        inner = np.zeros(shape, dtype=bool)
        inner[border:height - border, border:width - border] = True
        return inner, ~inner

    @staticmethod
    def _largest_component_ratio(mask, area: float) -> float:
        if not np.any(mask):
            return 0.0
        count, _, statistics, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return 0.0
        largest = int(np.max(statistics[1:, cv2.CC_STAT_AREA]))
        return float(largest) / max(area, 1.0)

    @staticmethod
    def _empty() -> InstantColor:
        zeros = {name: 0.0 for name in LIT_COLORS}
        return InstantColor(
            label="UNKNOWN",
            confidence=0.0,
            ratios=zeros.copy(),
            value_threshold=0,
            raw_ratios=zeros.copy(),
            background_ratios=zeros.copy(),
            brightness_contrast=zeros.copy(),
            channel_shares=zeros.copy(),
            decision_reason="empty_roi",
        )


class TemporalAnalyzer:
    """Infer Task 1 state from the history belonging to exactly one track."""

    def __init__(
        self,
        history_seconds: float,
        blink_min_window: float,
        blink_min_transitions: int,
        blink_min_hz: float,
        blink_max_hz: float,
        solid_min_seconds: float,
        off_min_seconds: float,
        color_purity_min: float,
        mode: str = "transitions",
        template_target_hz: float = 0.5,
        template_frequency_tolerance: float = 0.10,
        template_min_agreement: float = 0.78,
        template_phase_steps: int = 40,
        template_missing_weight: float = 0.55,
        photometric_signal_delta_min: float = 0.025,
        photometric_low_percentile: float = 20.0,
        photometric_high_percentile: float = 82.0,
        photometric_low_threshold: float = 0.32,
        photometric_high_threshold: float = 0.68,
        photometric_min_samples: int = 12,
        photometric_min_agreement: float = 0.72,
        photometric_color_purity_min: float = 0.55,
        photometric_color_level_min: float = 0.015,
        photometric_blue_level_min: float = 0.18,
        photometric_color_delta_min: float = 0.008,
        photometric_color_phase_mode: str = "independent",
        photometric_max_gap_seconds: float = 0.35,
        photometric_blue_purity_min: float = 0.0,
    ) -> None:
        self.history_seconds = history_seconds
        self.blink_min_window = blink_min_window
        self.blink_min_transitions = blink_min_transitions
        self.blink_min_hz = blink_min_hz
        self.blink_max_hz = blink_max_hz
        self.solid_min_seconds = solid_min_seconds
        self.off_min_seconds = off_min_seconds
        self.color_purity_min = color_purity_min
        if mode not in (
            "transitions", "duration_template", "photometric_template", "aperture_signal", "emitter_signal"
        ):
            raise ValueError(
                "temporal mode must be transitions, duration_template "
                "photometric_template, aperture_signal or emitter_signal"
            )
        self.mode = mode
        self.template_target_hz = template_target_hz
        self.template_frequency_tolerance = template_frequency_tolerance
        self.template_min_agreement = template_min_agreement
        self.template_phase_steps = max(8, int(template_phase_steps))
        self.template_missing_weight = float(template_missing_weight)
        self.photometric_signal_delta_min = float(
            photometric_signal_delta_min
        )
        self.photometric_low_percentile = float(
            photometric_low_percentile
        )
        self.photometric_high_percentile = float(
            photometric_high_percentile
        )
        self.photometric_low_threshold = float(
            photometric_low_threshold
        )
        self.photometric_high_threshold = float(
            photometric_high_threshold
        )
        self.photometric_min_samples = max(
            6, int(photometric_min_samples)
        )
        self.photometric_min_agreement = float(
            photometric_min_agreement
        )
        self.photometric_color_purity_min = float(
            photometric_color_purity_min
        )
        self.photometric_blue_purity_min = max(
            self.photometric_color_purity_min, float(photometric_blue_purity_min))
        if not 0 <= self.photometric_blue_purity_min <= 1:
            raise ValueError("photometric_blue_purity_min must be within [0, 1]")
        self.photometric_color_level_min = float(
            photometric_color_level_min
        )
        self.photometric_blue_level_min = float(
            photometric_blue_level_min
        )
        self.photometric_color_delta_min = float(
            photometric_color_delta_min
        )
        if photometric_color_phase_mode not in (
            "independent", "shared_luminance"
        ):
            raise ValueError(
                "photometric_color_phase_mode must be independent "
                "or shared_luminance"
            )
        self.photometric_color_phase_mode = photometric_color_phase_mode
        self.physical_decoder = (PhotometricSignalDecoder(self, photometric_max_gap_seconds)
                                 if mode in ("aperture_signal", "emitter_signal") else None)
        self.samples: Deque[Tuple[float, str]] = deque()
        self.photometric_samples: Deque[
            Tuple[float, float, str, Dict[str, float]]
        ] = deque()

    def update(
        self,
        timestamp: float,
        color: str,
        signal: Optional[float] = None,
        color_scores: Optional[Dict[str, float]] = None,
        phase: str = "LEGACY",
    ) -> TemporalResult:
        self.samples.append((timestamp, color))
        numeric_signal = (
            float(signal)
            if signal is not None and np.isfinite(signal)
            else float("nan")
        )
        self.photometric_samples.append((
            timestamp,
            numeric_signal,
            color,
            dict(color_scores or {}),
        ))
        while (
            self.samples
            and timestamp - self.samples[0][0] > self.history_seconds
        ):
            self.samples.popleft()
        while (
            self.photometric_samples
            and timestamp - self.photometric_samples[0][0]
            > self.history_seconds
        ):
            self.photometric_samples.popleft()
        if self.physical_decoder is not None:
            return TemporalResult(**self.physical_decoder.update(timestamp, phase, numeric_signal, color_scores or {}))
        if self.mode == "photometric_template":
            return self._infer_photometric()
        return self.infer()

    def reset(self) -> None:
        """Discard temporal evidence while preserving the tracker ID."""
        self.samples.clear()
        self.photometric_samples.clear()
        if self.physical_decoder is not None:
            self.physical_decoder.reset()

    def trim_before(self, timestamp: float) -> None:
        """Drop stale pre-change evidence while retaining the new segment."""
        if self.physical_decoder is not None:
            self.physical_decoder.trim_before(timestamp)
        while self.samples and self.samples[0][0] < timestamp:
            self.samples.popleft()
        while (
            self.photometric_samples
            and self.photometric_samples[0][0] < timestamp
        ):
            self.photometric_samples.popleft()

    def infer(self) -> TemporalResult:
        if not self.samples:
            return TemporalResult()
        end_time = self.samples[-1][0]
        span = end_time - self.samples[0][0]
        current_color = self.samples[-1][1]
        run_start = len(self.samples) - 1
        while (
            run_start > 0
            and self.samples[run_start - 1][1] == current_color
        ):
            run_start -= 1
        run_seconds = end_time - self.samples[run_start][0]

        if (
            current_color in ("DARK", "OFF")
            and run_seconds >= self.off_min_seconds
        ):
            return TemporalResult(
                state="OFF", confidence=1.0, history_span=span,
                method=self.mode,
            )

        lit = [color for _, color in self.samples if color in LIT_COLORS]
        if not lit:
            return TemporalResult(history_span=span, method=self.mode)
        dominant_color, dominant_count = Counter(lit).most_common(1)[0]
        purity = dominant_count / len(lit)
        timeline = [
            (stamp, 1 if color == dominant_color else 0, color)
            for stamp, color in self.samples
            if color == dominant_color or color in DARK_OBSERVATIONS
        ]
        if len(timeline) < 3 or purity < self.color_purity_min:
            return TemporalResult(
                dominant_color=dominant_color, history_span=span,
                method=self.mode,
            )

        raw = [state for _, state, _ in timeline]
        smooth = []
        for index, state in enumerate(raw):
            neighborhood = raw[max(0, index - 1):min(len(raw), index + 2)]
            smooth.append(
                1 if sum(neighborhood) * 2 >= len(neighborhood) else 0
            )
        transitions = sum(a != b for a, b in zip(smooth, smooth[1:]))
        timeline_span = timeline[-1][0] - timeline[0][0]
        on_ratio = sum(smooth) / len(smooth)
        blink_hz = transitions / max(2.0 * timeline_span, 1e-6)
        blinking = (
            timeline_span >= self.blink_min_window
            and transitions >= self.blink_min_transitions
            and self.blink_min_hz <= blink_hz <= self.blink_max_hz
            and 0.10 <= on_ratio <= 0.90
        )
        template_agreement = 0.0
        template_hz = 0.0
        if self.mode == "duration_template":
            template_agreement, template_hz = self._template_score(
                timeline, smooth
            )
            blinking = (
                blinking
                and template_agreement >= self.template_min_agreement
            )
        if blinking:
            confidence = float(np.clip(
                0.45 * purity
                + 0.35 * min(
                    1.0,
                    transitions / (self.blink_min_transitions + 2.0),
                )
                + 0.20 * (1.0 - abs(on_ratio - 0.5) / 0.5),
                0.0,
                1.0,
            ))
            return TemporalResult(
                state=f"BLINK_{dominant_color}",
                confidence=float(np.clip(
                    confidence * (
                        template_agreement
                        if self.mode == "duration_template"
                        else 1.0
                    ),
                    0.0,
                    1.0,
                )),
                dominant_color=dominant_color, on_ratio=on_ratio,
                transitions=transitions,
                blink_hz=(template_hz or blink_hz),
                history_span=timeline_span,
                method=self.mode,
                template_agreement=template_agreement,
            )
        if (
            current_color == "BLUE"
            and run_seconds >= self.solid_min_seconds
            and purity >= self.color_purity_min
        ):
            return TemporalResult(
                state="SOLID_BLUE", confidence=purity,
                dominant_color="BLUE", on_ratio=on_ratio,
                transitions=transitions, history_span=timeline_span,
                method=self.mode,
                template_agreement=template_agreement,
            )
        return TemporalResult(
            dominant_color=dominant_color, on_ratio=on_ratio,
            transitions=transitions, blink_hz=blink_hz,
            history_span=timeline_span,
            method=self.mode,
            template_agreement=template_agreement,
        )

    def _infer_photometric(self) -> TemporalResult:
        """Fit a 0.5 Hz square wave to continuous per-track light energy."""
        fallback = self.infer()
        # V7 blink states must never bypass the continuous template through
        # the legacy categorical transition path. The fallback is retained
        # only for stable OFF/SOLID_BLUE decisions.
        if fallback.state.startswith("BLINK_"):
            fallback = TemporalResult(
                dominant_color=fallback.dominant_color,
                history_span=fallback.history_span,
                method=self.mode,
            )
        samples = list(self.photometric_samples)
        if not samples:
            return replace(fallback, method=self.mode)
        span = samples[-1][0] - samples[0][0]
        finite = np.asarray([
            np.isfinite(signal) for _, signal, _, _ in samples
        ], dtype=bool)
        if (
            np.count_nonzero(finite) < self.photometric_min_samples
            or span < self.blink_min_window
        ):
            return replace(fallback, method=self.mode, history_span=span)

        signals = np.asarray([
            signal if np.isfinite(signal) else np.nan
            for _, signal, _, _ in samples
        ], dtype=np.float64)
        low = float(np.nanpercentile(
            signals, self.photometric_low_percentile
        ))
        high = float(np.nanpercentile(
            signals, self.photometric_high_percentile
        ))
        delta = high - low
        if delta < self.photometric_signal_delta_min:
            return replace(fallback, method=self.mode, history_span=span)

        normalized = np.clip(
            (signals - low) / max(delta, 1e-9), 0.0, 1.0
        )
        states = []
        previous = 0
        for value in normalized:
            if not np.isfinite(value):
                current = 0
            elif value >= self.photometric_high_threshold:
                current = 1
            elif value <= self.photometric_low_threshold:
                current = 0
            else:
                current = previous
            states.append(current)
            previous = current
        transitions = sum(
            first != second
            for first, second in zip(states, states[1:])
        )
        on_ratio = float(np.mean(states))

        offsets = np.asarray([
            stamp - samples[0][0] for stamp, _, _, _ in samples
        ], dtype=np.float64)
        observed = np.where(finite, normalized, 0.0)
        weights = np.where(
            finite, 1.0, self.template_missing_weight
        )
        lower = max(
            self.blink_min_hz,
            self.template_target_hz
            - self.template_frequency_tolerance,
        )
        upper = min(
            self.blink_max_hz,
            self.template_target_hz
            + self.template_frequency_tolerance,
        )
        best_agreement = 0.0
        best_frequency = 0.0
        best_template = None
        for frequency in np.linspace(lower, upper, 9):
            for phase in np.linspace(
                0.0, 1.0, self.template_phase_steps, endpoint=False
            ):
                predicted = (
                    np.mod(offsets * frequency + phase, 1.0) < 0.5
                ).astype(np.float64)
                agreement = 1.0 - float(
                    np.sum(weights * np.abs(predicted - observed))
                    / max(np.sum(weights), 1e-9)
                )
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_frequency = float(frequency)
                    best_template = predicted

        if (
            self.photometric_color_phase_mode == "shared_luminance"
            and best_template is not None
        ):
            shared = self._infer_shared_phase_colors(
                fallback=fallback,
                samples=samples,
                span=span,
                template=np.asarray(best_template, dtype=bool),
                best_agreement=best_agreement,
                best_frequency=best_frequency,
                transitions=transitions,
                on_ratio=on_ratio,
            )
            if shared is not None:
                return shared
            if fallback.state != "UNKNOWN":
                return replace(
                    fallback,
                    method=self.mode,
                    template_agreement=best_agreement,
                )
            return TemporalResult(
                on_ratio=on_ratio,
                transitions=transitions,
                blink_hz=best_frequency,
                history_span=span,
                method=self.mode,
                template_agreement=best_agreement,
            )

        # Fit each absolute color-support curve independently.  This is
        # deliberately not a vote among normalized hue shares: during a red
        # dark phase a small blue plastic lens can own 100% of weak chroma,
        # while its absolute support remains far below a true blue emitter.
        color_fits = {}
        missing_weights = np.asarray([
            self.template_missing_weight if color == "MISSING" else 1.0
            for _, _, color, _ in samples
        ], dtype=np.float64)
        for name in LIT_COLORS:
            evidence = np.asarray([
                max(float(scores.get(name, 0.0)), 0.0)
                for _, _, _, scores in samples
            ], dtype=np.float64)
            evidence_low = float(np.percentile(
                evidence, self.photometric_low_percentile
            ))
            evidence_high = float(np.percentile(
                evidence, self.photometric_high_percentile
            ))
            evidence_delta = evidence_high - evidence_low
            if evidence_delta <= 1e-9:
                normalized_color = np.zeros_like(evidence)
            else:
                normalized_color = np.clip(
                    (evidence - evidence_low) / evidence_delta,
                    0.0,
                    1.0,
                )
            color_states = []
            previous_color_state = 0
            for level in normalized_color:
                if level >= self.photometric_high_threshold:
                    color_state = 1
                elif level <= self.photometric_low_threshold:
                    color_state = 0
                else:
                    color_state = previous_color_state
                color_states.append(color_state)
                previous_color_state = color_state
            color_transitions = sum(
                first != second
                for first, second
                in zip(color_states, color_states[1:])
            )
            color_on_ratio = float(np.mean(color_states))
            color_agreement = 0.0
            color_frequency = 0.0
            for frequency in np.linspace(lower, upper, 9):
                for phase in np.linspace(
                    0.0, 1.0, self.template_phase_steps, endpoint=False
                ):
                    predicted = (
                        np.mod(offsets * frequency + phase, 1.0) < 0.5
                    ).astype(np.float64)
                    agreement = 1.0 - float(
                        np.sum(
                            missing_weights
                            * np.abs(predicted - normalized_color)
                        )
                        / max(np.sum(missing_weights), 1e-9)
                    )
                    if agreement > color_agreement:
                        color_agreement = agreement
                        color_frequency = float(frequency)
            minimum_level = (
                self.photometric_blue_level_min
                if name == "BLUE"
                else self.photometric_color_level_min
            )
            valid = (
                evidence_high >= minimum_level
                and evidence_delta >= self.photometric_color_delta_min
                and color_transitions >= self.blink_min_transitions
                and 0.10 <= color_on_ratio <= 0.90
                and color_agreement >= self.photometric_min_agreement
            )
            color_fits[name] = {
                "valid": valid,
                "agreement": color_agreement,
                "frequency": color_frequency,
                "transitions": color_transitions,
                "on_ratio": color_on_ratio,
                "active_ratio": float(np.mean(
                    evidence >= minimum_level
                )),
                "high": evidence_high,
                "delta": evidence_delta,
                "minimum": minimum_level,
            }

        valid_colors = [
            name for name, fit in color_fits.items() if fit["valid"]
        ]
        dominant_color = "UNKNOWN"
        if valid_colors:
            dominant_color = max(
                valid_colors,
                key=lambda name: (
                    color_fits[name]["agreement"],
                    color_fits[name]["high"]
                    / max(color_fits[name]["minimum"], 1e-9),
                ),
            )
            fit = color_fits[dominant_color]
            confidence = float(np.clip(
                0.60 * fit["agreement"]
                + 0.25 * min(
                    1.0,
                    fit["high"] / max(fit["minimum"] * 2.0, 1e-9),
                )
                + 0.15 * min(
                    1.0,
                    fit["delta"]
                    / max(
                        self.photometric_color_delta_min * 3.0, 1e-9
                    ),
                ),
                0.0,
                1.0,
            ))
            return TemporalResult(
                state=f"BLINK_{dominant_color}",
                confidence=confidence,
                dominant_color=dominant_color,
                on_ratio=fit["on_ratio"],
                transitions=fit["transitions"],
                blink_hz=fit["frequency"],
                history_span=span,
                method=self.mode,
                template_agreement=fit["agreement"],
            )

        plausible_colors = [
            name for name, fit in color_fits.items()
            if fit["high"] >= fit["minimum"]
        ]
        if plausible_colors:
            dominant_color = max(
                plausible_colors,
                key=lambda name: color_fits[name]["high"]
                / max(color_fits[name]["minimum"], 1e-9),
            )
        blue_fit = color_fits["BLUE"]
        other_peak = max(
            color_fits["RED"]["high"],
            color_fits["GREEN"]["high"],
        )
        if (
            span >= self.solid_min_seconds
            and blue_fit["high"] >= self.photometric_blue_level_min
            and blue_fit["active_ratio"] >= 0.45
            and other_peak < self.photometric_color_level_min
        ):
            return TemporalResult(
                state="SOLID_BLUE",
                confidence=float(np.clip(
                    blue_fit["active_ratio"], 0.0, 1.0
                )),
                dominant_color="BLUE",
                on_ratio=blue_fit["active_ratio"],
                transitions=blue_fit["transitions"],
                history_span=span,
                method=self.mode,
                template_agreement=blue_fit["agreement"],
            )
        if fallback.state != "UNKNOWN":
            return replace(
                fallback,
                method=self.mode,
                template_agreement=best_agreement,
            )
        return TemporalResult(
            dominant_color=dominant_color,
            on_ratio=on_ratio,
            transitions=transitions,
            blink_hz=best_frequency,
            history_span=span,
            method=self.mode,
            template_agreement=best_agreement,
        )

    def _infer_shared_phase_colors(
        self,
        fallback: TemporalResult,
        samples,
        span: float,
        template: np.ndarray,
        best_agreement: float,
        best_frequency: float,
        transitions: int,
        on_ratio: float,
    ) -> Optional[TemporalResult]:
        """Judge all colors on the single luminance-derived ON phase."""
        if (
            best_agreement < self.photometric_min_agreement
            or transitions < self.blink_min_transitions
            or not 0.10 <= on_ratio <= 0.90
        ):
            return None
        observed = np.asarray([
            color != "MISSING" for _, _, color, _ in samples
        ], dtype=bool)
        on_mask = template & observed
        off_mask = (~template) & observed
        if np.count_nonzero(on_mask) < 3 or np.count_nonzero(off_mask) < 3:
            return None

        fits = {}
        for name in LIT_COLORS:
            evidence = np.asarray([
                max(float(scores.get(name, 0.0)), 0.0)
                for _, _, _, scores in samples
            ], dtype=np.float64)
            on_values = evidence[on_mask]
            off_values = evidence[off_mask]
            on_level = float(np.percentile(
                on_values, self.photometric_high_percentile
            ))
            off_level = float(np.percentile(
                off_values, self.photometric_high_percentile
            ))
            phase_delta = on_level - off_level
            minimum = (
                self.photometric_blue_level_min
                if name == "BLUE"
                else self.photometric_color_level_min
            )
            fits[name] = {
                "on": on_level,
                "off": off_level,
                "delta": phase_delta,
                "minimum": minimum,
            }

        total_positive = sum(max(fit["delta"], 0.0) for fit in fits.values())
        for fit in fits.values():
            fit["purity"] = (
                max(fit["delta"], 0.0) / max(total_positive, 1e-9)
            )
            fit["valid"] = (
                fit["on"] >= fit["minimum"]
                and fit["delta"] >= self.photometric_color_delta_min
                and fit["purity"] >= self.photometric_color_purity_min
            )
        valid = [name for name, fit in fits.items() if fit["valid"]]
        if not valid:
            return None
        dominant = max(
            valid,
            key=lambda name: (
                fits[name]["purity"],
                fits[name]["delta"],
                fits[name]["on"] / max(fits[name]["minimum"], 1e-9),
            ),
        )
        fit = fits[dominant]
        confidence = float(np.clip(
            0.45 * best_agreement
            + 0.30 * fit["purity"]
            + 0.25 * min(
                1.0,
                fit["delta"]
                / max(self.photometric_color_delta_min * 4.0, 1e-9),
            ),
            0.0,
            1.0,
        ))
        return TemporalResult(
            state=f"BLINK_{dominant}",
            confidence=confidence,
            dominant_color=dominant,
            on_ratio=on_ratio,
            transitions=transitions,
            blink_hz=best_frequency,
            history_span=span,
            method=self.mode,
            template_agreement=best_agreement,
        )

    def _template_score(self, timeline, smooth) -> Tuple[float, float]:
        """Fit a one-second ON/OFF duration template to irregular samples."""
        start = timeline[0][0]
        lower = max(
            self.blink_min_hz,
            self.template_target_hz - self.template_frequency_tolerance,
        )
        upper = min(
            self.blink_max_hz,
            self.template_target_hz + self.template_frequency_tolerance,
        )
        frequencies = np.linspace(lower, upper, 9)
        phases = np.linspace(
            0.0, 1.0, self.template_phase_steps, endpoint=False
        )
        weights = np.asarray([
            self.template_missing_weight if color == "MISSING" else 1.0
            for _, _, color in timeline
        ], dtype=np.float64)
        observed = np.asarray(smooth, dtype=np.int8)
        offsets = np.asarray([
            stamp - start for stamp, _, _ in timeline
        ], dtype=np.float64)
        best_score = 0.0
        best_frequency = 0.0
        for frequency in frequencies:
            for phase in phases:
                predicted = (
                    np.mod(offsets * frequency + phase, 1.0) < 0.5
                ).astype(np.int8)
                score = float(
                    np.sum(weights * (predicted == observed))
                    / max(np.sum(weights), 1e-9)
                )
                if score > best_score:
                    best_score = score
                    best_frequency = float(frequency)
        return best_score, best_frequency


@dataclass
class Track:
    track_id: int
    bbox: BBox
    detector_confidence: float
    created_at: float
    last_seen: float
    analyzer: TemporalAnalyzer
    detector_bbox: Optional[BBox] = None
    hits: int = 1
    visible: bool = True
    instant: InstantColor = field(default_factory=HsvColorClassifier._empty)
    raw_instant: InstantColor = field(default_factory=HsvColorClassifier._empty)
    temporal: TemporalResult = field(default_factory=TemporalResult)
    confirmed_instant_label: str = "UNKNOWN"
    pending_instant_label: str = "UNKNOWN"
    pending_instant_count: int = 0
    last_reliable_color: str = "UNKNOWN"
    previous_detector_bbox: Optional[BBox] = None
    previous_seen: Optional[float] = None
    association_stage: str = "NEW"
    emitter_center_hint: Optional[Tuple[float, float]] = None
    decision: StateDecision = field(default_factory=StateDecision)
    blink_reference: BlinkPhotometricReference = field(
        default_factory=BlinkPhotometricReference
    )
    steady_off: SteadyNonBlueOff = field(default_factory=SteadyNonBlueOff)
    consecutive_hits: int = 1
    weak_birth_floor: float = 0.0
    measurement_bbox: Optional[BBox] = None
    measurement_time: float = 0.0
    motion_quality: float = 0.0
    detector_stamp: float = 0.0
    roi_last_observed: float = 0.0
    roi_tracking_active: bool = False
    roi_epoch: int = 0


    def __post_init__(self) -> None:
        if self.detector_bbox is None:
            self.detector_bbox = self.bbox

    def stabilize_instant(
        self,
        observation: InstantColor,
        lit_confirm_frames: int,
        dark_confirm_frames: int,
        color_switch_frames: int,
    ) -> InstantColor:
        """Apply per-track hysteresis without inventing missing evidence."""
        target = observation.label
        if target == "UNKNOWN":
            self.pending_instant_label = "UNKNOWN"
            self.pending_instant_count = 0
            return observation
        if target == self.confirmed_instant_label:
            self.pending_instant_label = "UNKNOWN"
            self.pending_instant_count = 0
            if target in LIT_COLORS:
                self.last_reliable_color = target
            return observation

        if target == self.pending_instant_label:
            self.pending_instant_count += 1
        else:
            self.pending_instant_label = target
            self.pending_instant_count = 1
        if target in ("DARK", "OFF"):
            required = dark_confirm_frames
        elif (
            self.confirmed_instant_label in LIT_COLORS
            and target in LIT_COLORS
        ):
            required = color_switch_frames
        else:
            required = lit_confirm_frames
        if self.pending_instant_count < max(1, required):
            return replace(
                observation, label="UNKNOWN", confidence=0.0
            )

        self.confirmed_instant_label = target
        self.pending_instant_label = "UNKNOWN"
        self.pending_instant_count = 0
        if target in LIT_COLORS:
            self.last_reliable_color = target
        return observation

    def gate_dark_without_lit_history(
        self,
        observation: InstantColor,
        minimum_hits: int,
        minimum_detector_confidence: float,
    ) -> InstantColor:
        """Reject weak never-lit DARK tracks that are likely background."""
        if (
            observation.label not in ("DARK", "OFF")
            or self.last_reliable_color in LIT_COLORS
        ):
            return observation
        if (
            self.hits >= max(1, minimum_hits)
            and self.detector_confidence >= minimum_detector_confidence
        ):
            return observation
        if (self.weak_birth_floor > 0
                and self.consecutive_hits >= max(1, minimum_hits)
                and self.detector_confidence >= self.weak_birth_floor):
            return observation
        return replace(
            observation, label="UNKNOWN", confidence=0.0
        )


def analysis_bbox(track: Track, source: str) -> BBox:
    """Select a current analysis ROI while leaving the display box smoothed."""
    if track.measurement_bbox is not None:
        return track.measurement_bbox
    if source == "smoothed":
        return track.bbox
    if source == "detector":
        return track.detector_bbox or track.bbox
    raise ValueError("analysis bbox source must be smoothed or detector")


def temporal_color_scores(
    observation: InstantColor, mode: str
) -> Dict[str, float]:
    """Build temporal color evidence without discarding alternate channels."""
    if observation.photometric_phase != "LEGACY":
        return dict(observation.color_energy) if observation.photometric_phase == "ON" else {name: 0.0 for name in LIT_COLORS}
    if mode == "legacy_stabilized":
        scores = {name: 0.0 for name in LIT_COLORS}
        if observation.label in scores:
            scores[observation.label] = max(
                float(observation.raw_ratios.get(
                    observation.label, 0.0
                )),
                0.0,
            )
        return scores
    if mode == "all_colors":
        return {
            name: max(float(observation.raw_ratios.get(name, 0.0)), 0.0)
            for name in LIT_COLORS
        }
    raise ValueError(
        "temporal color evidence mode must be legacy_stabilized "
        "or all_colors"
    )


def update_missing_track(track: Track, timestamp: float) -> None:
    """Advance a retained but currently unmeasured track."""
    track.steady_off.reset()
    if (
        track.analyzer.samples
        and timestamp + 1e-6 < track.analyzer.samples[-1][0]
    ):
        track.analyzer.reset()
        track.blink_reference.reset()
        track.decision.reset(timestamp=timestamp, identity_reset=True)
    track.temporal = track.analyzer.update(timestamp, "MISSING")
    track.blink_reference.learn(track.analyzer, track.temporal)
    track.decision.update(
        timestamp,
        observation_label="MISSING",
        observation_confidence=0.0,
        temporal=track.temporal,
        visible=False,
    )


def update_track_observation(
    track: Track,
    observation: InstantColor,
    timestamp: float,
    *,
    lit_confirm_frames: int,
    dark_confirm_frames: int,
    color_switch_frames: int,
    off_without_lit_min_hits: int,
    off_without_lit_min_detector_confidence: float,
    emitter_hint_update_confidence_min: float,
    emitter_hint_update_alpha: float,
    color_evidence_mode: str = "legacy_stabilized",
    guard_online_dark: bool = False,
) -> None:
    """Apply one measured ROI identically in ROS and offline replay."""
    online = observation.emission_diagnostics.get("online_light", {})
    if online.get("phase_reference_established",
                  online.get("learned_now") and online.get("periodic", {}).get("valid")):
        # Clear bootstrap hue once when measured levels first distinguish dark.
        # Later periodic confirmation must retain those already-clean ON pulses.
        track.analyzer.reset()
        track.blink_reference.reset()
        track.steady_off.reset()
        track.decision.reset(timestamp=timestamp, identity_reset=False)
        track.temporal = TemporalResult()
        track.confirmed_instant_label = track.pending_instant_label = "UNKNOWN"
        track.pending_instant_count = 0
        track.last_reliable_color = online.get("reference_color", "UNKNOWN")
    if (
        track.analyzer.samples
        and timestamp + 1e-6 < track.analyzer.samples[-1][0]
    ):
        track.analyzer.reset()
        track.blink_reference.reset()
        track.decision.reset(timestamp=timestamp, identity_reset=True)
    # Gate the raw color with a reference learned only from earlier trusted
    # blink cycles. Keep the raw observation visible for diagnostics.
    if (track.steady_off._samples
            and timestamp <= track.steady_off._samples[-1][0]):
        track.steady_off.reset()
    track.blink_reference.learn(track.analyzer, track.temporal)
    if guard_online_dark and observation.photometry_mode == "EMITTER_ONLINE" and observation.photometric_phase == "OFF":
        guarded = track.gate_dark_without_lit_history(observation,
            minimum_hits=off_without_lit_min_hits,
            minimum_detector_confidence=off_without_lit_min_detector_confidence)
        if guarded.label == "UNKNOWN":
            observation = replace(guarded, photometric_phase="UNCERTAIN",
                                  decision_reason="NEVER_LIT_IDENTITY_PENDING")
    track.raw_instant = observation
    # Physical ON/OFF evidence cannot be overridden by stale per-color levels.
    effective_observation = (observation if observation.photometric_phase != "LEGACY"
                             else track.blink_reference.gate(observation))

    track.instant = track.stabilize_instant(
        effective_observation,
        lit_confirm_frames=lit_confirm_frames,
        dark_confirm_frames=dark_confirm_frames,
        color_switch_frames=color_switch_frames,
    )
    if (
        (observation.photometric_phase == "ON"
         and observation.measurement_quality >= emitter_hint_update_confidence_min)
        or (observation.photometric_phase == "LEGACY"
            and track.instant.label in LIT_COLORS
            and track.instant.label == observation.label
            and observation.confidence >= emitter_hint_update_confidence_min)
    ):
        measured_hint = (
            observation.emitter_center_x,
            observation.emitter_center_y,
        )
        if track.emitter_center_hint is None:
            track.emitter_center_hint = measured_hint
        else:
            alpha = float(np.clip(
                emitter_hint_update_alpha, 0.0, 1.0
            ))
            track.emitter_center_hint = tuple(
                (1.0 - alpha) * previous + alpha * measured
                for previous, measured in zip(
                    track.emitter_center_hint, measured_hint
                )
            )
    if observation.photometry_mode != "EMITTER_ONLINE":
        track.instant = track.gate_dark_without_lit_history(
            track.instant,
            minimum_hits=off_without_lit_min_hits,
            minimum_detector_confidence=off_without_lit_min_detector_confidence,
        )
    scores = temporal_color_scores(track.instant, color_evidence_mode)
    track.temporal = track.analyzer.update(
        timestamp,
        track.instant.label,
        signal=track.instant.luminance_signal,
        color_scores=scores,
        phase=(observation.photometric_phase if observation.photometry_mode == "EMITTER_ONLINE"
               or track.instant.label != "UNKNOWN" or observation.photometric_phase == "ON" else "UNCERTAIN"),
    )
    track.blink_reference.learn(track.analyzer, track.temporal)
    steady_off = track.steady_off.update(
        timestamp, observation, track.temporal, track.detector_bbox
    )
    decision_temporal = track.temporal
    decision_label = track.instant.label
    decision_confidence = track.instant.confidence
    if steady_off:
        # Keep raw measurements intact; only the documented task-rule
        # evidence is fed to the committed-state machine.
        decision_temporal = TemporalResult(
            state="OFF", confidence=0.65,
            dominant_color=observation.label,
            history_span=track.steady_off.span,
            method="steady_nonblue_task_rule",
        )
        decision_label = "DARK"
        decision_confidence = 0.65
    track.decision.update(
        timestamp,
        observation_label=decision_label,
        observation_confidence=decision_confidence,
        temporal=decision_temporal,
        visible=True,
    )
    if steady_off and track.decision.committed_state == "OFF":
        track.decision.source = "steady_nonblue_task_rule"
    if track.decision.consume_history_reset_request():
        trim_at = (
            track.decision.pending_since
            if track.decision.pending_since is not None
            else timestamp
        )
        track.analyzer.trim_before(trim_at)


class MultiBeaconTracker:
    """Configurable baseline or two-stage association for a few beacons."""

    def __init__(
        self,
        iou_threshold: float,
        center_distance_ratio: float,
        max_age_seconds: float,
        analyzer_parameters: Dict[str, object],
        bbox_smoothing_alpha: float = 1.0,
        association_mode: str = "greedy",
        high_confidence_threshold: float = 0.40,
        low_confidence_threshold: float = 0.20,
        low_stage_iou_threshold: float = 0.10,
        low_stage_center_distance_ratio: float = 0.75,
        motion_horizon_seconds: float = 0.35,
        temporal_reset_distance_ratio: float = 0.0,
        decision_parameters: Optional[Dict[str, object]] = None,
        reference_parameters: Optional[Dict[str, object]] = None,
        stationary_reid_enabled: bool = False,
        stationary_reid_max_age_seconds: float = 6.0,
        stationary_reid_min_hits: int = 3,
        stationary_reid_iou_threshold: float = 0.10,
        stationary_reid_center_distance_ratio: float = 0.40,
        stationary_reid_size_similarity_min: float = 0.60,
        steady_off_parameters: Optional[Dict[str, object]] = None,
        duplicate_suppression_enabled: bool = False,
        duplicate_iou_threshold: float = 0.45,
        duplicate_containment_threshold: float = 0.85,
        duplicate_center_distance_ratio: float = 0.30,
        weak_birth_enabled: bool = False,
        weak_birth_confidence: float = 0.18,
        weak_birth_min_hits: int = 4,
        weak_birth_min_seconds: float = 0.60,
        weak_birth_max_gap_seconds: float = 0.65,
    ) -> None:
        if not 0.0 < bbox_smoothing_alpha <= 1.0:
            raise ValueError(
                "bbox_smoothing_alpha must be in the interval (0, 1]"
            )
        self.iou_threshold = iou_threshold
        self.center_distance_ratio = center_distance_ratio
        self.max_age_seconds = max_age_seconds
        self.analyzer_parameters = analyzer_parameters
        self.decision_parameters = dict(decision_parameters or {})
        self.reference_parameters = dict(reference_parameters or {})
        self.steady_off_parameters = dict(steady_off_parameters or {})
        self.duplicate_suppression_enabled = duplicate_suppression_enabled
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.duplicate_containment_threshold = duplicate_containment_threshold
        self.duplicate_center_distance_ratio = duplicate_center_distance_ratio
        self.weak_birth_enabled = weak_birth_enabled
        self.weak_birth_confidence = weak_birth_confidence
        self.weak_birth_min_hits = max(2, int(weak_birth_min_hits))
        self.weak_birth_min_seconds = weak_birth_min_seconds
        self.weak_birth_max_gap_seconds = weak_birth_max_gap_seconds
        self._weak_candidates = []
        self._last_timestamp = None
        self.suppressed_detections = 0
        self.bbox_smoothing_alpha = bbox_smoothing_alpha
        if association_mode not in ("greedy", "two_stage_global"):
            raise ValueError(
                "association_mode must be greedy or two_stage_global"
            )
        if not 0.0 <= low_confidence_threshold <= high_confidence_threshold:
            raise ValueError(
                "low confidence threshold must not exceed high threshold"
            )
        if not 0.0 <= stationary_reid_size_similarity_min <= 1.0:
            raise ValueError(
                "stationary re-ID size similarity must be in [0, 1]"
            )
        self.association_mode = association_mode
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.low_stage_iou_threshold = low_stage_iou_threshold
        self.low_stage_center_distance_ratio = (
            low_stage_center_distance_ratio
        )
        self.motion_horizon_seconds = max(0.0, motion_horizon_seconds)
        self.temporal_reset_distance_ratio = max(
            0.0, float(temporal_reset_distance_ratio)
        )
        self.stationary_reid_enabled = bool(stationary_reid_enabled)
        self.stationary_reid_max_age_seconds = max(
            self.max_age_seconds,
            float(stationary_reid_max_age_seconds),
        )
        self.stationary_reid_min_hits = max(
            1, int(stationary_reid_min_hits)
        )
        self.stationary_reid_iou_threshold = max(
            0.0, float(stationary_reid_iou_threshold)
        )
        self.stationary_reid_center_distance_ratio = max(
            0.0, float(stationary_reid_center_distance_ratio)
        )
        self.stationary_reid_size_similarity_min = float(
            stationary_reid_size_similarity_min
        )
        self.tracks: Dict[int, Track] = {}
        self.dormant_tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(
        self, detections: Sequence[Detection], timestamp: float, update_weak_candidates: bool = True
    ) -> List[Track]:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            self.tracks.clear()
            self.dormant_tracks.clear()
            self._weak_candidates.clear()
        self._last_timestamp = timestamp
        self.suppressed_detections = 0
        if self.duplicate_suppression_enabled:
            detections = self._suppress_detections(detections)
        self._retire_stale_tracks(timestamp)
        for track in self.tracks.values():
            track.visible = False
            track.association_stage = "MISSING"
        if self.association_mode == "two_stage_global":
            matched_detections = self._update_two_stage(
                detections, timestamp
            )
        else:
            matched_detections = self._update_greedy(
                detections, timestamp
            )

        if self.stationary_reid_enabled:
            reid_candidates = [
                index for index, detection in enumerate(detections)
                if (
                    index not in matched_detections
                    and detection.confidence
                    >= self.low_confidence_threshold
                )
            ]
            for track_id, detection_index in self._stationary_reid_matches(
                reid_candidates, detections, timestamp
            ):
                track = self.dormant_tracks.pop(track_id)
                self._prepare_dormant_reid(track, timestamp)
                self.tracks[track_id] = track
                self._apply_match(
                    track, detections[detection_index], timestamp, "REID"
                )
                matched_detections.add(detection_index)

        if self.association_mode == "two_stage_global":
            new_detection_indices = [
                index for index, detection in enumerate(detections)
                if (
                    index not in matched_detections
                    and detection.confidence
                    >= self.high_confidence_threshold
                )
            ]
        else:
            new_detection_indices = [
                index for index in range(len(detections))
                if index not in matched_detections
            ]
        for index in new_detection_indices:
            if self._near_existing(detections[index].bbox):
                self.suppressed_detections += 1
                continue
            self._create_track(detections[index], timestamp)
        # A crop does not observe distant weak candidates and cannot erase them.
        if update_weak_candidates:
            self._update_weak_births(detections, matched_detections, timestamp)
        for track in self.tracks.values():
            if not track.visible:
                track.consecutive_hits = 0
        if self.duplicate_suppression_enabled:
            self._suppress_tracks()
        return sorted(self.tracks.values(), key=lambda track: track.track_id)

    def _duplicate_boxes(self, first, second):
        x = max(0, min(first[2], second[2]) - max(first[0], second[0]))
        y = max(0, min(first[3], second[3]) - max(first[1], second[1]))
        minimum_area = max(1, min(
            (first[2] - first[0]) * (first[3] - first[1]),
            (second[2] - second[0]) * (second[3] - second[1]),
        ))
        scale = max(1, min(first[2] - first[0], first[3] - first[1],
                           second[2] - second[0], second[3] - second[1]))
        a, b = bbox_center(first), bbox_center(second)
        close = hypot(a[0] - b[0], a[1] - b[1]) / scale
        return (
            bbox_iou(first, second) >= self.duplicate_iou_threshold
            or (x * y / minimum_area >= self.duplicate_containment_threshold
                and close <= 0.75)
            or (close <= self.duplicate_center_distance_ratio
                and bbox_size_similarity(first, second) >= 0.45)
        )

    def _near_existing(self, bbox):
        return self.duplicate_suppression_enabled and any(
            self._duplicate_boxes(bbox, analysis_bbox(track, "detector"))
            for track in self.tracks.values()
        )

    def _suppress_detections(self, detections):
        # Prefer spatial continuity over a small confidence fluctuation.
        def rank(detection):
            overlap = max((bbox_iou(detection.bbox, analysis_bbox(t, "detector"))
                           for t in self.tracks.values() if t.hits >= 3), default=0)
            return detection.confidence + 0.15 * overlap
        kept = []
        for detection in sorted(detections, key=rank, reverse=True):
            if any(self._duplicate_boxes(detection.bbox, item.bbox) for item in kept):
                self.suppressed_detections += 1
            else:
                kept.append(detection)
        return kept

    def _suppress_tracks(self):
        kept = []
        for track in sorted(self.tracks.values(),
                            key=lambda t: (not t.visible, t.created_at, t.track_id)):
            if any(self._duplicate_boxes(analysis_bbox(track, "detector"),
                                         analysis_bbox(other, "detector"))
                   for other in kept):
                # Never combine two IDs' histories. Also suppress in JSON.
                del self.tracks[track.track_id]
            else:
                kept.append(track)

    def _update_weak_births(self, detections, matched, timestamp):
        if not self.weak_birth_enabled:
            return
        remaining = [dict(item) for item in self._weak_candidates
                     if 0 < timestamp - item["last"] <= self.weak_birth_max_gap_seconds]
        pending = []
        for index, detection in enumerate(detections):
            if (index in matched or not self.weak_birth_confidence <= detection.confidence
                    < self.high_confidence_threshold or self._near_existing(detection.bbox)):
                continue
            options = [(bbox_iou(item["anchor"], detection.bbox), j)
                       for j, item in enumerate(remaining)
                       if bbox_size_similarity(item["anchor"], detection.bbox) >= 0.6
                       and normalized_center_distance(item["anchor"], detection.bbox) <= 0.2]
            if options:
                _, j = max(options)
                item = remaining.pop(j)
                item["hits"] += 1
            else:
                item = {"anchor": detection.bbox, "since": timestamp, "hits": 1}
            item["last"] = timestamp
            if (item["hits"] >= self.weak_birth_min_hits
                    and timestamp - item["since"] + 1e-6 >= self.weak_birth_min_seconds):
                self._create_track(detection, timestamp)
                self.tracks[self.next_id - 1].association_stage = "WEAK_CONFIRMED"
                self.tracks[self.next_id - 1].weak_birth_floor = self.weak_birth_confidence
            else:
                pending.append(item)
        self._weak_candidates = pending

    def _retire_stale_tracks(self, timestamp: float) -> None:
        expired = [
            track_id for track_id, track in self.tracks.items()
            if timestamp - (max(track.last_seen, track.roi_last_observed)
                            if track.roi_tracking_active else track.last_seen) > self.max_age_seconds
        ]
        for track_id in expired:
            track = self.tracks.pop(track_id)
            if (
                self.stationary_reid_enabled
                and track.hits >= self.stationary_reid_min_hits
            ):
                track.visible = False
                track.association_stage = "DORMANT"
                self.dormant_tracks[track_id] = track
        stale_dormant = [
            track_id for track_id, track in self.dormant_tracks.items()
            if (
                timestamp - track.last_seen
                > self.stationary_reid_max_age_seconds
            )
        ]
        for track_id in stale_dormant:
            del self.dormant_tracks[track_id]

    def _stationary_reid_matches(
        self,
        detection_indices: Sequence[int],
        detections: Sequence[Detection],
        timestamp: float,
    ) -> List[Tuple[int, int]]:
        """Strict one-to-one recovery for a stationary camera/vehicle."""
        track_ids = [
            track_id
            for track_id, track in sorted(self.dormant_tracks.items())
            if (
                timestamp - track.last_seen
                <= self.stationary_reid_max_age_seconds
            )
        ]
        if not track_ids or not detection_indices:
            return []
        candidates = {}
        for track_id in track_ids:
            track = self.dormant_tracks[track_id]
            reference = track.detector_bbox or track.bbox
            options = []
            for detection_index in detection_indices:
                detection = detections[detection_index]
                iou = bbox_iou(reference, detection.bbox)
                distance = normalized_center_distance(
                    reference, detection.bbox
                )
                size_similarity = bbox_size_similarity(
                    reference, detection.bbox
                )
                if (
                    size_similarity
                    < self.stationary_reid_size_similarity_min
                    or (
                        iou < self.stationary_reid_iou_threshold
                        and distance
                        > self.stationary_reid_center_distance_ratio
                    )
                ):
                    continue
                distance_support = max(
                    0.0,
                    1.0 - distance / max(
                        self.stationary_reid_center_distance_ratio, 1e-6
                    ),
                )
                score = (
                    iou
                    + 0.30 * distance_support
                    + 0.25 * size_similarity
                    + 0.01 * detection.confidence
                )
                options.append((detection_index, score))
            candidates[track_id] = options

        detection_bits = {
            detection_index: 1 << position
            for position, detection_index in enumerate(detection_indices)
        }

        @lru_cache(maxsize=None)
        def solve(position: int, used_mask: int):
            if position == len(track_ids):
                return 0, 0.0, ()
            best = solve(position + 1, used_mask)
            track_id = track_ids[position]
            for detection_index, candidate_score in candidates[track_id]:
                bit = detection_bits[detection_index]
                if used_mask & bit:
                    continue
                count, score, pairs = solve(
                    position + 1, used_mask | bit
                )
                candidate = (
                    count + 1,
                    score + candidate_score,
                    ((track_id, detection_index),) + pairs,
                )
                if candidate[:2] > best[:2]:
                    best = candidate
            return best

        _, _, best_pairs = solve(0, 0)
        return list(best_pairs)

    @staticmethod
    def _prepare_dormant_reid(track: Track, timestamp: float) -> None:
        # The same physical box keeps its ID and learned ON/OFF reference,
        # but its light state may have changed while YOLO was absent.
        track.analyzer.reset()
        track.steady_off.reset()
        track.decision.reset(timestamp=timestamp, identity_reset=False)
        track.temporal = TemporalResult()
        track.instant = HsvColorClassifier._empty()
        track.raw_instant = HsvColorClassifier._empty()
        track.confirmed_instant_label = "UNKNOWN"
        track.pending_instant_label = "UNKNOWN"
        track.pending_instant_count = 0
        track.previous_detector_bbox = None
        track.previous_seen = None

    def _update_greedy(
        self, detections: Sequence[Detection], timestamp: float
    ) -> Set[int]:
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                reference_bbox = track.detector_bbox or track.bbox
                iou = bbox_iou(reference_bbox, detection.bbox)
                distance = normalized_center_distance(
                    reference_bbox, detection.bbox
                )
                if (
                    iou >= self.iou_threshold
                    or distance <= self.center_distance_ratio
                ):
                    candidates.append(
                        (iou - 0.1 * distance, track_id, detection_index)
                    )
        matched_tracks = set()
        matched_detections = set()
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if (
                track_id in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            detection = detections[detection_index]
            track = self.tracks[track_id]
            self._apply_match(
                track, detection, timestamp, "BASELINE"
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
        return matched_detections

    def _update_two_stage(
        self, detections: Sequence[Detection], timestamp: float
    ) -> Set[int]:
        high = [
            index for index, detection in enumerate(detections)
            if detection.confidence >= self.high_confidence_threshold
        ]
        low = [
            index for index, detection in enumerate(detections)
            if self.low_confidence_threshold
            <= detection.confidence
            < self.high_confidence_threshold
        ]
        track_ids = sorted(self.tracks)
        first_matches = self._optimal_matches(
            track_ids,
            high,
            detections,
            timestamp,
            self.iou_threshold,
            self.center_distance_ratio,
        )
        matched_tracks = set()
        matched_detections = set()
        for track_id, detection_index in first_matches:
            self._apply_match(
                self.tracks[track_id], detections[detection_index],
                timestamp, "HIGH",
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
        remaining_tracks = [
            track_id for track_id in track_ids
            if track_id not in matched_tracks
        ]
        second_matches = self._optimal_matches(
            remaining_tracks,
            low,
            detections,
            timestamp,
            self.low_stage_iou_threshold,
            self.low_stage_center_distance_ratio,
        )
        for track_id, detection_index in second_matches:
            self._apply_match(
                self.tracks[track_id], detections[detection_index],
                timestamp, "LOW",
            )
            matched_detections.add(detection_index)
        return matched_detections

    def _optimal_matches(
        self,
        track_ids: Sequence[int],
        detection_indices: Sequence[int],
        detections: Sequence[Detection],
        timestamp: float,
        iou_threshold: float,
        distance_threshold: float,
    ) -> List[Tuple[int, int]]:
        """Find a maximum-cardinality, maximum-score gated assignment."""
        if not track_ids or not detection_indices:
            return []
        candidates = {}
        for track_id in track_ids:
            reference = self._predicted_bbox(
                self.tracks[track_id], timestamp
            )
            options = []
            for detection_index in detection_indices:
                detection = detections[detection_index]
                iou = bbox_iou(reference, detection.bbox)
                distance = normalized_center_distance(
                    reference, detection.bbox
                )
                if (
                    iou >= iou_threshold
                    or distance <= distance_threshold
                ):
                    distance_support = max(
                        0.0,
                        1.0 - distance / max(distance_threshold, 1e-6),
                    )
                    score = (
                        iou + 0.20 * distance_support
                        + 0.01 * detection.confidence
                    )
                    options.append((detection_index, score))
            candidates[track_id] = options

        detection_bits = {
            detection_index: 1 << position
            for position, detection_index in enumerate(detection_indices)
        }

        @lru_cache(maxsize=None)
        def solve(position: int, used_mask: int):
            if position == len(track_ids):
                return 0, 0.0, ()
            best = solve(position + 1, used_mask)
            track_id = track_ids[position]
            for detection_index, candidate_score in candidates[track_id]:
                bit = detection_bits[detection_index]
                if used_mask & bit:
                    continue
                count, score, pairs = solve(
                    position + 1, used_mask | bit
                )
                candidate = (
                    count + 1,
                    score + candidate_score,
                    ((track_id, detection_index),) + pairs,
                )
                if candidate[:2] > best[:2]:
                    best = candidate
            return best

        _, _, best_pairs = solve(0, 0)
        return list(best_pairs)

    def _predicted_bbox(self, track: Track, timestamp: float) -> BBox:
        if track.measurement_bbox is not None and abs(timestamp - track.measurement_time) < 1e-5:
            return track.measurement_bbox
        current = track.detector_bbox or track.bbox
        previous = track.previous_detector_bbox
        if previous is None or track.previous_seen is None:
            return current
        observation_dt = track.last_seen - track.previous_seen
        elapsed = timestamp - track.last_seen
        if observation_dt <= 1e-6 or elapsed <= 0.0:
            return current
        horizon = min(elapsed, self.motion_horizon_seconds)
        previous_center = bbox_center(previous)
        current_center = bbox_center(current)
        dx = (
            (current_center[0] - previous_center[0])
            * horizon / observation_dt
        )
        dy = (
            (current_center[1] - previous_center[1])
            * horizon / observation_dt
        )
        width = max(1, current[2] - current[0])
        height = max(1, current[3] - current[1])
        limit = 1.5 * max(width, height)
        magnitude = hypot(dx, dy)
        if magnitude > limit:
            scale = limit / magnitude
            dx *= scale
            dy *= scale
        return tuple(int(round(value)) for value in (
            current[0] + dx,
            current[1] + dy,
            current[2] + dx,
            current[3] + dy,
        ))

    def _apply_match(
        self,
        track: Track,
        detection: Detection,
        timestamp: float,
        stage: str,
    ) -> None:
        reference_bbox = track.measurement_bbox or track.detector_bbox or track.bbox
        association_jump = normalized_center_distance(
            reference_bbox, detection.bbox
        )
        if (
            self.temporal_reset_distance_ratio > 0.0
            and association_jump > self.temporal_reset_distance_ratio
        ):
            track.analyzer.reset()
            track.blink_reference.reset()
            track.steady_off.reset()
            track.decision.reset(
                timestamp=timestamp, identity_reset=True
            )
            track.temporal = TemporalResult()
            track.instant = HsvColorClassifier._empty()
            track.raw_instant = HsvColorClassifier._empty()
            track.confirmed_instant_label = "UNKNOWN"
            track.pending_instant_label = "UNKNOWN"
            track.pending_instant_count = 0
            track.last_reliable_color = "UNKNOWN"
            track.emitter_center_hint = None
            track.roi_epoch += 1
        track.bbox = smooth_bbox(
            track.bbox, detection.bbox, self.bbox_smoothing_alpha
        )
        track.previous_detector_bbox = track.detector_bbox
        track.previous_seen = track.last_seen
        track.detector_bbox = detection.bbox
        track.detector_confidence = detection.confidence
        track.last_seen = timestamp
        track.hits += 1
        track.consecutive_hits += 1
        track.visible = True
        track.association_stage = stage

    def _create_track(
        self, detection: Detection, timestamp: float
    ) -> None:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = Track(
            track_id=track_id,
            bbox=detection.bbox,
            detector_confidence=detection.confidence,
            detector_bbox=detection.bbox,
            created_at=timestamp,
            last_seen=timestamp,
            analyzer=TemporalAnalyzer(**self.analyzer_parameters),
            blink_reference=BlinkPhotometricReference(
                **self.reference_parameters
            ),
            decision=StateDecision(**self.decision_parameters),
            steady_off=SteadyNonBlueOff(**self.steady_off_parameters),
            association_stage=(
                "HIGH"
                if self.association_mode == "two_stage_global"
                else "BASELINE"
            ),
        )


def smooth_bbox(previous: BBox, current: BBox, alpha: float) -> BBox:
    """Blend a new YOLO box with the displayed box to reduce frame jitter."""
    return tuple(
        int(round(alpha * new + (1.0 - alpha) * old))
        for old, new in zip(previous, current)
    )


def bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1)


def bbox_size_similarity(first: BBox, second: BBox) -> float:
    """Symmetric width/height agreement used by stationary re-ID."""
    first_width = max(1, first[2] - first[0])
    first_height = max(1, first[3] - first[1])
    second_width = max(1, second[2] - second[0])
    second_height = max(1, second[3] - second[1])
    width_similarity = min(first_width, second_width) / max(
        first_width, second_width
    )
    height_similarity = min(first_height, second_height) / max(
        first_height, second_height
    )
    return float(width_similarity * height_similarity)


def normalized_center_distance(first: BBox, second: BBox) -> float:
    first_center = bbox_center(first)
    second_center = bbox_center(second)
    scale = max(
        first[2] - first[0], first[3] - first[1],
        second[2] - second[0], second[3] - second[1], 1,
    )
    distance = hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    return distance / scale


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    return (
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
    )
