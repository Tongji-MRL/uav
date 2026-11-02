"""Per-ROI online two-level light reference, independent of detector scores.

Fit unlabelled, spatially valid intensity samples. Do not require an already
recognized blink in order to learn its OFF level. Single-level scenes never
manufacture a second level. Unknown observations never establish OFF.
"""
from collections import deque
from dataclasses import replace
import math
import numpy as np

from .periodic_light import measure_periodicity
from .light_reference_bank import LightReferenceBank

COLORS = ("RED", "GREEN", "BLUE")
DEFAULTS = {
    "enabled": False,
    "window_seconds": 6.0,
    "min_samples": 24,
    "min_cluster_seconds": 0.30,
    "min_separation": 0.025,
    "noise_multiplier": 3.5,
    "off_ratio": 0.35,
    "on_ratio": 0.65,
    "adapt_seconds": 3.0,
    "max_gap_seconds": 0.35,
    "reference_hold_seconds": 2.5,
    "bootstrap_off_fraction": 0.45,
    "bootstrap_on_frames": 2,
    "bootstrap_on_signal_min": 0.05,
    "bootstrap_on_contrast_min": 12.0,
    "min_motion_quality": 0.65,
    "lock_emitter_center": True,
    "center_x": 0.5,
    "center_y": 0.5,
    "color_change_frames": 3,
    "reset_generation": 0,
    "min_measurement_quality": 0.55,
    "require_emitted_on_for_learning": True,
    "fit_interval_seconds": 0.2,
    "require_periodic": False,
}


class OnlineLightReference:
    def __init__(self, parameters, periodic_parameters=None):
        self.p = dict(DEFAULTS)
        self.p.update(parameters)
        numeric = {k: float(v) for k, v in self.p.items() if k not in ("enabled", "lock_emitter_center", "require_emitted_on_for_learning", "require_periodic")}
        if not all(math.isfinite(v) for v in numeric.values()):
            raise ValueError("Online light parameters must be finite")
        if not 2 <= self.p["window_seconds"] <= 30 or not 6 <= self.p["min_samples"] <= 360:
            raise ValueError("online_light window must be 2..30 s and min_samples 6..360")
        if not 0 < self.p["off_ratio"] < self.p["on_ratio"] < 1:
            raise ValueError("Require 0 < online_light.off_ratio < on_ratio < 1")
        if not 0.005 <= self.p["min_separation"] <= 0.5 or not 1 <= self.p["noise_multiplier"] <= 10:
            raise ValueError("Invalid online light separation/noise multiplier")
        if not 0.1 <= self.p["min_cluster_seconds"] <= 1 or not 0.5 <= self.p["adapt_seconds"] <= 30:
            raise ValueError("Invalid online light dwell/adaptation time")
        if not 0 < self.p["max_gap_seconds"] < 1 or not 0 <= self.p["min_motion_quality"] <= 1:
            raise ValueError("Invalid online light gap/ROI quality")
        if not self.p["max_gap_seconds"] < self.p["reference_hold_seconds"] <= 10:
            raise ValueError("Light reference hold must exceed sampling gap and be <= 10 s")
        if not 0.1 <= self.p["bootstrap_off_fraction"] <= 0.75 or not 2 <= self.p["bootstrap_on_frames"] <= 12:
            raise ValueError("Invalid bootstrap OFF fraction/ON confirmation")
        if not 0.01 <= self.p["bootstrap_on_signal_min"] <= 0.5 or not 4 <= self.p["bootstrap_on_contrast_min"] <= 80:
            raise ValueError("Invalid bootstrap ON signal/channel contrast")
        if not 0 <= self.p["center_x"] <= 1 or not 0 <= self.p["center_y"] <= 1:
            raise ValueError("Online light center_x/center_y must be in [0, 1]")
        if not 1 <= self.p["color_change_frames"] <= 30 or self.p["reset_generation"] < 0:
            raise ValueError("Invalid color confirmation/reset generation")
        if not 0 <= self.p["min_measurement_quality"] <= 1:
            raise ValueError("min_measurement_quality must be within [0, 1]")
        if not isinstance(self.p["require_emitted_on_for_learning"], bool):
            raise ValueError("require_emitted_on_for_learning must be true or false")
        if not 0.1 <= self.p["fit_interval_seconds"] <= 2.0:
            raise ValueError("fit_interval_seconds must be 0.1..2 seconds")
        if not isinstance(self.p["require_periodic"], bool):
            raise ValueError("require_periodic must be true or false")
        self.periodic_parameters = dict(min_hz=0.45, max_hz=0.55,
            min_agreement=0.82, assist_min_agreement=0.78, phase_steps=40)
        self.periodic_parameters.update(periodic_parameters or {})
        pp = self.periodic_parameters
        if (not all(math.isfinite(float(v)) for v in pp.values())
                or not 0 < pp["min_hz"] <= pp["max_hz"]
                or not 0.5 <= pp["assist_min_agreement"] <= pp["min_agreement"] <= 1
                or not 4 <= pp["phase_steps"] <= 360):
            raise ValueError("Invalid periodic reference parameters")
        self.periodic_evidence = {"valid": False, "status": "WAITING_FOR_CYCLES"}
        self.periodic_attempt = dict(self.periodic_evidence)
        self.peer_references = []
        self.learning_allowed = True
        self.samples = deque(maxlen=600)
        self.levels = None
        self.phase_levels = None
        self.phase_fit_stamp = None
        self.phase_fit_quality = 0.0
        self.phase_reference_established = False
        self.phase = "UNCERTAIN"
        self.color = "UNKNOWN"
        self.pending_color, self.color_frames = "UNKNOWN", 0
        self.last_stamp = None
        self.last_valid_stamp = None
        self.last_box = None
        self.last_fit = None
        self.fit_stamp = None
        self.fit_quality = 0.0
        self.reset_count = 0
        self.reason = "warming_up"
        self.outside_since = None
        self.learned_now = False
        self.on_samples = deque(maxlen=12)
        self.on_reference = None
        self.on_stamp = None
        self.on_count = 0
        self.source_on_valid = False
        self.last_lamp_epoch = None

    def reset(self, reason):
        self.periodic_evidence = {"valid": False, "status": reason}
        self.periodic_attempt = dict(self.periodic_evidence)
        self.samples.clear()
        self.on_samples.clear()
        self.on_reference = None
        self.on_stamp = None
        self.on_count = 0
        self.source_on_valid = False
        self.levels = None
        self.phase_levels = None
        self.phase_fit_stamp = None
        self.phase_fit_quality = 0.0
        self.phase_reference_established = False
        self.phase = "UNCERTAIN"
        self.last_valid_stamp = None
        self.last_fit = None
        self.fit_stamp = None
        self.fit_quality = 0.0
        self.outside_since = None
        self.reason = reason
        self.reset_count += 1

    @staticmethod
    def _geometry_changed(previous, current):
        if previous is None:
            return False
        pw, ph = max(previous[2] - previous[0], 1), max(previous[3] - previous[1], 1)
        cw, ch = max(current[2] - current[0], 1), max(current[3] - current[1], 1)
        shift = math.hypot((current[0] + current[2] - previous[0] - previous[2]) * 0.5,
                           (current[1] + current[3] - previous[1] - previous[3]) * 0.5)
        return max(cw / pw, pw / cw, ch / ph, ph / ch) > 1.6 or shift > 0.6 * math.hypot(pw, ph)

    def _fit(self, stamp):
        if len(self.samples) < self.p["min_samples"]:
            return
        # Refitting and per-frame phase measurement have independent rates.
        if self.last_fit is not None and stamp - self.last_fit < self.p["fit_interval_seconds"]:
            return
        self.last_fit = stamp
        t = np.asarray([s[0] for s in self.samples], dtype=np.float64)
        x = np.asarray([s[1] for s in self.samples], dtype=np.float64)
        lo, hi = np.percentile(x, [20, 80])
        for _ in range(5):
            low = x <= (lo + hi) * 0.5
            if min(np.count_nonzero(low), np.count_nonzero(~low)) < 3:
                return
            lo, hi = float(np.median(x[low])), float(np.median(x[~low]))
        low_noise = 1.4826 * float(np.median(np.abs(x[low] - lo)))
        high_noise = 1.4826 * float(np.median(np.abs(x[~low] - hi)))
        separation = hi - lo
        noise = max(low_noise, high_noise, 0.002)
        if separation < max(self.p["min_separation"], self.p["noise_multiplier"] * noise):
            return
        # A numerical two-cluster fit cannot prove that a background object emitted light.
        if self.p["require_emitted_on_for_learning"] and sum(
                bool(s[2]) for s, is_low in zip(self.samples, low) if not is_low) < self.p["bootstrap_on_frames"]:
            self.reason = "waiting_for_emitted_on"
            return
        gaps = np.diff(t)
        # Retained light references must not fill a gap with invented samples.
        dt = np.where(gaps <= self.p["max_gap_seconds"], gaps, 0.0)
        weight = np.zeros(len(x))
        weight[:-1] += dt * 0.5
        weight[1:] += dt * 0.5
        # Require at least LOW-HIGH-LOW or HIGH-LOW-HIGH with sustained plateaus.
        # An exposure step or a constant illuminated surface is not a blink.
        confident = np.where(x <= lo + 0.35 * separation, 0,
                             np.where(x >= lo + 0.65 * separation, 1, -1))
        plateaus = []
        active, duration = -1, 0.0
        for index, (phase, seconds) in enumerate(zip(confident, weight)):
            separated = index > 0 and gaps[index - 1] > self.p["max_gap_seconds"]
            if phase != active or separated:
                if active >= 0 and duration >= self.p["min_cluster_seconds"]:
                    plateaus.append(active)
                if separated:
                    plateaus.append(None)
                active, duration = int(phase), 0.0
            duration += float(seconds)
        if active >= 0 and duration >= self.p["min_cluster_seconds"]:
            plateaus.append(active)
        switches = sum(a is not None and b is not None and a != b for a, b in zip(plateaus, plateaus[1:]))
        sustained = min(np.sum(weight[low]), np.sum(weight[~low])) >= self.p["min_cluster_seconds"]
        if self.p["require_periodic"] and switches >= 1 and sustained:
            # A measured high-to-low change can already suppress a dark hue.
            # This provisional boundary is not a periodic reference or a donor.
            self.phase_reference_established = self._phase_levels(stamp) is None
            self.phase_levels, self.phase_fit_stamp = (lo, hi), stamp
            self.phase_fit_quality = float(np.clip(1-noise/separation, 0.0, 1.0))
            if self.phase_reference_established:
                self.color = self._high_color(lo, hi)
                self.pending_color, self.color_frames = "UNKNOWN", 0
        if switches < 2 or not sustained:
            return
        if self.p["require_periodic"]:
            settings = dict(self.periodic_parameters)
            settings["min_agreement"] = settings["assist_min_agreement"]
            evidence = measure_periodicity(t, x, lo, hi, self.p["max_gap_seconds"], settings)
            support = LightReferenceBank.compare(self.peer_references,
                evidence.get("frequency_hz") or 0.0, separation/max(abs(hi), 1e-9))
            independent = bool(evidence["valid"] and
                               evidence["agreement"] >= self.periodic_parameters["min_agreement"])
            accepted = bool(evidence["valid"] and (independent or support["support_count"] >= 2))
            evidence.update(valid=accepted, independent=independent, peer_support=support)
            if accepted:
                evidence["status"] = "INDEPENDENT_PERIODIC" if independent else "PEER_SUPPORTED_PERIODIC"
            elif evidence.get("cycles", 0) >= 2 and not independent:
                evidence["status"] = "WAITING_FOR_STRONGER_PERIODIC_EVIDENCE"
            self.periodic_attempt = evidence
            if not accepted:
                self.reason = evidence["status"]
                return
            self.periodic_evidence = evidence
        if self.levels is None:
            if self.p["require_periodic"]:
                # Only a verified high plateau may establish the reference hue.
                # Lower-plateau reflections stay out of this vote.
                self.color = self._high_color(lo, hi)
                self.pending_color, self.color_frames = "UNKNOWN", 0
                self.on_reference, self.on_stamp = hi, stamp
                self.on_samples.clear()
                self.on_count = 0
            self.levels = (lo, hi)
            self.learned_now = True
        else:
            elapsed = max(0.0, stamp - (self.fit_stamp if self.fit_stamp is not None else stamp))
            alpha = min(0.25, 1.0 - math.exp(-elapsed / self.p["adapt_seconds"]))
            self.levels = tuple((1.0 - alpha) * old + alpha * new
                                for old, new in zip(self.levels, (lo, hi)))
        self.fit_stamp = stamp
        self.fit_quality = float(np.clip(1 - noise / max(separation, 1e-9), 0.0, 1.0))
        self.reason = "two_levels_learned"

    def update(self, observation, stamp, box, motion_quality, *, allow_learning=True, peer_references=()):
        if not self.p["enabled"]:
            return observation
        self.learned_now = False
        self.phase_reference_established = False
        self.peer_references = list(peer_references)
        self.learning_allowed = bool(allow_learning)
        self.source_on_valid = False
        reset_event = False
        lamp = observation.emission_diagnostics.get("lamp_region")
        if lamp is not None:
            if self.last_lamp_epoch is not None and lamp["epoch"] != self.last_lamp_epoch:
                self.reset("lamp_geometry_changed")
                reset_event = True
            self.last_lamp_epoch = lamp["epoch"]
        if (self.p["require_periodic"] and self.fit_stamp is not None
                and stamp-self.fit_stamp > self.p["reference_hold_seconds"]):
            self.reset("periodic_reference_expired")
            reset_event = True
        if self.last_stamp is not None:
            gap = stamp - self.last_stamp
            if gap <= 0 or gap > self.p["reference_hold_seconds"]:
                self.reset("time_reversed" if gap <= 0 else "reference_expired")
                reset_event = True
            elif gap > self.p["max_gap_seconds"]:
                # Retain measured levels across a brief loss, but never retain a
                # phase across it or add OFF observations for missing frames.
                self.phase = "UNCERTAIN"
                self.on_count = 0
        if self._geometry_changed(self.last_box, box):
            self.reset("roi_geometry_changed")
            reset_event = True
        self.last_stamp, self.last_box = stamp, tuple(box)
        signal = float(observation.luminance_signal)
        # Color confidence can disappear on a valid dark ROI. Gate sampling by
        # spatial tracking; require color/emission evidence only for ON seeds.
        spatial_valid = (math.isfinite(signal) and math.isfinite(motion_quality)
                         and motion_quality >= self.p["min_motion_quality"]
                          and observation.photometric_phase != "UNOBSERVED"
                          and (lamp is None or (lamp["valid"] and
                               observation.measurement_quality >= self.p["min_measurement_quality"])))
        if not spatial_valid:
            self.phase = "UNCERTAIN"
            self.on_count = 0
            self.on_samples.clear()
            result = self._output(observation, "UNKNOWN", 0.0, "roi_quality_low", reset_event)
            if lamp is not None:
                result = replace(result, photometric_phase="UNOBSERVED", measurement_quality=0.0)
            return result
        if self.last_valid_stamp is not None and stamp - self.last_valid_stamp > self.p["reference_hold_seconds"]:
            self.reset("valid_measurement_gap")
            reset_event = True
        if allow_learning:
            self.last_valid_stamp = stamp
        # Dark observations may teach the low brightness plateau, never a hue.
        phase_levels = self._phase_levels(stamp)
        if phase_levels is None:
            color_off = self._below_on_reference(signal)
        else:
            lo, hi = phase_levels
            z = (signal-lo)/max(hi-lo, self.p["min_separation"])
            color_off = (z <= self.p["off_ratio"]
                         or (self.phase == "OFF" and z < self.p["on_ratio"]))
        source_on = not color_off and self._source_on(observation, signal)
        self.source_on_valid = source_on
        candidate = (self._on_color(observation) if self.p["require_periodic"] else observation.label) if source_on else "UNKNOWN"
        if not allow_learning and candidate in COLORS and self.color in COLORS and candidate != self.color:
            self.phase = "UNCERTAIN"
            return self._output(observation, "UNKNOWN", 0.0, "recovered_color_requires_confirmation", reset_event)
        if allow_learning:
            collecting_periodic = self.p["require_periodic"] and self.levels is None
            # Before own cycles are established, a low-amplitude dark reflection
            # may look emitted. Keep its brightness without committing its hue.
            # Color changes start a new light reference rather than applying a red
            # threshold to a dimmer blue lamp. OFF colors do not switch the reference.
            if not collecting_periodic and candidate in COLORS and candidate != self.color:
                self.color_frames = self.color_frames + 1 if self.pending_color == candidate else 1
                self.pending_color = candidate
                if self.color_frames >= self.p["color_change_frames"]:
                    if self.color in COLORS:
                        self.reset("emission_color_changed")
                        reset_event = True
                    self.color = candidate
                    self.pending_color, self.color_frames = "UNKNOWN", 0
            else:
                self.pending_color, self.color_frames = "UNKNOWN", 0
            if not collecting_periodic and candidate in COLORS and self.color in COLORS and candidate != self.color:
                # A newly emitted hue can be dimmer than the previous hue. Do not
                # call it OFF using the previous color's threshold while confirming.
                self.phase = "UNCERTAIN"
                self.on_count = 0
                return self._output(observation, "UNKNOWN", 0.0, "emission_color_pending", reset_event)
            # A measured fall from a confirmed colored ON can supply the first OFF
            # before a full two-level fit exists. Detector presence never participates.
            relative_dark = self._below_on_reference(signal)
            if candidate in COLORS and not relative_dark:
                if self.on_count == 0:
                    self.on_samples.clear()
                self.on_count += 1
                self.on_samples.append(signal)
                self.on_stamp = stamp
                if self.on_count >= self.p["bootstrap_on_frames"]:
                    self.on_reference = float(np.median(self.on_samples))
            else:
                self.on_count = 0
            relative_dark = self._below_on_reference(signal)
            if self.levels is not None:
                lo, hi = self.levels
                span = hi - lo
                outside = signal < lo - 2 * span or signal > hi + 2 * span
                if outside:
                    self.outside_since = stamp if self.outside_since is None else self.outside_since
                    if stamp - self.outside_since >= 0.5:
                        self.reset("intensity_range_changed")
                        reset_event = True
                else:
                    self.outside_since = None
            if (self.p["require_periodic"] and self.samples
                    and stamp-self.samples[-1][0] > self.p["max_gap_seconds"]):
                # Neither clustering nor emission seeds may span an observation gap.
                self.samples.clear()
            # Emission and hue are separate evidence: clipped/ambiguous ON can
            # teach brightness without inventing a color.
            sample = (stamp, signal, source_on if self.p["require_periodic"] else source_on and candidate in COLORS)
            self.samples.append(sample + (candidate,) if self.p["require_periodic"] else sample)
            while self.samples and self.samples[0][0] < stamp - self.p["window_seconds"]:
                self.samples.popleft()
            self._fit(stamp)
        else:
            # A housing recovery can measure darkness using an earlier reference,
            # but cannot seed/adapt levels or prolong the trusted reference lease.
            self.on_count = 0
            self.on_samples.clear()
        phase_levels = self._phase_levels(stamp)
        if phase_levels is None:
            relative_dark = self._below_on_reference(signal)
            # The V7 accepted-color shortcut must not force ON before learning.
            # Strong spatial emission can bootstrap a solid lamp; otherwise
            # report uncertainty until temporal brightness levels separate.
            self.phase = "OFF" if relative_dark else (
                "ON" if source_on else ("OFF" if observation.photometric_phase == "OFF" else "UNCERTAIN"))
            color = self._on_color(observation) if self.p["require_periodic"] else observation.label
            label = color if self.phase == "ON" and color in COLORS else (
                "DARK" if self.phase == "OFF" else "UNKNOWN")
            color_confidence = observation.confidence
            if self.p["require_periodic"] and label in COLORS:
                color_confidence = max(color_confidence,
                    float(observation.ratios.get(label, 0))*observation.measurement_quality)
            confidence = 0.75 if relative_dark else (color_confidence if label != "UNKNOWN" else 0.0)
            return self._output(observation, label, confidence,
                                "online_relative_drop_off" if relative_dark else "online_bootstrap_" + self.phase.lower(),
                                reset_event)
        lo, hi = phase_levels
        quality = self.fit_quality if self.levels is not None else self.phase_fit_quality
        z = (signal - lo) / max(hi - lo, self.p["min_separation"])
        if z <= self.p["off_ratio"]:
            self.phase = "OFF"
        elif z >= self.p["on_ratio"]:
            self.phase = "ON"
        # Inside the hysteresis band keep the last observed phase; never use
        # the latest YOLO confidence, the presence of a box, or remembered hue.
        if self.phase == "OFF":
            label, confidence = "DARK", quality
        elif self.phase == "ON":
            label = self._on_color(observation)
            confidence = min(quality, float(observation.ratios.get(label, 0))) if label in COLORS else 0.0
        else:
            label, confidence = "UNKNOWN", 0.0
        prefix = "online_learned_" if self.levels is not None else "online_measured_levels_"
        return self._output(observation, label, confidence, prefix + self.phase.lower(), reset_event)

    def _phase_levels(self, stamp):
        if self.levels is not None:
            return self.levels
        if (self.p["require_periodic"] and self.phase_fit_stamp is not None
                and 0 <= stamp-self.phase_fit_stamp <= self.p["reference_hold_seconds"]):
            return self.phase_levels
        return None

    def _high_color(self, lo, hi):
        counts = {c: sum(s[2] and s[3] == c for s in self.samples
                         if s[1] >= lo+self.p["on_ratio"]*(hi-lo)) for c in COLORS}
        winner = max(COLORS, key=counts.get)
        return winner if counts[winner] >= max(1, 0.62*sum(counts.values())) else "UNKNOWN"

    def _on_color(self, observation):
        if observation.emission_diagnostics.get("color_valid") is False:
            return "UNKNOWN"
        if observation.label in COLORS:
            return observation.label
        votes = observation.ratios
        first, second = sorted(COLORS, key=lambda c: votes.get(c, 0), reverse=True)[:2]
        share, margin = votes.get(first, 0), votes.get(first, 0)-votes.get(second, 0)
        # Preserve the original recovery. Relax only measured ON red/green,
        # with current chroma/quality; never borrow a previous hue or relax blue.
        if share >= 0.62 and margin >= 0.18:
            return first
        if (self.p["require_periodic"] and first in ("RED", "GREEN")
                and observation.color_mass >= 0.018
                and observation.measurement_quality >= self.p["min_measurement_quality"]
                and share >= 0.55 and margin >= 0.12):
            return first
        return "UNKNOWN"

    def _source_on(self, observation, signal):
        if observation.measurement_quality < self.p["min_measurement_quality"]:
            return False
        diagnostics = observation.emission_diagnostics
        if diagnostics.get("source_on", False):
            return True
        # The strict fusion flag requires a large peak-to-ring contrast. A
        # colored halo can raise that ring while the LED face remains clearly
        # luminous. Permit an already accepted, spatially supported emitter to
        # establish its own ON reference; color alone is insufficient.
        if observation.label not in COLORS or observation.photometric_phase != "ON":
            return False
        votes = observation.ratios
        winner = observation.label
        runner_up = max(float(votes.get(c, 0.0)) for c in COLORS if c != winner)
        return (math.isfinite(signal)
                and signal >= self.p["bootstrap_on_signal_min"]
                and observation.measurement_quality >= 0.65
                and float(votes.get(winner, 0.0)) >= 0.62
                and float(votes.get(winner, 0.0)) - runner_up >= 0.18
                and float(diagnostics.get("core_peak", 0.0)) >= 110.0
                and float(diagnostics.get("channel_peak", 0.0)) >= self.p["bootstrap_on_contrast_min"]
                and float(diagnostics.get("core_color_support", 0.0)) >= 0.02
                and observation.color_mass >= 0.018)

    def _below_on_reference(self, signal):
        return (self.on_reference is not None
                and self.on_reference - signal >= self.p["min_separation"]
                and signal <= self.p["bootstrap_off_fraction"] * self.on_reference)

    def _output(self, observation, label, confidence, reason, reset_event):
        diagnostics = dict(observation.emission_diagnostics)
        lo, hi = self.levels if self.levels is not None else (None, None)
        diagnostics["online_light"] = {
            "ready": self.levels is not None, "status": self.reason,
            "low": lo, "high": hi,
            "off_threshold": None if lo is None else lo + self.p["off_ratio"] * (hi - lo),
            "on_threshold": None if lo is None else lo + self.p["on_ratio"] * (hi - lo),
            "samples": len(self.samples), "quality": round(self.fit_quality, 3),
            "reference_color": self.color, "reset_count": self.reset_count,
            "bootstrap_on_reference": self.on_reference,
            "source_on": bool(self.source_on_valid and self.phase == "ON"),
            "bootstrap_off_threshold": (None if self.on_reference is None else
                min(self.p["bootstrap_off_fraction"] * self.on_reference,
                    self.on_reference - self.p["min_separation"])),
            "reference_hold_seconds": self.p["reference_hold_seconds"],
            # The original profile retains measured temporal history. Periodic
            # mode separately clears bootstrap color votes once, on learned_now.
            "history_reset": reset_event,
            "learned_now": self.learned_now,
            "learning_allowed": self.learning_allowed,
        }
        if self.p["require_periodic"]:
            diagnostics["online_light"]["periodic"] = dict(self.periodic_evidence,
                last_attempt_status=self.periodic_attempt["status"],
                fit_stamp=self.fit_stamp,
                reference_kind="PERIODIC" if self.levels is not None else "BOOTSTRAP_NOT_REFERENCE")

        phase_levels = self._phase_levels(self.last_stamp)
        if self.p["require_periodic"]:
            plo, phi = phase_levels if phase_levels is not None else (None, None)
            diagnostics["online_light"]["phase_reference_established"] = self.phase_reference_established
            diagnostics["online_light"]["phase_reference"] = {
                "kind": "PERIODIC" if lo is not None else ("MEASURED_LEVELS" if plo is not None else "BOOTSTRAP"),
                "low": plo, "high": phi,
                "off_threshold": None if plo is None else plo+self.p["off_ratio"]*(phi-plo),
                "on_threshold": None if plo is None else plo+self.p["on_ratio"]*(phi-plo),
            }
        strength = (phase_levels[1]-phase_levels[0]) if phase_levels is not None else observation.luminance_signal
        quality = (self.fit_quality if lo is not None else self.phase_fit_quality
                   if phase_levels is not None else observation.measurement_quality)
        if self.phase == "OFF" and self.on_reference is not None:
            quality = max(quality, 0.75)
        energy = {c: (max(0.0, float(observation.ratios.get(c, 0))) * max(strength, 0.0) * quality
                      if self.phase == "ON" and label in COLORS else 0.0) for c in COLORS}
        return replace(observation, label=label, confidence=float(confidence),
                       photometric_phase=self.phase, photometry_mode="EMITTER_ONLINE",
                       color_energy=energy, decision_reason=reason,
                       measurement_quality=float(quality), emission_diagnostics=diagnostics)
