"""V7.5: retain the V7 emitter result; recover LED arrays with original color.

Background RGB is used ONLY for a brightness contrast measurement. It is never
subtracted from pixels used for hue. A contaminated ring cannot by itself vote
OFF. All work runs on a bounded CPU ROI; no model or GPU operation is added.
"""
from dataclasses import replace
from functools import lru_cache
import math

import cv2
import numpy as np

COLORS = ("RED", "GREEN", "BLUE")
DEFAULTS = {
    "fusion_max_roi_size": 256,
    "fusion_source_radius_ratio": 0.18,
    "fusion_color_radius_ratio": 0.26,
    "fusion_peak_min": 140.0,
    "fusion_channel_contrast_min": 24.0,
    "fusion_color_share_min": 0.62,
    "fusion_color_margin_min": 0.18,
    "fusion_core_support_min": 0.04,
    "fusion_dark_value_max": 110.0,
}


@lru_cache(maxsize=8)
def _coordinates(height, width):
    return np.ogrid[:height, :width]


class EmitterFusion:
    def __init__(self, parameters):
        self.p = {k: float(parameters.get(k, v)) for k, v in DEFAULTS.items()}
        if not all(math.isfinite(v) and v > 0 for v in self.p.values()):
            raise ValueError("Emitter fusion parameters must be finite and positive")
        if not 96 <= self.p["fusion_max_roi_size"] <= 640:
            raise ValueError("fusion_max_roi_size must be in [96, 640]")
        if not 0.06 <= self.p["fusion_source_radius_ratio"] < self.p["fusion_color_radius_ratio"] <= 0.35:
            raise ValueError("Emitter source/color radii must be ordered within [0.06, 0.35]")
        if not 0.5 < self.p["fusion_color_share_min"] <= 1:
            raise ValueError("fusion_color_share_min must be in (0.5, 1]")
        if not 0 < self.p["fusion_color_margin_min"] < 1 or not 0 < self.p["fusion_core_support_min"] < 1:
            raise ValueError("Emitter margin/support must be in (0, 1)")
        if not self.p["fusion_dark_value_max"] < self.p["fusion_peak_min"] < 255:
            raise ValueError("Emitter dark/bright levels must be ordered below 255")
        self.saturation_min = int(parameters.get("saturation_min", 35))
        self.hue = {k: int(parameters.get(k, v)) for k, v in {
            "red_low_max": 16, "red_high_min": 164,
            "green_min": 30, "green_max": 88, "blue_min": 92, "blue_max": 142,
        }.items()}

    def bound(self, roi):
        scale = min(1.0, self.p["fusion_max_roi_size"] / max(roi.shape[:2]))
        if scale < 1.0:
            roi = cv2.resize(roi, (max(1, round(roi.shape[1] * scale)),
                                   max(1, round(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return roi, scale

    def refine(self, bgr, hsv, base, scale):
        p = self.p
        h, w = bgr.shape[:2]
        if min(h, w) < 8:
            return replace(base, label="UNKNOWN", confidence=0.0,
                           photometric_phase="UNCERTAIN", photometry_mode="EMITTER_FUSION",
                           decision_reason="emitter_roi_too_small")
        yy, xx = _coordinates(h, w)
        cx, cy = base.emitter_center_x * (w - 1), base.emitter_center_y * (h - 1)
        size = min(h, w)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / float(size * size)
        source = d2 <= p["fusion_source_radius_ratio"] ** 2
        region = d2 <= p["fusion_color_radius_ratio"] ** 2
        ring = (d2 >= 0.30 ** 2) & (d2 <= 0.44 ** 2)
        if np.count_nonzero(source) < 4 or np.count_nonzero(ring) < 12:
            return replace(base, photometric_phase="ON" if base.label in COLORS else "UNCERTAIN",
                           photometry_mode="EMITTER_FUSION", decision_reason="emitter_roi_edge")
        pixels = bgr.astype(np.float32)
        hue, sat, value = cv2.split(hsv)
        # V can be saturated in both a blue LED and its blue halo. Channelwise
        # PEAK contrast still detects the green/red content of the bright core.
        # It is a measurement, never a color correction or an OFF veto.
        core_rgb = np.percentile(pixels[source], 90, axis=0)
        bg_rgb = np.percentile(pixels[ring], 60, axis=0)
        channel_peak = float(np.max(np.maximum(core_rgb - bg_rgb, 0)))
        channel_mean = float(np.max(np.maximum(np.mean(pixels[source], axis=0)
                                               - np.mean(pixels[ring], axis=0), 0)))
        peak_value = float(np.percentile(value[source], 90))
        bg_value = float(np.percentile(value[ring], 60))
        channel_signal = (0.75 * channel_peak + 0.25 * channel_mean) / 255.0
        signal = float(np.clip(max(base.luminance_signal, channel_signal), 0, 1))
        # Use original color and a local brightness percentile; do not reject
        # colored LED pixels because an illuminated enclosure raised ring noise.
        threshold = max(30.0, float(np.percentile(value[region], 40)))
        white = (value >= 245) & (sat <= 45)
        valid = region & (sat >= self.saturation_min) & (value >= threshold) & ~white
        radial = np.exp(-1.5 * d2 / p["fusion_color_radius_ratio"] ** 2)
        weight = radial * (sat.astype(np.float32) / 255.0) * (value.astype(np.float32) / 255.0) ** 1.5
        # Partly clipped pixels still carry color. Prefer unsaturated chroma,
        # but retain colored edges when every LED center reaches the limit.
        weight *= np.where(value >= 250, 0.65, 1.0) * valid
        masks = {
            "RED": (hue <= self.hue["red_low_max"]) | (hue >= self.hue["red_high_min"]),
            "GREEN": (hue >= self.hue["green_min"]) & (hue <= self.hue["green_max"]),
            "BLUE": (hue >= self.hue["blue_min"]) & (hue <= self.hue["blue_max"]),
        }
        hue_mass = {c: float(np.sum(weight * masks[c])) for c in COLORS}
        total_hue = sum(hue_mass.values())
        hue_share = {c: hue_mass[c] / max(total_hue, 1e-9) for c in COLORS}
        chroma = pixels - np.min(pixels, axis=2, keepdims=True)
        channel = np.sum(chroma * weight[..., None], axis=(0, 1))[::-1]
        channel /= max(float(np.sum(channel)), 1e-9)
        shares = dict(zip(COLORS, map(float, channel)))
        combined = {c: 0.65 * hue_share[c] + 0.35 * shares[c] for c in COLORS}
        ordered = sorted(COLORS, key=combined.get, reverse=True)
        candidate, second = ordered[:2]
        chroma_mass = float(np.sum(weight) / max(np.sum(radial * region), 1e-9))
        source_area = max(np.count_nonzero(source), 1)
        core_support = float(np.count_nonzero(source & valid & masks[candidate]
                                             & (value >= p["fusion_peak_min"])) / source_area)
        any_color_support = float(np.count_nonzero(source & valid
            & (masks["RED"] | masks["GREEN"] | masks["BLUE"])
            & (value >= p["fusion_peak_min"])) / source_area)
        white_support = float(np.count_nonzero(source & white) / source_area)
        bright_support = float(np.count_nonzero(source & (value >= p["fusion_peak_min"])) / source_area)
        neutral_peak = float(np.percentile(np.min(pixels[source], axis=1), 90))
        neutral_bg = float(np.percentile(np.min(pixels[ring], axis=1), 60))
        colored_source = core_support >= p["fusion_core_support_min"]
        clipped_source = white_support >= 0.04 and neutral_peak - neutral_bg >= 30.0
        source_on = (peak_value >= p["fusion_peak_min"]
                     and channel_peak >= p["fusion_channel_contrast_min"]
                     and (colored_source or clipped_source))
        color_clear = (chroma_mass >= 0.018 and combined[candidate] >= p["fusion_color_share_min"]
                       and combined[candidate] - combined[second] >= p["fusion_color_margin_min"]
                       and shares[candidate] >= 0.40
                       and shares[candidate] - max(shares[c] for c in COLORS if c != candidate) >= 0.04)
        # Preserve the exact accepted V7 emitter color. Recovery only operates
        # on its DARK/UNKNOWN outcomes, so an already good color is not retuned.
        if base.label in COLORS:
            label, phase, reason = base.label, "ON", "emitter_v7_accepted"
            color_confidence = max(base.confidence, float(base.ratios.get(label, 0)))
            evidence = dict(base.ratios)
        elif source_on:
            phase = "ON"
            label = candidate if color_clear else "UNKNOWN"
            reason = "emitter_array_recovered" if color_clear else "emitter_on_color_ambiguous"
            color_confidence = combined[candidate] if color_clear else 0.0
            evidence = combined
        else:
            # Low aperture contrast alone is NOT darkness. Bright colored or
            # clipped central pixels veto OFF until emission can be resolved.
            bright_color = any_color_support >= p["fusion_core_support_min"]
            measured_dim = (peak_value < p["fusion_dark_value_max"]
                            and channel_peak < p["fusion_channel_contrast_min"]
                            and base.luminance_signal < 0.04)
            dark_supported = (not bright_color and not clipped_source
                              and (measured_dim or base.label == "DARK"))
            phase = "OFF" if dark_supported else "UNCERTAIN"
            label = "DARK" if dark_supported else "UNKNOWN"
            reason = "emitter_measured_dark" if dark_supported else "emitter_bright_unresolved"
            color_confidence = 0.75 if dark_supported else 0.0
            evidence = combined
        quality = float(np.clip(max(base.confidence, channel_peak / 80.0)
                                if phase == "ON" else (0.75 if phase == "OFF" else 0.0), 0, 1))
        # Unknown hue must not silently vote a color into the protocol decoder.
        energy = {c: (float(evidence.get(c, 0)) * signal * quality
                      if phase == "ON" and label in COLORS else 0.0) for c in COLORS}
        return replace(
            base, label=label, confidence=float(np.clip(color_confidence * quality, 0, 1)),
            ratios=evidence, channel_shares=shares if base.label not in COLORS else base.channel_shares,
            hue_shares=hue_share if base.label not in COLORS else base.hue_shares,
            decision_reason=reason, luminance_signal=signal,
            photometric_phase=phase, photometry_mode="EMITTER_FUSION",
            flux_signed=signal, flux_snr=0.0, spatial_support=bright_support,
            peak_excess=channel_peak, color_mass=chroma_mass, color_energy=energy,
            measurement_quality=quality, aperture_radius_px=size * p["fusion_source_radius_ratio"] / scale,
            emitter_center_source="v7_emitter", color_value_threshold=threshold,
            emission_diagnostics={"legacy_label": base.label, "legacy_reason": base.decision_reason,
                "legacy_signal": round(base.luminance_signal, 4), "channel_peak": round(channel_peak, 1),
                "core_peak": round(peak_value, 1), "background_value": round(bg_value, 1),
                "core_color_support": round(core_support, 4), "any_core_color_support": round(any_color_support, 4),
                "white_support": round(white_support, 4),
                "background_bright": bool(bg_value >= peak_value - 12),
                "color_clear": bool(color_clear), "source_on": bool(source_on)},
        )
