"""Local, signed aperture photometry on camera pixels; no GPU dependency.

Intensity here is camera-code excess, not calibrated radiometric flux.
Subtract background before summing, and never rectify individual pixels in
our brightness integral. Normalized chromaticity cannot establish emission.
"""

import cv2
import numpy as np


DEFAULTS = {
    "aperture_radius_ratio": 0.16,
    "aperture_min_radius_px": 2.0,
    "aperture_background_inner": 1.65,
    "aperture_background_outer": 2.65,
    "aperture_max_roi_size": 320,
    "aperture_on_flux": 0.060,
    "aperture_off_flux": 0.022,
    "aperture_min_snr": 4.0,
    "aperture_noise_floor": 3.0,
    "aperture_peak_min": 22.0,
    "aperture_support_min": 0.10,
    "aperture_color_mass_min": 0.012,
    "aperture_color_share_min": 0.56,
    "aperture_color_margin_min": 0.16,
    "aperture_channel_margin_min": 0.06,
    "aperture_hint_search_ratio": 0.12,
    "aperture_relocate_gain": 1.25,
}
COLORS = ("RED", "GREEN", "BLUE")


class AperturePhotometer:
    def __init__(self, parameters):
        self.p = {k: float(parameters.get(k, v)) for k, v in DEFAULTS.items()}
        self.hue = {k: int(parameters.get(k, v)) for k, v in {
            "red_low_max": 16, "red_high_min": 164, "green_min": 30,
            "green_max": 88, "blue_min": 92, "blue_max": 142,
            "saturation_min": 35,
        }.items()}
        if not all(np.isfinite(v) and v > 0 for v in self.p.values()):
            raise ValueError("Aperture parameters must be finite and positive")
        if not 0 < self.p["aperture_off_flux"] < self.p["aperture_on_flux"] < 1:
            raise ValueError("Require 0 < aperture_off_flux < aperture_on_flux < 1")
        if not 1 < self.p["aperture_background_inner"] < self.p["aperture_background_outer"]:
            raise ValueError("Background annulus must be outside the aperture")
        if not 0.03 <= self.p["aperture_radius_ratio"] <= 0.22:
            raise ValueError("aperture_radius_ratio must be in [0.03, 0.22]")
        if self.p["aperture_max_roi_size"] < 32:
            raise ValueError("aperture_max_roi_size must be >= 32")

    def measure(self, roi, center_hint=None):
        p = self.p
        h0, w0 = roi.shape[:2]
        scale = min(1.0, p["aperture_max_roi_size"] / max(h0, w0))
        image = (cv2.resize(roi, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                            interpolation=cv2.INTER_AREA) if scale < 1 else roi)
        # No blur before photometry: preserve integrated energy of tiny emitters.
        bgr = image.astype(np.float32)
        value = np.max(bgr, axis=2)
        h, w = value.shape
        size = min(h, w)
        if size < 7:
            return dict(label="UNKNOWN", confidence=0.0, ratios={}, value_threshold=0,
                        photometric_phase="UNCERTAIN", decision_reason="too_few_camera_pixels")
        radius = max(p["aperture_min_radius_px"], size * p["aperture_radius_ratio"])
        yy, xx = np.ogrid[:h, :w]
        origin = ((w - 1) * 0.5, (h - 1) * 0.5)
        has_hint = center_hint is not None and np.all(np.isfinite(center_hint))
        if has_hint:
            origin = (float(np.clip(center_hint[0], 0, 1)) * (w - 1),
                      float(np.clip(center_hint[1], 0, 1)) * (h - 1))

        def sample(center):
            d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            aperture = d2 <= radius ** 2
            ring = ((d2 >= (radius * p["aperture_background_inner"]) ** 2)
                    & (d2 <= (radius * p["aperture_background_outer"]) ** 2))
            bg = value[ring]
            if bg.size < 12 or np.count_nonzero(aperture) < 5:
                return None
            median = float(np.median(bg))
            mad = max(p["aperture_noise_floor"], 1.4826 * float(np.median(np.abs(bg - median))))
            # Symmetric robust background; no foreground/background percentile mismatch.
            keep = np.abs(bg - median) <= 3.0 * mad
            bg_pixels = bgr[ring][keep]
            if len(bg_pixels) < 8:
                return None
            bg_v = bg[keep]
            level = float(np.mean(bg_v))
            noise = max(p["aperture_noise_floor"], float(np.std(bg_v)))
            weight = np.exp(-0.5 * d2 / max((0.7 * radius) ** 2, 1.0)) * aperture
            area = float(np.sum(weight))
            residual = value - level
            flux = float(np.sum(weight * residual))
            variance_area = float(np.sum(weight ** 2)) + area ** 2 / len(bg_v)
            snr = flux / max(noise * np.sqrt(variance_area), 1e-6)
            normalized = flux / (255.0 * area)
            support = float(np.sum(weight * (residual > max(6.0, 2.5 * noise))) / area)
            peak = float(np.percentile(value[aperture], 90) - level)
            signal_pixels = int(np.count_nonzero(aperture & (residual > max(6.0, 2.5 * noise))))
            on = (normalized >= p["aperture_on_flux"] and snr >= p["aperture_min_snr"]
                  and support >= p["aperture_support_min"] and signal_pixels >= 2
                  and peak >= p["aperture_peak_min"])
            off = normalized <= p["aperture_off_flux"] and peak < p["aperture_peak_min"]
            return dict(center=center, d2=d2, aperture=aperture, weight=weight, area=area,
                        bg=level, bg_bgr=np.mean(bg_pixels, axis=0), noise=noise,
                        flux=flux, normalized=normalized, snr=snr, support=support,
                        peak=peak, phase="ON" if on else ("OFF" if off else "UNCERTAIN"))

        fixed = sample(origin)
        # Search on an integrated brightness map, never a single brightest RGB pixel.
        sigma = max(0.8, radius * 0.65)
        smooth = cv2.GaussianBlur(value, (0, 0), sigma)
        search = (((xx - (w - 1) / 2) / max(w * 0.38, 1)) ** 2
                  + ((yy - (h - 1) / 2) / max(h * 0.38, 1)) ** 2 <= 1)
        if has_hint:
            search &= ((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2
                       <= max(2.0, size * p["aperture_hint_search_ratio"]) ** 2)
        selected = fixed
        source = "tracked_aperture" if has_hint else "box_center"
        if np.any(search):
            cy, cx = np.unravel_index(np.argmax(np.where(search, smooth, -1)), value.shape)
            candidate = sample((float(cx), float(cy)))
            # Dark phases retain the physical position. A new highlight must pass
            # ON evidence before moving a previously learned center.
            if candidate is not None and candidate["phase"] == "ON" and (
                    fixed is None or (not has_hint and candidate["normalized"] > fixed["normalized"])
                    or candidate["normalized"] > max(p["aperture_on_flux"],
                                                      fixed["normalized"] * p["aperture_relocate_gain"])):
                selected, source = candidate, "verified_emitter"
            elif not has_hint and candidate is not None and candidate["phase"] != "OFF":
                # A dubious peak must prevent a false OFF, without teaching a center.
                if selected is not None and selected["phase"] == "OFF":
                    selected = dict(selected, phase="UNCERTAIN")
        if selected is None:
            return dict(label="UNKNOWN", confidence=0.0, ratios={}, value_threshold=0,
                        photometric_phase="UNCERTAIN", decision_reason="insufficient_local_background")
        m = selected
        quality = float(np.clip(min(m["snr"] / (2 * p["aperture_min_snr"]),
                                    max(m["normalized"], 0) / (2 * p["aperture_on_flux"]),
                                    m["support"] / 0.30), 0, 1))
        # The core can clip white. Include nearby non-neutral halo pixels for color.
        color_region = m["d2"] <= (1.35 * radius) ** 2
        color_weight = np.exp(-0.5 * m["d2"] / max(radius ** 2, 1.0)) * color_region
        delta_bgr = bgr - m["bg_bgr"]
        corrected = np.clip(delta_bgr, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        neutral_clip = np.min(bgr, axis=2) >= 245
        valid = (color_region & ~neutral_clip & (sat >= self.hue["saturation_min"])
                 & (value - m["bg"] > max(6.0, 2.0 * m["noise"])))
        # Integrate channel excess before clamping; keep its absolute amount.
        sums = np.maximum(np.sum(delta_bgr * (color_weight * valid)[..., None], axis=(0, 1)), 0)
        sums = np.maximum(sums - np.min(sums), 0)
        energy_bgr = sums / max(255 * float(np.sum(color_weight)), 1)
        total = float(np.sum(energy_bgr))
        shares = dict(zip(COLORS, (energy_bgr[::-1] / max(total, 1e-9)).tolist()))
        masks = {"RED": ((hue <= self.hue["red_low_max"]) | (hue >= self.hue["red_high_min"])),
                 "GREEN": ((hue >= self.hue["green_min"]) & (hue <= self.hue["green_max"])),
                 "BLUE": ((hue >= self.hue["blue_min"]) & (hue <= self.hue["blue_max"]))}
        hue_weights = {c: float(np.sum(color_weight * valid * masks[c] * (sat / 255.0)
                                       * np.maximum(value - m["bg"], 0))) for c in COLORS}
        hue_total = sum(hue_weights.values())
        hue_shares = {c: hue_weights[c] / max(hue_total, 1e-9) for c in COLORS}
        combined = {c: 0.6 * shares[c] + 0.4 * hue_shares[c] for c in COLORS}
        ordered = sorted(COLORS, key=combined.get, reverse=True)
        first, second = ordered[:2]
        channel_order = sorted(COLORS, key=shares.get, reverse=True)
        phase = m["phase"]
        label, reason = "UNKNOWN", "uncertain_brightness"
        confidence = 0.0
        if phase == "OFF":
            label, reason = "DARK", "measured_low_flux"
            confidence = float(np.clip(1 - max(m["normalized"], 0) / p["aperture_on_flux"], 0, 1))
        elif phase == "ON":
            reason = "on_without_color_mass"
            if total >= p["aperture_color_mass_min"]:
                reason = "on_ambiguous_color"
                if (combined[first] >= p["aperture_color_share_min"]
                        and combined[first] - combined[second] >= p["aperture_color_margin_min"]
                        and channel_order[0] == first
                        and shares[channel_order[0]] - shares[channel_order[1]] >= p["aperture_channel_margin_min"]):
                    label, reason = first, "aperture_emission_color"
                    confidence = min(combined[first], shares[first]) * quality
        color_energy = dict(zip(COLORS, (energy_bgr[::-1] * quality).tolist()))
        if phase != "ON" or total < p["aperture_color_mass_min"]:
            color_energy = {c: 0.0 for c in COLORS}
        core = m["aperture"]
        return dict(label=label, confidence=float(confidence), ratios=combined,
                    raw_ratios=color_energy, channel_shares=shares, hue_shares=hue_shares,
                    value_threshold=int(np.clip(m["bg"] + m["noise"] * 2, 0, 255)),
                    luminance_signal=float(np.clip(m["normalized"], 0, 1)),
                    decision_reason=reason, photometric_phase=phase,
                    photometry_mode="POINT_SOURCE" if radius / scale <= 3.5 else "APERTURE",
                    flux_signed=m["flux"] / max(scale ** 2, 1e-9),
                    flux_snr=m["snr"], spatial_support=m["support"], peak_excess=m["peak"],
                    color_mass=total, color_energy=color_energy,
                    measurement_quality=(quality if phase == "ON" else confidence if phase == "OFF" else 0.0),
                    aperture_radius_px=radius / scale, emitter_center_source=source,
                    emitter_center_x=m["center"][0] / max(w - 1, 1),
                    emitter_center_y=m["center"][1] / max(h - 1, 1),
                    emitter_core_value=float(np.percentile(value[core], 90)),
                    center_value_p90=float(np.percentile(value[core], 90)),
                    center_saturation_p75=float(np.percentile(sat[core], 75)),
                    emitter_background_value=m["bg"],
                    emitter_chroma_fraction=float(np.count_nonzero(valid) / max(np.count_nonzero(color_region), 1)),
                    neutral_clip_ratio=float(np.mean(neutral_clip[core])),
                    required_signal=p["aperture_on_flux"])
