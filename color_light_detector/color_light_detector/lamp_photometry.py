"""Measure a geometrically located lamp aperture without moving its pixels."""
import math

import cv2
import numpy as np

from .lamp_region import ellipse_mask
from .task1_beacon_core import InstantColor

COLORS = ("RED", "GREEN", "BLUE")
DEFAULTS = {
    "aperture_shrink": 0.94, "halo_scale": 1.70,
    "min_aperture_pixels": 16, "min_aperture_coverage": 0.80,
    "saturation_min": 50, "color_value_min": 40,
    "source_value_min": 140, "source_support_min": 0.04,
    "channel_contrast_min": 24, "color_share_min": 0.62,
    "color_margin_min": 0.18, "dark_value_max": 65,
    "clipped_support_min": 0.60, "halo_support_min": 0.30,
    "halo_sector_coverage_min": 0.67,
}


def _color_votes(hsv, mask, p):
    hue, sat, value = cv2.split(hsv)
    valid = mask & (sat >= p["saturation_min"]) & (value >= p["color_value_min"])
    weights = (sat/255.0) * (value/255.0) * np.where(value >= 250, 0.65, 1.0) * valid
    hue_masks = {"RED": (hue <= 16) | (hue >= 164),
                 "GREEN": (hue >= 35) & (hue <= 88),
                 "BLUE": (hue >= 92) & (hue <= 142)}
    total = max(float(np.sum(weights)), 1e-9)
    votes = {c: float(np.sum(weights*hue_masks[c]))/total for c in COLORS}
    mass = float(np.sum(weights))/max(int(np.count_nonzero(mask)), 1)
    return votes, mass, valid, hue_masks


class LampPhotometer:
    def __init__(self, parameters=None):
        self.p = dict(DEFAULTS)
        self.p.update(parameters or {})
        if not all(math.isfinite(float(v)) for v in self.p.values()):
            raise ValueError("Lamp photometry parameters must be finite")
        for name in ("aperture_shrink", "min_aperture_coverage", "source_support_min",
                     "color_share_min", "color_margin_min", "clipped_support_min",
                     "halo_support_min", "halo_sector_coverage_min"):
            if not 0 < self.p[name] <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 1 < self.p["halo_scale"] <= 3 or self.p["min_aperture_pixels"] < 4:
            raise ValueError("Invalid lamp halo scale or minimum aperture area")
        if not 0 <= self.p["dark_value_max"] < self.p["source_value_min"] <= 255:
            raise ValueError("Lamp dark/source levels must be ordered within 0..255")
        if not 0 < self.p["saturation_min"] < 255 or not 0 < self.p["color_value_min"] < 255:
            raise ValueError("Invalid lamp color thresholds")
        if not 0 < self.p["channel_contrast_min"] <= 255:
            raise ValueError("Invalid lamp channel contrast")

    @staticmethod
    def _unobserved(region, reason):
        return InstantColor("UNKNOWN", 0.0, {c: 0.0 for c in COLORS}, 0,
            photometric_phase="UNOBSERVED", photometry_mode="RING_APERTURE",
            decision_reason=reason, emitter_center_source="lamp_region",
            emission_diagnostics={"lamp_region": region.diagnostics(),
                                  "source_on": False, "color_valid": False})

    def measure(self, frame, region, roi_bbox=None, visualization=None):
        """Optionally expose the actual native-pixel masks for diagnostic drawing."""
        if visualization is not None:
            visualization.clear()
        p = self.p
        if not region.valid or region.aperture is None or region.outer is None:
            return self._unobserved(region, "lamp_geometry_invalid")
        h, w = frame.shape[:2]
        (ox, oy), outer_d, _ = region.outer
        radius = 0.5*max(outer_d)+2
        x1, y1 = max(0, int(math.floor(ox-radius))), max(0, int(math.floor(oy-radius)))
        x2, y2 = min(w, int(math.ceil(ox+radius))), min(h, int(math.ceil(oy+radius)))
        if x2 <= x1 or y2 <= y1:
            return self._unobserved(region, "lamp_outside_image")
        crop = frame[y1:y2, x1:x2]
        shape, origin = crop.shape[:2], (x1, y1)
        outer = ellipse_mask(shape, region.outer, origin=origin, shrink=0.94)
        aperture = ellipse_mask(shape, region.aperture, origin=origin, shrink=p["aperture_shrink"])
        aperture &= outer
        count = int(np.count_nonzero(aperture))
        # Rasterized area avoids rounding artifacts in small, partly clipped holes.
        side = int(math.ceil(max(region.aperture[1])*p["aperture_shrink"]))+6
        complete = ((side//2, side//2), region.aperture[1], region.aperture[2])
        expected = int(np.count_nonzero(ellipse_mask((side, side), complete, shrink=p["aperture_shrink"])))
        coverage = min(1.0, count/max(expected, 1))
        if count < p["min_aperture_pixels"] or coverage < p["min_aperture_coverage"]:
            return self._unobserved(region, "lamp_aperture_unresolved_or_clipped")
        halo = outer & ellipse_mask(shape, region.aperture, origin=origin, shrink=p["halo_scale"])
        halo &= ~ellipse_mask(shape, region.aperture, origin=origin, shrink=1.06)
        rim = outer & ellipse_mask(shape, region.aperture, origin=origin)
        rim &= ~ellipse_mask(shape, region.aperture, origin=origin, shrink=0.85)
        background = outer & ~ellipse_mask(shape, region.aperture, origin=origin, shrink=p["halo_scale"]+0.25)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, value = cv2.split(hsv)
        rgb = crop.astype(np.float32)
        core = rgb[aperture]
        peak, mean = float(np.percentile(value[aperture], 90)), float(np.mean(value[aperture]))
        # Fixed aperture pixels, including stable occlusions, define every sample.
        # Background subtraction and color-dependent masks never change this flux.
        signal = float(np.clip((0.70*mean+0.30*peak)/255.0, 0, 1))
        neutral = (value >= 245) & (sat <= 45)
        white_fraction = float(np.mean(neutral[aperture]))
        core_votes, core_mass, core_color, hue_masks = _color_votes(hsv, aperture, p)
        halo_votes, halo_mass, halo_color, _ = _color_votes(hsv, halo, p)
        rim_votes, rim_mass, rim_color, _ = _color_votes(hsv, rim, p)
        color_source = "APERTURE" if np.count_nonzero(core_color) >= 3 and core_mass >= 0.018 else "HALO"
        votes, mass = (core_votes, core_mass) if color_source == "APERTURE" else (halo_votes, halo_mass*0.55)
        if color_source == "HALO" and np.count_nonzero(rim_color) >= 3 and rim_mass*0.75 >= 0.018:
            color_source, votes, mass = "APERTURE_RIM", rim_votes, rim_mass*0.75
        candidate, second = sorted(COLORS, key=votes.get, reverse=True)[:2]
        clear = (mass >= 0.018 and votes[candidate] >= p["color_share_min"]
                 and votes[candidate]-votes[second] >= p["color_margin_min"])
        # Contaminated black-pan pixels cannot supply a trustworthy baseline.
        # HSV saturation is unstable near black. Dim tinted plate pixels still
        # measure brightness; visibly bright colored spill remains excluded.
        clean_background = background & ((sat < p["saturation_min"])
            | (value < p["dark_value_max"])) & ~neutral
        clean_count = int(np.count_nonzero(clean_background))
        bg_quality = clean_count/max(int(np.count_nonzero(background)), 1) if clean_count >= 12 else 0.0
        bg_value = float(np.median(value[clean_background])) if clean_count else 0.0
        bg_rgb = np.percentile(rgb[clean_background], 60, axis=0) if clean_count else np.zeros(3)
        channel_peak = float(np.max(np.percentile(core, 90, axis=0)-bg_rgb)) if bg_quality >= 0.5 else 0.0
        support = float(np.count_nonzero(core_color & hue_masks[candidate]
            & (value >= p["source_value_min"]))) / count
        colored_on = (color_source == "APERTURE" and clear and support >= p["source_support_min"]
                      and peak >= p["source_value_min"] and bg_quality >= 0.5
                      and channel_peak >= p["channel_contrast_min"])
        # A clipped face needs strong, consistent near-hole color around it.
        # This combination is a spatial heuristic, not proof against reflection.
        strong_halo = halo_color & hue_masks[candidate] & (value >= p["source_value_min"])
        halo_support = float(np.count_nonzero(strong_halo))/max(int(np.count_nonzero(halo)), 1)
        (cx, cy), diameter, _ = region.aperture
        yy, xx = np.indices(shape)
        sectors = ((np.arctan2(yy+y1-cy, xx+x1-cx)+math.pi)*12/(2*math.pi)).astype(int) % 12
        sector_coverage = sum(np.count_nonzero(strong_halo & (sectors == s)) >= 2 for s in range(12))/12
        strong_rim = rim_color & hue_masks[candidate] & (value >= p["source_value_min"])
        rim_support = float(np.count_nonzero(strong_rim))/max(int(np.count_nonzero(rim)), 1)
        rim_coverage = sum(np.count_nonzero(strong_rim & (sectors == s)) >= 2 for s in range(12))/12
        halo_trusted = (halo_support >= p["halo_support_min"]
            and sector_coverage >= p["halo_sector_coverage_min"]
            and halo_votes[candidate] >= p["color_share_min"])
        rim_trusted = (rim_support >= p["halo_support_min"]
            and rim_coverage >= p["halo_sector_coverage_min"]
            and rim_votes[candidate] >= p["color_share_min"])
        # Clipping removes the face's hue: a few colored wire pixels must not
        # substitute for a consistent luminous rim or a surrounding halo.
        if white_fraction >= p["clipped_support_min"]:
            colored_on = colored_on and (rim_trusted or halo_trusted)
        neutral_peak = float(np.percentile(np.min(core, axis=1), 90))
        halo_neutral = float(np.percentile(np.min(rgb[halo], axis=1), 70)) if np.any(halo) else 255.0
        clipped_on = (clear and white_fraction >= p["clipped_support_min"] and peak >= 250
                      and (rim_trusted or halo_trusted)
                      and ((bg_quality >= 0.5 and channel_peak >= 40) or neutral_peak-halo_neutral >= 70))
        # Temporal ON may recover a hue, but a weak/one-sided reflection cannot
        # borrow that ON reference to turn its own halo vote into a valid color.
        color_valid = bool(clear and ((color_source == "APERTURE"
            and white_fraction < p["clipped_support_min"]) or rim_trusted or halo_trusted))
        if visualization is not None:
            color_region, color_pixels = {
                "APERTURE": (aperture, core_color),
                "APERTURE_RIM": (rim, rim_color),
                "HALO": (halo, halo_color),
            }[color_source]
            visualization.update(bounds=(x1, y1, x2, y2), aperture=aperture,
                color_region=color_region, color_pixels=color_pixels, color_source=color_source)
        source_on = bool(colored_on or clipped_on)
        if source_on:
            phase, label = "ON", candidate if clear else "UNKNOWN"
            reason = ("lamp_clipped_face_with_rim" if rim_trusted else "lamp_clipped_face_with_halo") if clipped_on else "lamp_aperture_colored_emission"
        elif peak < p["dark_value_max"] and core_mass < 0.018:
            phase, label, reason = "OFF", "DARK", "lamp_aperture_dim"
        else:
            phase, label, reason = "UNCERTAIN", "UNKNOWN", "lamp_bright_emission_unresolved"
        quality = float(np.clip(region.quality*coverage, 0, 1))
        confidence = quality*(votes[candidate]*(0.80 if color_source == "HALO" else 1)) if source_on else (
            0.8*quality if phase == "OFF" else 0.0)
        bx1, by1, bx2, by2 = roi_bbox if roi_bbox is not None else (x1, y1, x2, y2)
        center_x = float(np.clip((cx-bx1)/max(bx2-bx1-1, 1), 0, 1))
        center_y = float(np.clip((cy-by1)/max(by2-by1-1, 1), 0, 1))
        diagnostics = {"lamp_region": region.diagnostics(), "source_on": source_on,
            "source_heuristic": (("clipped_face_and_aperture_rim" if rim_trusted else "clipped_face_and_surrounding_color")
                if clipped_on else "colored_aperture_contrast"),
            "color_valid": color_valid, "color_source": color_source, "core_peak": peak,
            "core_mean": mean, "channel_peak": channel_peak, "core_color_support": support,
            "background_quality": bg_quality, "background_used": bool(bg_quality >= 0.5),
            "white_support": white_fraction, "halo_support": halo_support,
            "halo_sector_coverage": sector_coverage, "aperture_pixels": count,
            "rim_support": rim_support, "rim_sector_coverage": rim_coverage,
            "aperture_coverage": coverage, "halo_color_weight": 0.55 if color_source == "HALO" else 0.0,
            "rim_color_weight": 0.75 if color_source == "APERTURE_RIM" else 0.0}
        return InstantColor(label, confidence, votes, int(p["source_value_min"]),
            raw_ratios=core_votes, background_ratios=halo_votes, hue_shares=votes,
            center_value_p90=peak, center_saturation_p75=float(np.percentile(sat[aperture], 75)),
            neutral_clip_ratio=white_fraction, color_value_threshold=p["color_value_min"],
            decision_reason=reason, luminance_signal=signal, flux_signed=signal,
            emitter_center_x=center_x, emitter_center_y=center_y,
            emitter_core_value=peak, emitter_background_value=bg_value,
            emitter_chroma_fraction=float(np.mean(core_color[aperture])),
            photometric_phase=phase, photometry_mode="RING_APERTURE", peak_excess=channel_peak,
            spatial_support=support, color_mass=mass,
            color_energy={c: votes[c]*signal*quality if source_on and label in COLORS else 0.0 for c in COLORS},
            measurement_quality=quality, aperture_radius_px=math.sqrt(diameter[0]*diameter[1])/2,
            emitter_center_source="lamp_region", emission_diagnostics=diagnostics)
