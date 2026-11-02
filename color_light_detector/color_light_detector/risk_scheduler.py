"""Bounded discovery bursts and per-track risk for the existing YOLO worker."""
import math

from .task1_beacon_core import analysis_bbox

DEFAULTS = {
    "enabled": False,
    "candidate_burst_seconds": 1.0,
    "candidate_cooldown_seconds": 4.0,
    "detector_age_seconds": 2.5,
    "small_target_pixels": 32.0,
    "edge_margin_ratio": 0.04,
}


class RiskScheduler:
    def __init__(self, parameters):
        self.p = dict(DEFAULTS, **parameters)
        if not isinstance(self.p["enabled"], bool):
            raise ValueError("risk.enabled must be true or false")
        bounds = {"candidate_burst_seconds": (0.25, 2.0),
                  "candidate_cooldown_seconds": (2.0, 10.0),
                  "detector_age_seconds": (1.0, 5.0),
                  "small_target_pixels": (8.0, 96.0),
                  "edge_margin_ratio": (0.01, 0.15)}
        for key, (lo, hi) in bounds.items():
            value = self.p[key]
            if isinstance(value, bool) or not math.isfinite(float(value)) or not lo <= value <= hi:
                raise ValueError("risk." + key + " outside supported range")
        if self.p["candidate_cooldown_seconds"] <= self.p["candidate_burst_seconds"]:
            raise ValueError("Candidate cooldown must exceed burst duration")
        self.reset()

    def reset(self):
        self.seen = set()
        self.weak_active = False
        self.burst_until = float("-inf")
        self.next_burst = float("-inf")
        self.stable_since = None

    def choose(self, tracks, previous, stamp, shape, minimum_hits, bbox_source,
               stable_quality, speed_limit, hold_seconds, weak_candidates=False,
               require_light_reference=True):
        live = {t.track_id for t in tracks}
        candidates = {t.track_id for t in tracks if t.hits < minimum_hits}
        fresh = bool(candidates - self.seen) or (weak_candidates and not self.weak_active)
        self.seen = live
        self.weak_active = weak_candidates
        if fresh and stamp >= self.next_burst:
            self.burst_until = stamp + self.p["candidate_burst_seconds"]
            self.next_burst = stamp + self.p["candidate_cooldown_seconds"]
        confirmed = [t for t in tracks if t.hits >= minimum_hits]
        reasons = set()
        force_full = not confirmed
        if not confirmed:
            reasons.add("NO_CONFIRMED_TARGET")
        if stamp < self.burst_until:
            reasons.add("CANDIDATE_BURST")
            force_full = True
        h, w = shape[:2]
        for track in confirmed:
            if not track.visible:
                reasons.add("ROI_UNOBSERVED")
                force_full = True
                continue
            box = analysis_bbox(track, bbox_source)
            side = min(box[2] - box[0], box[3] - box[1])
            if side < self.p["small_target_pixels"]:
                reasons.add("SMALL_TARGET")
            margin = self.p["edge_margin_ratio"]
            if min(box[0] / w, box[1] / h, (w-box[2]) / w, (h-box[3]) / h) < margin:
                reasons.add("IMAGE_EDGE")
                force_full = True
            if track.motion_quality < stable_quality:
                reasons.add("MOTION_QUALITY")
                force_full = True
            if track.association_stage == "CONTEXT_RECOVERED":
                reasons.add("CONTEXT_RECOVERY")
            lamp = track.raw_instant.emission_diagnostics.get("lamp_region")
            if lamp is not None and (not lamp["valid"] or track.raw_instant.photometric_phase == "UNOBSERVED"):
                reasons.add("LAMP_REGION_UNOBSERVED")
                force_full = True
            if stamp - track.detector_stamp > self.p["detector_age_seconds"]:
                # Keep measuring a reliably tracked dark lamp while reacquiring it.
                reasons.add("DETECTOR_CORRECTION_DUE")
            old = previous.get(track.track_id)
            speed = float("inf")
            if old is not None and stamp > old[0]:
                last = old[1]
                delta = math.hypot((box[0]+box[2]-last[0]-last[2]) * .5,
                                   (box[1]+box[3]-last[1]-last[3]) * .5)
                speed = delta / max(min(last[2]-last[0], last[3]-last[1]), 1) / (stamp-old[0])
            if speed > speed_limit:
                reasons.add("TARGET_MOTION")
                force_full = True
            reference = track.raw_instant.emission_diagnostics.get("online_light", {})
            if require_light_reference and not (reference.get("ready") or reference.get("bootstrap_on_reference") is not None):
                reasons.add("LIGHT_REFERENCE_PENDING")
        if reasons:
            self.stable_since = None
            mode = "SEARCH" if not confirmed or "CANDIDATE_BURST" in reasons else "MOTION"
        else:
            self.stable_since = stamp if self.stable_since is None else self.stable_since
            mode = "TRACK" if stamp - self.stable_since >= hold_seconds else "MOTION"
            if mode != "TRACK":
                reasons.add("STABILITY_HOLD")
        return mode, force_full, {
            "enabled": True, "mode": mode, "reasons": sorted(reasons) or ["STABLE"],
            "confirmed_tracks": len(confirmed), "tentative_tracks": len(candidates),
            "candidate_burst_remaining_seconds": round(max(0., self.burst_until-stamp), 3),
        }
