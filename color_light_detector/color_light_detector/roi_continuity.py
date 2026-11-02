"""Recover a measured lamp ROI through detector/optical-flow misses.

An ON observation authorizes a bounded search, never an OFF vote. Current housing
structure must still verify the location before photometry can read the lamp.
Recent valid ROIs bridge illumination changes without extending the ON lease.
"""
import math
import cv2
import numpy as np

DEFAULTS = {
    "enabled": False,
    "anchor_seconds": 2.4,
    "confirm_on_frames": 1,
    "search_ratio": 0.22,
    "min_correlation": 0.74,
    "reference_seconds": 0.6,
    "max_anchor_shift_ratio": 0.65,
}


class RoiContinuity:
    def __init__(self, parameters):
        self.p = dict(DEFAULTS, **parameters)
        if not all(math.isfinite(float(v)) for v in self.p.values()):
            raise ValueError("ROI continuity parameters must be finite")
        if not 0.5 <= self.p["anchor_seconds"] <= 3.0:
            raise ValueError("roi_continuity.anchor_seconds must be 0.5..3 s")
        if not 1 <= self.p["confirm_on_frames"] <= 10:
            raise ValueError("roi_continuity.confirm_on_frames must be 1..10")
        if not 0.02 <= self.p["search_ratio"] <= 0.3 or not 0.60 <= self.p["min_correlation"] <= 0.98:
            raise ValueError("Invalid ROI continuity search/correlation")
        if not 0.1 <= self.p["reference_seconds"] <= 1.0:
            raise ValueError("roi_continuity.reference_seconds must be 0.1..1 s")
        if not 0.1 <= self.p["max_anchor_shift_ratio"] <= 1.0:
            raise ValueError("roi_continuity.max_anchor_shift_ratio must be 0.1..1")
        self.anchors = {}
        self.references = {}
        self.pending = {}
        self.last_status = {}

    def clear(self):
        self.anchors.clear()
        self.references.clear()
        self.pending.clear()
        self.last_status.clear()

    def prune(self, keys, stamp):
        self.anchors = {k: v for k, v in self.anchors.items()
                        if k in keys and 0 <= stamp - v[0] <= self.p["anchor_seconds"]}
        self.references = {k: v for k, v in self.references.items()
                           if k in self.anchors and 0 <= stamp - v[0] <= self.p["reference_seconds"]}
        self.pending = {k: v for k, v in self.pending.items() if k in keys and 0 <= stamp - v[0] <= 0.5}
        self.last_status = {k: v for k, v in self.last_status.items() if k in keys}

    def diagnostics(self, key, stamp):
        anchor, reference = self.anchors.get(key), self.references.get(key)
        age = None if anchor is None else stamp - anchor[0]
        return {
            "armed": age is not None and 0 <= age <= self.p["anchor_seconds"],
            "anchor_age": None if age is None else round(age, 3),
            "reference_age": None if reference is None else round(stamp - reference[0], 3),
            "pending_on_frames": self.pending.get(key, (None, 0))[1],
            "status": self.last_status.get(key, "waiting_for_colored_on"),
        }

    def _within_anchor(self, box, anchor_box):
        old_w, old_h = anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1]
        width, height = box[2] - box[0], box[3] - box[1]
        if min(old_w, old_h, width, height) <= 0:
            return False
        if not (0.7 <= width / old_w <= 1.4 and 0.7 <= height / old_h <= 1.4):
            return False
        distance = math.hypot((box[0] + box[2] - anchor_box[0] - anchor_box[2]) * 0.5,
                              (box[1] + box[3] - anchor_box[1] - anchor_box[3]) * 0.5)
        return distance <= max(2.0, self.p["max_anchor_shift_ratio"] * min(old_w, old_h))

    def observe(self, key, stamp, box, gray, observation, quality, min_quality,
                allow_anchor_refresh=True):
        if not self.p["enabled"] or gray is None:
            return
        spatial_valid = math.isfinite(quality) and quality >= min_quality
        diagnostics = observation.emission_diagnostics
        learned = diagnostics.get("online_light", {}).get("ready", False)
        label = observation.label
        votes = observation.ratios
        # A learned bright level plus current clear hue is a second legitimate
        # ON source. Requiring the bootstrap source_on flag again blocked dim
        # blue lamps even after their two measured levels were well separated.
        share = float(votes.get(label, 0.0))
        margin = share - max((float(votes.get(c, 0.0)) for c in ("RED", "GREEN", "BLUE") if c != label), default=0.0)
        trusted_on = (spatial_valid and observation.photometric_phase == "ON"
                      and label in ("RED", "GREEN", "BLUE")
                      and (diagnostics.get("source_on", False)
                           or diagnostics.get("online_light", {}).get("source_on", False)
                           or (learned and share >= 0.55 and margin >= 0.12)))
        if trusted_on and allow_anchor_refresh:
            previous = self.pending.get(key)
            count = previous[1] + 1 if previous and 0 < stamp - previous[0] <= 0.5 else 1
            self.pending[key] = (stamp, count)
            self.last_status[key] = "confirming_colored_on"
            if count >= self.p["confirm_on_frames"]:
                # gray is immutable after capture; objects share the same frame.
                self.anchors[key] = (stamp, tuple(box), gray)
                self.last_status[key] = "armed"
        else:
            self.pending.pop(key, None)
        anchor = self.anchors.get(key)
        if (spatial_valid and anchor is not None
                and 0 <= stamp - anchor[0] <= self.p["anchor_seconds"]
                and self._within_anchor(box, anchor[1])):
            # An OFF or recovered image can refresh geometry, never ON expiry.
            # This avoids repeatedly comparing dark housing with a bright halo.
            self.references[key] = (stamp, tuple(box), gray)
        elif anchor is None:
            self.last_status[key] = "waiting_for_colored_on"

    def recover(self, key, stamp, current, full_shape, min_quality=0.0):
        anchor = self.anchors.get(key)
        if not self.p["enabled"] or anchor is None:
            self.last_status[key] = "waiting_for_colored_on"
            return None
        when, box, reference = anchor
        if not 0 < stamp - when <= self.p["anchor_seconds"]:
            self.last_status[key] = "anchor_expired"
            return None
        candidates = []
        recent = self.references.get(key)
        if recent is not None and 0 < stamp - recent[0] <= self.p["reference_seconds"]:
            candidates.append(("recent_housing", recent[1], recent[2]))
        if not any(item[1] == box and item[2] is reference for item in candidates):
            candidates.append(("on_housing", box, reference))
        for name, reference_box, reference_gray in candidates:
            if reference_gray.shape != current.shape:
                continue
            matched = self._match_housing(reference_box, reference_gray, current, full_shape)
            if matched is not None and matched[1] >= min_quality and self._within_anchor(matched[0], box):
                self.last_status[key] = name + "_recovered"
                return matched
        self.last_status[key] = "housing_not_verified"
        return None

    @staticmethod
    def _shell_score(a, b, bx, by, bw, bh, dx, dy):
        """Validate one translation on textured, distributed housing pixels."""
        left, top = round(bx), round(by)
        right, bottom = round(bx + bw), round(by + bh)
        dx, dy = int(round(dx)), int(round(dy))
        if (min(left, top, left + dx, top + dy) < 0
                or right > a.shape[1] or bottom > a.shape[0]
                or right + dx > b.shape[1] or bottom + dy > b.shape[0]):
            return None
        first = a[top:bottom, left:right]
        second = b[top + dy:bottom + dy, left + dx:right + dx]
        if min(first.shape, default=0) < 6 or first.shape != second.shape:
            return None
        yy, xx = np.indices(first.shape, dtype=np.float32)
        xx, yy = (xx + 0.5) / first.shape[1], (yy + 0.5) / first.shape[0]
        shell = (xx < .22) | (xx > .78) | (yy < .22) | (yy > .78)
        threshold = max(.5, float(np.percentile(np.abs(first[shell]), 45)))
        mask = shell & (np.abs(first) >= threshold)
        if np.count_nonzero(mask) < 16:
            return None
        locations = np.column_stack((xx[mask], yy[mask]))
        if min(np.ptp(locations, axis=0)) < .5:
            return None
        av, bv = first[mask].astype(np.float64), second[mask].astype(np.float64)
        av, bv = av - np.mean(av), bv - np.mean(bv)
        if min(float(np.std(av)), float(np.std(bv))) < .6:
            return None
        denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denominator <= 1e-6:
            return None
        score = float(np.dot(av, bv) / denominator)
        return score if math.isfinite(score) else None

    def _match_housing(self, box, reference, current, full_shape):
        h, w = current.shape
        sx, sy = w / full_shape[1], h / full_shape[0]
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or min(x2 - x1, y2 - y1) < 16:
            return None
        radius = max(3, math.ceil(self.p["search_ratio"] * min(x2 - x1, y2 - y1)))
        left, top = max(0, math.floor(x1) - radius - 4), max(0, math.floor(y1) - radius - 4)
        right, bottom = min(w, math.ceil(x2) + radius + 4), min(h, math.ceil(y2) + radius + 4)
        a, b = reference[top:bottom, left:right], current[top:bottom, left:right]
        scale = min(1.0, 192.0 / max(a.shape))
        if scale < 1:
            shape = (max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale)))
            a, b = cv2.resize(a, shape, interpolation=cv2.INTER_AREA), cv2.resize(b, shape, interpolation=cv2.INTER_AREA)

        def structure(patch):
            smooth = cv2.GaussianBlur(patch.astype(np.float32), (3, 3), 0)
            return smooth - cv2.GaussianBlur(smooth, (0, 0), 3.0)

        a, b = structure(a), structure(b)
        bx, by = (x1 - left) * scale, (y1 - top) * scale
        bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
        radius = max(2, round(radius * scale))
        tiles = ((0, 0, .28, .28), (.36, 0, .64, .20), (.72, 0, 1, .28),
                 (0, .36, .20, .64), (.80, .36, 1, .64),
                 (0, .72, .28, 1), (.36, .80, .64, 1), (.72, .72, 1, 1))
        votes = []
        for u1, v1, u2, v2 in tiles:
            l, t, r, d = round(bx + u1 * bw), round(by + v1 * bh), round(bx + u2 * bw), round(by + v2 * bh)
            template = a[t:d, l:r]
            if min(template.shape, default=0) < 4 or float(np.std(template)) < .7:
                continue
            sl, st = max(0, l - radius), max(0, t - radius)
            sr, sd = min(b.shape[1], r + radius), min(b.shape[0], d + radius)
            search = b[st:sd, sl:sr]
            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, peak = cv2.minMaxLoc(response)
            if not math.isfinite(score) or score < self.p["min_correlation"]:
                continue
            votes.append((peak[0] + sl - l, peak[1] + st - t, score, (u1 + u2) * .5, (v1 + v2) * .5))

        accepted = None
        if len(votes) >= 2:
            votes = np.asarray(votes, dtype=np.float64)
            tolerance = max(1.5, .025 * min(bw, bh))
            groups = []
            for vote in votes:
                good = votes[np.linalg.norm(votes[:, :2] - vote[:2], axis=1) <= tolerance]
                spread = np.ptp(good[:, 3:5], axis=0)
                if len(good) >= 2 and float(np.linalg.norm(spread)) >= .55:
                    groups.append(good)
            # At most three candidate translations, then one stationary check.
            # The whole textured shell corroborates even a two-patch recovery.
            groups.sort(key=lambda group: (len(group), float(np.median(group[:, 2]))), reverse=True)
            seen = set()
            for good in groups:
                shift = np.median(good[:, :2], axis=0)
                location = tuple(np.round(shift).astype(int))
                if location in seen:
                    continue
                seen.add(location)
                shell_score = self._shell_score(a, b, bx, by, bw, bh, *location)
                required = max(.56, self.p["min_correlation"] - (.12 if len(good) == 2 else .18))
                if shell_score is not None and shell_score >= required:
                    accepted = (shift, float(.88 * np.median(good[:, 2]) + .12 * shell_score))
                    break
                if len(seen) >= 3:
                    break
        if accepted is None:
            # Small lamps may have too few corners for LK or small tile matches.
            # No-motion recovery still needs current distributed shell texture.
            score = self._shell_score(a, b, bx, by, bw, bh, 0, 0)
            if score is not None and score >= max(.76, self.p["min_correlation"] + .02):
                accepted = (np.zeros(2, dtype=np.float64), .90 * score)
        if accepted is None:
            return None
        shift, quality = accepted
        shift = shift / scale
        dx, dy = shift[0] / sx, shift[1] / sy
        moved = tuple(round(v) for v in (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
        if not (0 <= moved[0] < moved[2] <= full_shape[1] and 0 <= moved[1] < moved[3] <= full_shape[0]):
            return None
        return moved, float(np.clip(quality, 0.0, 1.0))
