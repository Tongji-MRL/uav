"""Locate the physical top plate and aperture without using the brightest pixel."""
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULTS = {
    "max_roi_side": 640,
    "roi_margin_fraction": 0.06,
    "min_outer_diameter": 20.0,
    "min_outer_fraction": 0.20,
    "min_aperture_diameter": 6.0,
    "min_aperture_fraction": 0.03,
    "min_axis_ratio": 0.30,
    "max_fit_error": 0.17,
    "min_arc_coverage": 0.70,
    "min_inner_ratio": 0.10,
    "max_inner_ratio": 0.55,
    "max_tracking_gap": 0.35,
    "tracking_min_support": 0.60,
    "max_center_shift": 0.30,
    "max_scale_change": 0.35,
}


@dataclass
class LampRegion:
    outer: object = None
    aperture: object = None
    quality: float = 0.0
    valid: bool = False
    source: str = "invalid"
    reason: str = "no_geometry"
    epoch: int = 0
    reset: bool = False

    def diagnostics(self):
        def serial(ellipse):
            if ellipse is None:
                return None
            return [[float(v) for v in ellipse[0]],
                    [float(v) for v in ellipse[1]], float(ellipse[2])]
        return {"outer": serial(self.outer), "aperture": serial(self.aperture),
                "quality": float(self.quality), "valid": bool(self.valid),
                "source": self.source, "reason": self.reason,
                "epoch": int(self.epoch), "reset": bool(self.reset)}


def ellipse_mask(shape, ellipse, origin=(0, 0), shrink=1.0):
    mask = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    if ellipse is not None:
        center, diameters, angle = ellipse
        center = tuple(int(round(center[i] - origin[i])) for i in range(2))
        axes = tuple(max(1, int(round(d * shrink * 0.5))) for d in diameters)
        cv2.ellipse(mask, center, axes, float(angle), 0, 360, 1, -1)
    return mask.astype(bool)


def _points(ellipse, count=64):
    center, axes, angle = ellipse
    theta = np.arange(count) * (2 * np.pi / count)
    pts = np.column_stack((np.cos(theta), np.sin(theta))) * (np.asarray(axes) / 2)
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    return pts @ rotation.T + center


def _normal_points(pts, ellipse):
    center, axes, angle = ellipse
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    return (pts - center) @ rotation / (np.asarray(axes) / 2)


def _move(ellipse, matrix):
    points = _points(ellipse).astype(np.float32)
    transformed = points @ matrix[:2, :2].T + matrix[:2, 2]
    center, axes, angle = cv2.fitEllipse(transformed.astype(np.float32))
    return (tuple(map(float, center)), tuple(map(float, axes)), float(angle))


def _similarity(transform):
    if transform is None:
        return None
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape not in ((2, 3), (3, 3)) or not np.isfinite(matrix).all():
        return None
    if matrix.shape == (3, 3) and not np.allclose(matrix[2], [0, 0, 1], atol=1e-5):
        return None
    scales = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    if (np.linalg.det(matrix[:2, :2]) <= 0 or min(scales) < 0.5 or
            max(scales) > 2 or max(scales) / min(scales) > 1.02):
        return None
    return matrix


def _blend(old, new, alpha):
    delta_angle = (new[2] - old[2] + 90) % 180 - 90
    center = tuple((1 - alpha) * np.asarray(old[0]) + alpha * np.asarray(new[0]))
    axes = tuple((1 - alpha) * np.asarray(old[1]) + alpha * np.asarray(new[1]))
    return (center, axes, float((old[2] + alpha * delta_angle) % 180))


def _edge_map(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = np.zeros(image.shape[:2], dtype=np.uint8)
    for channel in cv2.split(blurred):
        edges |= cv2.Canny(channel, 25, 75)
    return edges


class LampRegionLocator:
    def __init__(self, parameters=None):
        self.p = dict(DEFAULTS)
        self.p.update(parameters or {})
        self._states = {}

    def clear(self):
        self._states.clear()

    def prune(self, keys):
        keep = set(keys)
        self._states = {key: state for key, state in self._states.items() if key in keep}

    def _candidates(self, gray, scale, edges=None):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        if edges is None:
            edges = cv2.Canny(blurred, 25, 75)
        otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        levels = np.unique(np.append(np.percentile(blurred, [15, 30, 45, 60, 75]), otsu))
        binaries = [edges] + [(blurred <= level).astype(np.uint8) * 255 for level in levels]
        candidates = []
        for binary in binaries:
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            plausible = []
            minimum_span = max(self.p["min_aperture_diameter"] * scale,
                               self.p["min_aperture_fraction"] * min(gray.shape))
            for contour in contours:
                _, _, width, height = cv2.boundingRect(contour)
                if len(contour) >= 16 and max(width, height) >= minimum_span and min(width, height) >= minimum_span * .3:
                    plausible.append((width * height, contour))
            for _, contour in sorted(plausible, key=lambda row: row[0], reverse=True)[:128]:
                ellipse = cv2.fitEllipse(contour)
                axes = np.asarray(ellipse[1])
                if (min(axes) < max(3.0, self.p["min_aperture_diameter"] * scale) or
                        min(axes) / max(axes) < self.p["min_axis_ratio"] or
                        max(axes) > max(gray.shape) * 1.05):
                    continue
                points = contour[:, 0, :].astype(float)
                normalized = _normal_points(points, ellipse)
                radii = np.linalg.norm(normalized, axis=1)
                fit_error = float(np.percentile(abs(radii - 1), 90))
                bins = ((np.arctan2(normalized[:, 1], normalized[:, 0]) + np.pi)
                        * 20 / (2 * np.pi)).astype(int) % 20
                coverage = len(np.unique(bins)) / 20
                fill = cv2.contourArea(contour) / (np.pi * axes[0] * axes[1] / 4)
                boundary = _points(ellipse)
                if (fit_error > self.p["max_fit_error"] or
                        coverage < self.p["min_arc_coverage"] or not 0.60 <= fill <= 1.15 or
                        np.any(boundary < 1) or np.any(boundary[:, 0] >= gray.shape[1] - 1) or
                        np.any(boundary[:, 1] >= gray.shape[0] - 1)):
                    continue
                quality = float(coverage * (1 - fit_error / self.p["max_fit_error"]))
                duplicate = next((i for i, (e, _) in enumerate(candidates)
                                  if np.linalg.norm(np.asarray(e[0]) - ellipse[0]) < min(axes) * .04
                                  and max(abs(np.asarray(e[1]) / axes - 1)) < .08), None)
                if duplicate is None:
                    candidates.append((ellipse, quality))
                elif quality > candidates[duplicate][1]:
                    candidates[duplicate] = (ellipse, quality)
        return sorted(candidates, key=lambda row: row[1], reverse=True)[:80], edges

    def _recover_outer_arcs(self, gray, candidates, edges):
        # Wires connect the plate edge to the body: a complete contour is optional,
        # but real edge support around most of the ellipse remains mandatory.
        points = np.column_stack(np.nonzero(edges)[::-1]).astype(float)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        recovered = []
        seeds = sorted((row for row in candidates
                        if row[1] >= .20 and max(row[0][1]) >= min(gray.shape) * .035),
                       key=lambda row: row[1], reverse=True)[:12]
        for inner, quality in seeds:
            if quality < .20 or max(inner[1]) < min(gray.shape) * .035:
                continue
            nearby = points[np.linalg.norm(points - inner[0], axis=1) < max(inner[1]) * 3]
            for factor in np.arange(2.5, 5.1, .5):
                ellipse = (inner[0], tuple(np.asarray(inner[1]) * factor), inner[2])
                for iteration in range(4):
                    normalized = _normal_points(nearby, ellipse)
                    error = abs(np.linalg.norm(normalized, axis=1) - 1)
                    bins = ((np.arctan2(normalized[:, 1], normalized[:, 0]) + np.pi)
                            * 40 / (2 * np.pi)).astype(int) % 40
                    eligible = error < (.25 if iteration == 0 else .14)
                    best = np.full(40, np.inf)
                    np.minimum.at(best, bins[eligible], error[eligible])
                    chosen = eligible & (error <= best[bins] + 1e-8)
                    if np.isfinite(best).sum() < self.p["min_arc_coverage"] * 40:
                        break
                    ellipse = cv2.fitEllipse(nearby[chosen].astype(np.float32))
                else:
                    boundary = _points(ellipse)
                    if (np.any(boundary < 1) or np.any(boundary[:, 0] >= gray.shape[1] - 1) or
                            np.any(boundary[:, 1] >= gray.shape[0] - 1) or
                            min(ellipse[1]) / max(ellipse[1]) < self.p["min_axis_ratio"]):
                        continue
                    support = self._supported(edges, ellipse, distance)
                    if support >= max(.75, self.p["min_arc_coverage"]):
                        recovered.append((ellipse, support * .85))
        return candidates + recovered

    def _pairs(self, gray, candidates, scale):
        pairs = []
        for outer, outer_quality in candidates:
            if (min(outer[1]) < self.p["min_outer_diameter"] * scale or
                    max(outer[1]) < self.p["min_outer_fraction"] * min(gray.shape)):
                continue
            for inner, inner_quality in candidates:
                ratio = max(inner[1]) / max(outer[1])
                if not self.p["min_inner_ratio"] <= ratio <= self.p["max_inner_ratio"]:
                    continue
                if np.max(np.linalg.norm(_normal_points(_points(inner), outer), axis=1)) > .90:
                    continue
                rim = ellipse_mask(gray.shape, outer, shrink=.9) & ~ellipse_mask(gray.shape, inner, shrink=1.2)
                if rim.sum() < 20:
                    continue
                values = gray[rim]
                dark = 1 - np.median(values) / 255
                uniform = 1 - min(1, (np.percentile(values, 90) - np.percentile(values, 10)) / 128)
                offset = np.linalg.norm(_normal_points(np.asarray([inner[0]]), outer))
                quality = .45 * outer_quality + .45 * inner_quality + .05 * dark + .05 * uniform
                pairs.append((float(quality - .05 * offset), outer, inner))
        return pairs

    def _supported(self, edges, ellipse, distance=None):
        if distance is None:
            distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        points = np.rint(_points(ellipse)).astype(int)
        if (points.min() < 1 or points[:, 0].max() >= edges.shape[1] - 1 or
                points[:, 1].max() >= edges.shape[0] - 1):
            return 0.0
        return float(np.mean(distance[points[:, 1], points[:, 0]] <= 2.5))

    def update(self, key, frame, bbox, stamp, transform=None, polygon=None):
        """Measure on this image; motion only propagates recent, supported geometry."""
        previous = self._states.get(key)
        epoch = previous[0].epoch if previous else 0
        matrix = _similarity(transform)
        motion_reset = (matrix is not None and
                        abs(np.linalg.norm(matrix[:2, 0]) - 1) > self.p["max_scale_change"])
        predicted = None
        if previous:
            prior, last_measured = previous
            if prior.outer is not None and prior.aperture is not None:
                predicted = (prior.outer, prior.aperture)
                if matrix is not None:
                    predicted = tuple(_move(e, matrix) for e in predicted)
        x0, y0, x1, y1 = map(float, bbox)
        h, w = frame.shape[:2]
        margin = max(3, int(max(x1 - x0, y1 - y0) * self.p["roi_margin_fraction"]))
        left, top = max(0, int(np.floor(x0)) - margin), max(0, int(np.floor(y0)) - margin)
        right, bottom = min(w, int(np.ceil(x1)) + margin), min(h, int(np.ceil(y1)) + margin)
        if right - left < 12 or bottom - top < 12:
            if previous:
                self._states[key] = (LampRegion(epoch=epoch), last_measured)
            return LampRegion(epoch=epoch, reason="bbox_too_small")
        if predicted and matrix is not None and not motion_reset and 0 <= stamp - last_measured <= self.p["max_tracking_gap"]:
            boundary = _points(predicted[0])
            padding = max(8, max(predicted[0][1]) * .15)
            search_low, search_high = boundary.min(0) - padding, boundary.max(0) + padding
            left, top = max(left, int(search_low[0])), max(top, int(search_low[1]))
            right, bottom = min(right, int(np.ceil(search_high[0]))), min(bottom, int(np.ceil(search_high[1])))
            if right - left < 12 or bottom - top < 12:
                self._states[key] = (LampRegion(epoch=epoch), last_measured)
                return LampRegion(epoch=epoch, reason="motion_outside_bbox")
        roi = frame[top:bottom, left:right]
        scale = min(1.0, self.p["max_roi_side"] / max(roi.shape[:2]))
        if scale < 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = _edge_map(roi)
        def full(ellipse):
            return ((ellipse[0][0] / scale + left, ellipse[0][1] / scale + top),
                    tuple(a / scale for a in ellipse[1]), float(ellipse[2]))
        def local(ellipse):
            return (((ellipse[0][0] - left) * scale, (ellipse[0][1] - top) * scale),
                    tuple(a * scale for a in ellipse[1]), float(ellipse[2]))
        if predicted and matrix is not None and not motion_reset and 0 <= stamp - last_measured <= self.p["max_tracking_gap"]:
            distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
            support = min(self._supported(edges, local(e), distance) for e in predicted)
            if support >= self.p["tracking_min_support"]:
                result = LampRegion(*predicted, support * .7, True, "tracked", "supported_motion", epoch)
                self._states[key] = (result, last_measured)
                return result
        candidates, edges = self._candidates(gray, scale, edges)
        candidates = self._recover_outer_arcs(gray, candidates, edges)
        pairs = self._pairs(gray, candidates, scale)
        if pairs:
            ranked = []
            for quality, outer, inner in pairs:
                outer, inner = full(outer), full(inner)
                continuity = 0.0
                if predicted:
                    shift = np.linalg.norm(np.asarray(outer[0]) - predicted[0][0]) / max(predicted[0][1])
                    continuity = .15 * max(0, 1 - shift / self.p["max_center_shift"])
                ranked.append((quality + continuity, quality, outer, inner))
            _, quality, outer, inner = max(ranked, key=lambda row: row[0])
            reset = previous is None or predicted is None
            if previous and predicted is None:
                epoch += 1
            if predicted:
                shift = np.linalg.norm(np.asarray(inner[0]) - predicted[1][0]) / max(predicted[0][1])
                relative_shift = np.linalg.norm((np.asarray(inner[0]) - outer[0]) -
                                                (np.asarray(predicted[1][0]) - predicted[0][0])) / min(predicted[1][1])
                scale_change = max(abs(np.asarray(outer[1]) / predicted[0][1] - 1))
                aperture_scale_change = max(abs(np.asarray(inner[1]) / predicted[1][1] - 1))
                reset = (motion_reset or shift > self.p["max_center_shift"] or
                         relative_shift > self.p["max_center_shift"] or
                         aperture_scale_change > self.p["max_scale_change"] or
                         scale_change > self.p["max_scale_change"] or
                         stamp - last_measured > self.p["max_tracking_gap"] or stamp < last_measured)
                if reset:
                    epoch += 1
                elif matrix is not None:
                    outer = _blend(predicted[0], outer, .5)
                    inner = _blend(predicted[1], inner, .25)
            result = LampRegion(outer, inner, quality, True, "measured", "ellipse_pair", epoch, reset)
            self._states[key] = (result, float(stamp))
            return result
        if previous:
            self._states[key] = (LampRegion(epoch=epoch), last_measured)
        return LampRegion(epoch=epoch, reason="no_supported_aperture")
