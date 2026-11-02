"""Sparse CPU enclosure/context motion for per-frame photometry.

No object identities are created here. A failed forward/backward check means
unobserved, never darkness. Detector results are transported from their own
capture frame before association with the current image.
"""
import cv2
import numpy as np


class RoiMotion:
    def __init__(self, max_width=1280, max_fb_error=1.5, min_points=4, stationary_fallback=False):
        self.max_width = int(max_width)
        self.max_fb_error = float(max_fb_error)
        self.min_points = max(4, int(min_points))
        self.stationary_fallback = bool(stationary_fallback)
        self.last_transforms = {}
        if self.max_width < 320 or not np.isfinite(self.max_fb_error) or self.max_fb_error <= 0:
            raise ValueError("Invalid ROI motion parameters")

    def gray(self, frame):
        if frame.shape[1] > self.max_width:
            frame = cv2.resize(frame, (self.max_width, round(frame.shape[0] * self.max_width / frame.shape[1])),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _transport_flow(self, boxes, previous, current, full_shape, exclusions=()):
        self.last_transforms = {}
        if previous is None or previous.shape != current.shape or not boxes:
            return {}
        if previous is current:
            self.last_transforms = {key: np.eye(3) for key in boxes}
            return {key: (box, 1.0) for key, box in boxes.items()}
        h, w = current.shape
        sx, sy = w / full_shape[1], h / full_shape[0]
        points, ranges = [], {}
        for key, box in boxes.items():
            x1, y1, x2, y2 = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            # Once located, suppress the physical head instead of the box center.
            # Before acquisition the legacy center exclusion remains conservative.
            pad = max(10, int(0.45 * max(x2 - x1, y2 - y1)))
            left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
            right, bottom = min(w, int(x2) + pad), min(h, int(y2) + pad)
            patch = previous[top:bottom, left:right]
            if min(patch.shape, default=0) < 5:
                continue
            mask = np.full(patch.shape, 255, dtype=np.uint8)
            cx, cy = (x1 + x2) / 2 - left, (y1 + y2) / 2 - top
            rx, ry = max(1, (x2 - x1) * 0.25), max(1, (y2 - y1) * 0.25)
            if not exclusions:
                cv2.rectangle(mask, (round(cx - rx), round(cy - ry)), (round(cx + rx), round(cy + ry)), 0, -1)
            for ex1, ey1, ex2, ey2 in exclusions:
                cv2.rectangle(mask, (round(ex1 * sx - left), round(ey1 * sy - top)),
                              (round(ex2 * sx - left), round(ey2 * sy - top)), 0, -1)
            corners = cv2.goodFeaturesToTrack(patch, maxCorners=24, qualityLevel=0.02,
                                             minDistance=3, mask=mask, blockSize=3)
            if corners is None or len(corners) < self.min_points:
                continue
            corners = corners.reshape(-1, 2) + np.asarray([left, top], dtype=np.float32)
            ranges[key] = (len(points), len(points) + len(corners))
            points.extend(corners)
        if not points:
            return {}
        p0 = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, status1, _ = cv2.calcOpticalFlowPyrLK(previous, current, p0, None, **options)
        if p1 is None:
            return {}
        pback, status2, _ = cv2.calcOpticalFlowPyrLK(current, previous, p1, None, **options)
        if pback is None:
            return {}
        good = ((status1.ravel() == 1) & (status2.ravel() == 1)
                & (np.linalg.norm(pback.reshape(-1, 2) - p0.reshape(-1, 2), axis=1) <= self.max_fb_error))
        result = {}
        for key, (start, end) in ranges.items():
            keep = good[start:end]
            if np.count_nonzero(keep) < self.min_points or np.mean(keep) < 0.5:
                continue
            source = p0[start:end].reshape(-1, 2)[keep]
            dest = p1[start:end].reshape(-1, 2)[keep]
            affine, inliers = cv2.estimateAffinePartial2D(source, dest, method=cv2.RANSAC,
                                                        ransacReprojThreshold=2.0, maxIters=100, confidence=0.95)
            if affine is None or inliers is None or np.mean(inliers) < 0.65 or not np.all(np.isfinite(affine)):
                continue
            scale = float(np.hypot(affine[0, 0], affine[1, 0]))
            if not 0.75 <= scale <= 1.33:
                continue
            box = boxes[key]
            corners = np.asarray([[box[0] * sx, box[1] * sy, 1], [box[2] * sx, box[1] * sy, 1],
                                  [box[2] * sx, box[3] * sy, 1], [box[0] * sx, box[3] * sy, 1]])
            warped = corners @ affine.T
            left, top = np.min(warped, axis=0) / [sx, sy]
            right, bottom = np.max(warped, axis=0) / [sx, sy]
            # A mostly out-of-image box cannot establish an OFF state.
            if left < 0 or top < 0 or right > full_shape[1] or bottom > full_shape[0]:
                continue
            candidate = tuple(map(round, (left, top, right, bottom)))
            if candidate[2] - candidate[0] < 3 or candidate[3] - candidate[1] < 3:
                continue
            result[key] = (candidate, float(np.mean(keep) * np.mean(inliers)))
            matrix = np.eye(3)
            matrix[:2] = affine
            scaling = np.diag([sx, sy, 1.0])
            self.last_transforms[key] = np.linalg.inv(scaling) @ matrix @ scaling
        return result


    def transport(self, boxes, previous, current, full_shape, exclusions=()):
        result = self._transport_flow(boxes, previous, current, full_shape, exclusions)
        if (not self.stationary_fallback or previous is None or previous.shape != current.shape):
            return result
        for key, box in boxes.items():
            if key not in result:
                quality = self._stationary_context(box, previous, current, full_shape, exclusions)
                if quality > 0:
                    result[key] = (box, quality)
                    self.last_transforms[key] = np.eye(3)
        return result

    @staticmethod
    def _stationary_context(box, previous, current, full_shape, exclusions=()):
        """Recover sparse-corner failures only with measured stationary context.

        Compare the enclosure/context with the flashing center masked out. This
        never accepts a stale box merely because the detector produced it.
        """
        if not (0 <= box[0] < box[2] <= full_shape[1] and 0 <= box[1] < box[3] <= full_shape[0]):
            return 0.0
        h, w = current.shape
        sx, sy = w / full_shape[1], h / full_shape[0]
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        pad = max(6, int(0.3 * max(x2 - x1, y2 - y1)))
        left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
        right, bottom = min(w, int(x2) + pad), min(h, int(y2) + pad)
        if min(right - left, bottom - top) < 8:
            return 0.0
        a, b = previous[top:bottom, left:right], current[top:bottom, left:right]
        mask = np.ones(a.shape, dtype=np.uint8)
        cx, cy = (x1 + x2) * 0.5 - left, (y1 + y2) * 0.5 - top
        rx, ry = 0.4 * (x2 - x1), 0.4 * (y2 - y1)
        if not exclusions:
            cv2.rectangle(mask, (round(cx - rx), round(cy - ry)),
                          (round(cx + rx), round(cy + ry)), 0, -1)
        for ex1, ey1, ex2, ey2 in exclusions:
            cv2.rectangle(mask, (round(ex1 * sx - left), round(ey1 * sy - top)),
                          (round(ex2 * sx - left), round(ey2 * sy - top)), 0, -1)
        scale = min(1.0, 96.0 / max(a.shape))
        if scale < 1.0:
            shape = (max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale)))
            a = cv2.resize(a, shape, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, shape, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        valid = mask.astype(bool)
        if np.count_nonzero(valid) < 48:
            return 0.0
        a, b = a[valid].astype(np.float32), b[valid].astype(np.float32)
        # Flat patches cannot prove spatial alignment, even if their means match.
        if min(float(np.std(a)), float(np.std(b))) < 6.0:
            return 0.0
        da, db = a - np.mean(a), b - np.mean(b)
        correlation = float(np.dot(da, db) / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-9))
        difference = a - b
        residual = np.abs(difference - np.median(difference))
        if correlation < 0.985 or np.median(residual) > 4.0 or np.percentile(residual, 90) > 12.0:
            return 0.0
        return 0.8 * correlation
