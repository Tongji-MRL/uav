"""One inference region per scheduled slot; native-pixel photometry is separate.

Full-frame visits discover new objects. Between them an existing small beacon
can fill more of a smaller YOLO input. Predictor letterboxing preserves aspect
ratio; no enlarged pixels are used for light or color decisions.
"""
import math
import cv2
import numpy as np

DEFAULTS = {
    "enabled": False,
    "image_size": 640,
    "full_frame_seconds": 1.0,
    "context_ratio": 3.0,
    "min_side_pixels": 192,
    "max_area_fraction": 0.45,
    "max_target_side_pixels": 320,
    "preview_enabled": False,
}


def validate(values):
    p = dict(DEFAULTS, **values)
    if not all(math.isfinite(float(v)) for v in p.values()):
        raise ValueError("Local zoom parameters must be finite")
    if not 320 <= p["image_size"] <= 1280 or int(p["image_size"]) % 32:
        raise ValueError("local_zoom.image_size must be 320..1280 and a multiple of 32")
    if not .5 <= p["full_frame_seconds"] <= 2 or not 1.5 <= p["context_ratio"] <= 5:
        raise ValueError("Invalid full-frame interval or crop context")
    if not 96 <= p["min_side_pixels"] <= 640 or not .05 <= p["max_area_fraction"] <= .7:
        raise ValueError("Invalid minimum crop size or maximum crop area")
    if not 16 <= p["max_target_side_pixels"] <= 640:
        raise ValueError("Invalid local zoom target size")
    return p


class InferenceRegionPlanner:
    """Owned by the single predictor thread; context travels with its frame."""
    def __init__(self):
        self.last_full = None
        self.epoch = None
        self.cursor = 0
        self.last_reason = "STARTUP_FULL"

    def choose(self, shape, context, parameters, now, fixed_engine=False):
        # Kept for callers of older planners. Fixed engines constrain tensor
        # shape, not the source image region; the worker resolves that shape.
        del fixed_engine
        h, w = shape[:2]
        full = (0, 0, w, h)
        epoch = context.get("epoch")
        if epoch != self.epoch:
            self.last_full, self.epoch, self.cursor = None, epoch, 0
        due = self.last_full is None or now - self.last_full >= parameters["full_frame_seconds"]
        if not parameters["enabled"] or context.get("force_full", False) or due:
            self.last_reason = ("ZOOM_DISABLED" if not parameters["enabled"] else
                                "SEARCH_OR_UNSTABLE" if context.get("force_full", False) else
                                "PERIODIC_FULL")
            self.last_full = now
            return full, False
        regions = []
        for box in context.get("boxes", ()):
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                continue
            bw, bh = x2 - x1, y2 - y1
            if max(bw, bh) > parameters["max_target_side_pixels"]:
                continue
            cw = min(w, max(parameters["min_side_pixels"], bw * parameters["context_ratio"]))
            ch = min(h, max(parameters["min_side_pixels"], bh * parameters["context_ratio"]))
            if cw * ch > parameters["max_area_fraction"] * w * h:
                continue
            left = max(0, min(w - math.ceil(cw), round((x1 + x2 - cw) * .5)))
            top = max(0, min(h - math.ceil(ch), round((y1 + y2 - ch) * .5)))
            regions.append((left, top, left + math.ceil(cw), top + math.ceil(ch)))
        if not regions:
            self.last_reason = "NO_SUITABLE_CROP"
            self.last_full = now
            return full, False
        region = regions[self.cursor % len(regions)]
        self.cursor += 1
        self.last_reason = "TRACKED_TARGET_CROP"
        return region, True


def fit_canvas(frame, size):
    """Resize by a single scale and pad, including non-16:10 cameras."""
    w, h = size
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    scale = min(w / frame.shape[1], h / frame.shape[0])
    rw = max(1, min(w, round(frame.shape[1] * scale)))
    rh = max(1, min(h, round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - rw) // 2, (h - rh) // 2
    canvas[y:y + rh, x:x + rw] = resized
    return canvas


def add_roi_panel(preview, frame, items):
    """At most two native-frame crops, CPU only at the debug publication rate."""
    if not items:
        return preview
    height = preview.shape[0]
    panel_width = min(320, max(160, preview.shape[1] // 3))
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
    count = min(2, len(items))
    slot = height // count
    for index, (box, title, detail) in enumerate(items[:count]):
        x1, y1, x2, y2 = box
        padx, pady = round((x2 - x1) * .12), round((y2 - y1) * .12)
        crop = frame[max(0, y1 - pady):min(frame.shape[0], y2 + pady),
                     max(0, x1 - padx):min(frame.shape[1], x2 + padx)]
        if crop.size == 0 or slot < 64:
            continue
        top = index * slot
        canvas = fit_canvas(crop, (panel_width - 8, slot - 48))
        panel[top + 44:top + slot - 4, 4:panel_width - 4] = canvas
        cv2.putText(panel, title, (6, top + 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, detail, (6, top + 35), cv2.FONT_HERSHEY_SIMPLEX, .40, (100, 220, 250), 1, cv2.LINE_AA)
    return np.concatenate((preview, panel), axis=1)
