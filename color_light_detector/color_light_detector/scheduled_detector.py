"""One replaceable frame, one YOLO worker; live time-based scheduling.

With the budget enabled, each completed prediction is followed by an idle
period proportional to its measured wall time. This limits this worker's
long-run duty, not the GPU utilization of other processes or individual kernels.
FPS limits and minimum idle can only make the pause longer. With the budget
disabled, a zero rate runs on each newest frame at actual hardware speed.

The prediction callback must return CPU detections after its device work has
completed, so that asynchronous GPU launches are not mistaken for fast work.
"""
import math
import threading
import time

CONTROL_DEFAULTS = {
    "search_fps": 4.0, "track_fps": 3.0, "motion_fps": 6.0,
    "budget_enabled": True, "duty_cycle": 0.55, "min_idle_seconds": 0.04,
}


def validate_control(values):
    p = dict(CONTROL_DEFAULTS)
    p.update(values)
    if not all(math.isfinite(float(p[k])) for k in p if k != "budget_enabled"):
        raise ValueError("YOLO scheduling parameters must be finite")
    if min(p["search_fps"], p["track_fps"], p["motion_fps"], p["min_idle_seconds"]) < 0:
        raise ValueError("YOLO rates and idle must be nonnegative; FPS 0 means uncapped")
    if not 0 < p["duty_cycle"] <= 0.9:
        raise ValueError("duty_cycle must be in (0, 0.9]; disabled budget ignores this value")
    return p


class ScheduledDetector:
    def __init__(self, predict, search_fps=4.0, track_fps=3.0, duty_cycle=0.55,
                 min_idle_seconds=0.04, budget_enabled=True, motion_fps=6.0):
        self.predict = predict
        self.control = validate_control(dict(search_fps=search_fps, track_fps=track_fps,
            duty_cycle=duty_cycle, min_idle_seconds=min_idle_seconds,
            budget_enabled=budget_enabled, motion_fps=motion_fps))
        self.condition = threading.Condition()
        self.latest = None
        self.result = None
        self.closed = False
        self.last_finished = 0.0
        self.last_elapsed = 0.0
        self.retry_at = 0.0
        self.stats = dict(inference_count=0, replaced_inputs=0, inference_ms=0.0,
                          idle_seconds=0.0, error=None, schedule="SEARCH", target_fps=search_fps,
                          budget_enabled=budget_enabled,
                          duty_budget=duty_cycle if budget_enabled else None)
        self.thread = threading.Thread(target=self._run, name="task1-yolo-worker", daemon=True)
        self.thread.start()

    def configure(self, values):
        with self.condition:
            self.control = validate_control({**self.control, **values})
            self.stats.update(budget_enabled=self.control["budget_enabled"],
                duty_budget=self.control["duty_cycle"] if self.control["budget_enabled"] else None)
            self.condition.notify_all()

    @staticmethod
    def _mode(tracking):
        if isinstance(tracking, str) and tracking in ("SEARCH", "MOTION", "TRACK"):
            return tracking
        return "TRACK" if tracking else "SEARCH"

    def _idle(self, mode):
        p = self.control
        fps = p[{"SEARCH": "search_fps", "MOTION": "motion_fps", "TRACK": "track_fps"}[mode]]
        rate_idle = max(0.0, 1.0 / fps - self.last_elapsed) if fps > 0 else 0.0
        if p["budget_enabled"]:
            idle = max(rate_idle, p["min_idle_seconds"], self.last_elapsed * (1 / p["duty_cycle"] - 1))
        else:
            idle = rate_idle
        return idle, fps

    def submit(self, frame, gray, stamp, epoch, tracking, context=None):
        """Replace pending input together with its matching geometry context.

        Callers must treat the submitted frame, gray image and context as
        immutable. A crop choice therefore cannot be paired with a newer frame
        or a newer set of ROI coordinates while this worker is busy.
        """
        with self.condition:
            if self.closed:
                return
            if self.latest is not None:
                self.stats["replaced_inputs"] += 1
            self.latest = (frame, gray, stamp, epoch, self._mode(tracking), context)
            self.condition.notify_all()

    def poll(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    def snapshot(self):
        with self.condition:
            return dict(self.stats)

    def close(self):
        with self.condition:
            self.closed = True
            self.latest = None
            self.condition.notify_all()
        self.thread.join(timeout=3.0)

    def _run(self):
        while True:
            with self.condition:
                while not self.closed:
                    if self.latest is None:
                        self.condition.wait()
                        continue
                    idle, fps = self._idle(self.latest[4])
                    remaining = max(self.last_finished + idle, self.retry_at) - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(timeout=remaining)
                if self.closed:
                    return
                frame, gray, stamp, epoch, mode, context = self.latest
                self.latest = None
            started = time.monotonic()
            detections, error = [], None
            try:
                detections = self.predict(frame, context) if context is not None else self.predict(frame)
            except Exception as exc:
                error = str(exc)
            finished = time.monotonic()
            elapsed = max(0.001, finished - started)
            with self.condition:
                if self.closed:
                    return
                self.last_finished, self.last_elapsed = finished, elapsed
                # Error backoff is recovery behavior, not a GPU load budget.
                self.retry_at = finished + 2.0 if error else 0.0
                idle, fps = self._idle(mode)
                self.result = dict(detections=detections, gray=gray, stamp=stamp,
                                   epoch=epoch, inference_ms=elapsed * 1000,
                                   completed_at=finished, error=error, context=context)
                self.stats.update(inference_count=self.stats["inference_count"] + 1,
                                  inference_ms=round(elapsed * 1000, 2), idle_seconds=round(idle, 3),
                                  error=error, schedule=mode, target_fps=fps,
                                  risk_reasons=(context or {}).get("risk", {}).get("reasons", []))
            del frame, gray, detections, context
