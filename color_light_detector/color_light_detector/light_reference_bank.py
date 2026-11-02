"""Session-local per-target periodic references; never share absolute thresholds."""
import math
import statistics


class LightReferenceBank:
    def __init__(self):
        self.entries = {}
        self.last_stamp = None

    def clear(self):
        self.entries.clear()
        self.last_stamp = None

    def prune(self, active_keys, stamp):
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.clear()
        self.last_stamp = stamp
        self.entries = {key: row for key, row in self.entries.items()
                        if key in active_keys and stamp <= row["expires_at"]}

    def record(self, key, learner, stamp, eligible):
        evidence = learner.periodic_evidence
        fresh = (learner.fit_stamp is not None
                 and 0 <= stamp-learner.fit_stamp <= learner.p["reference_hold_seconds"])
        if not eligible or learner.levels is None or not evidence.get("valid") or not fresh:
            self.entries.pop(key, None)
            return False
        low, high = learner.levels
        self.entries[key] = {
            "reference_id": f"{key[0]}:{key[1]}:{learner.reset_count}",
            "track_id": int(key[0]), "roi_epoch": int(key[1]),
            "low": float(low), "high": float(high),
            "modulation": float((high-low)/max(abs(high), 1e-9)),
            "frequency_hz": evidence["frequency_hz"],
            "period_seconds": 1.0/evidence["frequency_hz"],
            "agreement": evidence["agreement"], "cycles": evidence["cycles"],
            "independent": bool(evidence["independent"]),
            "updated_at": learner.fit_stamp,
            "expires_at": learner.fit_stamp+learner.p["reference_hold_seconds"],
        }
        if len(self.entries) > 32:
            oldest = min(self.entries, key=lambda k: self.entries[k]["updated_at"])
            del self.entries[oldest]
        return key in self.entries

    def peers(self, key):
        # One target cannot support itself, even after its ROI identity changes.
        # An assisted reference cannot recursively certify more targets.
        return [dict(row) for other, row in self.entries.items()
                if other[0] != key[0] and row["independent"]]

    @staticmethod
    def compare(peers, frequency, modulation):
        matched = []
        seen = set()
        for row in peers:
            track_id = row["track_id"]
            if track_id in seen:
                continue
            seen.add(track_id)
            f, m = row["frequency_hz"], row["modulation"]
            if not all(math.isfinite(v) for v in (frequency, modulation, f, m)):
                continue
            if (abs(frequency-f) <= 0.10*max(f, 1e-9)
                    and abs(modulation-m) <= max(0.05, 0.35*max(m, 1e-9))):
                matched.append(row)
        return {
            "support_count": len(matched),
            "reference_ids": [row["reference_id"] for row in matched],
            "frequency_hz": statistics.median(row["frequency_hz"] for row in matched) if matched else None,
            "modulation": statistics.median(row["modulation"] for row in matched) if matched else None,
            "absolute_thresholds_shared": False,
        }

    def snapshot(self):
        return {"scope": "CURRENT_SESSION", "reference_count": len(self.entries),
                "independent_count": sum(row["independent"] for row in self.entries.values()),
                "references": [dict(row) for row in self.entries.values()]}
