"""Task 1 protocol decoder: measured light phase, time-weighted ON color."""
from collections import deque
import numpy as np

COLORS = ("RED", "GREEN", "BLUE")


class PhotometricSignalDecoder:
    def __init__(self, analyzer, max_gap_seconds=0.35):
        self.a = analyzer
        self.max_gap = float(max_gap_seconds)
        if not 0 < self.max_gap < 1.0:
            raise ValueError("photometric_max_gap_seconds must be between 0 and 1")
        self.samples = deque()
        self.diagnostics = {}

    def reset(self):
        self.samples.clear()
        self.diagnostics = {}

    def trim_before(self, stamp):
        while self.samples and self.samples[0][0] < stamp:
            self.samples.popleft()

    def update(self, stamp, phase, signal, scores):
        if self.samples and stamp <= self.samples[-1][0]:
            self.reset()
        elif self.samples and stamp - self.samples[-1][0] > self.max_gap:
            self.reset()
        if phase not in ("ON", "OFF", "UNCERTAIN"):
            phase = "UNCERTAIN"
        self.samples.append((stamp, phase, float(signal), dict(scores)))
        self.trim_before(stamp - self.a.history_seconds)
        samples = list(self.samples)
        t = np.asarray([s[0] - samples[0][0] for s in samples])
        span = float(t[-1])
        phase_on = np.asarray([s[1] == "ON" for s in samples])
        phase_off = np.asarray([s[1] == "OFF" for s in samples])
        valid = phase_on | phase_off
        signal = np.asarray([s[2] for s in samples])
        valid &= np.isfinite(signal)
        # Voronoi time weights prevent a burst of frames from owning a pulse.
        dt = np.diff(t)
        weight = np.ones(len(t)) * 1e-3
        if len(dt):
            weight[:-1] += np.minimum(dt, self.max_gap) * 0.5
            weight[1:] += np.minimum(dt, self.max_gap) * 0.5
        energy = np.asarray([[max(s[3].get(c, 0), 0) for c in COLORS] for s in samples])
        energy[~phase_on] = 0
        totals = np.sum(energy * weight[:, None], axis=0)
        dominant = COLORS[int(np.argmax(totals))] if np.sum(totals) > 0 else "UNKNOWN"
        purity = float(np.max(totals) / max(np.sum(totals), 1e-9))
        levels = np.sum(energy * weight[:, None], axis=0) / max(float(np.sum(weight[phase_on])), 1e-9)
        pulse_colors = []
        pulse = np.zeros(3)
        # A pulse never integrates across OFF or missing/uncertain observations.
        for i in range(len(samples)):
            if phase_on[i]:
                pulse += energy[i] * weight[i]
            elif np.sum(pulse) > 0:
                pulse_colors.append(COLORS[int(np.argmax(pulse))])
                pulse = np.zeros(3)
        if np.sum(pulse) > 0:
            pulse_colors.append(COLORS[int(np.argmax(pulse))])
        coverage = float(np.sum(weight[valid]) / max(np.sum(weight), 1e-9))
        self.diagnostics = dict(expected_period_seconds=1.0/self.a.template_target_hz,
                                observed_fraction=round(coverage, 4),
                                pulse_colors=pulse_colors, color_purity=round(purity, 4),
                                on_color_energy=dict(zip(COLORS, totals.tolist())))
        result = dict(state="UNKNOWN", confidence=0.0, dominant_color=dominant,
                      history_span=span, method=self.a.mode)
        # Solid/OFF require a continuous measured phase. Uncertain and MISSING
        # samples terminate the run: they never establish a one-second OFF gap.
        last_phase = samples[-1][1]
        start = len(samples) - 1
        while start > 0 and samples[start - 1][1] == last_phase:
            start -= 1
        run = float(t[-1] - t[start])
        if last_phase == "OFF" and run >= self.a.off_min_seconds:
            return dict(result, state="OFF", confidence=coverage)
        if last_phase == "ON" and run >= self.a.solid_min_seconds:
            run_energy = np.sum(energy[start:] * weight[start:, None], axis=0)
            blue_purity = float(run_energy[2] / max(np.sum(run_energy), 1e-9))
            blue_level = float(run_energy[2] / max(np.sum(weight[start:]), 1e-9))
            if blue_purity >= self.a.photometric_blue_purity_min and blue_level >= self.a.photometric_blue_level_min:
                return dict(result, state="SOLID_BLUE", confidence=blue_purity,
                            dominant_color="BLUE", on_ratio=1.0)
        if (span < self.a.blink_min_window or np.count_nonzero(valid) < self.a.photometric_min_samples
                or coverage < 0.80 or np.count_nonzero(phase_on) < 3 or np.count_nonzero(phase_off) < 3):
            return result
        known_phase = phase_on[valid]
        transitions = int(np.count_nonzero(known_phase[1:] != known_phase[:-1]))
        on_ratio = float(np.sum(weight[phase_on]) / max(np.sum(weight[valid]), 1e-9))
        if transitions < self.a.blink_min_transitions or not 0.15 <= on_ratio <= 0.85:
            return dict(result, transitions=transitions, on_ratio=on_ratio)
        low = float(np.median(signal[phase_off & valid]))
        high = float(np.median(signal[phase_on & valid]))
        self.diagnostics.update(signal_delta=high-low,
                                required_signal_delta=self.a.photometric_signal_delta_min)
        if high - low < self.a.photometric_signal_delta_min:
            return result
        observed = np.clip((np.nan_to_num(signal) - low) / (high - low), 0, 1)
        # Vectorize only the existing lightweight timing templates, on the CPU.
        frequencies = np.linspace(max(self.a.blink_min_hz, self.a.template_target_hz - self.a.template_frequency_tolerance),
                                  min(self.a.blink_max_hz, self.a.template_target_hz + self.a.template_frequency_tolerance), 9)
        phases = np.linspace(0, 1, self.a.template_phase_steps, endpoint=False)
        predicted = np.mod(frequencies[:, None, None] * t[None, None, :] + phases[None, :, None], 1) < 0.5
        weights = weight * valid
        agreement = 1 - np.sum(np.abs(predicted - observed) * weights, axis=2) / max(np.sum(weights), 1e-9)
        fi, pi = np.unravel_index(np.argmax(agreement), agreement.shape)
        best = float(agreement[fi, pi])
        # Color evidence belongs to measured ON pulses and must also agree with
        # the fitted ON phase; a reflected OFF color cannot own this decision.
        aligned = phase_on & predicted[fi, pi]
        pulse_totals = np.sum(energy * (weight * aligned)[:, None], axis=0)
        index = int(np.argmax(pulse_totals))
        pulse_purity = float(pulse_totals[index] / max(np.sum(pulse_totals), 1e-9))
        color_level = float(pulse_totals[index] / max(np.sum(weight[aligned]), 1e-9))
        minimum = self.a.photometric_blue_level_min if index == 2 else self.a.photometric_color_level_min
        purity_minimum = self.a.photometric_blue_purity_min if index == 2 else self.a.photometric_color_purity_min
        self.diagnostics.update(template_agreement=best,
                                required_agreement=self.a.photometric_min_agreement,
                                aligned_color_purity=pulse_purity,
                                required_color_purity=purity_minimum,
                                aligned_color_level=color_level, required_color_level=minimum)
        result.update(transitions=transitions, on_ratio=on_ratio,
                      blink_hz=float(frequencies[fi]), template_agreement=best)
        if (best >= self.a.photometric_min_agreement and pulse_purity >= purity_minimum
                and color_level >= minimum):
            result.update(state="BLINK_" + COLORS[index], dominant_color=COLORS[index],
                          confidence=float(0.6 * best + 0.4 * pulse_purity))
        return result
