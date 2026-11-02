"""Require repeated, locally measured brightness cycles before learning levels."""
import numpy as np


def measure_periodicity(t, x, lo, hi, max_gap, periodic):
    """Check the latest continuous segment; neither colors nor identities are inferred."""
    result = dict(valid=False, status="INVALID_INPUT", frequency_hz=None,
                  agreement=0.0, cycles=0)
    try:
        t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
        low_hz, high_hz = float(periodic["min_hz"]), float(periodic["max_hz"])
        minimum = float(periodic["min_agreement"])
        phase_steps = int(periodic["phase_steps"])
        lo, hi, max_gap = float(lo), float(hi), float(max_gap)
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    if (t.ndim != 1 or x.shape != t.shape or not np.all(np.isfinite(t))
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite([lo, hi, max_gap, low_hz, high_hz, minimum]))
            or hi <= lo or max_gap <= 0 or not 0 < low_hz <= high_hz
            or not 0 < minimum <= 1 or phase_steps < 4):
        return result
    if len(t) < 2:
        return dict(result, status="INSUFFICIENT_SAMPLES")
    gaps = np.diff(t)
    if np.any(gaps <= 0):
        return result
    breaks = np.flatnonzero(gaps > max_gap)
    if len(breaks):
        start = int(breaks[-1]) + 1
        t, x = t[start:], x[start:]
    if len(t) < 6:
        return dict(result, status="INSUFFICIENT_SAMPLES")
    t = t - t[0]
    gaps = np.diff(t)
    observed = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    confident = np.where(observed <= 0.35, 0, np.where(observed >= 0.65, 1, -1))
    # Hysteresis ignores uncertain transition samples without inventing an edge
    # at the start of the window. Three same-direction edges span two cycles.
    edges = ([], [])
    previous, previous_index = -1, 0
    for index, phase in enumerate(confident):
        if phase < 0:
            continue
        if previous >= 0 and phase != previous:
            edges[int(phase)].append(float((t[previous_index] + t[index]) * 0.5))
        previous, previous_index = int(phase), index
    cycles = max(len(edges[0]), len(edges[1])) - 1
    result["cycles"] = max(cycles, 0)
    if cycles < 2:
        return dict(result, status="INSUFFICIENT_CYCLES")
    periods = np.concatenate([np.diff(edge) for edge in edges])
    period = float(np.median(periods))
    # Sampling uncertainty is bounded by the observed frame gap. Do not permit
    # an arbitrary pair of repeated excursions to count as a stable cycle.
    tolerance = max(0.20 * period, float(np.max(gaps)))
    if np.any(np.abs(periods - period) > tolerance + 1e-9):
        return dict(result, status="IRREGULAR_PERIOD")
    if period < 1.0 / high_hz - np.max(gaps) or period > 1.0 / low_hz + np.max(gaps):
        return dict(result, status="FREQUENCY_OUT_OF_RANGE")
    last_edge = max(edge[-1] for edge in edges if edge)
    if t[-1] - last_edge > 0.75 * period + 0.5 * np.median(gaps):
        return dict(result, status="PERIODICITY_STALE")
    # Use the same 50-percent timing templates and Voronoi time weights as
    # PhotometricSignalDecoder. Bursts of frames cannot outweigh long intervals.
    weight = np.zeros(len(t), dtype=float)
    weight[:-1] += gaps * 0.5
    weight[1:] += gaps * 0.5
    frequencies = np.linspace(low_hz, high_hz, 9)
    measured_hz = 1.0 / period
    if low_hz <= measured_hz <= high_hz:
        frequencies = np.unique(np.append(frequencies, measured_hz))
    phases = np.linspace(0, 1, phase_steps, endpoint=False)
    predicted = np.mod(frequencies[:, None, None] * t[None, None, :]
                       + phases[None, :, None], 1) < 0.5
    agreement = 1 - np.sum(np.abs(predicted - observed) * weight, axis=2) / np.sum(weight)
    fi, pi = np.unravel_index(np.argmax(agreement), agreement.shape)
    best, frequency = float(agreement[fi, pi]), float(frequencies[fi])
    result.update(agreement=best, frequency_hz=frequency)
    if abs(1.0 / frequency - period) > tolerance + 1e-9:
        return dict(result, status="IRREGULAR_PERIOD")
    if best < minimum:
        return dict(result, status="LOW_TEMPLATE_AGREEMENT")
    return dict(result, valid=True, status="PERIODIC")
