"""Harmonic post-processing, following the definitions given in the paper.

The two quantities the whole validation hangs on are defined in Sect. 5.3:

* the **peak-to-peak amplitude** is "2 times the FFT amplitude at the frequency of
  interest", i.e. only the first harmonic of the platform motion, and
* the **phase shift** is "computed based on the real and imaginary part of the complex
  FFT at the frequency of interest", and is ~90 deg for a quasi-steady response.

Both are expressed here against a single *canonical motion phase* so that surge and
pitch load cases, simulations and experiments are all directly comparable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "canonical_phase_deg",
    "harmonic",
    "lowpass_fft",
    "peak_to_peak",
    "phase_average",
    "phase_shift_deg",
]


# --------------------------------------------------------------------------------------
# Canonical motion phase
# --------------------------------------------------------------------------------------
# theta = 0 deg  -> platform at neutral, moving downwind  -> rotor apparent wind minimum
# theta = 180 deg -> platform at neutral, moving upwind   -> rotor apparent wind maximum
#
# i.e. the hub displacement (positive downwind) is  x_hub(theta) = A * sin(theta).
#
# The definition document (Sect. 6.1) prescribes opposite starting directions for the two
# motion types: surge starts as a negative sine, platform pitch as a positive sine. That
# is why NREL applied a 180 deg shift to the surge results before comparing them with the
# pitch results, and why the paper's Fig. 13 shows the thrust at a minimum at phase 0.

def canonical_phase_deg(time_s, frequency_hz: float, motion: str) -> np.ndarray:
    """Canonical motion phase in degrees for a prescribed simulation time vector."""
    time_s = np.asarray(time_s, dtype=float)
    if motion == "none" or frequency_hz == 0:
        return np.full(time_s.shape, np.nan)
    offset = {"surge": 180.0, "pitch": 0.0}[motion]
    return (360.0 * frequency_hz * time_s + offset) % 360.0


def canonical_phase_from_measurement(displacement, frequency_hz: float,
                                     sample_rate_hz: float) -> np.ndarray:
    """Canonical motion phase recovered from a measured displacement record.

    ``displacement`` must be positive downwind. The phase is taken from the first
    harmonic so that noise and drift in the record do not perturb the binning.
    """
    displacement = np.asarray(displacement, dtype=float)
    n = displacement.size
    coefficient = harmonic(displacement, frequency_hz, sample_rate_hz)
    # x = D * sin(w t + phi0) has FFT phase phi0 - pi/2.
    phi0 = np.angle(coefficient) + np.pi / 2
    t = np.arange(n) / sample_rate_hz
    return np.degrees(2 * np.pi * frequency_hz * t + phi0) % 360.0


# --------------------------------------------------------------------------------------
# Single-harmonic extraction
# --------------------------------------------------------------------------------------

def _bin_index(n: int, frequency_hz: float, sample_rate_hz: float) -> int:
    exact = frequency_hz * n / sample_rate_hz
    index = int(round(exact))
    if index < 1 or index >= n // 2:
        raise ValueError(f"frequency {frequency_hz} Hz is outside the resolvable range")
    return index


def harmonic(signal, frequency_hz: float, sample_rate_hz: float,
             order: int = 1) -> complex:
    """Complex single-sided amplitude of ``signal`` at ``order * frequency_hz``.

    The magnitude is the amplitude of the sinusoid (not the FFT bin value), so that
    ``2 * abs(harmonic(...))`` is the peak-to-peak value of that harmonic.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    n = signal.size
    index = _bin_index(n, order * frequency_hz, sample_rate_hz)
    spectrum = np.fft.rfft(signal - signal.mean())
    return 2 * spectrum[index] / n


def _period_harmonic(signal, order: int = 1) -> complex:
    """Harmonic of a record that covers exactly one period, endpoint excluded."""
    signal = np.asarray(signal, dtype=float)
    if signal.size > 1 and np.isclose(signal[0], signal[-1], rtol=1e-6, atol=1e-9):
        signal = signal[:-1]
    n = signal.size
    spectrum = np.fft.rfft(signal - signal.mean())
    return 2 * spectrum[order] / n


def peak_to_peak(signal, frequency_hz: float | None = None,
                 sample_rate_hz: float | None = None, order: int = 1) -> float:
    """Peak-to-peak amplitude of the first harmonic, as defined in the paper.

    With ``frequency_hz``/``sample_rate_hz`` omitted the signal is assumed to cover
    exactly one motion period, which is how the participants' files are written.
    """
    if frequency_hz is None:
        return float(2 * abs(_period_harmonic(signal, order)))
    return float(2 * abs(harmonic(signal, frequency_hz, sample_rate_hz, order)))


def phase_shift_deg(load, phase_deg, frequency_hz: float | None = None,
                    sample_rate_hz: float | None = None) -> float:
    """Phase shift of ``load`` with respect to the platform motion, in degrees.

    ``phase_deg`` is the canonical motion phase sampled at the same instants as
    ``load``. A quasi-steady response returns ~90 deg.
    """
    reference = np.sin(np.radians(np.asarray(phase_deg, dtype=float)))
    if frequency_hz is None:
        h_ref = _period_harmonic(reference)
        h_load = _period_harmonic(load)
    else:
        h_ref = harmonic(reference, frequency_hz, sample_rate_hz)
        h_load = harmonic(load, frequency_hz, sample_rate_hz)
    return float(np.degrees(np.angle(h_ref) - np.angle(h_load)) % 360.0)


# --------------------------------------------------------------------------------------
# Filtering and phase averaging (used for the measured records)
# --------------------------------------------------------------------------------------

def lowpass_fft(signal, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass filter performed in the frequency domain.

    Paper Sect. 5.3: "The complex fast Fourier transform (FFT) was performed, the
    frequency of interest was kept (i.e., from 0 to 3 Hz), and the inverse of the FFT was
    applied to reconstruct the time domain signal. The main advantage of this approach is
    that it does not introduce a phase lag in the signal."

    The 3 Hz cut-off keeps the fastest platform motion (2 Hz) while removing the 1P
    response at 4 Hz caused by the rotor mass imbalance, and the 3P tower-shadow response.
    The cut-off frequency itself is removed too: in the 1 Hz cases the Experiment 1 record
    carries a third harmonic of the motion at exactly 3 Hz (0.12 N m on a 0.57 N m torque
    response), which survives phase averaging and which the paper's curves (Fig. 16) do
    not show.
    """
    signal = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    spectrum[frequencies >= cutoff_hz - 1e-9] = 0.0
    return np.fft.irfft(spectrum, n=signal.size)


def phase_average(phase_deg, signal, n_bins: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Bin ``signal`` by canonical motion phase and average within each bin.

    Returns the bin centres in degrees and the mean of ``signal`` in each bin.
    """
    phase_deg = np.asarray(phase_deg, dtype=float) % 360.0
    signal = np.asarray(signal, dtype=float)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    index = np.clip(np.digitize(phase_deg, edges) - 1, 0, n_bins - 1)
    totals = np.bincount(index, weights=signal, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return (edges[:-1] + edges[1:]) / 2, means
