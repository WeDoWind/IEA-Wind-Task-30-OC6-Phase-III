"""Reduce the measured wind-tunnel records to the quantities the paper plots.

The OC6 Phase III paper compares the simulations against two campaigns, both run in the
Politecnico di Milano wind tunnel on the same 1:75 DTU 10 MW rotor:

* **Experiment 1** - the UNAFLOW campaign (Fontanella et al., 2021), published openly at
  https://doi.org/10.5281/zenodo.4740005 and processed here.
* **Experiment 2** - the 2021 follow-on campaign, published on the Wind Data Hub in
  October 2026, in ``archive.experiment2.tower_top_loads``. Reduced here by
  :func:`experiment2_forces` and :func:`experiment2_summary`; see the section at the end.

The input is the UNAFLOW archive tables (``archive.experiment1.*``), which hold the
record unchanged: a 2 kHz record of the tower-top balance, not the hub loads the paper
plots. The reduction applied here is:

1. ``thrust = -TTFx`` and ``torque = -TTMx``. The tower-top frame has *x* pointing
   upwind (companion document Fig. 1), so both channels are negative in the raw file.
2. For the oscillating cases the record is low-pass filtered at 3 Hz in the frequency
   domain (paper Sect. 4 and 5.3) and phase-averaged over all motion periods against the
   canonical motion phase recovered from the measured displacement.

**No inertial-load subtraction is needed**: the published records already have it removed
(R. Bergua, pers. comm., 2026). That matches what the data shows - the measured load is in
phase with the platform *velocity*, not its acceleration.

**The projection onto the hub is implemented but off by default.**
:func:`tower_top_to_hub` is the exact two-step recipe - a 5 deg rotation into
ground-parallel axes, then a translation by X = 0.0979 m and Z = 0.0365 m. Applied to the
*published* UNAFLOW channels it improves the steady torque (3.32 -> 3.24 N m against the
paper's 3.25) but degrades the thrust (35.91 -> 36.70 against the paper's 36.1), because
the rotation mixes the vertical channel into the thrust and that channel still carries the
static weight above the balance - only the inertial part was removed. Leaving it off
reproduces the published values to within 1 %. Pass ``project_to_hub=True`` for records
whose vertical channel is gravity-free.

Treat these columns as a faithful reduction of the open UNAFLOW data rather than a
byte-for-byte reproduction of the paper's Experiment 1 curves.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..harmonics import (
    canonical_phase_from_measurement,
    harmonic,
    lowpass_fft,
    phase_average,
)
from ..registry import (
    HOTWIRE_CROSSWIND_Y_EXP1_OFFSET_M,
    HUB_OFFSET_X_M,
    HUB_OFFSET_Z_M,
    HUB_TO_PITCH_AXIS_M,
    TILT_ANGLE_DEG,
    load_cases,
)

SAMPLE_RATE_HZ = 2000.0
LOWPASS_CUTOFF_HZ = 3.0
N_PHASE_BINS = 72

EXPERIMENT = "Experiment 1 (UNAFLOW)"
EXPERIMENT2 = "Experiment 2"

_CONDITION = re.compile(
    r"(?P<motion>Steady|Surge)\d*_V(?P<speed>[0-9d]+)"
    r"(?:_A(?P<amplitude>[0-9d]+)_F(?P<frequency>[0-9d]+))?"
)


def _number(token: str | None) -> float:
    return float(token.replace("d", ".")) if token else 0.0


def parse_condition(name: str) -> dict | None:
    """Extract ``(motion, wind speed, amplitude, frequency)`` from an UNAFLOW path.

    The whole path is searched, not just the file name: force and wake records carry the
    condition in the file name (``Surge50_V4_A0d035_F1.dat``) while PIV planes carry it in
    the directory (``PIV/Steady_V4/AVERAGED/psi30.dat``).
    """
    match = _CONDITION.search(str(name))
    if match is None:
        return None
    parts = match.groupdict()
    return {
        "motion": "none" if parts["motion"] == "Steady" else "surge",
        "wind_speed_ms": _number(parts["speed"]),
        "amplitude_m": _number(parts["amplitude"]),
        "frequency_hz": _number(parts["frequency"]),
    }


def match_load_case(condition: dict) -> str | None:
    """Map an UNAFLOW test condition onto an OC6 load case, or ``None``."""
    cases = load_cases()
    for case in cases.itertuples():
        if case.motion != condition["motion"]:
            continue
        if not np.isclose(case.wind_speed_measured_ms or np.nan,
                          condition["wind_speed_ms"], atol=0.05):
            continue
        if condition["motion"] == "none":
            return case.load_case
        if np.isclose(case.frequency_hz, condition["frequency_hz"], atol=1e-6) and np.isclose(
            case.amplitude, condition["amplitude_m"], atol=1e-4
        ):
            return case.load_case
    return None


def tower_top_to_hub(fx, fy, fz, mx, my, mz, tilt_deg: float = TILT_ANGLE_DEG,
                     offset_x_m: float = HUB_OFFSET_X_M,
                     offset_z_m: float = HUB_OFFSET_Z_M):
    """Project tower-top loads onto the hub, in ground-parallel coordinates.

    The recipe is R. Bergua's (pers. comm., 2026), in two steps.

    **Rotate** out of the tilted tower-top frame into one parallel to the ground::

        [Fx]         [ cos a   0  -sin a ] [Fx]
        [Fy]ground = [   0     1     0   ] [Fy]tilted
        [Fz]         [ sin a   0   cos a ] [Fz]

    **Translate** to the hub. The forces are unchanged; the moments pick up the offset,
    X = Overhang + Twr2Shft sin(a) = 0.0979 m and Z = Twr2Shft cos(a) = 0.0365 m::

        Mx_hub = Mx + Fy Z        My_hub = My - Fx Z - Fz X        Mz_hub = Mz + Fy X

    Inputs must already be in the tower-top frame with *x downwind and z up*, and must
    have the inertial loading removed - the measured OC6 records do.
    """
    angle = np.radians(tilt_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    fx_g, fy_g, fz_g = fx * cos - fz * sin, fy, fx * sin + fz * cos
    mx_g, my_g, mz_g = mx * cos - mz * sin, my, mx * sin + mz * cos
    return (
        fx_g, fy_g, fz_g,
        mx_g + fy_g * offset_z_m,
        my_g - fx_g * offset_z_m - fz_g * offset_x_m,
        mz_g + fy_g * offset_x_m,
    )


def _hub_loads(data: pd.DataFrame, project_to_hub: bool):
    """Thrust and torque from an UNAFLOW record, optionally projected onto the hub.

    The published tower-top channels have *x pointing upwind* (companion document Fig. 1),
    so they are negated into the downwind-positive frame the projection expects.

    ``project_to_hub`` is off by default. Applied to the published UNAFLOW channels it
    moves the steady torque from 3.32 to 3.24 N m against the paper's 3.25, but the thrust
    from 35.91 to 36.70 against the paper's 36.1 - because the published vertical channel
    still carries the static weight above the balance, which the rotation then mixes into
    the thrust. Only the inertial loading was removed. Turn it on for records whose
    vertical channel is gravity-free.
    """
    fx, fy, fz = -data.ttfx_n.to_numpy(), -data.ttfy_n.to_numpy(), data.ttfz_n.to_numpy()
    mx, my, mz = -data.ttmx_nm.to_numpy(), -data.ttmy_nm.to_numpy(), data.ttmz_nm.to_numpy()
    if project_to_hub:
        fx, _, _, mx, _, _ = tower_top_to_hub(fx, fy, fz, mx, my, mz)
    return fx, mx




def oc6_tests(tests: pd.Series) -> dict[str, str]:
    """``{test: load_case}`` for the UNAFLOW tests that correspond to an OC6 load case."""
    matched = {}
    for test in pd.unique(tests):
        condition = parse_condition(test)
        load_case = match_load_case(condition) if condition else None
        if load_case is not None:
            matched[test] = load_case
    return matched


# --------------------------------------------------------------------------------------
# Forces
# --------------------------------------------------------------------------------------

def forces(tower_loads: pd.DataFrame, project_to_hub: bool = False) -> pd.DataFrame:
    """Phase-averaged hub thrust and torque per load case. See :func:`_hub_loads`.

    ``tower_loads`` is ``archive.experiment1.tower_loads``, or any subset of its tests.
    """
    blocks = []
    for test, load_case in oc6_tests(tower_loads.test).items():
        data = tower_loads[tower_loads.test == test]
        condition = parse_condition(test)
        thrust, torque = _hub_loads(data, project_to_hub)
        if condition["motion"] == "none":
            block = pd.DataFrame(
                {"motion_phase_deg": [np.nan],
                 "fx_n": [thrust.mean()], "mx_nm": [torque.mean()]}
            )
        else:
            phase = canonical_phase_from_measurement(
                data.platform_displacement_m.to_numpy(), condition["frequency_hz"],
                SAMPLE_RATE_HZ,
            )
            filtered_thrust = lowpass_fft(thrust, SAMPLE_RATE_HZ, LOWPASS_CUTOFF_HZ)
            filtered_torque = lowpass_fft(torque, SAMPLE_RATE_HZ, LOWPASS_CUTOFF_HZ)
            centres, mean_thrust = phase_average(phase, filtered_thrust, N_PHASE_BINS)
            _, mean_torque = phase_average(phase, filtered_torque, N_PHASE_BINS)
            block = pd.DataFrame(
                {"motion_phase_deg": centres, "fx_n": mean_thrust, "mx_nm": mean_torque}
            )
        block.insert(0, "experiment", EXPERIMENT)
        block.insert(1, "load_case", load_case)
        block["test"] = test
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["load_case", "motion_phase_deg"], ignore_index=True
    )


def force_summary(tower_loads: pd.DataFrame, project_to_hub: bool = False) -> pd.DataFrame:
    """Mean, peak-to-peak and phase of the measured thrust and torque per load case."""
    rows = []
    for test, load_case in oc6_tests(tower_loads.test).items():
        data = tower_loads[tower_loads.test == test]
        condition = parse_condition(test)
        thrust, torque = _hub_loads(data, project_to_hub)
        row = {
            "experiment": EXPERIMENT,
            "load_case": load_case,
            "fx_mean_n": float(thrust.mean()),
            "mx_mean_nm": float(torque.mean()),
            "fx_peak_to_peak_n": np.nan,
            "mx_peak_to_peak_nm": np.nan,
            "fx_phase_deg": np.nan,
            "mx_phase_deg": np.nan,
            "test": test,
        }
        if condition["motion"] != "none":
            frequency = condition["frequency_hz"]
            h_disp = harmonic(data.platform_displacement_m.to_numpy(), frequency,
                              SAMPLE_RATE_HZ)
            h_thrust = harmonic(thrust, frequency, SAMPLE_RATE_HZ)
            h_torque = harmonic(torque, frequency, SAMPLE_RATE_HZ)
            row["fx_peak_to_peak_n"] = float(2 * abs(h_thrust))
            row["mx_peak_to_peak_nm"] = float(2 * abs(h_torque))
            row["fx_phase_deg"] = float(np.degrees(np.angle(h_disp) - np.angle(h_thrust)) % 360)
            row["mx_phase_deg"] = float(np.degrees(np.angle(h_disp) - np.angle(h_torque)) % 360)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("load_case", ignore_index=True)


# --------------------------------------------------------------------------------------
# Hot wire
# --------------------------------------------------------------------------------------

def hotwire(hotwire_wake: pd.DataFrame, steady_only: bool = True) -> pd.DataFrame:
    """Time-averaged wake velocities at each traverse point.

    ``hotwire_wake`` is ``archive.experiment1.hotwire_wake``: 11 along-wind or 33 crosswind
    points per test (companion document, WAKE section). Only the steady cases are reduced
    by default, which is what the paper uses in Figs. 7 and 8. Positions are expressed
    relative to the hub, as for the simulations.
    """
    from ..registry import (
        HOTWIRE_ALONGWIND_X_M,
        HOTWIRE_ALONGWIND_Y_M,
        HOTWIRE_CROSSWIND_X_M,
        HOTWIRE_CROSSWIND_Y_M,
    )

    rows = []
    for test, load_case in oc6_tests(hotwire_wake.test).items():
        condition = parse_condition(test)
        if steady_only and condition["motion"] != "none":
            continue
        data = hotwire_wake[hotwire_wake.test == test]
        means = data.groupby("point")[["u_ms", "v_ms", "w_ms"]].mean().sort_index()
        n_points = len(means)
        direction = data.traverse.iloc[0]
        if direction == "aw":
            probe_x = np.asarray(HOTWIRE_ALONGWIND_X_M, dtype=float)[:n_points]
            probe_y = np.full(n_points, HOTWIRE_ALONGWIND_Y_M)
        else:
            probe_x = np.full(n_points, HOTWIRE_CROSSWIND_X_M)
            probe_y = np.asarray(HOTWIRE_CROSSWIND_Y_M, dtype=float)[:n_points]
            if condition["motion"] == "none":
                # Paper Sect. 5.1.2: for the steady case the crosswind probes sat
                # 0.13 m further along y, effectively covering -1.47 m to 1.73 m.
                probe_y = probe_y + HOTWIRE_CROSSWIND_Y_EXP1_OFFSET_M
        rows.append(
            pd.DataFrame(
                {
                    "experiment": EXPERIMENT,
                    "load_case": load_case,
                    "direction": direction,
                    "probe_x_m": probe_x,
                    "probe_y_m": probe_y,
                    "u_ms": means.u_ms.to_numpy(),
                    "v_ms": means.v_ms.to_numpy(),
                    "w_ms": means.w_ms.to_numpy(),
                    "test": test,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# PIV
# --------------------------------------------------------------------------------------

def _blank_mask(block: pd.DataFrame) -> np.ndarray:
    """Cells with no PIV data, grown by one cell in every direction."""
    from scipy.ndimage import binary_dilation

    blank = (block.u_ms == 0).to_numpy()
    n_x = block.x_m.nunique()
    if blank.size % n_x:  # not a full rectangular grid; fall back to the raw mask
        return blank
    grid = blank.reshape(-1, n_x)
    return binary_dilation(grid, iterations=1).reshape(-1)


def piv(piv_planes: pd.DataFrame, load_case: str = "1.1") -> pd.DataFrame:
    """Averaged PIV planes for one load case, one frame per blade azimuth angle, blanked.

    ``piv_planes`` is ``archive.experiment1.piv_wake_plane``. The PIV laser sheet did not
    cover the whole window; the companion document asks for the u = 0 rectangle to be
    blanked. The ring of cells bordering that rectangle carries a spurious vorticity spike
    from the one-sided difference across the edge, so it is blanked as well.
    """
    tests = [test for test, lc in oc6_tests(piv_planes.test).items() if lc == load_case]
    selection = piv_planes[piv_planes.test.isin(tests)
                           & (piv_planes.averaging == "averaged")
                           & piv_planes.azimuth_deg.notna()]
    blocks = []
    for azimuth, block in selection.groupby("azimuth_deg", sort=True):
        block = block[["x_m", "z_m", "u_ms", "w_ms", "vorticity_1_s"]].copy()
        block.loc[_blank_mask(block), ["u_ms", "w_ms", "vorticity_1_s"]] = np.nan
        block.insert(0, "experiment", EXPERIMENT)
        block.insert(1, "load_case", load_case)
        block.insert(2, "azimuth_deg", azimuth)
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    frame["experiment"] = frame["experiment"].astype("category")
    return frame.sort_values(["azimuth_deg", "z_m", "x_m"], ignore_index=True)


# --------------------------------------------------------------------------------------
# Experiment 2
# --------------------------------------------------------------------------------------
# The published Experiment 2 channels reproduce the paper's Experiment 2 values as they
# are: steady thrust 33.7 N and torque 3.23 N m (Figs. 3, 4), and every normalised
# amplitude and phase shift of Figs. 17 and 18 to within a few percent and a degree. So
# thrust is Fx and torque Mx, with no sign change and no projection - projecting with the
# Experiment 2 geometry of paper Table 2 would move the steady torque to 3.10 N m, away
# from the published 3.22. The record is filtered and phase-averaged like Experiment 1.
#
# The amplitudes are normalised by the motion the robot actually achieved, not the one
# prescribed: at 2 Hz in pitch it reached 0.26 deg of the prescribed 0.3, and the paper's
# Fig. 17a (249 N/m) uses the achieved value.


def _experiment2_cases(tower_top: pd.DataFrame):
    cases = load_cases()
    for load_case, data in tower_top.groupby(tower_top.load_case.astype(str), sort=False):
        time = data.time_s.to_numpy()
        yield cases.loc[load_case], data, 1.0 / float(np.median(np.diff(time)))


def _surge_equivalent_m(case, amplitude: float) -> float:
    """Achieved motion amplitude as a hub displacement in metres."""
    if case.motion == "pitch":
        return HUB_TO_PITCH_AXIS_M * np.sin(np.radians(amplitude))
    return amplitude


def experiment2_forces(tower_top: pd.DataFrame) -> pd.DataFrame:
    """Phase-averaged thrust and torque per load case (Experiment 2).

    ``tower_top`` is ``archive.experiment2.tower_top_loads``.
    """
    blocks = []
    for case, data, sample_rate in _experiment2_cases(tower_top):
        thrust, torque = data.ttfx_n.to_numpy(), data.ttmx_nm.to_numpy()
        if case.motion == "none":
            block = pd.DataFrame({"motion_phase_deg": [np.nan],
                                  "fx_n": [thrust.mean()], "mx_nm": [torque.mean()]})
        else:
            phase = canonical_phase_from_measurement(
                data.platform_displacement.to_numpy(), case.frequency_hz, sample_rate
            )
            centres, mean_thrust = phase_average(
                phase, lowpass_fft(thrust, sample_rate, LOWPASS_CUTOFF_HZ), N_PHASE_BINS)
            _, mean_torque = phase_average(
                phase, lowpass_fft(torque, sample_rate, LOWPASS_CUTOFF_HZ), N_PHASE_BINS)
            block = pd.DataFrame({"motion_phase_deg": centres, "fx_n": mean_thrust,
                                  "mx_nm": mean_torque})
        block.insert(0, "experiment", EXPERIMENT2)
        block.insert(1, "load_case", case.load_case)
        block["test"] = str(data.test.iloc[0])
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def experiment2_summary(tower_top: pd.DataFrame) -> pd.DataFrame:
    """Mean, peak-to-peak, phase and normalised amplitude per load case (Experiment 2)."""
    rows = []
    for case, data, sample_rate in _experiment2_cases(tower_top):
        thrust, torque = data.ttfx_n.to_numpy(), data.ttmx_nm.to_numpy()
        row = {
            "experiment": EXPERIMENT2,
            "load_case": case.load_case,
            "fx_mean_n": float(thrust.mean()),
            "mx_mean_nm": float(torque.mean()),
            "fx_peak_to_peak_n": np.nan,
            "mx_peak_to_peak_nm": np.nan,
            "fx_phase_deg": np.nan,
            "mx_phase_deg": np.nan,
            "motion_amplitude_m": np.nan,
            "fx_peak_to_peak_per_amplitude_n_per_m": np.nan,
            "mx_peak_to_peak_per_amplitude_nm_per_m": np.nan,
            "test": str(data.test.iloc[0]),
        }
        if case.motion != "none":
            f = case.frequency_hz
            h_disp = harmonic(data.platform_displacement.to_numpy(), f, sample_rate)
            h_thrust = harmonic(thrust, f, sample_rate)
            h_torque = harmonic(torque, f, sample_rate)
            amplitude = _surge_equivalent_m(case, abs(h_disp))
            row.update(
                fx_peak_to_peak_n=float(2 * abs(h_thrust)),
                mx_peak_to_peak_nm=float(2 * abs(h_torque)),
                fx_phase_deg=float(np.degrees(np.angle(h_disp) - np.angle(h_thrust)) % 360),
                mx_phase_deg=float(np.degrees(np.angle(h_disp) - np.angle(h_torque)) % 360),
                motion_amplitude_m=float(amplitude),
                fx_peak_to_peak_per_amplitude_n_per_m=float(2 * abs(h_thrust) / amplitude),
                mx_peak_to_peak_per_amplitude_nm_per_m=float(2 * abs(h_torque) / amplitude),
            )
        rows.append(row)
    return pd.DataFrame(rows)
