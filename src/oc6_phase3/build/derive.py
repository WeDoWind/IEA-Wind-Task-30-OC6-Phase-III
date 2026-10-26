"""Derived quantities: the per-case load summary and the blade-tip vortex trajectories.

These are the two things every summary figure in the paper needs and that would otherwise
have to be recomputed from the full time series (and, for the vortices, from ~900 MB of
PIV planes) on every notebook run.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from ..harmonics import _period_harmonic
from ..registry import HUB_RADIUS_M, ROTOR_RADIUS_M, load_cases
from ..wake import piv_field, tip_vortex_centres

SUMMARY_KEYS = ["participant", "label", "model_code", "wake_model", "airfoil_model",
                "load_case"]


def blade_aero_loads(blade_aero: pd.DataFrame) -> pd.DataFrame:
    """Rotor thrust and torque integrated from the sectional blade quantities.

    A consistency check on the reported hub loads (every case with blade aerodynamics also
    reports them), and a fallback where a model reports only the former. The
    blade-aerodynamics files give the normal and tangential force per unit length at nine
    stations from 18.9 % to 98.3 % span (18 for one file); integrating over the blade and
    multiplying by three blades gives::

        Fx = 3 * integral(Fn dr)      Mx = 3 * integral(Ft * r dr)

    The reported stations do not reach the root or the tip, so the integrand is extended
    linearly to zero at both ends. The median agreement with the reported loads is about
    1 % on thrust and torque; the few submissions that disagree strongly are listed in the
    repository README.
    """
    rows = []
    # dropna=False: several steady-wind files give their single row no time stamp.
    grouped = blade_aero.groupby(
        ["participant", "label", "model_code", "wake_model", "airfoil_model", "load_case",
         "time_s"],
        observed=True, sort=False, dropna=False,
    )
    for keys, block in grouped:
        block = block.sort_values("radius_m")
        radius = np.concatenate(([HUB_RADIUS_M], block.radius_m.to_numpy(), [ROTOR_RADIUS_M]))
        normal = np.concatenate(([0.0], block.fn_n_per_m.to_numpy(), [0.0]))
        tangential = np.concatenate(([0.0], block.ft_n_per_m.to_numpy(), [0.0]))
        rows.append(
            dict(
                zip(SUMMARY_KEYS + ["time_s"], keys, strict=True),
                fx_from_blade_aero_n=3 * np.trapezoid(normal, radius),
                mx_from_blade_aero_nm=3 * np.trapezoid(tangential * radius, radius),
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # One participant reports the tangential force with the opposite sign convention to
    # the definition document; flag rather than silently flip it.
    signs = frame.groupby(SUMMARY_KEYS, observed=True).mx_from_blade_aero_nm.transform("mean")
    frame["blade_aero_sign_corrected"] = signs < 0
    frame.loc[frame.blade_aero_sign_corrected, "mx_from_blade_aero_nm"] *= -1
    return frame


def _one_period(block: pd.DataFrame, case) -> pd.DataFrame:
    """Drop the wrapped endpoint of a single-period record.

    The submissions cover exactly one motion period with N+1 samples, the last repeating
    the first. Averaging all N+1 double-counts that sample and biases the mean by
    roughly amplitude/N - 0.06 N on Load Case 2.5, but 0.5 N on the large-amplitude
    verification cases, which is not negligible. The harmonics already discount it.
    """
    if case.motion == "none" or len(block) < 3:
        return block
    wrapped = np.isclose(
        (block.motion_phase_deg.iloc[-1] - block.motion_phase_deg.iloc[0]) % 360.0,
        0.0, atol=1e-6,
    )
    return block.iloc[:-1] if wrapped else block


def load_summary(forces: pd.DataFrame, blade_aero: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """One row per model and load case with the paper's validation metrics.

    * ``fx_mean_n`` / ``mx_mean_nm`` - mean aerodynamic rotor thrust and torque.
    * ``fx_peak_to_peak_n`` / ``mx_peak_to_peak_nm`` - twice the first-harmonic amplitude
      at the platform motion frequency (paper Sect. 5.3).
    * ``fx_phase_deg`` / ``mx_phase_deg`` - phase shift with respect to the platform
      motion; ~90 deg for a quasi-steady response.
    * ``*_per_amplitude`` - normalised by the platform motion amplitude in metres, so that
      surge and platform-pitch load cases are directly comparable (paper Sect. 5.4).

    Passing ``blade_aero`` adds ``fx_from_blade_aero_n`` and ``mx_from_blade_aero_nm``,
    integrated from the sectional forces by :func:`blade_aero_loads`, and rows for the
    models that reported blade aerodynamics but no hub loads.
    """
    cases = load_cases()
    rows = []
    grouped = forces.groupby(
        ["participant", "label", "model_code", "wake_model", "airfoil_model", "load_case"],
        observed=True, sort=False,
    )
    for keys, block in grouped:
        case = cases.loc[keys[-1]]
        block = _one_period(block.sort_values("time_s"), case)
        row = dict(zip(SUMMARY_KEYS, keys, strict=True))
        row["motion"] = case.motion
        row["frequency_hz"] = case.frequency_hz
        row["reduced_frequency"] = case.reduced_frequency
        row["motion_amplitude_m"] = case.surge_equivalent_amplitude_m
        row["fx_mean_n"] = float(np.nanmean(block.fx_n))
        row["mx_mean_nm"] = float(np.nanmean(block.mx_nm))
        if case.motion == "none" or len(block) < 4:
            row.update(
                fx_peak_to_peak_n=np.nan, mx_peak_to_peak_nm=np.nan,
                fx_phase_deg=np.nan, mx_phase_deg=np.nan,
            )
        else:
            # Submissions cover one motion period, except CENER m43 LC 2.5, which covers
            # eight: the motion frequency is then harmonic 8 of the record.
            step = float(np.median(np.diff(block.time_s.to_numpy())))
            periods = max(1, int(round(len(block) * step * case.frequency_hz)))
            reference = np.sin(np.radians(block.motion_phase_deg.to_numpy()))
            h_ref = _period_harmonic(reference, order=periods)
            for channel, column, unit in (("fx", "fx_n", "n"), ("mx", "mx_nm", "nm")):
                signal = block[column].to_numpy()
                if not np.isfinite(signal).all():
                    row[f"{channel}_peak_to_peak_{unit}"] = np.nan
                    row[f"{channel}_phase_deg"] = np.nan
                    continue
                h_signal = _period_harmonic(signal, order=periods)
                row[f"{channel}_peak_to_peak_{unit}"] = float(2 * abs(h_signal))
                row[f"{channel}_phase_deg"] = float(
                    np.degrees(np.angle(h_ref) - np.angle(h_signal)) % 360
                )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if blade_aero is not None and not blade_aero.empty:
        summary = _merge_blade_aero(summary, blade_aero_loads(blade_aero))
    amplitude = summary.motion_amplitude_m.replace(0.0, np.nan)
    summary["fx_peak_to_peak_per_amplitude_n_per_m"] = (
        summary.fx_peak_to_peak_n / amplitude
    )
    summary["mx_peak_to_peak_per_amplitude_nm_per_m"] = (
        summary.mx_peak_to_peak_nm / amplitude
    )
    return summary.sort_values(["load_case", "wake_model", "label"], ignore_index=True)


def _merge_blade_aero(summary: pd.DataFrame, integrated: pd.DataFrame) -> pd.DataFrame:
    """Add the integrated loads to the summary, including models with no hub loads."""
    if integrated.empty:
        return summary
    cases = load_cases()
    aggregated = (
        integrated.groupby(SUMMARY_KEYS, observed=True)
        .agg(fx_from_blade_aero_n=("fx_from_blade_aero_n", "mean"),
             mx_from_blade_aero_nm=("mx_from_blade_aero_nm", "mean"),
             blade_aero_sign_corrected=("blade_aero_sign_corrected", "first"))
        .reset_index()
    )
    merged = summary.merge(aggregated, on=SUMMARY_KEYS, how="outer")
    # Rows that came only from the blade-aerodynamics side have no load-case columns yet.
    missing = merged.motion.isna()
    for column in ("motion", "frequency_hz", "reduced_frequency"):
        merged.loc[missing, column] = merged.loc[missing, "load_case"].map(cases[column])
    merged.loc[missing, "motion_amplitude_m"] = merged.loc[missing, "load_case"].map(
        cases.surge_equivalent_amplitude_m
    )
    return merged


def tip_vortices(piv: pd.DataFrame, group_columns=("participant", "model_code"),
                 **kwargs) -> pd.DataFrame:
    """Blade-tip vortex centres in every PIV plane of a steady-wind load case.

    Centres are the local maxima of Graftieaux's ``Gamma1`` (paper Sect. 5.1.3), computed
    per group and azimuth angle. Pool them across azimuth angles and bin by ``x`` to get
    the averaged trajectory the paper plots in Fig. 10.
    """
    group_columns = list(group_columns)
    blocks = []
    for keys, frame in piv.groupby([*group_columns, "azimuth_deg"], observed=True):
        x, z, u = piv_field(frame, "u_ms")
        _, _, w = piv_field(frame, "w_ms")
        _, _, vorticity = piv_field(frame, "vorticity_1_s")
        if np.isnan(u).all():
            continue
        centres = tip_vortex_centres(x, z, np.nan_to_num(u), np.nan_to_num(w),
                                     vorticity=vorticity, **kwargs)
        if centres.empty:
            continue
        for column, value in zip([*group_columns, "azimuth_deg"], keys, strict=True):
            centres.insert(0, column, value)
        blocks.append(centres)
    if not blocks:
        return pd.DataFrame(columns=[*group_columns, "azimuth_deg", "x_m", "z_m", "gamma1"])
    return pd.concat(blocks, ignore_index=True)


#: The detector searches the PIV window minus a 0.05 m border (2 x 5 px of 5 mm). A centre
#: on the outermost ring of that area may be the clipped flank of a peak in the border,
#: not a vortex core.
SEARCH_EDGE_M = 0.05
PIV_STEP_M = 0.005


def off_search_edge(centres: pd.DataFrame) -> pd.Series:
    """True for centres clear of the edge of the detector's search area."""
    from ..registry import PIV_X_RANGE_M, PIV_Z_RANGE_M

    edge = SEARCH_EDGE_M + PIV_STEP_M / 2
    return ((centres.x_m > PIV_X_RANGE_M[0] + edge) & (centres.x_m < PIV_X_RANGE_M[1] - edge)
            & (centres.z_m < PIV_Z_RANGE_M[1] - edge))


def vortex_track(centres: pd.DataFrame, tolerance_m: float = 0.012,
                 max_slope: float = 0.25) -> pd.DataFrame:
    """The dominant tip-vortex track among the centres detected in one model's planes.

    Besides the tip vortices, the detector picks up a second structure in some planes - a
    cluster 0.1-0.15 m lower in several FVW models. The tip vortices of all azimuth planes
    lie on one slowly rising path (the wake expansion), so the track is the straight line,
    rising by at most ``max_slope``, that the most centres lie within ``tolerance_m`` of;
    its inliers are returned.
    """
    x, z = centres.x_m.to_numpy(), centres.z_m.to_numpy()
    best, best_count, best_error = np.zeros(len(x), dtype=bool), 0, np.inf
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] == x[i]:
                continue
            slope = (z[j] - z[i]) / (x[j] - x[i])
            if not 0.0 <= slope <= max_slope:
                continue
            residual = np.abs(z - (z[i] + slope * (x - x[i])))
            inliers = residual <= tolerance_m
            count, error = int(inliers.sum()), float(residual[inliers].sum())
            if count > best_count or (count == best_count and error < best_error):
                best, best_count, best_error = inliers, count, error
    return centres[best]


def average_trajectory(centres: pd.DataFrame, group_columns=("participant", "model_code"),
                       bin_width_m: float = 0.05, track: bool = False) -> pd.DataFrame:
    """Bin tip-vortex centres by ``x`` and summarise ``z`` within each bin.

    ``z_m`` is the median, which is what the trajectory should be read from: pooling
    across blade azimuths puts vortices of different ages in the same bin. ``z_min_m``
    and ``z_max_m`` bound the centres in the bin. ``n`` is the number of centres behind
    each point - bins with one detection are not worth plotting.

    With ``track``, centres on the edge of the detector's search area are dropped first
    and only the dominant track of each group is kept (:func:`vortex_track`), which is
    what the averaged trajectory of paper Fig. 10 shows.
    """
    group_columns = list(group_columns)
    frame = centres.copy()
    if track:
        frame = frame[off_search_edge(frame)]
        frame = pd.concat([vortex_track(block) for _, block
                           in frame.groupby(group_columns, observed=True)],
                          ignore_index=True)
    frame["x_bin"] = (frame.x_m / bin_width_m).round() * bin_width_m
    grouped = (
        frame.groupby([*group_columns, "x_bin"], observed=True)
        .agg(x_m=("x_m", "median"), z_m=("z_m", "median"), z_mean_m=("z_m", "mean"),
             z_min_m=("z_m", "min"), z_max_m=("z_m", "max"), n=("z_m", "size"))
        .reset_index()
    )
    return grouped.sort_values([*group_columns, "x_m"], ignore_index=True)


#: Detector settings a careful analyst might equally have chosen: the Gamma1 threshold,
#: the vorticity percentile a candidate must exceed, and whether the centre follows the
#: vorticity peak. The spread of a trajectory across them is its extraction uncertainty.
DETECTOR_SETTINGS = [
    dict(threshold=threshold, vorticity_percentile=percentile, refine_px=refine)
    for threshold in (0.8, 0.85, 0.9) for percentile in (90.0, 95.0) for refine in (0, 10)
]


def trajectory_sensitivity(piv: pd.DataFrame, group_columns=("participant", "model_code"),
                           settings=None, x_m=(0.8, 1.2), min_centres: int = 10
                           ) -> pd.DataFrame:
    """How far each tip-vortex trajectory moves across detector settings.

    For every setting the dominant track of each group (:func:`vortex_track`) is fitted
    with a straight line and evaluated at ``x_m``; the result gives, per group and
    position, the lowest and highest ``z`` over the settings, their spread, in how many
    settings the track was found at all, and ``z_default_m``, the height with the
    detector's default settings - those of the published tip-vortex centres.
    """
    settings = DETECTOR_SETTINGS if settings is None else settings
    group_columns = list(group_columns)
    defaults = {name: parameter.default for name, parameter
                in inspect.signature(tip_vortex_centres).parameters.items()}
    default = next((index for index, setting in enumerate(settings)
                    if all(defaults[name] == value for name, value in setting.items())), None)
    rows = []
    for index, setting in enumerate(settings):
        centres = tip_vortices(piv, group_columns, **setting)
        centres = centres[off_search_edge(centres)]
        for keys, block in centres.groupby(group_columns, observed=True):
            track = vortex_track(block)
            if len(track) < min_centres:
                continue
            slope, intercept = np.polyfit(track.x_m, track.z_m, 1)
            keys = keys if isinstance(keys, tuple) else (keys,)
            for x in x_m:
                rows.append({**dict(zip(group_columns, keys, strict=True)), "setting": index,
                             "x_m": x, "z_m": intercept + slope * x})
    frame = pd.DataFrame(rows)
    keys = [*group_columns, "x_m"]
    summary = (frame.groupby(keys, observed=True)
               .agg(settings_found=("setting", "nunique"), z_min_m=("z_m", "min"),
                    z_max_m=("z_m", "max"))
               .assign(spread_m=lambda d: d.z_max_m - d.z_min_m,
                       settings_tried=len(settings)))
    by_default = frame[frame.setting == default].set_index(keys).z_m.rename("z_default_m")
    return summary.join(by_default).reset_index()
