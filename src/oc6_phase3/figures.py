"""The figures of Bergua et al. (2023), each one a short recipe.

Every function here is a handful of calls to :mod:`oc6_phase3.plots`, so that the
interesting part - which data, which encoding - is visible at a glance and you can write
your own variant in a notebook cell without opening this file.

**They all take their data as the first argument**, defaulting to the packaged tables::

    figures.steady_loads()                        # the packaged data
    figures.steady_loads(my_summary)              # a frame you filtered, or extended

That is the seam for comparing your own results against the campaign: build a frame with
the same columns (see :mod:`oc6_phase3.contribute`), concatenate, pass it in.

The two models absent from the paper's Table 3 - TUD m52 and EDF2 m42, which took part in
the wake analysis - are drawn too, and named with a "*".

Pass ``measured=False`` to any figure to drop the measured reference series. The
measurements of both campaigns are drawn where they exist: Experiment 1 (UNAFLOW) solid,
Experiment 2 dashed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import data, plots, style
from .registry import ROTOR_RADIUS_M, WAKE_MODEL_ORDER
from .wake import average_wake_deficit, inflection_recovery_point, piv_field

__all__ = [
    "amplitude_and_phase",
    "approach_medians_over_period",
    "controller_cases",
    "hotwire_alongwind",
    "hotwire_crosswind",
    "lift_hysteresis",
    "normalised_amplitude_vs_frequency",
    "piv_vorticity",
    "platform_motion",
    "steady_loads",
    "steady_quartiles",
    "thrust_over_period",
    "tip_vortex_trajectory",
    "wake_deficit",
    "wake_recovery_point",
]

#: Channel -> (column, unit, name), so a figure can serve thrust and torque alike.
QUANTITIES = {
    "thrust": dict(mean="fx_mean_n", series="fx_n", amplitude="fx_peak_to_peak_n",
                   phase="fx_phase_deg",
                   normalised="fx_peak_to_peak_per_amplitude_n_per_m",
                   unit="N", per_metre="N/m", name="Aerodynamic rotor thrust"),
    "torque": dict(mean="mx_mean_nm", series="mx_nm", amplitude="mx_peak_to_peak_nm",
                   phase="mx_phase_deg",
                   normalised="mx_peak_to_peak_per_amplitude_nm_per_m",
                   unit="N m", per_metre="N m/m", name="Aerodynamic rotor torque"),
}

#: Reported hub loads, and the blade-aerodynamics estimate that stands in for them.
BLADE_AERO_FALLBACK = {"fx_mean_n": "fx_from_blade_aero_n",
                       "mx_mean_nm": "mx_from_blade_aero_nm"}


# --------------------------------------------------------------------------------------
# resolving arguments
# --------------------------------------------------------------------------------------

def _use(frame, loader, load_case: str | None = None) -> pd.DataFrame:
    """The caller's frame if given, otherwise the packaged one, filtered to a load case."""
    frame = loader() if frame is None else frame
    if load_case is not None and "load_case" in frame.columns:
        frame = frame[frame.load_case == load_case]
    return frame.copy()


def _measured(measured, loader) -> pd.DataFrame | None:
    """``None``/default loads the packaged measurements; ``False`` suppresses them."""
    if measured is False:
        return None
    if measured is not None:
        return measured
    try:
        return loader()
    except FileNotFoundError:
        return None


def _by_experiment(measured: pd.DataFrame | None):
    """``(name, frame)`` per measurement campaign, named as in the legend."""
    if measured is None:
        return
    if "experiment" not in measured.columns:
        yield "Experiment 1", measured
        return
    for name, block in measured.groupby("experiment", observed=True, sort=True):
        yield str(name).split(" (")[0], block


def _measured_value(measured, load_case: str, column: str) -> float | None:
    if measured is None or column not in measured.columns:
        return None
    row = measured[measured.load_case == load_case]
    if row.empty or not np.isfinite(row[column].iloc[0]):
        return None
    return float(row[column].iloc[0])


def _with_blade_aero(frame: pd.DataFrame, column: str, enabled: bool) -> pd.DataFrame:
    """Fill ``column`` from the integrated blade loads, marking which rows were filled."""
    frame = frame.copy()
    frame["estimated"] = False
    substitute = BLADE_AERO_FALLBACK.get(column)
    if enabled and substitute in frame.columns:
        missing = frame[column].isna() & frame[substitute].notna()
        frame.loc[missing, column] = frame.loc[missing, substitute]
        frame.loc[missing, "estimated"] = True
    return frame.dropna(subset=[column])


MODEL_KEYS = ["label", "model_code", "wake_model", "airfoil_model"]


def _fold_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """One period per model: records covering several periods are averaged per phase.

    Every submission covers one motion period, except CENER m43 in LC 2.5, which covers
    eight; drawn against phase, its eight periods would otherwise retrace each other.
    """
    keys = [c for c in [*MODEL_KEYS, "load_case"] if c in frame.columns]
    numeric = [c for c in frame.select_dtypes("number").columns
               if c not in keys and c != "motion_phase_deg"]
    return (frame.dropna(subset=["motion_phase_deg"])
            .groupby([*keys, "motion_phase_deg"], observed=True)[numeric].mean()
            .reset_index())


def _on_phase_grid(frame: pd.DataFrame, columns: list[str],
                   step_deg: float = 5.0) -> pd.DataFrame:
    """Each model interpolated onto one phase grid, so medians across models compare
    like with like - the models report at different phases."""
    grid = np.arange(0.0, 360.0 + step_deg, step_deg)
    blocks = []
    for keys, block in _fold_periods(frame).groupby(MODEL_KEYS, observed=True):
        block = block.sort_values("motion_phase_deg")
        out = pd.DataFrame(dict(zip(MODEL_KEYS, keys, strict=True)), index=range(len(grid)))
        out["motion_phase_deg"] = grid
        for column in columns:
            out[column] = np.interp(grid, block.motion_phase_deg, block[column], period=360)
        blocks.append(out)
    frame = pd.concat(blocks, ignore_index=True)
    frame["wake_model"] = pd.Categorical(frame.wake_model, WAKE_MODEL_ORDER, ordered=True)
    return frame


def _model_note(frame: pd.DataFrame, text: str) -> str:
    """``text``, plus the "*" footnote when a model absent from the paper is drawn."""
    marked = "model" in frame.columns and frame.model.astype(str).str.endswith("*").any()
    return f"{text}; {plots.NOT_IN_PAPER_NOTE}" if marked else text


# --------------------------------------------------------------------------------------
# Steady wind
# --------------------------------------------------------------------------------------

def steady_loads(summary=None, quantity: str = "thrust", load_case: str = "1.1",
                 measured=None, use_blade_aero: bool = True) -> go.Figure:
    """Paper Figs. 3 and 4 - rotor thrust or torque per model in steady wind.

    The paper draws these as bars from zero, which puts every model's 35 N inside an
    identical column and hides the 3 % spread that is the subject of the figure. A point
    encoding carries no zero-baseline obligation, so the axis can cover the data's range.

    ``use_blade_aero`` includes models that reported sectional blade forces but no hub
    loads, using the integrated estimate, drawn as open markers.
    """
    q = QUANTITIES[quantity]
    frame = _with_blade_aero(_use(summary, data.summary, load_case), q["mean"],
                             use_blade_aero)
    frame = plots.sort_models(frame)
    measured = _measured(measured, data.experiment_summary)

    fig = plots.figure(f"{q['name']} in steady wind (Load Case {load_case})",
                       height=520, margin=dict(b=130),
                       legend=dict(orientation="h", y=-0.4, x=0))
    plots.markers(fig, frame, x="model", y=q["mean"], unit=q["unit"],
                  value_label=q["name"], open_when="estimated")
    for name, block in _by_experiment(measured):
        plots.reference_line(fig, _measured_value(block, load_case, q["mean"]), name=name)
    fig.update_layout(yaxis_title=f"{q['name']} [{q['unit']}]")
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=10))

    estimated = int(frame.estimated.sum())
    note = f"{len(frame) - estimated} models reported hub loads"
    if estimated:
        note += f", {estimated} more recovered from blade aerodynamics (open markers)"
    return plots.subtitle(fig, _model_note(frame, f"{note}; marker shape = airfoil model"))


def steady_quartiles(summary=None, load_case: str = "1.1", measured=None) -> go.Figure:
    """Paper Fig. 5 - median and quartiles of thrust and torque per modelling approach."""
    frame = _use(summary, data.summary, load_case)
    measured = _measured(measured, data.experiment_summary)

    fig = plots.subplots(("(a) Thrust", "(b) Torque"),
                         f"Thrust and torque by modelling approach, steady wind "
                         f"(LC {load_case})", height=470, showlegend=False)
    for col, quantity in enumerate(("thrust", "torque"), start=1):
        q = QUANTITIES[quantity]
        plots.quartiles_per_approach(fig, frame, q["mean"], unit=q["unit"], row=1, col=col)
        for name, block in _by_experiment(measured):
            plots.reference_line(fig, _measured_value(block, load_case, q["mean"]),
                                 name=name, row=1, col=col)
        fig.update_yaxes(title_text=f"[{q['unit']}]", row=1, col=col)
    return plots.subtitle(fig, "dot = median, thick bar = interquartile range, whiskers = "
                               "full range; subscript = number of models")


# --------------------------------------------------------------------------------------
# Wake in steady wind
# --------------------------------------------------------------------------------------

def _hotwire_profile(hotwire, load_case, direction, position):
    """Average each model's hot-wire record over time, one value per probe position."""
    frame = _use(hotwire, data.hotwire, load_case)
    frame = frame[frame.direction == direction]
    keys = ["label", "model_code", "wake_model", "airfoil_model", position]
    return frame.groupby(keys, observed=True).u_ms.mean().reset_index()


def _measured_hotwire(measured, direction, load_case="1.1"):
    measured = _measured(measured, data.experiment_hotwire)
    if measured is None:
        return None
    frame = measured[(measured.direction == direction) & (measured.load_case == load_case)]
    return frame.sort_values("probe_x_m" if direction == "aw" else "probe_y_m")


def hotwire_alongwind(hotwire=None, load_case: str = "1.1", measured=None) -> go.Figure:
    """Paper Fig. 7a - longitudinal velocity along the wake centreline traverse."""
    frame = _hotwire_profile(hotwire, load_case, "aw", "probe_x_m")
    measured = _measured_hotwire(measured, "aw", load_case)

    fig = plots.figure("Wake velocity along the wind direction, steady wind", height=470)
    plots.lines(fig, frame, x="probe_x_m", y="u_ms", unit="m/s", value_label="u",
                x_label="x", width=2)
    if measured is not None:
        plots.measured(fig, measured.probe_x_m, measured.u_ms, unit="m/s",
                       mode="lines+markers")
    style.approach_legend(fig, frame.wake_model.tolist())
    fig.update_layout(xaxis_title="x from hub [m]   (y = 0.9 m, z = hub height)",
                      yaxis_title="u-velocity [m/s]")
    models = frame.groupby(["label", "model_code"], observed=True).ngroups
    return plots.subtitle(fig, f"{models} models reporting along-wind hot-wire output")


def hotwire_crosswind(hotwire=None, load_case: str = "1.1", measured=None) -> go.Figure:
    """Paper Fig. 8a - longitudinal velocity across the wake, 2.3 diameters downwind."""
    frame = _hotwire_profile(hotwire, load_case, "cw", "probe_y_m")
    measured = _measured_hotwire(measured, "cw", load_case)

    fig = plots.figure("Wake velocity across the wind direction, steady wind", height=470)
    fig.add_vrect(x0=-ROTOR_RADIUS_M, x1=ROTOR_RADIUS_M, fillcolor=style.INK["grid"],
                  opacity=0.55, line_width=0, layer="below", annotation_text="rotor",
                  annotation_position="bottom",
                  annotation_font=dict(size=10, color=style.INK["muted"]))
    plots.lines(fig, frame, x="probe_y_m", y="u_ms", unit="m/s", value_label="u",
                x_label="y", width=2)
    if measured is not None:
        plots.measured(fig, measured.probe_y_m, measured.u_ms, unit="m/s")
    style.approach_legend(fig, frame.wake_model.tolist())
    fig.update_layout(xaxis_title="y from hub [m]   (x = 5.48 m, z = hub height)",
                      yaxis_title="u-velocity [m/s]")
    models = frame.groupby(["label", "model_code"], observed=True).ngroups
    return plots.subtitle(fig, f"{models} models reporting crosswind hot-wire output")


def wake_recovery_point(hotwire=None, load_case: str = "1.1", measured=None,
                        approaches: tuple[str, ...] = ("CFD",)) -> go.Figure:
    """Paper Fig. 7b - the inflection wake recovery point along the wind direction.

    The station where the along-wind velocity is a minimum and starts to recover. As in
    the paper, only CFD is shown: the FVW wakes carry no viscous diffusion, so where they
    recover depends on the wake length each participant chose. A minimum at the last
    station (5.48 m) means "at 5.48 m or beyond", drawn as an open marker.
    """
    frame = _hotwire_profile(hotwire, load_case, "aw", "probe_x_m")
    frame = frame[frame.wake_model.astype(str).isin(approaches)]
    last = float(frame.probe_x_m.max())
    points = (
        frame.groupby(MODEL_KEYS, observed=True)
        .apply(lambda b: inflection_recovery_point(b.probe_x_m, b.u_ms), include_groups=False)
        .rename("recovery_m").reset_index()
    )
    points = plots.sort_models(points)
    points["beyond_window"] = points.recovery_m >= last - 1e-9
    measured = _measured_hotwire(measured, "aw", load_case)

    fig = plots.figure(f"Inflection wake recovery point, steady wind (LC {load_case})",
                       height=470, margin=dict(b=110),
                       legend=dict(orientation="h", y=-0.3, x=0),
                       yaxis_title="Inflection wake recovery point x [m]")
    plots.markers(fig, points, x="model", y="recovery_m", unit="m",
                  value_label="recovery point", open_when="beyond_window", size=14)
    if measured is not None and not measured.empty:
        plots.reference_line(fig, inflection_recovery_point(measured.probe_x_m,
                                                            measured.u_ms))
    fig.update_yaxes(range=[2.0, last + 0.3])
    return plots.subtitle(fig, _model_note(points, f"open marker = minimum at the last "
                                                   f"station, i.e. at {last:.2f} m or beyond"))


def wake_deficit(hotwire=None, load_case: str = "1.1", measured=None) -> go.Figure:
    """Paper Fig. 8b - radius-weighted average wake deficit inside the rotor region."""
    free_stream = float(data.load_cases(indexed=True).loc[load_case, "wind_speed_ms"])
    frame = _hotwire_profile(hotwire, load_case, "cw", "probe_y_m")
    deficits = (
        frame.groupby(["label", "model_code", "wake_model", "airfoil_model"],
                      observed=True)
        .apply(lambda b: average_wake_deficit(b.probe_y_m, b.u_ms, free_stream),
               include_groups=False)
        .rename("deficit_ms").reset_index()
    )
    measured = _measured_hotwire(measured, "cw", load_case)

    fig = plots.figure("Average wake deficit within the rotor region, steady wind",
                       height=470, margin=dict(b=120),
                       legend=dict(orientation="h", y=-0.45, x=0))
    plots.markers(fig, plots.sort_models(deficits), x="model", y="deficit_ms",
                  unit="m/s", value_label="average deficit")
    if measured is not None:
        plots.reference_line(
            fig, average_wake_deficit(measured.probe_y_m, measured.u_ms, free_stream)
        )
    fig.update_layout(yaxis_title="Average wake deficit [m/s]")
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    return plots.subtitle(fig, "Eq. (4): radius-weighted mean of u - U0 over |y| <= R")


def piv_vorticity(piv=None, source: str = "experiment", azimuth_deg: float = 30.0,
                  participant: str = "nrel2", centres=None) -> go.Figure:
    """Paper Fig. 9b - vorticity magnitude in the PIV plane at one blade azimuth."""
    if piv is None and source == "experiment":
        piv, name = data.experiment_piv(azimuth_deg), "Experiment 1 (UNAFLOW)"
    elif piv is None:
        piv = data.piv("1.1", participant=participant, azimuth_deg=azimuth_deg)
        name = f"{participant} simulation"
    else:
        name = source
    x, z, field = piv_field(piv, "vorticity_1_s")

    fig = plots.figure(
        f"Vorticity in the PIV plane at {azimuth_deg:.0f} deg azimuth - {name}",
        # A velocity field is only readable at equal aspect, so the figure is sized to the
        # measurement window (0.75 m by 0.79 m) rather than stretched to the page width.
        autosize=False, width=640, height=680,
        legend=dict(orientation="h", y=-0.13, x=0),
    )
    fig.add_trace(go.Heatmap(
        x=x, y=z, z=np.abs(field), colorscale="Blues", zmin=0, zmax=250,
        colorbar=dict(title="|vorticity|<br>[1/s]", thickness=12, len=0.85),
        hovertemplate="x %{x:.3f} m<br>z %{y:.3f} m<br>|w| %{z:.0f} 1/s<extra></extra>",
    ))

    centres = data.tip_vortices() if centres is None else centres
    picked = centres.source.eq("Experiment 1 (UNAFLOW)") if source == "experiment" else (
        centres.source.eq("simulation") & centres.participant.eq(participant)
    )
    picked = centres[picked & centres.azimuth_deg.eq(azimuth_deg)]
    if not picked.empty:
        fig.add_trace(go.Scatter(
            x=picked.x_m, y=picked.z_m, mode="markers", name="tip vortex centre",
            marker=dict(symbol="circle-open", size=16,
                        line=dict(width=2.5, color=style.WAKE_MODEL_COLOURS["DBEM"])),
            customdata=picked.gamma1,
            hovertemplate="Gamma1 %{customdata:.2f}<br>x %{x:.3f} m<br>z %{y:.3f} m"
                          "<extra>tip vortex</extra>",
        ))
    fig.update_layout(xaxis_title="x from hub [m]", yaxis_title="z from hub [m]")
    fig.update_xaxes(range=[x.min(), x.max()], constrain="domain")
    fig.update_yaxes(range=[z.min(), z.max()], scaleanchor="x", scaleratio=1,
                     constrain="domain")
    return plots.subtitle(fig, "circles: centres found with Graftieaux's Gamma1")


def tip_vortex_trajectory(centres=None, bin_width_m: float = 0.1, min_points: int = 1,
                          min_centres: int = 10) -> go.Figure:
    """Paper Fig. 10 - averaged blade-tip vortex trajectory in the PIV plane.

    Per model, centres on the edge of the detector's search area are dropped and only the
    dominant rising track is kept (:func:`oc6_phase3.build.derive.vortex_track`) - some
    planes also show a second, lower structure. Models with fewer than ``min_centres``
    centres on their track are left out.
    """
    from .build.derive import average_trajectory

    centres = (data.tip_vortices() if centres is None else centres).copy()
    registry = data.models().set_index(["participant", "model_code"])
    simulated = centres.source == "simulation"
    labels = [registry.label.get((p, c), p) for p, c in
              zip(centres.participant.astype(str), centres.model_code.astype(str),
                  strict=True)]
    centres["label"] = labels
    centres["model"] = np.where(
        simulated, centres.label.astype(str) + " " + centres.model_code.astype(str),
        "Experiment 1",
    )
    centres.loc[simulated, "model"] += np.where(
        plots.not_in_paper(centres[simulated].assign(label=centres.label)), "*", "")
    paths = average_trajectory(centres, ("model",), bin_width_m=bin_width_m, track=True)
    totals = paths.groupby("model").n.sum()
    dropped = sorted(m for m in totals.index[totals < min_centres] if m != "Experiment 1")
    dropped += sorted(set(centres.model) - set(totals.index))
    paths = paths[paths.model.isin(totals.index[totals >= min_centres])
                  & (paths.n >= min_points)]
    approach_of = {
        model: {"4": "FVW", "5": "CFD"}.get(str(code)[1], "FVW")
        for model, code in centres.drop_duplicates("model").set_index("model").model_code.items()
        if not pd.isna(code)
    }

    fig = plots.figure("Averaged blade-tip vortex trajectory, steady wind", height=470,
                       xaxis_title="x from hub [m]", yaxis_title="z from hub [m]")
    for model, block in paths.groupby("model", observed=True):
        block = block.sort_values("x_m")
        if model == "Experiment 1":
            fig.add_trace(go.Scatter(
                x=[*block.x_m, *block.x_m[::-1]], y=[*block.z_max_m, *block.z_min_m[::-1]],
                mode="lines", fill="toself", fillcolor="rgba(11, 11, 11, 0.12)",
                line=dict(width=0),
                name="Experiment 1, range of centres", hoverinfo="skip",
            ))
            plots.measured(fig, block.x_m, block.z_m, unit="m", mode="lines+markers")
            continue
        approach = approach_of.get(model, "FVW")
        fig.add_trace(go.Scatter(
            x=block.x_m, y=block.z_m, mode="lines", name=model, legendgroup=approach,
            showlegend=False,
            line=dict(color=style.colour(approach), width=2, dash=style.dash(approach)),
            hovertemplate=f"<b>{model}</b><br>x %{{x:.2f}}<br>z %{{y:.3f}}<extra></extra>",
        ))
    style.approach_legend(fig, list(approach_of.values()))
    note = "dominant track per model, binned in x, median per bin; shaded: measured centres"
    if dropped:
        note += f"; too few centres detected for {', '.join(dropped)}"
    return plots.subtitle(fig, _model_note(paths, note))


def tip_vortex_uncertainty(uncertainty=None) -> go.Figure:
    """How far each trajectory of Fig. 10 moves when the vortex detector is retuned.

    The centres depend on choices the paper leaves open: the Gamma1 threshold, how strong
    the vorticity must be, whether the centre follows the vorticity peak. The marker is the
    straight-line fit of a model's track with the default detector, at x = 0.8 and
    1.2 m; the bar runs over the 12 settings of
    :data:`oc6_phase3.build.derive.DETECTOR_SETTINGS`. Open markers: the track was found in
    fewer than half of them.
    """
    frame = (data.tip_vortex_uncertainty() if uncertainty is None else uncertainty).copy()
    measured = frame[frame.source != "simulation"].set_index("x_m")
    simulated = plots.sort_models(data.with_models(frame[frame.source == "simulation"]))
    simulated["z_m"] = simulated.z_default_m.fillna(
        (simulated.z_min_m + simulated.z_max_m) / 2)
    simulated["rarely_found"] = simulated.settings_found < simulated.settings_tried / 2
    positions = sorted(simulated.x_m.unique())
    order = list(dict.fromkeys(simulated.model))

    fig = plots.subplots(tuple(f"x = {x:.1f} m from hub" for x in positions),
                         "Blade-tip vortex trajectory across detector settings, steady wind",
                         height=500, margin=dict(b=130),
                         legend=dict(orientation="h", y=-0.38, x=0))
    for col, x in enumerate(positions, start=1):
        plots.markers(fig, simulated[simulated.x_m == x], x="model", y="z_m", unit="m",
                      value_label="z", open_when="rarely_found", span=("z_min_m", "z_max_m"),
                      size=10, legend=col == 1, row=1, col=col)
        if x in measured.index:
            plots.reference_line(fig, float(measured.z_default_m.get(x)),
                                 x=[order[0], order[-1]], row=1, col=col)
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(matches="y")
    fig.update_yaxes(title_text="z from hub [m]", row=1, col=1)
    return plots.subtitle(fig, _model_note(
        simulated, "marker: default detector; bar: lowest to highest of 12 settings; "
                   "open: track found in fewer than 6"))


# --------------------------------------------------------------------------------------
# Unsteady wind
# --------------------------------------------------------------------------------------

def platform_motion(load_case: str = "2.5") -> go.Figure:
    """Paper Figs. 11 and 12 - the prescribed motion, and why the loads lag it by 90 deg.

    Analytic, from Eqs. (5)-(7); nothing is read from the submissions. It fixes the phase
    convention every other unsteady figure uses: phase 0 is the platform at neutral moving
    downwind, where the rotor apparent wind - and so the thrust - is at a minimum.
    """
    case = data.load_cases(indexed=True).loc[load_case]
    amplitude, omega = case.surge_equivalent_amplitude_m, 2 * np.pi * case.frequency_hz
    phase = np.linspace(0, 360, 721)
    radians = np.radians(phase)
    channels = [
        (amplitude * np.sin(radians), "m", "BEM"),
        (omega * amplitude * np.cos(radians), "m/s", "DBEM"),
        (case.wind_speed_ms - omega * amplitude * np.cos(radians), "m/s", "GDW"),
    ]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("(a) Hub displacement, positive downwind",
                                        "(b) Hub velocity", "(c) Rotor apparent wind"))
    for row, (values, unit, slot) in enumerate(channels, start=1):
        fig.add_trace(
            go.Scatter(x=phase, y=values, mode="lines", showlegend=False,
                       line=dict(color=style.WAKE_MODEL_COLOURS[slot], width=2.5),
                       hovertemplate=f"phase %{{x:.0f}} deg<br>%{{y:.3f}} {unit}"
                                     "<extra></extra>"),
            row=row, col=1,
        )
        fig.update_yaxes(title_text=f"[{unit}]", row=row, col=1)
    fig.add_vline(x=180, line=dict(color=style.INK["muted"], width=1, dash="dot"))
    fig.update_xaxes(title_text="Platform motion phase [deg]",
                     tickvals=[0, 90, 180, 270, 360], row=3, col=1)
    fig.update_layout(title=f"Platform motion and rotor apparent wind, LC {load_case}",
                      height=620)
    return plots.subtitle(
        fig, "thrust follows (c): minimum at phase 0, maximum at 180 - 90 deg behind (a)"
    )


def thrust_over_period(forces=None, load_case: str = "2.5", quantity: str = "thrust",
                       measured=None) -> go.Figure:
    """Paper Fig. 13 - rotor load through one platform motion period."""
    q = QUANTITIES[quantity]
    frame = _fold_periods(_use(forces, data.forces, load_case))
    measured = _measured(measured, data.experiment_forces)
    steady = _measured(None, data.experiment_summary)

    fig = plots.figure(f"{q['name']} during one motion period, Load Case {load_case}",
                       height=500, xaxis_title="Platform motion phase [deg]",
                       yaxis_title=f"{q['name']} [{q['unit']}]")
    plots.lines(fig, frame, x="motion_phase_deg", y=q["series"], unit=q["unit"],
                value_label=q["name"], x_label="phase")
    for name, block in _by_experiment(measured):
        trace = block[block.load_case == load_case].dropna(
            subset=["motion_phase_deg"]).sort_values("motion_phase_deg")
        if not trace.empty:
            plots.measured(fig, trace.motion_phase_deg, trace[q["series"]], name=name,
                           unit=q["unit"])
    for name, block in _by_experiment(steady):
        plots.reference_line(fig, _measured_value(block, "1.1", q["mean"]),
                             name=f"{name}, steady wind", x=[0, 360])
    style.approach_legend(fig, frame.wake_model.tolist())
    fig.update_xaxes(tickvals=[0, 90, 180, 270, 360], range=[0, 360])
    models = frame.groupby(["label", "model_code"], observed=True).ngroups
    return plots.subtitle(fig, f"{models} models in the archive for this load case")


def amplitude_and_phase(summary=None, load_case: str = "2.5", quantity: str = "thrust",
                        measured=None) -> go.Figure:
    """Paper Figs. 14 and 19 - peak-to-peak amplitude and phase shift per model.

    The amplitude is twice the FFT amplitude at the motion frequency; the phase comes from
    the same complex coefficient. A quasi-steady rotor sits at 90 deg.
    """
    q = QUANTITIES[quantity]
    frame = plots.sort_models(_use(summary, data.summary, load_case))
    measured = _measured(measured, data.experiment_summary)

    fig = plots.subplots(("(a) Peak-to-peak amplitude", "(b) Phase shift w.r.t. motion"),
                         f"Rotor {quantity} response, Load Case {load_case}",
                         height=560, margin=dict(b=150),
                         legend=dict(orientation="h", y=-0.42, x=0))
    for col, (key, unit, label) in enumerate(
        [("amplitude", q["unit"], f"peak-to-peak {quantity}"),
         ("phase", "deg", "phase shift")], start=1
    ):
        plots.markers(fig, frame, x="model", y=q[key], unit=unit, value_label=label,
                      size=11, legend=col == 1, row=1, col=col)
        for name, block in _by_experiment(measured):
            plots.reference_line(fig, _measured_value(block, load_case, q[key]),
                                 name=name, row=1, col=col)
    fig.add_hline(y=90, line=dict(color=style.INK["muted"], width=1, dash="dot"),
                  row=1, col=2)
    fig.update_yaxes(title_text=f"Peak-to-peak {quantity} [{q['unit']}]", row=1, col=1)
    fig.update_yaxes(title_text="Phase shift [deg]", row=1, col=2)
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    return plots.subtitle(fig, _model_note(frame, "marker shape = airfoil model; dotted "
                                                  "line = 90 deg quasi-steady response"))


def approach_medians_over_period(forces=None, load_case: str = "2.5",
                                 measured=None) -> go.Figure:
    """Paper Fig. 16 - median thrust and torque per approach through one period.

    Each model is first interpolated onto a 5 deg phase grid: the models report at
    different phases, and a median taken at the raw phases would mix a different subset
    of models at every point.
    """
    frame = _on_phase_grid(_use(forces, data.forces, load_case), ["fx_n", "mx_nm"])
    measured = _measured(measured, data.experiment_forces)

    fig = plots.subplots(("(a) Thrust", "(b) Torque"),
                         f"Median response per modelling approach, LC {load_case}",
                         height=470)
    for col, quantity in enumerate(("thrust", "torque"), start=1):
        q = QUANTITIES[quantity]
        plots.lines_per_approach(fig, frame, x="motion_phase_deg", y=q["series"],
                                 unit=q["unit"], legend=col == 1, row=1, col=col)
        for name, block in _by_experiment(measured):
            trace = block[block.load_case == load_case].dropna(
                subset=["motion_phase_deg"]).sort_values("motion_phase_deg")
            if not trace.empty:
                plots.measured(fig, trace.motion_phase_deg, trace[q["series"]], name=name,
                               unit=q["unit"], legend=col == 1, row=1, col=col)
        fig.update_yaxes(title_text=f"[{q['unit']}]", row=1, col=col)
        fig.update_xaxes(title_text="Platform motion phase [deg]",
                         tickvals=[0, 90, 180, 270, 360], row=1, col=col)
    return plots.subtitle(fig, "median across the models of each approach; count in "
                               "brackets")


def lift_hysteresis(blade_aero=None, load_case: str = "2.5", station: int = 7,
                    models: list[str] | None = None) -> go.Figure:
    """Paper Fig. 15 - lift coefficient against angle of attack over one motion period.

    Static polars give one lift coefficient per angle of attack, so the trace is a line
    and the response is symmetric about the 180 deg phase. Unsteady airfoil aerodynamics
    opens it into a loop, and that broken symmetry is what pushes the thrust phase past
    90 deg.

    By default two models are shown, one of each airfoil model, preferring a pair from the
    same organisation so the airfoil treatment is the only difference. Pass
    ``models=["NREL1 m12", "NREL2 m13"]`` to choose, or ``models=[]`` for all.
    """
    frame = _use(blade_aero, data.blade_aero, load_case)
    frame = frame[frame.station == station].copy()
    frame["model"] = plots.model_name(frame)
    models = _default_hysteresis_pair(frame) if models is None else models
    if models:
        frame = frame[frame.model.isin(models)]

    palette = [style.WAKE_MODEL_COLOURS[k] for k in ("BEM", "DBEM", "GDW", "FVW")]
    span = frame.span_pct.iloc[0] if not frame.empty else float("nan")
    fig = plots.figure(
        f"Lift coefficient vs angle of attack, station {station} ({span:.0f}% span), "
        f"LC {load_case}", height=500, xaxis_title="Angle of attack [deg]",
        yaxis_title="Lift coefficient [-]", legend=dict(orientation="h", y=-0.22, x=0),
    )
    for index, (model, block) in enumerate(frame.groupby("model", observed=True)):
        block = block.sort_values("motion_phase_deg")
        unsteady = "Unsteady" in str(block.airfoil_model.iloc[0])
        closed = pd.concat([block, block.head(1)])
        fig.add_trace(go.Scatter(
            x=closed.aoa_deg, y=closed.cl, mode="lines",
            name=f"{model} - {'unsteady' if unsteady else 'static polars'}",
            line=dict(width=2.2, color=palette[index % len(palette)],
                      dash="solid" if unsteady else "dot"),
            customdata=closed.motion_phase_deg,
            hovertemplate=f"<b>{model}</b><br>AoA %{{x:.2f}} deg<br>Cl %{{y:.3f}}"
                          "<br>phase %{customdata:.0f} deg<extra></extra>",
        ))
    _annotate_motion_phase(fig, frame)
    return plots.subtitle(fig, "solid = unsteady airfoil aerodynamics, "
                               "dotted = static polars")


#: The pair of paper Fig. 15: NREL's lifting line with static polars and with unsteady
#: airfoil aerodynamics, otherwise identical.
PAPER_HYSTERESIS_PAIR = ["NREL1 m12", "NREL2 m13"]


def _default_hysteresis_pair(frame: pd.DataFrame) -> list[str]:
    """The paper's pair; else a static-polar and an unsteady model, same organisation."""
    if set(PAPER_HYSTERESIS_PAIR) <= set(frame.model):
        return PAPER_HYSTERESIS_PAIR
    registry = data.models()
    organisation = (
        registry.assign(model=registry.label + " " + registry.model_code)
        .set_index("model").organization
    )
    available = frame[["model", "airfoil_model"]].drop_duplicates()
    available["organization"] = available.model.map(organisation)
    static = available[available.airfoil_model == "Static polars"]
    unsteady = available[available.airfoil_model.astype(str).str.contains("Unsteady")]
    if static.empty or unsteady.empty:
        return list(available.model.head(2))
    shared = sorted(set(static.organization) & set(unsteady.organization))
    if shared:
        return [static[static.organization == shared[0]].model.iloc[0],
                unsteady[unsteady.organization == shared[0]].model.iloc[0]]
    return [static.model.iloc[0], unsteady.model.iloc[0]]


def _annotate_motion_phase(fig: go.Figure, frame: pd.DataFrame) -> None:
    """Mark where the platform moves downwind and upwind on the hysteresis loop."""
    if frame.empty:
        return
    first = frame[frame.model == frame.model.iloc[0]]
    for phase, text in ((0.0, "platform moving downwind (phase 0 deg)"),
                        (180.0, "platform moving upwind (phase 180 deg)")):
        point = first.iloc[(first.motion_phase_deg - phase).abs().argmin()]
        fig.add_annotation(x=point.aoa_deg, y=point.cl, text=text, showarrow=True,
                           arrowhead=0, arrowcolor=style.INK["muted"],
                           ax=40 if phase else -40, ay=40 if phase else -40,
                           font=dict(size=10, color=style.INK["secondary"]))


# --------------------------------------------------------------------------------------
# Across load cases
# --------------------------------------------------------------------------------------

#: Figs. 17 and 18 use "load cases from 2.1 to 3.7 (excluding Load Cases 2.16 and 2.17)":
#: those two add rotor speed and blade pitch variation, so their load amplitude is no
#: longer set by the platform motion alone. They get their own figure.
CONTROLLER_CASES = ["2.16", "2.17"]


def normalised_amplitude_vs_frequency(summary=None, quantity: str = "thrust",
                                      measured=None) -> go.Figure:
    """Paper Figs. 17 and 18 - load variation per metre of motion, against frequency.

    The simulation points falling on a straight line through the origin is the paper's
    central result: the load variation is proportional to the change in apparent wind, so
    the aerodynamic response is quasi-steady.
    """
    q = QUANTITIES[quantity]
    cases = data.load_cases(indexed=True)
    frame = _use(summary, data.summary).dropna(subset=[q["normalised"]])
    frame = frame[~frame.load_case.isin(CONTROLLER_CASES)]
    per_case = (
        frame.groupby("load_case", observed=True)[[q["normalised"], q["phase"]]]
        .median().join(cases[["motion", "frequency_hz", "validation"]]).reset_index()
    )
    per_case["group"] = np.where(~per_case.validation, "Simulation: surge (verification)",
                                 "Simulation: " + per_case.motion.astype(str))

    fig = plots.subplots(("(a) Normalised amplitude", "(b) Phase shift"),
                         f"Normalised rotor {quantity} variation, load cases 2.1 to 3.7",
                         height=490, legend=dict(orientation="h", y=-0.32, x=0))
    symbols = {"Simulation: surge": "circle", "Simulation: pitch": "square",
               "Simulation: surge (verification)": "circle-open"}
    for name, block in per_case.groupby("group", observed=True):
        for col, column in enumerate([q["normalised"], q["phase"]], start=1):
            fig.add_trace(
                go.Scatter(
                    x=block.frequency_hz, y=block[column], mode="markers", name=name,
                    legendgroup=name, showlegend=col == 1, customdata=block.load_case,
                    marker=dict(color=style.WAKE_MODEL_COLOURS["DBEM"], size=11,
                                symbol=symbols.get(name, "circle"),
                                line=dict(width=2,
                                          color=style.WAKE_MODEL_COLOURS["DBEM"])),
                    hovertemplate="LC %{customdata}<br>%{x} Hz<br>%{y:.4g}"
                                  f"<extra>{name}</extra>",
                ),
                row=1, col=col,
            )
    _regression_through_origin(fig, per_case[per_case.validation], q["normalised"])
    _measured_points(fig, _measured(measured, data.experiment_summary), cases, q)

    fig.add_hline(y=90, line=dict(color=style.INK["muted"], width=1, dash="dot"),
                  row=1, col=2)
    fig.update_yaxes(title_text=f"Peak-to-peak {quantity} / amplitude [{q['per_metre']}]",
                     row=1, col=1)
    fig.update_yaxes(title_text="Phase shift [deg]", row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="Platform motion frequency [Hz]", row=1, col=col)
    return plots.subtitle(fig, "each point is the median of the models reporting that "
                               "load case")


def _regression_through_origin(fig: go.Figure, points: pd.DataFrame, column: str) -> None:
    if len(points) < 2:
        return
    slope = float(np.sum(points.frequency_hz * points[column])
                  / np.sum(points.frequency_hz ** 2))
    line = np.linspace(0, points.frequency_hz.max() * 1.05, 20)
    fig.add_trace(
        go.Scatter(x=line, y=slope * line, mode="lines", hoverinfo="skip",
                   name="Linear regression (simulation)",
                   line=dict(color=style.INK["muted"], width=1.5, dash="dash")),
        row=1, col=1,
    )


#: Marker per (campaign, motion), after the paper: Experiment 1 filled, Experiment 2 open.
MEASURED_SYMBOLS = {("Experiment 1", "surge"): "diamond",
                    ("Experiment 2", "surge"): "diamond-open",
                    ("Experiment 2", "pitch"): "square-open"}


def _measured_points(fig: go.Figure, measured, cases: pd.DataFrame, q: dict) -> None:
    """The measured points of Figs. 17 and 18, per campaign and motion.

    A campaign that carries its own normalised amplitude (Experiment 2, divided by the
    achieved motion) keeps it; otherwise the prescribed amplitude is used.
    """
    if measured is None or q["amplitude"] not in measured.columns:
        return
    for name, block in _by_experiment(measured):
        frame = block.merge(cases[["motion", "frequency_hz", "surge_equivalent_amplitude_m"]],
                            left_on=block.load_case.astype(str), right_index=True,
                            how="left")
        frame = frame[frame.frequency_hz > 0]
        normalised = frame[q["amplitude"]] / frame.surge_equivalent_amplitude_m
        if q["normalised"] in frame.columns:
            normalised = frame[q["normalised"]].fillna(normalised)
        frame = frame.assign(normalised=normalised)
        for motion, points in frame.groupby("motion", observed=True):
            label = f"{name}: {motion}"
            for col, values in enumerate([points.normalised, points[q["phase"]]], start=1):
                fig.add_trace(
                    go.Scatter(
                        x=points.frequency_hz, y=values, mode="markers", name=label,
                        legendgroup=label, showlegend=col == 1,
                        marker=dict(color=style.experiment_line(name)["color"], size=11,
                                    symbol=MEASURED_SYMBOLS.get((name, motion), "diamond"),
                                    line=dict(width=2,
                                              color=style.experiment_line(name)["color"])),
                        customdata=points.load_case.astype(str),
                        hovertemplate="LC %{customdata}<br>%{x} Hz<br>%{y:.4g}"
                                      f"<extra>{label}</extra>",
                    ),
                    row=1, col=col,
                )


def controller_cases(summary=None, quantity: str = "thrust") -> go.Figure:
    """Paper Fig. 20 - the effect of rotor speed and blade pitch variation.

    LC 2.12 is platform motion alone, LC 2.16 adds rotor speed variation and LC 2.17 adds
    blade pitch actuation. Dynamic inflow only matters once the controller moves.
    """
    q = QUANTITIES[quantity]
    labels = {"2.12": "2.12<br>motion only", "2.16": "2.16<br>+ rotor speed",
              "2.17": "2.17<br>+ blade pitch"}
    frame = _use(summary, data.summary)
    frame = frame[frame.load_case.isin(labels)].copy()
    groups = ["load_case", "wake_model", "airfoil_model"]
    medians = (frame.groupby(groups, observed=True)[[q["amplitude"], q["phase"]]]
               .agg(["median", "size"]))
    medians.columns = [f"{c}_{s}" for c, s in medians.columns]
    medians = medians.reset_index()
    cases = list(labels)
    approaches = [a for a in WAKE_MODEL_ORDER if a in set(medians.wake_model.astype(str))]
    # side by side within a load case: approaches apart, airfoil models of one approach
    # closer together
    medians["x"] = medians.load_case.astype(str).map(cases.index).astype(float) + [
        0.14 * (approaches.index(str(a)) - (len(approaches) - 1) / 2)
        for a in medians.wake_model
    ]
    medians["x"] += medians.groupby(["load_case", "wake_model"], observed=True).cumcount() * 0.04

    fig = plots.subplots((f"(a) Peak-to-peak {quantity}", "(b) Phase shift"),
                         "Rotor speed and blade pitch variation (verification load cases)",
                         height=490, legend=dict(orientation="h", y=-0.22, x=0))
    for col, key in enumerate(("amplitude", "phase"), start=1):
        unit = q["unit"] if col == 1 else "deg"
        for (approach, airfoil), block in medians.groupby(["wake_model", "airfoil_model"],
                                                          observed=True):
            name = f"{approach}, {str(airfoil).lower()}"
            fig.add_trace(
                go.Scatter(
                    x=block.x, y=block[f"{q[key]}_median"], mode="markers", name=name,
                    legendgroup=name, showlegend=col == 1,
                    marker=dict(color=style.colour(approach), size=12,
                                symbol=style.symbol(airfoil),
                                line=dict(width=1.5, color=style.INK["surface"])),
                    customdata=np.stack([block.load_case.astype(str),
                                         block[f"{q[key]}_size"]], axis=-1),
                    hovertemplate=(f"<b>{name}</b><br>LC %{{customdata[0]}}<br>"
                                   f"median %{{y:.4g}} {unit} of %{{customdata[1]}} models"
                                   "<extra></extra>"),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(tickvals=list(range(len(cases))),
                         ticktext=[labels[c] for c in cases],
                         range=[-0.5, len(cases) - 0.5], row=1, col=col)
    fig.add_hline(y=90, line=dict(color=style.INK["muted"], width=1, dash="dot"),
                  row=1, col=2)
    fig.update_yaxes(title_text=f"Peak-to-peak {quantity} [{q['unit']}]", row=1, col=1)
    fig.update_yaxes(title_text="Phase shift [deg]", row=1, col=2)
    return plots.subtitle(fig, "median per modelling approach and airfoil model; no "
                               "measurements exist for these cases")
