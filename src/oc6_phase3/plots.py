"""The handful of drawing moves every figure in this project is made of.

:mod:`oc6_phase3.figures` is a set of short recipes written against these primitives, and
they are meant to be used directly too - building a figure in a notebook cell should be a
few lines, not a wall of ``go.Scatter`` calls.

Every primitive takes a **model frame**: a tidy :class:`~pandas.DataFrame` with at least

    label | model_code | wake_model | airfoil_model | <the columns you are plotting>

which is the schema of :func:`oc6_phase3.data.summary`, :func:`~oc6_phase3.data.forces`
and friends. Anything with those columns works, including your own results - see
:mod:`oc6_phase3.contribute`.

    import oc6_phase3 as oc6
    from oc6_phase3 import plots

    frame = oc6.data.summary("2.5")
    fig = plots.figure("Peak-to-peak thrust", "one point per model")
    plots.markers(fig, frame, x="model", y="fx_peak_to_peak_n", unit="N")
    plots.reference_line(fig, 4.34, name="Experiment 1")
    fig
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style
from .registry import NOT_IN_PAPER, PARTICIPANTS, WAKE_MODEL_ORDER

#: (label, model code) of the models absent from the paper's Table 3 - they took part in
#: the wake analysis (Cioni et al., 2023) - marked "*" wherever models are named.
NOT_IN_PAPER_LABELS = {(PARTICIPANTS[p].label, code) for p, code in NOT_IN_PAPER}
NOT_IN_PAPER_NOTE = "* not in the paper's Table 3"

__all__ = [
    "MODEL_COLUMNS",
    "figure",
    "lines",
    "lines_per_approach",
    "markers",
    "measured",
    "model_name",
    "quartiles_per_approach",
    "reference_line",
    "sort_models",
    "subplots",
    "subtitle",
]

#: The columns a model frame must carry for these primitives to work.
MODEL_COLUMNS = ["label", "model_code", "wake_model", "airfoil_model"]


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------

def model_name(frame: pd.DataFrame) -> pd.Series:
    """``"NREL2 m43"`` - the name a model is known by in every figure."""
    return frame.label.astype(str) + " " + frame.model_code.astype(str)


def not_in_paper(frame: pd.DataFrame) -> pd.Series:
    """True for rows of a model absent from the paper's Table 3."""
    pairs = zip(frame.label.astype(str), frame.model_code.astype(str), strict=True)
    return pd.Series([pair in NOT_IN_PAPER_LABELS for pair in pairs], index=frame.index)


def sort_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a ``model`` column and order rows by modelling approach, then by name.

    Models absent from the paper's Table 3 are named with a trailing "*".
    """
    frame = frame.copy()
    frame["model"] = model_name(frame) + np.where(not_in_paper(frame), "*", "")
    frame["wake_model"] = pd.Categorical(frame.wake_model, WAKE_MODEL_ORDER, ordered=True)
    return frame.sort_values(["wake_model", "model"], ignore_index=True)


def _hover(frame: pd.DataFrame, value_label: str, unit: str, x_label: str | None) -> dict:
    """Name the model, the value and the modelling approach in every tooltip."""
    columns = ["label", "model_code", "wake_model", "airfoil_model"]
    customdata = np.stack([frame[c].astype(str) for c in columns], axis=-1)
    x_part = f"{x_label} %{{x}}<br>" if x_label else ""
    unit_part = f" {unit}" if unit else ""
    return dict(
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]} %{customdata[1]}</b><br>"
            f"{x_part}{value_label} %{{y:.4g}}{unit_part}<br>"
            "%{customdata[3]}<extra>%{customdata[2]}</extra>"
        ),
    )


# --------------------------------------------------------------------------------------
# canvas
# --------------------------------------------------------------------------------------

def figure(title: str = "", subtitle_text: str = "", **layout) -> go.Figure:
    """An empty figure carrying the project style, a title and an optional caption."""
    fig = go.Figure()
    fig.update_layout(title=title, **layout)
    return subtitle(fig, subtitle_text) if subtitle_text else fig


def subplots(titles: tuple[str, ...], title: str = "", subtitle_text: str = "",
             **layout) -> go.Figure:
    """A one-row panel of subplots, e.g. ``("(a) Thrust", "(b) Torque")``."""
    fig = make_subplots(rows=1, cols=len(titles), horizontal_spacing=0.12,
                        subplot_titles=titles)
    fig.update_layout(title=title, **layout)
    return subtitle(fig, subtitle_text) if subtitle_text else fig


def subtitle(fig: go.Figure, text: str) -> go.Figure:
    """Caption under the title, in the native slot so it cannot hit subplot titles."""
    fig.update_layout(
        title=dict(text=fig.layout.title.text,
                   subtitle=dict(text=text, font=dict(size=11, color=style.INK["muted"])))
    )
    return fig


# --------------------------------------------------------------------------------------
# marks
# --------------------------------------------------------------------------------------

def markers(fig: go.Figure, frame: pd.DataFrame, x: str, y: str, *, unit: str = "",
            value_label: str = "value", x_label: str | None = None,
            open_when: str | None = None, span: tuple[str, str] | None = None,
            size: int = 12, legend: bool = True, row=None, col=None) -> go.Figure:
    """One marker per model: colour by wake model, shape by airfoil model.

    ``open_when`` names a boolean column; those rows are drawn as open markers, for values
    that are an estimate rather than a reported number. ``span`` names a (low, high) pair
    of columns, drawn as a bar through each marker: the range the value could as well
    have taken.
    """
    at = {} if row is None else dict(row=row, col=col)
    frame = frame.dropna(subset=[y])
    groups = ["wake_model"] + ([open_when] if open_when else [])
    for keys, block in frame.groupby(groups, observed=True):
        # pandas hands back a tuple even for a single grouper, so normalise before
        # unpacking - otherwise every marker reads as "open" and the legend disappears.
        keys = keys if isinstance(keys, tuple) else (keys,)
        approach = keys[0]
        is_open = bool(keys[1]) if open_when else False
        colour = style.colour(approach)
        bar = None if span is None else dict(
            type="data", symmetric=False, array=block[span[1]] - block[y],
            arrayminus=block[y] - block[span[0]], color=colour, thickness=1.5, width=5,
        )
        fig.add_trace(
            go.Scatter(
                x=block[x], y=block[y], mode="markers", name=str(approach), error_y=bar,
                legendgroup=str(approach), showlegend=legend and not is_open,
                marker=dict(
                    color=style.INK["surface"] if is_open else colour, size=size,
                    symbol=[style.symbol(a) for a in block.airfoil_model],
                    line=dict(width=2 if is_open else 1.5,
                              color=colour if is_open else style.INK["surface"]),
                ),
                **_hover(block, value_label, unit, x_label),
            ),
            **at,
        )
    return fig


def lines(fig: go.Figure, frame: pd.DataFrame, x: str, y: str, *, unit: str = "",
          value_label: str = "value", x_label: str | None = None, width: float = 1.6,
          close_loop: bool = False, row=None, col=None) -> go.Figure:
    """One line per model, coloured and dashed by modelling approach.

    Individual models are not put in the legend - there can be fifty of them. Call
    :func:`oc6_phase3.style.approach_legend` once to name the five approaches instead.
    """
    at = {} if row is None else dict(row=row, col=col)
    for (label, code, approach), block in frame.groupby(
        ["label", "model_code", "wake_model"], observed=True
    ):
        block = block.sort_values(x)
        if close_loop:
            block = pd.concat([block, block.head(1)])
        fig.add_trace(
            go.Scatter(
                x=block[x], y=block[y], mode="lines", name=f"{label} {code}",
                legendgroup=str(approach), showlegend=False,
                line=dict(color=style.colour(approach), width=width,
                          dash=style.dash(approach)),
                **_hover(block, value_label, unit, x_label),
            ),
            **at,
        )
    return fig


def measured(fig: go.Figure, x, y, *, name: str = "Experiment 1", unit: str = "",
             mode: str = "lines", legend: bool = True, row=None, col=None) -> go.Figure:
    """A measured series, in neutral ink - it is the reference, not another model."""
    at = {} if row is None else dict(row=row, col=col)
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode=mode, name=name, legendgroup=name, showlegend=legend,
            line=dict(width=2.5, **style.experiment_line(name)),
            marker=dict(color=style.experiment_line(name)["color"], size=7),
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.4g}} {unit}<extra></extra>",
        ),
        **at,
    )
    return fig


def reference_line(fig: go.Figure, value: float, *, name: str = "Experiment 1", x=None,
                   row=None, col=None) -> go.Figure:
    """A horizontal reference at ``value``.

    Drawn as a trace, not a shape, so that it takes part in the automatic y range: a shape
    at 35.9 N above data topping out at 35.5 N would otherwise sit on the frame.
    """
    if value is None or not np.isfinite(value):
        return fig
    if x is None:
        ends = [
            end for trace in fig.data
            if getattr(trace, "x", None) is not None and len(trace.x)
            and trace.x[0] is not None
            for end in (trace.x[0], trace.x[-1])
        ]
        if not ends:
            return fig
        x = [ends[0], ends[-1]]
    at = {} if row is None else dict(row=row, col=col)
    fig.add_trace(
        go.Scatter(
            x=list(x), y=[value, value], mode="lines", name=name, legendgroup=name,
            showlegend=not any(trace.name == name for trace in fig.data),
            line=dict(width=1.8, **style.experiment_line(name)),
            hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>",
        ),
        **at,
    )
    return fig


# --------------------------------------------------------------------------------------
# aggregates over the modelling approaches
# --------------------------------------------------------------------------------------

def quartiles_per_approach(fig: go.Figure, frame: pd.DataFrame, value: str, *,
                           unit: str = "", row=None, col=None) -> go.Figure:
    """Median, interquartile range and full range per modelling approach.

    The paper's Fig. 5 summary: the dot is the median, the thick bar spans the middle half
    of the models, the thin whiskers the rest, and the subscript on the category name is
    how many there were.
    """
    at = {} if row is None else dict(row=row, col=col)
    grouped = frame.dropna(subset=[value]).groupby("wake_model", observed=True)[value]
    stats = grouped.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).unstack()
    counts = grouped.size()
    for approach in stats.index:
        lowest, low, mid, high, highest = stats.loc[approach]
        name = f"{approach}<sub>{counts[approach]}</sub>"
        fig.add_trace(
            go.Scatter(
                x=[name] * 2, y=[lowest, highest], mode="lines+markers",
                showlegend=False, hoverinfo="skip",
                line=dict(color=style.colour(approach), width=1),
                marker=dict(symbol="line-ew", size=7,
                            line=dict(width=1, color=style.colour(approach))),
            ),
            **at,
        )
        fig.add_trace(
            go.Scatter(
                x=[name] * 3, y=[low, mid, high],
                mode="lines+markers", name=str(approach), showlegend=False,
                line=dict(color=style.colour(approach), width=4),
                marker=dict(color=style.colour(approach), size=[9, 13, 9],
                            symbol=["line-ew", "circle", "line-ew"],
                            line=dict(width=2, color=style.colour(approach))),
                hovertemplate=(f"{approach}, {counts[approach]} models<br>"
                               f"median %{{y:.4g}} {unit}<br>"
                               f"quartiles {low:.4g} - {high:.4g}<br>"
                               f"range {lowest:.4g} - {highest:.4g}<extra></extra>"),
            ),
            **at,
        )
    return fig


def lines_per_approach(fig: go.Figure, frame: pd.DataFrame, x: str, y: str, *,
                       unit: str = "", agg: str = "median", legend: bool = True,
                       row=None, col=None) -> go.Figure:
    """One line per modelling approach, aggregating the models within it."""
    at = {} if row is None else dict(row=row, col=col)
    for approach, block in frame.groupby("wake_model", observed=True):
        count = block.groupby(["label", "model_code"], observed=True).ngroups
        line = block.groupby(x, observed=True)[y].agg(agg).reset_index().sort_values(x)
        fig.add_trace(
            go.Scatter(
                x=line[x], y=line[y], mode="lines", name=f"{approach} ({count})",
                legendgroup=str(approach), showlegend=legend,
                line=dict(color=style.colour(approach), width=2.2,
                          dash=style.dash(approach)),
                hovertemplate=(f"{approach}<br>%{{x}}<br>%{{y:.4g}} {unit}"
                               f"<extra>{agg} of {count}</extra>"),
            ),
            **at,
        )
    return fig
