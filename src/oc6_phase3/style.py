"""One visual language for every figure.

The paper gives each of the 34 participants its own colour, which makes Figs. 3, 4, 13
and 14 unreadable as anything other than "a lot of models agree". The story those figures
actually tell is about the five **modelling approaches**, so that is what carries colour
here; individual models are separated by hover text and by the table underneath. Line
dash repeats the approach as a second, colour-independent channel, following the paper's
own convention (BEM dotted, DBEM/GDW dash-dot, FVW dashed, CFD solid).

Measurements are reference, not another series, so they are drawn in neutral ink.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
from plotly.io.base_renderers import HtmlRenderer

from .registry import WAKE_MODEL_ORDER

#: Categorical slots 1-5, validated for colour-vision deficiency on a light surface.
WAKE_MODEL_COLOURS: dict[str, str] = {
    "BEM": "#2a78d6",
    "DBEM": "#eb6834",
    "GDW": "#1baf7a",
    "FVW": "#eda100",
    "CFD": "#e87ba4",
}

WAKE_MODEL_DASH: dict[str, str] = {
    "BEM": "dot",
    "DBEM": "dashdot",
    "GDW": "dashdot",
    "FVW": "dash",
    "CFD": "solid",
}

AIRFOIL_MODEL_SYMBOLS: dict[str, str] = {
    "Disk": "square",
    "Static polars": "circle",
    "Unsteady airfoil aerodynamics": "diamond",
    "Blade resolved": "star",
}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#8a8880",
    "grid": "#e6e5e0",
    "surface": "#fcfcfb",
}

#: Measurements and reference lines.
EXPERIMENT_COLOURS: dict[str, str] = {
    "Experiment 1 (UNAFLOW)": INK["primary"],
    "Experiment 2": INK["secondary"],
}
EXPERIMENT_DASH: dict[str, str] = {
    "Experiment 1 (UNAFLOW)": "solid",
    "Experiment 2": "dash",
}

TEMPLATE_NAME = "oc6"


def experiment_line(name: str) -> dict:
    """Colour and dash of a measured series, by campaign: Experiment 1 solid black,
    Experiment 2 dashed grey, as in the paper. ``name`` may carry a suffix, e.g.
    "Experiment 2, steady wind"."""
    for experiment, colour in EXPERIMENT_COLOURS.items():
        if name.startswith(experiment.split(" (")[0]):
            return dict(color=colour, dash=EXPERIMENT_DASH[experiment])
    return dict(color=INK["primary"], dash="solid")


def template() -> go.layout.Template:
    """The project's Plotly template: recessive axes, thin marks, generous margins."""
    axis = dict(
        showgrid=True,
        gridcolor=INK["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=INK["grid"],
        ticks="outside",
        tickcolor=INK["grid"],
        ticklen=4,
        title_font=dict(size=13, color=INK["secondary"]),
        tickfont=dict(size=11, color=INK["secondary"]),
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="Inter, Helvetica Neue, Helvetica, Arial, sans-serif",
                size=12,
                color=INK["primary"],
            ),
            paper_bgcolor=INK["surface"],
            plot_bgcolor=INK["surface"],
            colorway=list(WAKE_MODEL_COLOURS.values()),
            title=dict(font=dict(size=15, color=INK["primary"]), x=0, xanchor="left"),
            margin=dict(l=70, r=30, t=105, b=70),
            hovermode="closest",
            hoverlabel=dict(font_size=11, namelength=-1),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=11, color=INK["secondary"]),
                itemsizing="constant",
            ),
            xaxis=axis,
            yaxis=axis,
        )
    )


#: Figures as HTML that loads plotly.js from its CDN - what any JupyterLab can draw.
HTML_RENDERER = "html_cdn"


def use() -> None:
    """Register the template and make it the default for this session.

    Also gives figures in a Jupyter kernel an HTML fallback. There Plotly sends only its
    own mime type, which JupyterLab draws only when plotly's extension is installed in the
    environment of the *server*, not the kernel; on RenkuLab the server has an environment
    of its own, so figures come back blank, without an error. With the fallback, a
    JupyterLab with the extension draws the figure as before and one without it draws the
    HTML. Other front ends (VS Code, Colab) keep the renderer Plotly chose for them.
    """
    pio.templates[TEMPLATE_NAME] = template()
    pio.templates.default = TEMPLATE_NAME
    pio.renderers[HTML_RENDERER] = HtmlRenderer(connected=True)
    if pio.renderers.default == "plotly_mimetype":
        pio.renderers.default = f"plotly_mimetype+{HTML_RENDERER}"


def colour(wake_model: str) -> str:
    return WAKE_MODEL_COLOURS.get(str(wake_model), INK["secondary"])


def dash(wake_model: str) -> str:
    return WAKE_MODEL_DASH.get(str(wake_model), "solid")


def symbol(airfoil_model: str) -> str:
    return AIRFOIL_MODEL_SYMBOLS.get(str(airfoil_model), "circle")


def legend_entries(present: list[str] | None = None) -> list[str]:
    """Wake models in fidelity order, restricted to those actually plotted."""
    if present is None:
        return list(WAKE_MODEL_ORDER)
    seen = {str(p) for p in present}
    return [name for name in WAKE_MODEL_ORDER if name in seen]


def approach_legend(figure: go.Figure, present: list[str] | None = None,
                    mode: str = "lines") -> go.Figure:
    """Add one legend entry per modelling approach.

    The individual model traces are drawn with ``showlegend=False`` so that the legend
    stays at five entries instead of fifty; this adds the proxy traces that name them.
    Model traces carry their approach as ``legendgroup``, so clicking an entry shows or
    hides every model of that approach.
    """
    for name in legend_entries(present):
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode=mode, name=name, legendgroup=name,
                line=dict(color=colour(name), width=2, dash=dash(name)),
                marker=dict(color=colour(name), size=9),
                hoverinfo="skip", showlegend=True,
            )
        )
    return figure
