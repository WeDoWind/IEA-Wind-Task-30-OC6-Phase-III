"""OC6 Phase III (IEA Wind Task 30): analysis-ready data and figure reproductions.

    import oc6_phase3 as oc6

    oc6.data.summary("2.5")        # per-model thrust/torque amplitude and phase
    oc6.figures.thrust_over_period()   # Fig. 13 of Bergua et al. (2023)

Figures take their data as the first argument, so your own results go in by
concatenation - see :mod:`oc6_phase3.contribute`. The drawing primitives the figures are
built from are in :mod:`oc6_phase3.plots`, and are meant to be used directly.

See ``notebooks/oc6_phase3_figures.ipynb`` for the guided tour.
"""

from . import (
    contribute,
    data,
    figures,
    harmonics,
    paths,
    plots,
    registry,
    style,
    wake,
)

__all__ = [
    "contribute", "data", "figures", "harmonics", "paths", "plots", "registry",
    "style", "wake",
]
__version__ = "0.2.0"
