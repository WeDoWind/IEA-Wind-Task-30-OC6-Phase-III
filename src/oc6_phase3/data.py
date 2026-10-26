"""Read the OC6 Phase III tables.

The record has two layers, told apart by the file names:

* ``oc6.phase3.archive.*`` - the source archives (Wind Data Hub, UNAFLOW) in Parquet,
  every value as submitted;
* ``oc6.phase3.derived.*`` - what is calculated from them: load metrics, phase-averaged
  experiment loads, tip-vortex centres.

Small reference tables - models, load cases, blade geometry, the test lists - travel inline
in ``croissant.json``.

The archive tables store only what the sources hold. What can be looked up is added on
read instead: the participant label and modelling approach (from the model registry), the
canonical motion phase (from the time and the load case) and the station radius (from the
blade geometry). Every function returns a tidy :class:`pandas.DataFrame` and caches the
file it read, so calling them repeatedly in a notebook is free.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .harmonics import canonical_phase_deg
from .paths import data_dir
from .registry import LOAD_CASE_ORDER, WAKE_MODEL_ORDER

__all__ = [
    "airfoil_polars",
    "airfoil_unsteady_parameters",
    "blade_aero",
    "blade_geometry",
    "describe",
    "experiment1_tower_loads",
    "experiment2_tower_top_loads",
    "experiment_forces",
    "experiment_hotwire",
    "experiment_piv",
    "experiment_summary",
    "extra_hub_load_columns",
    "extra_openfast_channels",
    "forces",
    "hotwire",
    "inline",
    "load_cases",
    "models",
    "piv",
    "sd7032_polars",
    "source_files",
    "summary",
    "tip_vortex_uncertainty",
    "tip_vortices",
]

ARCHIVE = "oc6.phase3.archive"
DERIVED = "oc6.phase3.derived"


@cache
def _read(name: str, **filters) -> pd.DataFrame:
    path = Path(data_dir()) / name
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing from {path.parent}")
    predicate = [(column, "==", value) for column, value in filters.items()] or None
    frame = pd.read_parquet(path, filters=predicate)
    return _order_categories(frame)


def _order_categories(frame: pd.DataFrame) -> pd.DataFrame:
    if "wake_model" in frame:
        frame["wake_model"] = pd.Categorical(
            frame.wake_model, WAKE_MODEL_ORDER, ordered=True
        )
    if "load_case" in frame:
        present = [c for c in LOAD_CASE_ORDER if c in set(frame.load_case.astype(str))]
        frame["load_case"] = pd.Categorical(frame.load_case.astype(str), present,
                                            ordered=True)
    return frame


def _select(frame: pd.DataFrame, **equal) -> pd.DataFrame:
    for column, value in equal.items():
        if value is not None:
            frame = frame[frame[column] == value]
    return frame.copy()


# --------------------------------------------------------------------------------------
# Inline reference tables (croissant.json)
# --------------------------------------------------------------------------------------

@cache
def _croissant() -> dict:
    return json.loads((Path(data_dir()) / "croissant.json").read_text())


@cache
def inline(record_set: str) -> pd.DataFrame:
    """A record set embedded in ``croissant.json``, e.g. ``"models"`` or ``"load_cases"``."""
    for entry in _croissant()["recordSet"]:
        if entry["@id"] == record_set:
            frame = pd.DataFrame(entry["data"])
            return frame.rename(columns=lambda c: c.split("/", 1)[-1])
    raise KeyError(f"croissant.json has no inline record set {record_set!r}")


def models() -> pd.DataFrame:
    """One row per numerical model: participant, code, wake model, airfoil model."""
    return _order_categories(inline("models").copy())


def load_cases(indexed: bool = False) -> pd.DataFrame:
    """One row per load case: motion, frequency, amplitude, reduced frequency."""
    frame = inline("load_cases").copy()
    frame["load_case"] = frame.load_case.astype(str)
    return frame.set_index("load_case", drop=True) if indexed else frame


def blade_geometry() -> pd.DataFrame:
    """The 20 blade stations of definition document Table 3."""
    return inline("blade_geometry").copy()


def airfoil_polars(station: int | None = None,
                   reynolds_number: float | None = None) -> pd.DataFrame:
    """Lift and drag per blade station and Reynolds number (``airfoil_data.mat``)."""
    return _select(_read(f"{ARCHIVE}.simulations.airfoil_polars.parquet"), station=station,
                   reynolds_number=reynolds_number)


def airfoil_unsteady_parameters(airfoil_id: int | None = None) -> pd.DataFrame:
    """AeroDyn unsteady-aerodynamics parameters per airfoil and Reynolds number (UNAFLOW)."""
    return _select(inline("airfoil_unsteady_parameters"), airfoil_id=airfoil_id)


def sd7032_polars(reynolds_number: float | None = None) -> pd.DataFrame:
    """Measured 2D polars of the SD7032 blade airfoil (UNAFLOW)."""
    return _select(_read(f"{ARCHIVE}.experiment1.sd7032_polars.parquet"),
                   reynolds_number=reynolds_number)


# --------------------------------------------------------------------------------------
# What is added on read
# --------------------------------------------------------------------------------------

def with_models(frame: pd.DataFrame, registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``label``, ``wake_model`` and ``airfoil_model`` after ``model_code``."""
    registry = models() if registry is None else registry
    lookup = registry.set_index(["participant", "model_code"])[
        ["label", "wake_model", "airfoil_model"]
    ]
    keys = pd.MultiIndex.from_arrays([frame.participant.astype(str),
                                      frame.model_code.astype(str)])
    added = lookup.reindex(keys)
    frame = frame.copy()
    at = frame.columns.get_loc("model_code")
    frame.insert(at, "label", added.label.to_numpy())
    frame.insert(at + 2, "wake_model", added.wake_model.to_numpy())
    frame.insert(at + 3, "airfoil_model", added.airfoil_model.to_numpy())
    return _order_categories(frame)


def with_motion_phase(frame: pd.DataFrame, cases: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """Add the canonical ``motion_phase_deg`` after ``time_s``; see ``harmonics``."""
    cases = load_cases(indexed=True) if cases is None else cases
    phases = np.full(len(frame), np.nan)
    load_case = frame.load_case.astype(str).to_numpy()
    time = frame.time_s.to_numpy()
    for case in np.unique(load_case):
        rows = load_case == case
        phases[rows] = canonical_phase_deg(time[rows], cases.loc[case, "frequency_hz"],
                                           cases.loc[case, "motion"])
    frame = frame.copy()
    frame.insert(frame.columns.get_loc("time_s") + 1, "motion_phase_deg", phases)
    return frame


def with_station_geometry(frame: pd.DataFrame, geometry: pd.DataFrame | None = None
                          ) -> pd.DataFrame:
    """Add ``radius_m`` and ``span_pct`` after ``station``."""
    geometry = blade_geometry() if geometry is None else geometry
    lookup = geometry.set_index("station")
    frame = frame.copy()
    at = frame.columns.get_loc("station")
    frame.insert(at + 1, "radius_m", frame.station.map(lookup.radius_m).to_numpy())
    frame.insert(at + 2, "span_pct", frame.station.map(lookup.span_pct).to_numpy())
    return frame


# --------------------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------------------

@cache
def _simulation(table: str) -> pd.DataFrame:
    frame = with_motion_phase(with_models(_read(f"{ARCHIVE}.simulations.{table}.parquet")))
    if "station" in frame:
        frame = with_station_geometry(frame)
    return frame


def forces(load_case: str | None = None) -> pd.DataFrame:
    """Hub loads over one platform motion period, as each participant reported them."""
    return _select(_simulation("hub_loads"), load_case=load_case)


def blade_aero(load_case: str | None = None) -> pd.DataFrame:
    """Sectional blade quantities at the reported stations (nine for all but one file)."""
    return _select(_simulation("blade_sections"), load_case=load_case)


def hotwire(load_case: str | None = None, direction: str | None = None) -> pd.DataFrame:
    """Simulated hot-wire wake velocities along the two traverse lines."""
    return _select(_simulation("hotwire_wake"), load_case=load_case, direction=direction)


def piv(load_case: str = "1.1", participant: str | None = None,
        azimuth_deg: float | None = None) -> pd.DataFrame:
    """Simulated PIV plane fields. Filter before use - these are the largest tables."""
    tag = load_case.replace(".", "")
    frame = _read(f"{ARCHIVE}.simulations.piv_wake_plane.lc{tag}.parquet")
    return _select(frame, participant=participant, azimuth_deg=azimuth_deg)


def summary(load_case: str | None = None) -> pd.DataFrame:
    """Per-model, per-load-case mean, peak-to-peak amplitude and phase shift."""
    return _select(_read(f"{DERIVED}.simulations.load_metrics.parquet"), load_case=load_case)


def tip_vortex_uncertainty() -> pd.DataFrame:
    """How far each tip-vortex trajectory moves across detector settings (x = 0.8, 1.2 m)."""
    simulated = _read(f"{DERIVED}.simulations.tip_vortex_uncertainty.parquet").assign(
        source="simulation")
    measured = _read(f"{DERIVED}.experiment1.tip_vortex_uncertainty.parquet")
    return pd.concat([simulated, measured], ignore_index=True)


def tip_vortices() -> pd.DataFrame:
    """Blade-tip vortex centres in every steady-wind PIV plane, simulated and measured."""
    simulated = _read(f"{DERIVED}.simulations.tip_vortex_centres.parquet")
    measured = _read(f"{DERIVED}.experiment1.tip_vortex_centres.parquet")
    return pd.concat([simulated, measured], ignore_index=True)


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------

#: The two measurement campaigns, as named in the ``experiment`` column.
EXPERIMENTS = ("Experiment 1 (UNAFLOW)", "Experiment 2")


@cache
def _experiment2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Experiment 2 reduced on read; see ``build.experiments.experiment2_summary``."""
    from .build.experiments import experiment2_forces, experiment2_summary

    tower_top = _read(f"{ARCHIVE}.experiment2.tower_top_loads.parquet")
    return experiment2_forces(tower_top), experiment2_summary(tower_top)


def _both(experiment1: pd.DataFrame, experiment2: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([experiment1.assign(load_case=experiment1.load_case.astype(str)),
                       experiment2], ignore_index=True)
    frame["experiment"] = pd.Categorical(frame.experiment, EXPERIMENTS)
    return _order_categories(frame)


def experiment_forces(load_case: str | None = None,
                      experiment: str | None = None) -> pd.DataFrame:
    """Phase-averaged measured thrust and torque, both campaigns (``experiment`` column)."""
    frame = _both(_read(f"{DERIVED}.experiment1.hub_loads.parquet"), _experiment2()[0])
    return _select(frame, load_case=load_case, experiment=experiment)


def experiment_summary(experiment: str | None = None) -> pd.DataFrame:
    """Measured mean, peak-to-peak amplitude and phase shift, both campaigns.

    Experiment 2 also carries the normalised amplitudes, divided by the motion the robot
    achieved; for Experiment 1 the achieved and prescribed amplitudes agree.
    """
    frame = _both(_read(f"{DERIVED}.experiment1.load_metrics.parquet"), _experiment2()[1])
    return _select(frame, experiment=experiment)


def experiment_hotwire(direction: str | None = None) -> pd.DataFrame:
    """Measured hot-wire wake velocities, time-averaged (Experiment 1)."""
    return _select(_read(f"{DERIVED}.experiment1.hotwire_wake.parquet"), direction=direction)


@cache
def _experiment_piv(load_case: str) -> pd.DataFrame:
    from .build.experiments import piv as reduce_piv

    return reduce_piv(_read(f"{ARCHIVE}.experiment1.piv_wake_plane.parquet"), load_case)


def experiment_piv(azimuth_deg: float | None = None, load_case: str = "1.1") -> pd.DataFrame:
    """Measured averaged PIV planes for one load case, blanked (Experiment 1)."""
    return _select(_experiment_piv(load_case), azimuth_deg=azimuth_deg)


def experiment1_tower_loads(test: str | None = None) -> pd.DataFrame:
    """The UNAFLOW balance record, 2 kHz, one test or all (large: ~8.5 M rows)."""
    name = f"{ARCHIVE}.experiment1.tower_loads.parquet"
    return _read(name, test=test).copy() if test else _read(name).copy()


def experiment2_tower_top_loads(load_case: str | None = None) -> pd.DataFrame:
    """The Experiment 2 tower-top loads and platform displacement, 2 kHz."""
    return _select(_read(f"{ARCHIVE}.experiment2.tower_top_loads.parquet"),
                   load_case=load_case)


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

def source_files() -> pd.DataFrame:
    """One row per source file: checksum, header, layout, and what the conversion did."""
    return _read(f"{ARCHIVE}.source_files.parquet")


def extra_openfast_channels(load_case: str | None = None) -> pd.DataFrame:
    """The other OpenFAST channels of UNIFI m23's two complete-output blade files."""
    return _select(_read(f"{ARCHIVE}.simulations.extra_openfast_channels.parquet"),
                   load_case=load_case)


def extra_hub_load_columns() -> pd.DataFrame:
    """Columns a hub-load file holds beyond Table 10 (UNIFI m52 LC 2.7: two, unlabelled)."""
    return _read(f"{ARCHIVE}.simulations.extra_hub_load_columns.parquet")


def describe() -> pd.DataFrame:
    """Inventory of the files in the data directory, for orientation."""
    rows = []
    for path in sorted(Path(data_dir()).glob("oc6.phase3.*")):
        layer = path.name.split(".")[2]
        rows.append({"file": path.name, "layer": layer,
                     "size_mb": round(path.stat().st_size / 1e6, 2)})
    return pd.DataFrame(rows)
