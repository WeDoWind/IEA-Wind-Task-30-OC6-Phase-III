"""Put your own model results next to the OC6 Phase III campaign.

Every figure takes its data as a frame, so comparing your own run against the 54 models is
concatenation::

    import pandas as pd, oc6_phase3 as oc6

    mine = oc6.contribute.model_results(
        label="MyModel", model_code="m23", load_case="2.5",
        time_s=t, fx_n=thrust, mx_nm=torque,
    )
    summary = pd.concat([oc6.data.summary("2.5"), mine.summary])
    oc6.figures.amplitude_and_phase(summary)

``model_results`` puts your time series through exactly the same post-processing as the
archive - canonical motion phase, first-harmonic amplitude, phase shift - so the numbers
are comparable rather than merely plotted together.

What you need per load case: a time vector covering **one full platform motion period**
(longer records are phase-averaged down to one), the rotor thrust ``fx_n`` and, optionally,
the torque ``mx_nm``. Loads are at the hub, thrust positive downwind, as in Sect. 6.1 of
the definition document.

If your code writes the output files the definition document asks for (Sect. 6, Tables
10-13: forces, blade aerodynamics, hot wire, PIV), :func:`read_outputs` reads them with the
same reader that converted the campaign's 4,122 submissions::

    mine = oc6.contribute.read_outputs("my_results/", label="MyModel")
    forces = pd.concat([oc6.data.forces("2.5"), mine.forces])
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .harmonics import canonical_phase_deg, phase_average
from .registry import BLADE_GEOMETRY, PARTICIPANTS, decode_model_code, load_cases

__all__ = ["ModelResults", "check", "example_model_results", "model_results", "read_outputs"]

#: Columns a frame needs before the figures will accept it.
REQUIRED = {
    "forces": ["label", "model_code", "wake_model", "airfoil_model", "load_case",
               "time_s", "motion_phase_deg", "fx_n", "mx_nm"],
    "summary": ["label", "model_code", "wake_model", "airfoil_model", "load_case",
                "fx_mean_n", "mx_mean_nm", "fx_peak_to_peak_n", "fx_phase_deg"],
}


@dataclass
class ModelResults:
    """One model's results, in the schema the figures expect.

    Each table has the columns of the :mod:`oc6_phase3.data` reader of the same name;
    a table the model did not report is ``None``.
    """

    forces: pd.DataFrame | None
    summary: pd.DataFrame | None
    blade_aero: pd.DataFrame | None = None
    hotwire: pd.DataFrame | None = None
    piv: pd.DataFrame | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        if self.summary is None or len(self.summary) != 1:
            tables = [name for name in ("forces", "blade_aero", "hotwire", "piv")
                      if getattr(self, name) is not None]
            return f"<ModelResults: {', '.join(tables)}>"
        row = self.summary.iloc[0]
        return (f"<ModelResults {row.label} {row.model_code} LC {row.load_case}: "
                f"Fx mean {row.fx_mean_n:.3f} N, peak-to-peak "
                f"{row.fx_peak_to_peak_n:.3f} N, phase {row.fx_phase_deg:.1f} deg>")


def check(frame: pd.DataFrame, kind: str = "summary") -> pd.DataFrame:
    """Raise if ``frame`` is missing anything the figures need. Returns it unchanged."""
    missing = [c for c in REQUIRED[kind] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{kind} frame is missing {missing}. Build one with "
            f"oc6_phase3.contribute.model_results(), or see REQUIRED['{kind}'] for the "
            f"full column list."
        )
    return frame


def model_results(label: str, model_code: str, load_case: str, *, time_s,
                  fx_n, mx_nm=None, participant: str | None = None,
                  n_phase_bins: int = 72) -> ModelResults:
    """Turn one model's time series into ``forces`` and ``summary`` rows.

    Parameters
    ----------
    label:
        How the model should be named in figures, e.g. ``"MyModel"``.
    model_code:
        ``m<XY>`` - wake model (1 BEM, 2 DBEM, 3 GDW, 4 FVW, 5 CFD) and airfoil model
        (1 disk, 2 static polars, 3 unsteady airfoil aerodynamics, 4 blade resolved).
        This is what gives the model its colour and marker in every figure.
    load_case:
        One of the OC6 load cases, e.g. ``"1.1"``, ``"2.5"``, ``"3.5"``.
    time_s, fx_n, mx_nm:
        The record. Thrust and torque at the hub. For a steady load case a single value
        each is enough.

    A record longer than one motion period is phase-averaged down to one, so you can pass
    a raw simulation output without trimming it first.
    """
    case = load_cases().loc[load_case]
    wake_model, airfoil_model = decode_model_code(model_code)

    time_s = np.atleast_1d(np.asarray(time_s, dtype=float))
    fx_n = np.atleast_1d(np.asarray(fx_n, dtype=float))
    mx_nm = (np.full_like(fx_n, np.nan) if mx_nm is None
             else np.atleast_1d(np.asarray(mx_nm, dtype=float)))
    if not (len(time_s) == len(fx_n) == len(mx_nm)):
        raise ValueError("time_s, fx_n and mx_nm must be the same length")

    phase = canonical_phase_deg(time_s, case.frequency_hz, case.motion)
    if case.motion != "none" and len(time_s) > 3:
        periods = (time_s[-1] - time_s[0]) * case.frequency_hz
        if periods > 1.5:
            # Fold a multi-period record onto one period, as the measurements are.
            phase, fx_n, mx_nm = _fold(phase, fx_n, mx_nm, n_phase_bins)
            time_s = phase / 360.0 * float(case.period_s)

    forces = pd.DataFrame({
        "participant": participant or label.lower().replace(" ", "_"),
        "label": label,
        "model_code": model_code,
        "wake_model": wake_model,
        "airfoil_model": airfoil_model,
        "load_case": load_case,
        "time_s": time_s,
        "motion_phase_deg": phase,
        "fx_n": fx_n,
        "mx_nm": mx_nm,
    }).sort_values("time_s", ignore_index=True)

    from .build.derive import load_summary  # same maths as the packaged tables

    return ModelResults(forces=forces, summary=load_summary(forces))


#: The table each output kind lands in, named after the :mod:`oc6_phase3.data` reader
#: whose columns it matches.
OUTPUT_TABLES = {"hub_loads": "forces", "blade_sections": "blade_aero",
                 "hotwire_wake": "hotwire", "piv_wake_plane": "piv"}


def read_outputs(paths, label: str | None = None) -> ModelResults:
    """Read output files written as the definition document asks.

    The files are those of Sect. 6 of the definition document: the columns of Tables
    10-13, whitespace- or tab-separated, an optional header line, named as the document
    asks (``MYCODE_M23_LC25_Forces.txt``, ``_BA``, ``_HW_AW``, ``_HW_CW``,
    ``_PIV_PSI<deg>``, ``_PIV_STEP<n>``) or as the archive does
    (``oc6.phase3.mycode.m23.lc25.forces.txt``). Time runs as the document prescribes -
    surge starting at 0 m and moving upwind, platform pitch starting at 0 deg and
    increasing - and the motion phase is derived from it exactly as for the archive.

    ``paths`` is a folder, a file, or a list of files. Returns ``forces``, ``blade_aero``,
    ``hotwire`` and ``piv`` with the columns of the matching :mod:`oc6_phase3.data`
    readers, and ``summary``, the validation metrics of the hub loads (with the blade loads
    integrated, where both are given). Each is ready to be concatenated with the
    campaign's own table.
    """
    from .build import archive_wdh
    from .build.derive import load_summary
    from .build.textfiles import read_numeric

    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    files = [f for p in paths for f in (sorted(p.iterdir()) if p.is_dir() else [p])]
    frames: dict[str, list[pd.DataFrame]] = {}
    skipped = []
    for path in files:
        record = archive_wdh.parse_name(_archive_name(path.name))
        if record is None:
            skipped.append(path.name)
            continue
        table = archive_wdh.TABLES[record["kind"]]
        text = read_numeric(path.read_bytes())
        layout = (archive_wdh.blade_layout(record, text.header, text.n_columns)
                  if table == "blade_sections"
                  else archive_wdh.table_layout(table, text.header, text.n_columns))
        if layout.extras:
            raise ValueError(f"{path.name}: {text.n_columns} columns, more than the "
                             "definition document asks for")
        frame = archive_wdh._standard_frame(table, record, text, layout)
        wake_model, airfoil_model = decode_model_code(record["model_code"])
        participant = PARTICIPANTS.get(record["participant"])
        identity = {
            "participant": record["participant"],
            "label": label or (participant.label if participant
                               else record["participant"].upper()),
            "model_code": record["model_code"],
            "wake_model": wake_model,
            "airfoil_model": airfoil_model,
            "load_case": record["load_case"],
        }
        for at, (column, value) in enumerate(identity.items()):
            frame.insert(at, column, value)
        frame["source_file"] = path.name
        frames.setdefault(table, []).append(frame)
    if not frames:
        raise ValueError(f"no output files named like MYCODE_M23_LC25_Forces.txt in "
                         f"{', '.join(str(p) for p in paths)}")
    if skipped:
        warnings.warn(f"not named like an output file, skipped: {', '.join(skipped)}",
                      stacklevel=2)

    tables = {name: pd.concat(found, ignore_index=True) for name, found in frames.items()}
    for name in ("hub_loads", "blade_sections", "hotwire_wake"):
        if name in tables:
            tables[name] = data.with_motion_phase(tables[name], load_cases())
    if "blade_sections" in tables:
        tables["blade_sections"] = data.with_station_geometry(tables["blade_sections"],
                                                              BLADE_GEOMETRY)
    result = {OUTPUT_TABLES[name]: frame for name, frame in tables.items()}
    forces = result.pop("forces", None)
    summary = (None if forces is None
               else load_summary(forces, result.get("blade_aero")))
    return ModelResults(forces=forces, summary=summary, **result)


#: The definition document's file names, e.g. ``NREL_M12_LC11_Forces.txt``.
DOCUMENT_NAME = re.compile(
    r"^(?P<participant>[A-Za-z0-9-]+)_M(?P<model>\d\d)_LC(?P<case>\d+)_"
    r"(?P<kind>Forces|BA|HW_AW|HW_CW|PIV_PSI\d+|PIV_STEP\d+)\.txt$",
    re.IGNORECASE,
)


def _archive_name(name: str) -> str:
    """``NREL_M12_LC11_Forces.txt`` -> ``oc6.phase3.nrel.m12.lc11.forces.txt``."""
    match = DOCUMENT_NAME.match(name.replace(" ", ""))
    if match is None:
        return name
    participant = match["participant"].lower().replace("-", "_")
    return (f"oc6.phase3.{participant}.m{match['model']}.lc{match['case']}."
            f"{match['kind'].lower()}.txt")


def _fold(phase, fx_n, mx_nm, n_bins):
    centres, folded_fx = phase_average(phase, fx_n, n_bins)
    _, folded_mx = phase_average(phase, mx_nm, n_bins)
    keep = np.isfinite(folded_fx)
    return centres[keep], folded_fx[keep], folded_mx[keep]


def example_model_results(label: str = "EXAMPLE my-model", model_code: str = "m23",
                          load_case: str = "2.5") -> ModelResults:
    """A synthetic, clearly-labelled model, for trying the workflow without your own data.

    It is a quasi-steady response with a deliberately high mean thrust, so that it stands
    out from the campaign rather than blending in. **Not a real submission.**
    """
    case = load_cases().loc[load_case]
    time_s = np.linspace(0.0, float(case.period_s), 41)
    phase = np.radians(canonical_phase_deg(time_s, case.frequency_hz, case.motion))
    return model_results(
        label=label, model_code=model_code, load_case=load_case, time_s=time_s,
        fx_n=37.5 - 2.6 * np.cos(phase),
        mx_nm=3.4 - 0.6 * np.cos(phase),
    )


def campaign_plus(results: ModelResults, table: str = "summary",
                  load_case: str | None = None) -> pd.DataFrame:
    """The packaged table with your results appended - the usual one-liner."""
    packaged = getattr(data, table)(load_case) if load_case else getattr(data, table)()
    return pd.concat([packaged, getattr(results, table)], ignore_index=True)
