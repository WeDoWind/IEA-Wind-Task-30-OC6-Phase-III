"""Convert the UNAFLOW record (Experiment 1) into Parquet, losing nothing.

Source: UNAFLOW, Fontanella, Bayati, Mikkelsen, Belloli and Zasso (2021),
https://doi.org/10.5281/zenodo.4740006 (CC BY 4.0). The campaign is the paper's
Experiment 1: the same 1:75 rotor in the Politecnico di Milano wind tunnel, steady and
surge tests at several wind speeds. The record holds, as text:

* ``FORCE.zip`` - tower-top and tower-base balance, 2 kHz, one file per test;
* ``WAKE.zip`` - three-component hot-wire velocities at 11 along-wind or 33 crosswind
  points, 2 kHz;
* ``PIV.zip`` - Tecplot planes, averaged and single-cycle;
* ``Blade_Airfoils.zip`` - AeroDyn airfoil files for the 20 blade stations;
* ``SD7032.zip`` - the measured 2D polars of the blade airfoil at 7 Reynolds numbers;
* ``CompanionDocument.pdf`` - the documentation, not converted.

Every test is converted, not only the ones OC6 Phase III uses. Column meanings follow the
companion document; the PIV columns are renamed to the OC6 convention (``Y`` -> ``z_m``,
``v`` -> ``w_ms``) so the measured and simulated planes share one schema.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..registry import REYNOLDS_NUMBERS
from .archive_wdh import PARQUET_OPTIONS, Result
from .experiments import match_load_case, parse_condition
from .textfiles import read_numeric

SOURCE_DOI = "10.5281/zenodo.4740006"

FORCE_HEADER = ["Time", "disp", "TTFx", "TTFy", "TTFz", "TTMx", "TTMy", "TTMz",
                "TBFx", "TBFy", "TBFz", "TBMx", "TBMy", "TBMz"]
FORCE_COLUMNS = ["time_s", "platform_displacement_m",
                 "ttfx_n", "ttfy_n", "ttfz_n", "ttmx_nm", "ttmy_nm", "ttmz_nm",
                 "tbfx_n", "tbfy_n", "tbfz_n", "tbmx_nm", "tbmy_nm", "tbmz_nm"]
WAKE_POINTS = {"aw": 11, "cw": 33}
#: The ten parameter rows of an AeroDyn v13 airfoil file, in file order. ``None`` marks the
#: three placeholder rows the files label "useless empty line"; they hold only zeros and
#: are checked, then skipped.
AERODYN_PARAMETERS = [
    "stall_angle_deg", None, None, None, "zero_lift_aoa_deg",
    "cn_slope", "cn_stall_positive", "cn_stall_negative", "aoa_min_cd_deg", "min_cd",
]


def _members(archive: zipfile.ZipFile, suffix: str = ".dat") -> list[str]:
    return sorted(
        name for name in archive.namelist()
        if name.endswith(suffix) and "__MACOSX" not in name
        and not Path(name).name.startswith("._")
    )


def _source_row(zip_name: str, name: str, raw: bytes, table: str, *, header=None,
                n_rows=None, n_columns=None, n_values=None, layout="standard",
                notes=()) -> dict:
    return {
        "file_name": name,
        "source_record": SOURCE_DOI,
        "source_path": f"{zip_name}/{name}",
        "source_received_utc": None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "encoding": None,
        "table": table,
        "participant": None,
        "model_code": None,
        "load_case": None,
        "header": header,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_values": n_values,
        "layout": layout,
        "notes": "; ".join(notes) or None,
    }


class _Sink:
    """Append frames to one Parquet file without holding the whole table in memory."""

    def __init__(self, path: Path):
        self.path, self.writer, self.rows = path, None, 0

    def append(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, **PARQUET_OPTIONS)
        self.writer.write_table(table.cast(self.writer.schema))
        self.rows += len(frame)

    def close(self) -> Path:
        self.writer.close()
        return self.path


def _test_record(measurement: str, test: str, source_file: str) -> dict:
    condition = parse_condition(test) or {}
    return {
        "experiment1_tests/measurement": measurement,
        "experiment1_tests/test": test,
        "experiment1_tests/motion": condition.get("motion"),
        "experiment1_tests/wind_speed_ms": condition.get("wind_speed_ms"),
        "experiment1_tests/amplitude_m": condition.get("amplitude_m"),
        "experiment1_tests/frequency_hz": condition.get("frequency_hz"),
        "experiment1_tests/oc6_load_case": match_load_case(condition) if condition else None,
        "experiment1_tests/source_file": source_file,
    }


def _forces(path: Path, out_dir: Path, result: Result, tests: list, quiet: bool) -> None:
    sink = _Sink(out_dir / "oc6.phase3.archive.experiment1.tower_loads.parquet")
    with zipfile.ZipFile(path) as archive:
        for name in _members(archive):
            raw = archive.read(name)
            data = read_numeric(raw)
            names = data.header[0].split() if data.header else []
            if names != FORCE_HEADER or data.n_columns != len(FORCE_COLUMNS):
                raise ValueError(f"{name}: unexpected columns {names}")
            if data.n_tokens != data.n_rows * data.n_columns:
                raise AssertionError(f"{name}: values do not fill the table")
            test = Path(name).stem
            frame = pd.DataFrame(data.values, columns=FORCE_COLUMNS)
            frame.insert(0, "test", test)
            frame["source_file"] = name
            sink.append(frame)
            tests.append(_test_record("force", test, name))
            result.source_files.append(_source_row(
                path.name, name, raw, "experiment1.tower_loads", header="\n".join(data.header),
                n_rows=data.n_rows, n_columns=data.n_columns, n_values=data.n_tokens,
                notes=data.notes,
            ))
    result.written.append(sink.close())
    if not quiet:
        print(f"  {sink.path.name:58s} {sink.rows:>11,} rows")


def _wake(path: Path, out_dir: Path, result: Result, tests: list, quiet: bool) -> None:
    sink = _Sink(out_dir / "oc6.phase3.archive.experiment1.hotwire_wake.parquet")
    with zipfile.ZipFile(path) as archive:
        for name in _members(archive):
            raw = archive.read(name)
            data = read_numeric(raw)
            stem = Path(name).stem
            traverse = stem[:2].lower()
            n_points = WAKE_POINTS[traverse]
            if data.header or data.n_columns != 2 + 3 * n_points:
                raise ValueError(f"{name}: {data.n_columns} columns for {n_points} points")
            if data.n_tokens != data.n_rows * data.n_columns:
                raise AssertionError(f"{name}: values do not fill the table")
            test = stem[3:]
            velocities = data.values[:, 2:].reshape(data.n_rows, n_points, 3)
            frame = pd.DataFrame({
                "test": stem,
                "traverse": traverse,
                "time_s": np.repeat(data.values[:, 0], n_points),
                "platform_displacement": np.repeat(data.values[:, 1], n_points),
                "point": np.tile(np.arange(1, n_points + 1, dtype=np.int16), data.n_rows),
                "u_ms": velocities[:, :, 0].reshape(-1),
                "v_ms": velocities[:, :, 1].reshape(-1),
                "w_ms": velocities[:, :, 2].reshape(-1),
                "source_file": name,
            })
            sink.append(frame)
            notes = list(data.notes)
            condition = parse_condition(test) or {}
            amplitude = condition.get("amplitude_m") or 0.0
            reach = np.nanmax(np.abs(data.values[:, 1]))
            if amplitude and np.isclose(reach, 1000 * amplitude, rtol=0.1):
                notes.append("the displacement column is in mm, although the companion "
                             "document gives m")
            tests.append(_test_record("wake", stem, name))
            result.source_files.append(_source_row(
                path.name, name, raw, "experiment1.hotwire_wake", n_rows=data.n_rows,
                n_columns=data.n_columns, n_values=data.n_tokens,
                layout=f"time, displacement, (u, v, w) x {n_points}", notes=notes,
            ))
    result.written.append(sink.close())
    if not quiet:
        print(f"  {sink.path.name:58s} {sink.rows:>11,} rows")


def _piv(path: Path, out_dir: Path, result: Result, tests: list, quiet: bool) -> None:
    sink = _Sink(out_dir / "oc6.phase3.archive.experiment1.piv_wake_plane.parquet")
    seen_tests = set()
    with zipfile.ZipFile(path) as archive:
        for name in _members(archive):
            raw = archive.read(name)
            data = read_numeric(raw)
            parts = Path(name).parts  # PIV / <test> / <AVERAGED|SINGLECYCLE> / file
            test, averaging, stem = parts[1], parts[2].lower(), Path(name).stem
            if data.n_columns != 5:
                raise ValueError(f"{name}: {data.n_columns} columns")
            if data.n_tokens != data.n_rows * data.n_columns:
                raise AssertionError(f"{name}: values do not fill the table")
            psi = re.search(r"psi(\d+)", stem)
            step = re.search(r"step(\d+)", stem)
            frame = pd.DataFrame(data.values, columns=["x_m", "z_m", "u_ms", "w_ms",
                                                       "vorticity_1_s"])
            frame.insert(0, "test", test)
            frame.insert(1, "averaging", averaging.replace("singlecycle", "single_cycle"))
            frame.insert(2, "azimuth_deg", float(psi.group(1)) if psi else np.nan)
            frame.insert(3, "step", pd.array([int(step.group(1)) if step else None]
                                             * len(frame), dtype="Int16"))
            frame["source_file"] = name
            sink.append(frame)
            if test not in seen_tests:
                seen_tests.add(test)
                tests.append(_test_record("piv", test, f"PIV/{test}/"))
            result.source_files.append(_source_row(
                path.name, name, raw, "experiment1.piv_wake_plane",
                header="\n".join(data.header), n_rows=data.n_rows,
                n_columns=data.n_columns, n_values=data.n_tokens,
                layout="tecplot", notes=["columns X, Y, u, v, vorticity stored as x_m, z_m, "
                                         "u_ms, w_ms, vorticity_1_s", *data.notes],
            ))
    result.written.append(sink.close())
    if not quiet:
        print(f"  {sink.path.name:58s} {sink.rows:>11,} rows")


def _blade_airfoils(path: Path, result: Result,
                    airfoil_polars: pd.DataFrame | None) -> list[dict]:
    """AeroDyn v13 airfoil files: unsteady parameters per Reynolds number, then polars.

    The parameters become inline records of ``croissant.json``. The polar tables are
    compared against ``airfoil_data.mat`` from the Wind Data Hub
    (``archive.simulations.airfoil_polars``); where they agree to the 4 decimals the
    AeroDyn files print, they are not stored twice.
    """
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in _members(archive):
            raw = archive.read(name)
            lines = raw.decode("latin-1").splitlines()
            airfoil_id = int(re.search(r"airfoilEL_(\d+)", name).group(1))
            n_tables = int(lines[2].split()[0])
            reynolds = [float(t) * 1e6 for t in lines[3].split()[:n_tables]]
            if not np.allclose(reynolds, REYNOLDS_NUMBERS):
                raise ValueError(f"{name}: Reynolds numbers {reynolds}")
            parameters = np.array([[float(t) for t in line.split()[:n_tables]]
                                   for line in lines[4:4 + len(AERODYN_PARAMETERS)]])
            table = np.array([[float(t) for t in line.split()]
                              for line in lines[4 + len(AERODYN_PARAMETERS):] if line.strip()
                              and line.split()[0].lstrip("+-").replace(".", "", 1).isdigit()])
            if table.shape[1] != 1 + 2 * n_tables:
                raise ValueError(f"{name}: polar table of width {table.shape[1]}")
            placeholders = [i for i, p in enumerate(AERODYN_PARAMETERS) if p is None]
            if np.any(parameters[placeholders] != 0):
                raise ValueError(f"{name}: a 'useless empty line' row is not all zeros")
            for r in range(n_tables):
                rows.append({
                    "airfoil_unsteady_parameters/airfoil_id": airfoil_id,
                    "airfoil_unsteady_parameters/reynolds_number": REYNOLDS_NUMBERS[r],
                    **{f"airfoil_unsteady_parameters/{p}": float(parameters[i, r])
                       for i, p in enumerate(AERODYN_PARAMETERS) if p is not None},
                })
            notes = [f"the {len(placeholders)} placeholder rows labelled 'useless empty "
                     f"line' hold only zeros and are not stored"]
            if airfoil_polars is not None:
                ref = airfoil_polars[airfoil_polars.airfoil_id == airfoil_id]
                expected = np.column_stack([
                    ref[ref.reynolds_number == REYNOLDS_NUMBERS[0]].aoa_deg.to_numpy(),
                    *[ref[ref.reynolds_number == re_].pipe(lambda d: np.column_stack(
                        [d.cl, d.cd])) for re_ in REYNOLDS_NUMBERS],
                ])
                if expected.shape == table.shape and np.allclose(
                    table, expected, atol=5.1e-5 * np.maximum(1, np.abs(expected))
                ):
                    notes.append("polar table equals airfoil_data.mat "
                                 "(archive.simulations.airfoil_polars) to the 4 decimals "
                                 "printed; not stored twice")
                else:
                    raise ValueError(f"{name}: polars differ from airfoil_data.mat; "
                                     "store them separately")
            result.source_files.append(_source_row(
                path.name, name, raw, "croissant:airfoil_unsteady_parameters",
                header="\n".join(lines[:4]), n_rows=table.shape[0],
                n_columns=table.shape[1], n_values=int(1 + 11 * n_tables + table.size),
                layout="aerodyn v13", notes=notes,
            ))
    return sorted(rows, key=lambda row: (row["airfoil_unsteady_parameters/airfoil_id"],
                                         row["airfoil_unsteady_parameters/reynolds_number"]))


def _sd7032(path: Path, result: Result) -> pd.DataFrame:
    blocks = []
    with zipfile.ZipFile(path) as archive:
        for name in _members(archive):
            raw = archive.read(name)
            data = read_numeric(raw)
            reynolds = float(re.search(r"Re(\d+)", name).group(1))
            if data.n_columns != 4 or data.n_tokens != data.values.size:
                raise ValueError(f"{name}: unexpected layout")
            frame = pd.DataFrame(data.values, columns=["aoa_deg", "cl", "cd", "cm"])
            frame.insert(0, "reynolds_number", reynolds)
            frame["source_file"] = name
            blocks.append(frame)
            result.source_files.append(_source_row(
                path.name, name, raw, "experiment1.sd7032_polars", n_rows=data.n_rows,
                n_columns=4, n_values=data.n_tokens, notes=data.notes,
            ))
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["reynolds_number", "aoa_deg"], ignore_index=True, kind="stable")


def hotwire_points() -> list[dict]:
    """Traverse positions in the wind-tunnel frame, mm (companion document, WAKE)."""
    rows = []
    for point in range(1, 12):
        rows.append({"traverse": "aw", "tests": "all", "point": point,
                     "x_mm": 2180.0 + 330.0 * (point - 1), "y_mm": 900.0, "z_mm": 2100.0})
    for point in range(1, 34):
        rows.append({"traverse": "cw", "tests": "all but Steady_V4", "point": point,
                     "x_mm": 5480.0, "y_mm": -1600.0 + 100.0 * (point - 1), "z_mm": 2085.0})
        rows.append({"traverse": "cw", "tests": "Steady_V4", "point": point,
                     "x_mm": 5480.0, "y_mm": -1470.0 + 100.0 * (point - 1), "z_mm": 2085.0})
    return [{f"experiment1_hotwire_points/{k}": v for k, v in row.items()} for row in rows]


def convert(files: dict[str, Path], out_dir: Path,
            airfoil_polars: pd.DataFrame | None = None, quiet: bool = False) -> Result:
    """Write the ``archive.experiment1`` tables and the UNAFLOW inline record sets.

    ``files`` maps each UNAFLOW file name (``FORCE.zip``, ``WAKE.zip``, ``PIV.zip``,
    ``Blade_Airfoils.zip``, ``SD7032.zip``, ``CompanionDocument.pdf``) to its local path.
    """
    result = Result()
    tests: list[dict] = []
    _forces(files["FORCE.zip"], out_dir, result, tests, quiet)
    _wake(files["WAKE.zip"], out_dir, result, tests, quiet)
    _piv(files["PIV.zip"], out_dir, result, tests, quiet)
    result.inline["airfoil_unsteady_parameters"] = _blade_airfoils(
        files["Blade_Airfoils.zip"], result, airfoil_polars
    )
    sd7032 = _sd7032(files["SD7032.zip"], result)
    path = out_dir / "oc6.phase3.archive.experiment1.sd7032_polars.parquet"
    sd7032.to_parquet(path, index=False, **PARQUET_OPTIONS)
    result.written.append(path)
    if not quiet:
        print(f"  {path.name:58s} {len(sd7032):>11,} rows")
    result.inline["experiment1_tests"] = tests
    result.inline["experiment1_hotwire_points"] = hotwire_points()
    raw = files["CompanionDocument.pdf"].read_bytes()
    result.source_files.append(_source_row(
        "", "CompanionDocument.pdf", raw, "not converted", layout="pdf",
        notes=["the UNAFLOW documentation; read it at the source record"],
    ) | {"source_path": "CompanionDocument.pdf"})
    return result
