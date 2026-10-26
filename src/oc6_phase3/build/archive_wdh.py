"""Convert the Wind Data Hub archive of OC6 Phase III into Parquet, losing nothing.

Source: ``oc6.phase3.zip``, version 2 of https://doi.org/10.5281/zenodo.23572116 - a
byte-identical copy of Wind Data Hub https://doi.org/10.21947/1922900 as served on
22 October 2026. It holds:

* 4,122 whitespace-delimited text files named
  ``oc6.phase3.<participant>.m<XY>.lc<NM>.<kind>.txt``, the participants' results. Their
  column layouts are fixed by Sect. 6 of the definition document but their headers are
  not: 132 spellings of the same columns, files without a header, two in UTF-16;
* the Experiment 2 measurements, seven MATLAB files ``oc6.phase3.experiment.m00.*.mat``;
* the airfoil polars, ``oc6.phase3.nrel.air.foil.data.mat`` (the campaign's
  ``airfoil_data.mat``, renamed by the hub).

Every numeric value lands in exactly one place: a column of a standard table, or - when a
file carries more than the definition document asks for - a table of extra columns
(``extra_openfast_channels``, ``extra_hub_load_columns``). Two kinds of extra are copies
of what is stored elsewhere, and are checked, then dropped: VULC's displacement column
and a MATLAB ``ans`` variable. Each file's layout is resolved by an explicit rule, and
:func:`convert` checks per file that the columns it accounts for add up to the columns
the file has. A layout or an extra no rule covers stops the conversion rather than being
guessed at.

The conversion changes no value. What it does do is listed per file in the source-files
table (``layout`` and ``notes``): the corrected file names, the participant renamed by
the hub owner, one truncated number restored, and one inferred station numbering.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import BLADE_GEOMETRY, REYNOLDS_NUMBERS
from .textfiles import NumericText, read_numeric

SOURCE_DOI = "10.5281/zenodo.23572116"
WDH_DOI = "10.21947/1922900"

FILENAME = re.compile(
    r"^oc6\.phase3\.(?P<participant>[a-z0-9_]+)"
    r"\.m(?P<model>\d\d)"
    r"\.lc(?P<major>\d)(?P<minor>\d+)"
    r"\.(?P<kind>forces|ba|hw_aw|hw_cw|piv_psi\d+|piv_step\d+|piv_\d+)"
    r"\.(?:txt|dat)$",
    re.IGNORECASE,
)

#: File names that do not follow the convention, and their corrections. (ON names its PIV
#: planes ``piv_015`` instead of ``piv_psi15``; ``parse_name`` reads those as azimuths.)
FILENAME_FIXES = {
    "oc6.phase3.dut.m23.lc12.forces_.txt": "oc6.phase3.dut.m23.lc12.forces.txt",
}

#: Participant tokens filed under the wrong name. EDF submitted its LC 1.1 wake results
#: as ``edf`` although they belong to EDF2 (R. Bergua, 2026-10-22, who renamed them on the
#: Wind Data Hub); the first Zenodo mirror still carries the old name.
PARTICIPANT_FIXES = {"edf": "edf2"}

#: Files with a corrupted value, and the literal repair. EDF2's LC 2.5 forces file ends
#: mid-number: its last time stamp lost the exponent (the series steps by 0.025 s).
CONTENT_FIXES = {
    "oc6.phase3.edf2.m43.lc25.forces.txt": ("1.000001e ", "1.000001e+00 "),
}

TABLES = {
    "forces": "hub_loads",
    "ba": "blade_sections",
    "hotwire": "hotwire_wake",
    "piv": "piv_wake_plane",
}

#: Standard columns per table, in file order (definition document Tables 10-13).
HUB_LOAD_COLUMNS = ["time_s", "platform_displacement",
                    "fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm"]
BLADE_QUANTITIES = ["fn_n_per_m", "ft_n_per_m", "vrel_ms", "aoa_deg", "cl", "cd"]
HOTWIRE_COLUMNS = ["time_s", "platform_displacement",
                   "probe_x_m", "probe_y_m", "u_ms", "v_ms", "w_ms"]
PIV_COLUMNS = ["x_m", "z_m", "u_ms", "w_ms", "vorticity_1_s"]
STANDARD_WIDTH = {"hub_loads": 8, "hotwire_wake": 7, "piv_wake_plane": 5}

OUTPUT_STATIONS = list(BLADE_GEOMETRY.station[BLADE_GEOMETRY.output_station])
OPENFAST_BLADE_CHANNELS = ["Fx", "Fy", "VRel", "Alpha", "Cl", "Cd"]


def parse_name(name: str) -> dict | None:
    """Decompose a participant file name, or return ``None`` if it is not one."""
    name = FILENAME_FIXES.get(name, name)
    match = FILENAME.match(name)
    if match is None:
        return None
    parts = match.groupdict()
    kind = parts["kind"].lower()
    participant = parts["participant"].lower()
    record = {
        "participant": PARTICIPANT_FIXES.get(participant, participant),
        "model_code": "m" + parts["model"],
        "load_case": f"{parts['major']}.{parts['minor']}",
        "kind": kind.split("_")[0],
        "azimuth_deg": None,
        "step": None,
        "direction": None,
    }
    if kind.startswith("piv_psi"):
        record["kind"] = "piv"
        record["azimuth_deg"] = float(kind.removeprefix("piv_psi"))
    elif kind.startswith("piv_step"):
        record["kind"] = "piv"
        record["step"] = int(kind.removeprefix("piv_step"))
    elif kind.startswith("piv_"):
        record["kind"] = "piv"
        record["azimuth_deg"] = float(kind.removeprefix("piv_"))
    elif kind.startswith("hw_"):
        record["kind"] = "hotwire"
        record["direction"] = kind.removeprefix("hw_")
    return record


# --------------------------------------------------------------------------------------
# Column layouts
# --------------------------------------------------------------------------------------

@dataclass
class Layout:
    """Which file columns feed the standard columns, and which are left over."""

    standard: list[int]
    extras: dict[int, str | None] = field(default_factory=dict)
    stations: list[int] | None = None
    code: str = "standard"
    note: str = ""

    def check(self, width: int) -> None:
        used = self.standard + list(self.extras)
        if sorted(used) != list(range(width)):
            raise AssertionError(f"layout assigns columns {sorted(used)} of {width}")


def header_names(header: list[str], width: int) -> list[str | None]:
    """Column names from the last header line, when there is one name per column."""
    if not header:
        return [None] * width
    line = header[-1]
    names = [n.strip() for n in (line.split("\t") if "\t" in line else line.split())]
    names = [n for n in names if n]
    return names if len(names) == width else [None] * width


def blade_layout(record: dict, header: list[str], width: int) -> Layout:
    """Resolve a blade-aerodynamics file: time, then 6 quantities x 9 stations."""
    standard_width = 1 + len(BLADE_QUANTITIES) * len(OUTPUT_STATIONS)
    if width == standard_width:
        return Layout(list(range(width)), stations=OUTPUT_STATIONS)
    names = header_names(header, width)
    bare = [n.split("_[")[0] if n else n for n in names]
    if "B1N1Fx" in bare:
        columns = [bare.index("Time")]
        for channel in OPENFAST_BLADE_CHANNELS:
            columns += [bare.index(f"B1N{i}{channel}") for i in range(1, 10)]
        extras = {i: names[i] for i in range(width) if i not in columns}
        return Layout(columns, extras, OUTPUT_STATIONS, "openfast_channels",
                      "full OpenFAST output; blade channels B1N1-B1N9 Fx, Fy, VRel, Alpha, "
                      "Cl, Cd mapped by name, the other channels in "
                      "archive.simulations.extra_openfast_channels")
    if record["participant"] == "vulc" and width == standard_width + 1:
        return Layout([0, *range(2, width)], {1: None}, OUTPUT_STATIONS, "extra_column_2",
                      "an unlabelled second column precedes the standard columns; it "
                      "equals the platform displacement of the matching forces file "
                      "(checked to 1e-12) and is not stored twice")
    if record["participant"] == "polimi" and width == 1 + len(BLADE_QUANTITIES) * 18:
        return Layout(list(range(width)), stations=list(range(1, 19)),
                      code="18_stations",
                      note="6 quantities at 18 stations instead of 9; taken to be "
                           "definition-document Table 3 stations 1-18 (best fit against "
                           "the other CFD models at the 9 standard stations)")
    raise ValueError(f"no layout rule for a {width}-column blade file of "
                     f"{record['participant']} {record['model_code']}")


def table_layout(table: str, header: list[str], width: int) -> Layout:
    """Resolve a forces, hot-wire or PIV file: the standard columns, then any extras."""
    standard_width = STANDARD_WIDTH[table]
    if width < standard_width:
        raise ValueError(f"{width} columns where {table} needs {standard_width}")
    if width == standard_width:
        return Layout(list(range(width)))
    names = header_names(header, width)
    return Layout(list(range(standard_width)),
                  {i: names[i] for i in range(standard_width, width)}, code="extra_columns",
                  note=f"{width - standard_width} columns beyond the {standard_width} the "
                       "definition document asks for, in "
                       f"archive.simulations.extra_{table.removesuffix('s')}_columns")


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------

@dataclass
class Result:
    """What one archive contributed: written tables, bookkeeping rows, inline data."""

    written: list[Path] = field(default_factory=list)
    source_files: list[dict] = field(default_factory=list)
    #: Columns a file holds beyond its table's layout, by destination table.
    extras: dict[str, list[pd.DataFrame]] = field(default_factory=dict)
    #: Field descriptions for the extras tables, written into the Parquet schema.
    extra_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)
    inline: dict[str, list[dict]] = field(default_factory=dict)
    #: ``airfoil_data.mat`` as a table, for checking the UNAFLOW airfoil files against it.
    airfoil_polars: pd.DataFrame | None = None


def _extras(table: str, record: dict, source_file: str, layout: Layout,
            data: NumericText, result: Result,
            displacement: dict[tuple, np.ndarray]) -> None:
    """Route the columns a file holds beyond its layout.

    Two kinds carry information and are kept, each in a table of its own; one is a copy of
    a column stored elsewhere and is checked, then dropped. Anything else stops the
    conversion.
    """
    if not layout.extras:
        return
    key = (record["participant"], record["model_code"], record["load_case"])
    columns, descriptions = {}, {}
    if layout.code == "openfast_channels":
        columns["time_s"] = data.values[:, layout.standard[0]]
        for column, name in layout.extras.items():
            channel, _, unit = name.partition("_[")
            unit = unit.rstrip("]")
            columns[channel] = data.values[:, column]
            descriptions[channel] = (f"OpenFAST output channel {channel}, "
                                     f"{'dimensionless' if unit in ('', '-') else unit}.")
        target = "extra_openfast_channels"
    elif layout.code == "extra_columns":
        columns["time_s"] = data.values[:, 0]
        for column, name in layout.extras.items():
            label = f"column_{column + 1}"
            columns[label] = data.values[:, column]
            descriptions[label] = (f"Column {column + 1} of the file"
                                   + (f", headed {name!r}." if name else
                                      ", unlabelled; the file does not say what it holds."))
        target = f"extra_{table.removesuffix('s')}_columns"
    elif layout.code == "extra_column_2":
        reference = displacement.get(key)
        copy = data.values[:, 1]
        if reference is None or reference.shape != copy.shape or not np.allclose(
                copy, reference, rtol=0, atol=1e-12):
            raise ValueError(f"{source_file}: column 2 is not the platform displacement of "
                             "the matching forces file")
        return
    else:
        raise ValueError(f"{source_file}: no rule for the extra columns of a "
                         f"{layout.code!r} layout")
    frame = pd.DataFrame({
        **{column: record[column] for column in ("participant", "model_code", "load_case")},
        **columns,
        "source_file": source_file,
    })
    result.extras.setdefault(target, []).append(frame)
    result.extra_descriptions.setdefault(target, {}).update(descriptions)


def _identify(frame: pd.DataFrame, record: dict, source_file: str) -> pd.DataFrame:
    frame.insert(0, "participant", record["participant"])
    frame.insert(1, "model_code", record["model_code"])
    frame.insert(2, "load_case", record["load_case"])
    frame["source_file"] = source_file
    return frame


def _standard_frame(table: str, record: dict, data: NumericText, layout: Layout
                    ) -> pd.DataFrame:
    values = data.values[:, layout.standard]
    if table == "hub_loads":
        return pd.DataFrame(values, columns=HUB_LOAD_COLUMNS)
    if table == "hotwire_wake":
        frame = pd.DataFrame(values, columns=HOTWIRE_COLUMNS)
        frame.insert(0, "direction", record["direction"])
        return frame
    if table == "piv_wake_plane":
        frame = pd.DataFrame(values, columns=PIV_COLUMNS)
        frame.insert(0, "azimuth_deg", record["azimuth_deg"])
        frame.insert(1, "step", pd.array([record["step"]] * len(frame), dtype="Int16"))
        return frame
    # blade sections: time, then each quantity at every station -> long, one row per
    # (time, station)
    stations = layout.stations
    n_time, n_station = values.shape[0], len(stations)
    frame = pd.DataFrame({
        "time_s": np.repeat(values[:, 0], n_station),
        "station": np.tile(np.asarray(stations, dtype=np.int16), n_time),
    })
    for i, quantity in enumerate(BLADE_QUANTITIES):
        start = 1 + i * n_station
        frame[quantity] = values[:, start:start + n_station].reshape(-1)
    return frame


#: zstd at level 9: a third smaller than the default level for these float64 tables, at
#: no cost in read time; higher levels gain ~2 % for 4-7x the write time.
PARQUET_OPTIONS = {"compression": "zstd", "compression_level": 9}


def _write(frame: pd.DataFrame, path: Path, categories: tuple[str, ...]) -> Path:
    frame = frame.copy()
    for column in categories:
        if column in frame:
            frame[column] = frame[column].astype("category")
    frame.to_parquet(path, index=False, **PARQUET_OPTIONS)
    return path


def _text_members(archive: zipfile.ZipFile) -> Iterator[tuple[str, dict]]:
    for name in sorted(archive.namelist()):
        record = parse_name(Path(name).name)
        if record is not None:
            yield name, record


def convert(zip_path: Path, out_dir: Path, manifest: Path | None = None,
            quiet: bool = False) -> Result:
    """Write the ``archive.simulations`` and ``archive.experiment2`` tables."""
    result = Result()
    hub = None
    if manifest is not None and Path(manifest).is_file():
        hub = pd.read_csv(manifest).set_index("filename")
    categories = ("participant", "model_code", "load_case", "direction", "source_file")
    displacement: dict[tuple, np.ndarray] = {}

    with zipfile.ZipFile(zip_path) as archive:
        members = list(_text_members(archive))
        by_table: dict[str, list] = {}
        for name, record in members:
            by_table.setdefault(TABLES[record["kind"]], []).append((name, record))

        def read(name: str, record: dict, table: str) -> tuple[pd.DataFrame, NumericText]:
            raw = archive.read(name)
            source_file = Path(name).name
            data = read_numeric(raw, CONTENT_FIXES.get(source_file))
            layout = (blade_layout(record, data.header, data.n_columns)
                      if table == "blade_sections"
                      else table_layout(table, data.header, data.n_columns))
            layout.check(data.n_columns)
            if data.n_tokens != data.n_rows * data.n_columns:
                raise AssertionError(f"{source_file}: {data.n_tokens} values read, "
                                     f"{data.n_rows * data.n_columns} cells stored")
            if table == "hub_loads":
                displacement[(record["participant"], record["model_code"],
                              record["load_case"])] = data.values[:, 1]
            _extras(table, record, source_file, layout, data, result, displacement)
            notes = list(data.notes)
            if source_file in FILENAME_FIXES:
                notes.append(f"file name read as {FILENAME_FIXES[source_file]}")
            if source_file.split(".")[2] in PARTICIPANT_FIXES:
                notes.append(f"participant read as {record['participant']}")
            if re.search(r"\.piv_\d+\.", source_file):
                notes.append("piv_<deg> read as the blade azimuth")
            if layout.note:
                notes.append(layout.note)
            result.source_files.append(_source_row(
                source_file, raw, record, f"simulations.{table}", data, layout, notes, hub,
            ))
            frame = _identify(_standard_frame(table, record, data, layout),
                              record, source_file)
            return frame, data

        for table in ("hub_loads", "blade_sections", "hotwire_wake"):
            frames = [read(name, record, table)[0] for name, record in by_table[table]]
            path = out_dir / f"oc6.phase3.archive.simulations.{table}.parquet"
            result.written.append(_write(pd.concat(frames, ignore_index=True), path,
                                         categories))
            if not quiet:
                print(f"  {path.name:58s} {sum(len(f) for f in frames):>11,} rows")

        piv = by_table["piv_wake_plane"]
        for load_case in sorted({record["load_case"] for _, record in piv},
                                key=lambda lc: [int(p) for p in lc.split(".")]):
            frames = [read(name, record, "piv_wake_plane")[0]
                      for name, record in piv if record["load_case"] == load_case]
            tag = load_case.replace(".", "")
            path = out_dir / f"oc6.phase3.archive.simulations.piv_wake_plane.lc{tag}.parquet"
            result.written.append(_write(pd.concat(frames, ignore_index=True), path,
                                         categories))
            if not quiet:
                print(f"  {path.name:58s} {sum(len(f) for f in frames):>11,} rows")

        mats = [n for n in sorted(archive.namelist()) if n.endswith(".mat")]
        experiment2, tests = [], []
        for name in mats:
            source_file = Path(name).name
            raw = archive.read(name)
            if source_file.startswith("oc6.phase3.experiment."):
                frame, test = _experiment2(source_file, raw, result)
                experiment2.append(frame)
                tests.append(test)
            elif "air.foil.data" in source_file:
                result.airfoil_polars = _airfoil_polars(source_file, raw, result)
                path = out_dir / "oc6.phase3.archive.simulations.airfoil_polars.parquet"
                result.written.append(_write(result.airfoil_polars, path, ()))
                if not quiet:
                    print(f"  {path.name:58s} {len(result.airfoil_polars):>11,} rows")
            else:
                raise ValueError(f"no reader for {source_file}")
        path = out_dir / "oc6.phase3.archive.experiment2.tower_top_loads.parquet"
        result.written.append(_write(pd.concat(experiment2, ignore_index=True), path,
                                     ("load_case", "test", "source_file")))
        result.inline["experiment2_tests"] = [
            {f"experiment2_tests/{key}": value for key, value in test.items()}
            for test in tests
        ]
        if not quiet:
            print(f"  {path.name:58s} {sum(len(f) for f in experiment2):>11,} rows")

    for row in result.source_files:
        if hub is not None and row["file_name"] in hub.index:
            entry = hub.loc[row["file_name"]]
            if entry.sha256 != row["sha256"]:
                raise AssertionError(f"{row['file_name']}: checksum differs from the hub")
            row["source_path"] = entry.wdh_path
            row["source_received_utc"] = entry.wdh_received_utc
    return result


def _source_row(source_file: str, raw: bytes, record: dict | None, table: str,
                data: NumericText | None, layout: Layout | None, notes: list[str],
                hub: pd.DataFrame | None) -> dict:
    return {
        "file_name": source_file,
        "source_record": SOURCE_DOI,
        "source_path": None,
        "source_received_utc": None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "encoding": data.encoding if data is not None else None,
        "table": table,
        "participant": record["participant"] if record else None,
        "model_code": record["model_code"] if record else None,
        "load_case": record["load_case"] if record else None,
        "header": "\n".join(data.header) if data is not None else None,
        "n_rows": data.n_rows if data is not None else None,
        "n_columns": data.n_columns if data is not None else None,
        "n_values": data.n_tokens if data is not None else None,
        "layout": layout.code if layout is not None else None,
        "notes": "; ".join(notes) or None,
    }


# --------------------------------------------------------------------------------------
# MATLAB files
# --------------------------------------------------------------------------------------

def _load_mat(raw: bytes) -> dict:
    import scipy.io

    return {k: v for k, v in scipy.io.loadmat(io.BytesIO(raw), squeeze_me=True).items()
            if not k.startswith("__")}


def _numeric_count(variables: dict) -> int:
    return sum(np.size(v) for v in variables.values() if np.asarray(v).dtype.kind in "iuf")


def _experiment2(source_file: str, raw: bytes, result: Result) -> tuple[pd.DataFrame, dict]:
    """One Experiment 2 test: 2 kHz tower-top loads and platform displacement."""
    variables = _load_mat(raw)
    match = re.match(r"oc6\.phase3\.experiment\.m00\.lc(\d)(\d+)\.(.+)\.mat$", source_file)
    load_case, test = f"{match.group(1)}.{match.group(2)}", match.group(3)
    loads = variables["FTTxyz"]
    labels = [str(label) for label in variables["FTTinfo"]]
    if labels != ["Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nm]", "My [Nm]", "Mz [Nm]"]:
        raise ValueError(f"{source_file}: unexpected channels {labels}")
    frame = pd.DataFrame({
        "load_case": load_case,
        "test": test,
        "time_s": variables["t"],
        "platform_displacement": variables["disp"],
        **{column: loads[:, i] for i, column in enumerate(
            ["ttfx_n", "ttfy_n", "ttfz_n", "ttmx_nm", "ttmy_nm", "ttmz_nm"])},
        "source_file": source_file,
    })
    consumed = {"t", "disp", "FTTxyz", "fsamp", "SGamp", "SGfreq"}
    leftover = {k: v for k, v in variables.items()
                if k not in consumed and np.asarray(v).dtype.kind in "iuf"}
    notes = []
    for key, value in leftover.items():
        if key == "ans" and np.allclose(value, loads.mean(axis=0), rtol=1e-12, atol=1e-12):
            notes.append("MATLAB leftover variable 'ans' equals the column means of FTTxyz "
                         "(checked to 1e-12) and is not stored")
        else:
            raise ValueError(f"{source_file}: no rule for variable {key!r}")
    scalars = [key for key in ("fsamp", "SGamp", "SGfreq") if key in variables]
    # every value is stored, or checked to be derivable from what is (ans)
    accounted = frame.shape[0] * 8 + len(scalars) + sum(np.size(v) for v in leftover.values())
    expected = _numeric_count(variables)
    if accounted != expected:
        raise AssertionError(f"{source_file}: {expected} values read, {accounted} accounted for")
    test_row = {
        "load_case": load_case,
        "test": test,
        "sampling_rate_hz": float(variables["fsamp"]),
        "motion_amplitude": float(variables["SGamp"]) if "SGamp" in variables else 0.0,
        "motion_amplitude_unit": str(variables.get("SGampInfo", "")) or None,
        "motion_frequency_hz": float(variables["SGfreq"]) if "SGfreq" in variables else 0.0,
        "platform_displacement_unit": str(variables["dispInfo"]),
        "source_file": source_file,
    }
    result.source_files.append({
        **_source_row(source_file, raw, None, "experiment2.tower_top_loads", None, None,
                      notes, None),
        "load_case": load_case,
        "header": "; ".join(f"{k} = {variables[k]}" for k in sorted(variables)
                            if np.asarray(variables[k]).dtype.kind in "UO"),
        "n_rows": int(frame.shape[0]),
        "n_columns": 8,
        "n_values": int(expected),
        "layout": "matlab",
    })
    return frame, test_row


def _airfoil_polars(source_file: str, raw: bytes, result: Result) -> pd.DataFrame:
    """``airfoil_data`` (128 AoA x [AoA + 7 x (Cl, Cd)] x 20 stations), one row per
    station, Reynolds number and angle of attack (definition document Fig. 4)."""
    variables = _load_mat(raw)
    data = np.asarray(variables["airfoil_data"], dtype=float)
    n_aoa, _, n_station = data.shape
    if data.shape != (128, 1 + 2 * len(REYNOLDS_NUMBERS), len(BLADE_GEOMETRY)):
        raise ValueError(f"{source_file}: unexpected shape {data.shape}")
    blocks = []
    for s, station in enumerate(BLADE_GEOMETRY.itertuples()):
        for r, reynolds in enumerate(REYNOLDS_NUMBERS):
            blocks.append(pd.DataFrame({
                "station": np.int16(station.station),
                "airfoil_id": np.int16(station.airfoil_id),
                "reynolds_number": reynolds,
                "aoa_deg": data[:, 0, s],
                "cl": data[:, 1 + 2 * r, s],
                "cd": data[:, 2 + 2 * r, s],
            }))
    result.source_files.append({
        **_source_row(source_file, raw, None, "simulations.airfoil_polars", None, None,
                      ["the campaign's airfoil_data.mat, renamed by the Wind Data Hub; "
                       "definition document Sect. 3.2 and Fig. 4"], None),
        "n_rows": int(n_aoa),
        "n_columns": int(data.shape[1] * n_station),
        "n_values": int(data.size),
        "layout": "matlab 128 x 15 x 20",
    })
    return pd.concat(blocks, ignore_index=True)
