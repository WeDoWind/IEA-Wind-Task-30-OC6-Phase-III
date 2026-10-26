"""Build the OC6 Phase III record in two steps.

    python scripts/convert_archive.py --raw data/raw --out data   # once
    python scripts/build_dataset.py --data data                   # derived tables

``convert_archive`` turns the two source records - the Wind Data Hub archive
(``oc6.phase3.zip``, Zenodo version 2) and UNAFLOW - into the ``oc6.phase3.archive.*``
Parquet tables, losing no value. Run it once; it is how the published archive layer was
made, and it downloads whatever is missing (~2.3 GB).

``build_dataset`` writes the ``oc6.phase3.derived.*`` tables from the archive Parquet
alone, so recalculating them needs no source archive at all.

Both refresh ``croissant.json``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import registry
from ..data import with_models, with_motion_phase, with_station_geometry
from ..paths import (
    RAW_SIMULATION_ZIP,
    UNAFLOW_FILES,
    UNAFLOW_RECORD_ID,
    raw_simulation_zip_url,
    repo_root,
)
from . import archive_unaflow, archive_wdh, derive, experiments
from .archive_wdh import PARQUET_OPTIONS
from .croissant import write_croissant

SIMULATION_ZIP = RAW_SIMULATION_ZIP


def _download(url: str, target: Path, quiet: bool = False) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"downloading {target.name} from {url}")
    urllib.request.urlretrieve(url, target)
    return target


def resolve_raw(raw_dir: Path, download: bool, quiet: bool = False) -> dict[str, Path]:
    """Locate (or fetch) the source archives."""
    def unaflow_url(name: str) -> str:
        return f"https://zenodo.org/api/records/{UNAFLOW_RECORD_ID}/files/{name}/content"

    # URLs are resolved only when a download is needed: the simulation one asks Zenodo
    # for the newest version of the record that holds it.
    wanted = {
        SIMULATION_ZIP: raw_simulation_zip_url,
        **{name: (lambda name=name: unaflow_url(name)) for name in UNAFLOW_FILES},
    }
    found: dict[str, Path] = {}
    for name, url_of in wanted.items():
        # is_file() follows symlinks, so a link whose target has gone is not a match
        matches = ([p for p in sorted(raw_dir.rglob(name)) if p.is_file()]
                   if raw_dir.is_dir() else [])
        if matches:
            found[name] = matches[0]
        elif download:
            found[name] = _download(url_of(), raw_dir / name, quiet=quiet)
        else:
            raise FileNotFoundError(
                f"{name} not found under {raw_dir}; drop --no-download to fetch it from Zenodo"
            )
    return found


# --------------------------------------------------------------------------------------
# Step 1: source archives -> archive layer
# --------------------------------------------------------------------------------------

def convert(raw: dict[str, Path], out_dir: Path, manifest: Path | None = None,
            quiet: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print("Wind Data Hub archive (simulations, Experiment 2, airfoil polars)")
    wdh = archive_wdh.convert(raw[SIMULATION_ZIP], out_dir, manifest=manifest, quiet=quiet)
    if not quiet:
        print("UNAFLOW (Experiment 1)")
    unaflow = archive_unaflow.convert({name: raw[name] for name in UNAFLOW_FILES}, out_dir,
                                      wdh.airfoil_polars, quiet=quiet)

    source_files = pd.DataFrame(wdh.source_files + unaflow.source_files)
    for column in ("n_rows", "n_columns", "n_values"):
        source_files[column] = source_files[column].astype("Int64")
    written = wdh.written + unaflow.written
    for name, frames in {**wdh.extras, **unaflow.extras}.items():
        path = out_dir / f"oc6.phase3.archive.simulations.{name}.parquet"
        descriptions = {**wdh.extra_descriptions, **unaflow.extra_descriptions}[name]
        written.append(_write_described(pd.concat(frames, ignore_index=True), path,
                                        descriptions))
        if not quiet:
            print(f"  {path.name:64s} {sum(len(f) for f in frames):>11,} rows")
    path = out_dir / "oc6.phase3.archive.source_files.parquet"
    source_files.to_parquet(path, index=False, **PARQUET_OPTIONS)
    written.append(path)
    if not quiet:
        print(f"  {path.name:64s} {len(source_files):>11,} rows")
    _warn_stale(out_dir, "archive", written)
    written.append(write_croissant(out_dir, inline={**wdh.inline, **unaflow.inline}))
    return written


def _write_described(frame: pd.DataFrame, path: Path, descriptions: dict[str, str]) -> Path:
    """Write ``frame`` with per-column descriptions in the Parquet schema, where
    croissant.json picks them up - for tables whose columns are named by the source."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema = pa.schema(
        [f.with_metadata({"description": descriptions[f.name]}) if f.name in descriptions
         else f for f in table.schema],
        metadata=table.schema.metadata,
    )
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), path,
                   **PARQUET_OPTIONS)
    return path


def convert_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=repo_root() / "data" / "raw",
                        help="directory holding (or to receive) the source archives")
    parser.add_argument("--out", type=Path, default=repo_root() / "data",
                        help="directory to write the archive tables into")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="optional Wind Data Hub manifest (filename, sha256, wdh_path, "
                             "wdh_received_utc) to record each file's hub path and upload "
                             "time and to check its checksum")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of fetching missing source archives")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    raw = resolve_raw(args.raw, download=not args.no_download, quiet=args.quiet)
    written = convert(raw, args.out, manifest=args.manifest, quiet=args.quiet)
    _report(written, args.out, args.quiet)
    return 0


# --------------------------------------------------------------------------------------
# Step 2: archive layer -> derived layer
# --------------------------------------------------------------------------------------

def _archive(data_dir: Path, table: str, **filters) -> pd.DataFrame:
    predicate = [(column, "in", list(values)) for column, values in filters.items()] or None
    return pd.read_parquet(data_dir / f"oc6.phase3.archive.{table}.parquet",
                           filters=predicate)


def _tests(data_dir: Path, table: str) -> list[str]:
    path = data_dir / f"oc6.phase3.archive.{table}.parquet"
    tests = pq.read_table(path, columns=["test"]).column("test").unique().to_pylist()
    return list(experiments.oc6_tests(pd.Series(tests)))


def derive_all(data_dir: Path, out_dir: Path, quiet: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    models, cases = registry.models(), registry.load_cases()

    def emit(frame: pd.DataFrame, name: str) -> None:
        target = out_dir / f"oc6.phase3.derived.{name}.parquet"
        frame.to_parquet(target, index=False, **PARQUET_OPTIONS)
        written.append(target)
        if not quiet:
            print(f"  {target.name:64s} {len(frame):>11,} rows")

    if not quiet:
        print("simulations")
    forces = with_motion_phase(with_models(_archive(data_dir, "simulations.hub_loads"),
                                           models), cases)
    blade = with_station_geometry(
        with_motion_phase(with_models(_archive(data_dir, "simulations.blade_sections"),
                                      models), cases),
        registry.BLADE_GEOMETRY,
    )
    emit(derive.load_summary(forces, blade), "simulations.load_metrics")
    steady_piv = _archive(data_dir, "simulations.piv_wake_plane.lc11")
    simulated = derive.tip_vortices(steady_piv)
    simulated.insert(0, "source", "simulation")
    emit(simulated, "simulations.tip_vortex_centres")
    emit(derive.trajectory_sensitivity(steady_piv), "simulations.tip_vortex_uncertainty")

    if not quiet:
        print("experiment 1 (UNAFLOW)")
    tower_loads = _archive(data_dir, "experiment1.tower_loads",
                           test=_tests(data_dir, "experiment1.tower_loads"))
    emit(experiments.forces(tower_loads), "experiment1.hub_loads")
    emit(experiments.force_summary(tower_loads), "experiment1.load_metrics")
    del tower_loads
    wake = _archive(data_dir, "experiment1.hotwire_wake",
                    test=_tests(data_dir, "experiment1.hotwire_wake"))
    emit(experiments.hotwire(wake), "experiment1.hotwire_wake")
    del wake
    planes = _archive(data_dir, "experiment1.piv_wake_plane",
                      test=_tests(data_dir, "experiment1.piv_wake_plane"))
    measured_planes = experiments.piv(planes, "1.1")
    measured = derive.tip_vortices(measured_planes, group_columns=("experiment",))
    emit(measured.rename(columns={"experiment": "source"}), "experiment1.tip_vortex_centres")
    emit(derive.trajectory_sensitivity(measured_planes, group_columns=("experiment",))
         .rename(columns={"experiment": "source"}), "experiment1.tip_vortex_uncertainty")

    _warn_stale(out_dir, "derived", written)
    written.append(write_croissant(out_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=repo_root() / "data",
                        help="directory holding the archive tables")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory to write the derived tables into (default: --data)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or args.data
    written = derive_all(args.data, out, quiet=args.quiet)
    _report(written, out, args.quiet)
    return 0


def _warn_stale(out_dir: Path, layer: str, written: list[Path]) -> None:
    """Name the files of ``layer`` this run did not write: croissant.json would describe
    them, and they would be uploaded with the record."""
    stale = sorted(set(out_dir.glob(f"oc6.phase3.{layer}.*.parquet")) - set(written))
    for path in stale:
        print(f"WARNING: {path.name} was not written by this build; delete it if it is "
              "left over from an earlier layout")


def _report(written: list[Path], out_dir: Path, quiet: bool) -> None:
    if not quiet:
        total = sum(path.stat().st_size for path in written)
        print(f"\n{len(written)} files, {total / 1e6:.1f} MB, in {out_dir}")
