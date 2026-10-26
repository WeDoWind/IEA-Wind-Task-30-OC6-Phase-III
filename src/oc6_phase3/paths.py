"""Locate the OC6 Phase III data, wherever this code happens to be running.

Three environments are supported, tried in this order:

1. **Explicit** - ``$OC6_DATA_DIR`` points at the directory holding the derived tables.
2. **Renku** - the Zenodo record is mounted as a sibling folder of the repository
   (RenkuLab names it after the record title plus its DOI). Any directory under ``$HOME``
   or beside the repository that contains the marker table is accepted.
3. **Local** - ``<repo>/data/`` , populated either by :func:`download_derived_dataset` or
   by ``scripts/convert_archive.py`` and ``scripts/build_dataset.py``.

Two Zenodo records are involved. The **dataset** this code reads - the benchmark tables -
and the **Wind Data Hub mirror**, which holds the raw ``oc6.phase3.zip`` the archive tables
were converted from. Each is identified by one of its version records: the Zenodo API
lists a record's versions by version id, not by concept id.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

#: The benchmark dataset: archive and derived tables, croissant.json. The concept DOI
#: resolves to the newest version; it is the one to give a RenkuLab data connector.
ZENODO_RECORD_ID = "23593301"
ZENODO_DOI = f"10.5281/zenodo.{ZENODO_RECORD_ID}"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.23593300"
#: The Wind Data Hub mirror (version 1: 766 files; version 2: all 4,130).
MIRROR_RECORD_ID = "22746850"
MIRROR_CONCEPT_DOI = "10.5281/zenodo.22746849"

#: File that must exist for a directory to count as a valid data directory.
MARKER = "oc6.phase3.archive.simulations.hub_loads.parquet"


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


def _candidate_roots() -> Iterator[Path]:
    root = repo_root()
    yield root / "data"
    for base in (Path.home(), root.parent, Path("/home/jovyan/work"), Path.cwd()):
        if not base.is_dir():
            continue
        yield base
        try:
            yield from (p for p in base.iterdir() if p.is_dir())
        except PermissionError:  # pragma: no cover - unreadable mount
            continue


def find_data_dir() -> Path | None:
    """Return the first directory that holds the derived tables, or ``None``."""
    env = os.environ.get("OC6_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in _candidate_roots():
        if (candidate / MARKER).is_file():
            return candidate.resolve()
        # RenkuLab data connectors mount the record contents one level down.
        nested = candidate / "data"
        if (nested / MARKER).is_file():
            return nested.resolve()
    return None


def data_dir(download: bool = False) -> Path:
    """Directory holding the derived tables.

    Parameters
    ----------
    download:
        When the data cannot be found locally, fetch the newest published version from
        Zenodo into ``<repo>/data/`` instead of raising.
    """
    found = find_data_dir()
    if found is not None:
        return found
    if download:
        return download_derived_dataset()
    raise FileNotFoundError(
        "Could not locate the OC6 Phase III data.\n"
        "  * On RenkuLab: add the Zenodo data connector for "
        f"{ZENODO_CONCEPT_DOI} to the session.\n"
        "  * Locally: call oc6_phase3.paths.data_dir(download=True), "
        "or set $OC6_DATA_DIR."
    )


def _newest_record_with(filename: str, record_id: str = ZENODO_RECORD_ID) -> dict:
    """Newest published version of a Zenodo record that contains ``filename``."""
    import json
    import urllib.request

    api = f"https://zenodo.org/api/records/{record_id}/versions?size=25"
    with urllib.request.urlopen(api, timeout=60) as response:
        versions = json.load(response)["hits"]["hits"]
    for record in sorted(versions, key=lambda record: record["created"], reverse=True):
        if any(entry["key"] == filename for entry in record["files"]):
            return record
    raise FileNotFoundError(f"no version of Zenodo record {record_id} contains {filename}")


def download_derived_dataset(dest: Path | None = None, quiet: bool = False) -> Path:
    """Download the published derived tables from Zenodo into ``dest``."""
    import urllib.request

    dest = Path(dest) if dest is not None else repo_root() / "data"
    dest.mkdir(parents=True, exist_ok=True)

    for entry in _newest_record_with(MARKER)["files"]:
        target = dest / entry["key"]
        if target.is_file() and target.stat().st_size == entry["size"]:
            continue
        if not quiet:
            print(f"downloading {entry['key']} ({entry['size'] / 1e6:.1f} MB)")
        urllib.request.urlretrieve(entry["links"]["self"], target)
    return dest.resolve()


# --------------------------------------------------------------------------------------
# Raw sources - only needed to rebuild the derived tables
# --------------------------------------------------------------------------------------

#: Participant submissions, Wind Data Hub 10.21947/1922900, mirrored on Zenodo. Version 1
#: of the mirror holds only the 766 files the hub served until October 2026; version 2
#: (10.5281/zenodo.23572116) holds all 4,130, byte-identical to the hub.
RAW_SIMULATION_ZIP = "oc6.phase3.zip"


def raw_simulation_zip_url() -> str:
    """Download link of ``oc6.phase3.zip`` in the newest version of the mirror."""
    record = _newest_record_with(RAW_SIMULATION_ZIP, MIRROR_RECORD_ID)
    entry = next(entry for entry in record["files"] if entry["key"] == RAW_SIMULATION_ZIP)
    return entry["links"]["self"]


#: UNAFLOW experimental campaign (Experiment 1), Fontanella et al. 2021.
UNAFLOW_RECORD_ID = "4740006"
UNAFLOW_DOI = "10.5281/zenodo.4740005"
UNAFLOW_FILES = ("FORCE.zip", "WAKE.zip", "PIV.zip", "Blade_Airfoils.zip", "SD7032.zip",
                 "CompanionDocument.pdf")
