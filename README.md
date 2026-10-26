# IEA Wind Task 30 — OC6 Phase III

Analysis-ready data and an interactive notebook for **OC6 Phase III**, the IEA Wind Task 30 campaign
that validated aerodynamic load predictions for a floating wind turbine undergoing large platform
motion.

[`notebooks/oc6_phase3_figures.ipynb`](notebooks/oc6_phase3_figures.ipynb) reproduces the figures of
[Bergua et al. (2023)](https://doi.org/10.5194/wes-8-465-2023) with Plotly, against both measurement
campaigns, from tidy tables rather than from the 4,122 raw text files — so the notebook reads as an
argument about aerodynamics, not as a parsing exercise. It is also the template for benchmarking your
own aerodynamics code against the 56 models and the two experiments.

The data is a Zenodo record of its own, [10.5281/zenodo.23593300](https://doi.org/10.5281/zenodo.23593300)
(~2.2 GB); this repository holds the code. The notebook is committed without outputs: run it to see
the figures.

## Run it

### On RenkuLab

1. In a Renku project, add this repository and a Zenodo data connector for
   **[10.5281/zenodo.23593300](https://doi.org/10.5281/zenodo.23593300)**, the benchmark dataset
   (this DOI always resolves to its newest version).
2. Add a session launcher with **Create from code**: this repository, environment type Python,
   interface JupyterLab. Renku builds the session image from two files at the repository root, once:
   `environment.yml` (the dependencies) and `jupyterlab_extensions.txt` (Plotly's renderer for
   JupyterLab). Rebuild it from the launcher when either changes.
3. Launch the session and open `notebooks/oc6_phase3_figures.ipynb`. Nothing needs installing: the
   notebook imports the package from `src/`, and `oc6_phase3.paths.data_dir()` finds the record,
   which Renku mounts as a folder beside the repository.

Renku builds from code only for public repositories.

### Locally with [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/WeDoWind/IEA-Wind-Task-30-OC6-Phase-III.git
cd IEA-Wind-Task-30-OC6-Phase-III

uv venv --python 3.12                 # creates .venv
uv pip install -e ".[notebook]"

uv run python -c "from oc6_phase3.paths import download_derived_dataset as d; d()"   # ~2.2 GB
uv run jupyter lab notebooks/oc6_phase3_figures.ipynb
```

`uv run` uses `.venv` automatically; `source .venv/bin/activate` first if you prefer.

### Locally with pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
python -c "from oc6_phase3.paths import download_derived_dataset as d; d()"
jupyter lab notebooks/oc6_phase3_figures.ipynb
```

The download goes to `./data`. Set `OC6_DATA_DIR` if your data lives somewhere else.

### In VS Code

VS Code picks the interpreter up on its own once `.venv` exists in the workspace folder:
open the notebook, click **Select Kernel** (top right) → **Python Environments** → the
`.venv` entry. Register it as a named kernel if you would rather see it under
**Jupyter Kernel**, or if VS Code does not find it:

```bash
uv run python -m ipykernel install --user --name oc6-phase3 \
    --display-name "OC6 Phase III (.venv)"
```

You need the Microsoft **Python** and **Jupyter** extensions installed. Nothing else has to
be configured — `oc6_phase3.paths.data_dir()` finds `./data` relative to the repository, so
no `OC6_DATA_DIR` and no working-directory fiddling. Plotly picks its own renderer for VS
Code; if a figure comes back blank, set `pio.renderers.default = "vscode"` in the setup
cell.

In JupyterLab, `style.use()` gives every figure an HTML copy as well, which loads plotly.js
from its CDN. A JupyterLab without Plotly's own renderer — on RenkuLab it runs in an
environment of its own — draws that instead of nothing.

## Working with the figures

Every figure takes its data as the first argument and draws whatever it is handed, so choosing
models is a pandas filter before the call:

```python
import oc6_phase3 as oc6

summary = oc6.data.summary("1.1")
oc6.figures.steady_loads(summary[summary.wake_model == "CFD"])          # only the CFD models

forces = oc6.data.forces("2.5")
oc6.figures.thrust_over_period(forces[forces.wake_model.isin(["FVW", "CFD"])])
```

Every table carries `wake_model` (BEM, DBEM, GDW, FVW, CFD), `airfoil_model`, the participant
`label` and the `model_code`. In the figures themselves, the legend names the modelling approaches:
click one to hide or show all its models, double-click it to show that approach alone.

Figures are built from `oc6_phase3.plots`, five primitives - `markers`, `lines`,
`measured`, `reference_line`, and the per-approach aggregates - so writing your own variant
in a notebook cell is a few lines rather than a wall of Plotly. The notebook has a worked
example.

## Comparing your own results

Run your code on the load cases of the OC6 Phase III definition document
([10.2172/1957538](https://doi.org/10.2172/1957538)): rotor and blade geometry in Sect. 3, load cases
in Sect. 5, outputs in Sect. 6. The blade geometry, airfoil polars and load cases are in the record
too (`data.blade_geometry()`, `data.airfoil_polars()`, `data.load_cases()`). The notebook's
*Benchmarking your own model* section maps every part of the setup to its section and table, and
spells out the conventions — above all the time zero of the motion, which every phase is computed
from.

If your code writes the output files the definition document asks for (Sect. 6, Tables 10-13:
`MYCODE_M23_LC25_Forces.txt`, `..._BA.txt`, `..._HW_AW.txt`, `..._PIV_PSI0.txt`, ...), read a folder
of them with the same reader that converted the campaign's submissions:

```python
import pandas as pd, oc6_phase3 as oc6

mine = oc6.contribute.read_outputs("my_results/", label="MyModel")
oc6.figures.thrust_over_period(pd.concat([oc6.data.forces("2.5"), mine.forces]))
oc6.figures.amplitude_and_phase(pd.concat([oc6.data.summary("2.5"), mine.summary]))
```

For hub loads alone, `contribute.model_results` takes the time series from memory:

```python
mine = oc6.contribute.model_results(
    label="MyModel", model_code="m23", load_case="2.5",
    time_s=t, fx_n=thrust, mx_nm=torque,          # one platform motion period
)
```

Either way your results go through the same post-processing as the archive - canonical motion
phase, first-harmonic amplitude, phase shift - so the numbers are comparable rather than merely
drawn on the same axes. The `model_code` (`m23` = dynamic BEM with unsteady airfoil aerodynamics,
`m43` = free vortex wake, `m54` = blade-resolved CFD) is what gives your model its colour and
marker.

## The data

Two layers, told apart by the file name:

- **`oc6.phase3.archive.*`** — the source records with every value as submitted. The
  participants' hub loads, blade sections, hot-wire wake and PIV planes; the Experiment 2
  tower-top loads; the UNAFLOW (Experiment 1) tower loads, hot-wire wake and PIV for every test.
  Plus a table of all 4,730 source files (checksum, header, layout, corrections), and the extra
  columns a few files carry beyond the standard layout.
- **`oc6.phase3.derived.*`** — calculated from the archive tables: the paper's load metrics, the
  phase-averaged Experiment 1 loads, the blade-tip vortex centres, and how far the vortex
  trajectories move with the detector settings.

| table | rows | what |
|---|---|---|
| `archive.simulations.hub_loads` | 40 430 | hub loads (Fx … Mz) of 55 models over the 11 load cases |
| `archive.simulations.blade_sections` | 349 371 | sectional forces, relative wind, angle of attack, lift and drag at the blade stations |
| `archive.simulations.hotwire_wake` | 449 380 | wake velocities along the two hot-wire traverses |
| `archive.simulations.piv_wake_plane.lc*` | 69 M | PIV plane fields, one file per load case |
| `archive.simulations.airfoil_polars` | 17 920 | lift and drag of the 20 blade stations at 7 Reynolds numbers, as given to the participants |
| `archive.simulations.extra_*` | 63 | the columns a few files carry beyond the standard layout |
| `archive.experiment1.*` | 79 M | UNAFLOW tower loads, hot-wire wake and PIV, every test, and the measured SD7032 polars |
| `archive.experiment2.tower_top_loads` | 744 000 | Experiment 2 tower-top loads, 2 kHz |
| `archive.source_files` | 4 730 | every source file: checksum, original header, layout, corrections |
| `derived.simulations.load_metrics` | 529 | mean, peak-to-peak amplitude and phase shift per model and load case, plus loads integrated from the blade sections |
| `derived.simulations.tip_vortex_centres` | 694 | blade-tip vortex centres found with Graftieaux's Gamma1 |
| `derived.simulations.tip_vortex_uncertainty` | 28 | how far each vortex trajectory moves over 12 detector settings |
| `derived.experiment1.*` | 337 | the same quantities measured in the UNAFLOW campaign |
| `croissant.json` | | mlcroissant JSON-LD describing every file, column and unit, with the models, load cases and blade geometry inline |

`oc6_phase3.data` reads them and adds what is looked up rather than stored: participant label,
modelling approach, canonical motion phase, station radius. `croissant.json` describes every
file and column, and `archive.source_files` every correction applied to a source file.

Built from two open records, both converted in full:

- **Simulations, Experiment 2, airfoil polars** — the Wind Data Hub archive,
  [10.5281/zenodo.23572116](https://doi.org/10.5281/zenodo.23572116), byte-identical to
  [10.21947/1922900](https://doi.org/10.21947/1922900): 4,122 whitespace-delimited text files whose
  column layouts are fixed by the definition document but whose headers are not (132 spellings,
  commas, UTF-16, a few files with extra columns), seven MATLAB files of Experiment 2, and the
  airfoil polars. `build/archive_wdh.py` lists every rule it applies.
- **Experiment 1** — UNAFLOW, [10.5281/zenodo.4740005](https://doi.org/10.5281/zenodo.4740005).
  The reduction to hub loads is in `build/experiments.py`, which states exactly which
  transformations are and are not applied.

### Rebuilding the data

The derived tables are recalculated from the archive tables alone (~3 min, most of it the
vortex-detector sensitivity):

```bash
uv pip install -e ".[dev]"
uv run python scripts/build_dataset.py --data data
uv run pytest
```

The archive tables themselves were made once from the two source records, the Wind Data Hub
zip (Zenodo version 2) and UNAFLOW. To repeat that conversion (~2.3 GB download, ~3 min):

```bash
uv run python scripts/convert_archive.py --raw data/raw --out data
```

Both refresh `croissant.json`.

## Layout

```
notebooks/oc6_phase3_figures.ipynb   the figures, with the physics written out
src/oc6_phase3/
  registry.py     participants, model codes, load cases, geometry - from the source documents
  paths.py        find the data: $OC6_DATA_DIR, a Renku mount, ./data, or Zenodo
  data.py         read the tidy tables
  harmonics.py    peak-to-peak amplitude, phase shift, 3 Hz filter, phase averaging
  wake.py         wake deficit, Graftieaux Gamma1 blade-tip vortex tracking
  plots.py        the five drawing primitives every figure is made of
  figures.py      one short recipe per paper figure
  contribute.py   put your own model results into the same schema
  style.py        colour by modelling approach, measurements in neutral ink
  build/          convert the source archives, calculate the derived tables
scripts/convert_archive.py           source archives -> archive tables (run once)
scripts/build_dataset.py             archive tables -> derived tables
tests/                               conventions, maths and conversion rules, no data needed
environment.yml                      RenkuLab session environment (the dependencies)
jupyterlab_extensions.txt            RenkuLab: Plotly's renderer for JupyterLab
```

## Known gaps

The record holds the complete campaign: every file the participants submitted to the Wind Data Hub
(55 models with hub loads, coverage 87 % of model × load case — not every model ran every case), all
of UNAFLOW, and the Experiment 2 loads. The notebook reproduces the paper's figures with its model
counts — plus TUD `m52`, see below — and both experiments' measured values match those plotted in
the paper to within 3 %.

What is still missing or differs:

- **Experiment 2 wake data** — the record has the 2021 campaign's loads only, no hot-wire or PIV.
  Its loads are reduced on read, not yet stored as derived tables.
- **Fig. 10** (tip-vortex trajectories) depends on detector settings the paper does not give. The
  curves here lie within about 0.01–0.02 m of the published ones; `derived.*.tip_vortex_uncertainty`
  records how far each moves over 12 settings, and the notebook draws it.
- **Fig. 15** (lift hysteresis) cannot be matched exactly: in the archive the unsteady-airfoil loop
  at LC 2.5 is about half as wide as the paper draws it and sits slightly below the static polar.
- **Two models are not in the paper's Table 3**: TUD m52 (CFD, it also submitted loads) and
  EDF2 m42 (FVW, wake data only). They are kept and marked `*` in the figures.
- **A few internally inconsistent submissions** — DTU1/DTU2 (blade forces about 1000× smaller than
  their hub loads), POLIMI, PRI, UoS — are kept as submitted; the notebook's
  steady-wind section quantifies them.

## Contributing

Run `uv run ruff check src tests` and `uv run pytest` before a pull request, and commit the notebook
without outputs:

```bash
uv run jupyter nbconvert --clear-output --inplace notebooks/oc6_phase3_figures.ipynb
```

## Documents

- Bergua, R. *et al.* (2023). *OC6 project Phase III*. Wind Energy Science 8, 465–485.
  <https://doi.org/10.5194/wes-8-465-2023>
- Robertson, A., Bergua, R., Fontanella, A., Jonkman, J. (2023). *OC6 Phase III Definition Document*.
  NREL/TP-5000-83102. <https://doi.org/10.2172/1957538>
- Fontanella, A. *et al.* (2021). *UNAFLOW*. Wind Energy Science 6, 1169–1190.
  <https://doi.org/10.5194/wes-6-1169-2021>
- Cioni, S. *et al.* (2023). *On the characteristics of the wake of a wind turbine undergoing large
  motions caused by a floating structure* — the companion OC6 Phase III wake analysis.
  Wind Energy Science 8, 1659–1691. <https://doi.org/10.5194/wes-8-1659-2023>

## Licence

Code MIT (see [LICENSE](LICENSE)). The dataset, [10.5281/zenodo.23593300](https://doi.org/10.5281/zenodo.23593300),
is CC BY 4.0: the OC6 Phase III submissions are CC0, the UNAFLOW data is CC BY 4.0 and requires
attribution to Fontanella et al. Cite the dataset, the paper and UNAFLOW if you use this work.
