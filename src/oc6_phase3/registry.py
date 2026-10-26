"""Static metadata for OC6 Phase III: model codes, participants, load cases, geometry.

Everything here comes from the two authoritative documents and is kept out of the
notebook so that the figures read as figures:

* Robertson, A., Bergua, R., Fontanella, A., Jonkman, J. (2023)
  *OC6 Phase III Definition Document*, NREL/TP-5000-83102. https://doi.org/10.2172/1957538
* Bergua, R. et al. (2023) *OC6 project Phase III*, Wind Energ. Sci. 8, 465-485.
  https://doi.org/10.5194/wes-8-465-2023
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------------------
# Modelling approach ("mXY" in every raw file name)
# --------------------------------------------------------------------------------------
# Definition document Sect. 6.1: the first digit is the wake/induction model, the second
# the blade/airfoil model.

WAKE_MODELS: dict[int, str] = {
    1: "BEM",
    2: "DBEM",
    3: "GDW",
    4: "FVW",
    5: "CFD",
}

WAKE_MODEL_LONG: dict[str, str] = {
    "BEM": "Blade element / momentum",
    "DBEM": "Dynamic (filtered) blade element / momentum",
    "GDW": "Generalized dynamic wake",
    "FVW": "Free vortex wake",
    "CFD": "Computational fluid dynamics",
}

#: Fidelity ordering used throughout the paper (BEM < DBEM ~ GDW < FVW < CFD).
WAKE_MODEL_ORDER: list[str] = ["BEM", "DBEM", "GDW", "FVW", "CFD"]

AIRFOIL_MODELS: dict[int, str] = {
    1: "Disk",
    2: "Static polars",
    3: "Unsteady airfoil aerodynamics",
    4: "Blade resolved",
}

AIRFOIL_MODEL_LONG: dict[str, str] = {
    "Disk": "Disk (thrust/torque)",
    "Static polars": "Lifting line with static polars",
    "Unsteady airfoil aerodynamics": "Lifting line with unsteady airfoil aerodynamics",
    "Blade resolved": "Blade resolved / surface mesh",
}


def decode_model_code(model_code: str) -> tuple[str, str]:
    """``"m43"`` -> ``("FVW", "Unsteady airfoil aerodynamics")``."""
    digits = model_code.removeprefix("m")
    return WAKE_MODELS[int(digits[0])], AIRFOIL_MODELS[int(digits[1])]


# --------------------------------------------------------------------------------------
# Participants
# --------------------------------------------------------------------------------------
# Keys are the lowercase tokens used in the raw file names. ``label`` is the name used in
# the paper's figures.

@dataclass(frozen=True)
class Participant:
    label: str
    organization: str
    country: str


PARTICIPANTS: dict[str, Participant] = {
    "bvmo": Participant("BVMO", "Bureau Veritas", "France"),
    "cener": Participant("CENER", "Centro Nacional de Energias Renovables", "Spain"),
    "cgc": Participant("CGC", "China General Certification Center", "China"),
    "cssc": Participant("CSSC", "China State Shipbuilding Corporation", "China"),
    "dnv": Participant("DNV", "Det Norske Veritas", "United Kingdom"),
    "dtu1": Participant("DTU1", "Technical University of Denmark", "Denmark"),
    "dtu2": Participant("DTU2", "Technical University of Denmark", "Denmark"),
    "dut": Participant("DUT", "Dalian University of Technology", "China"),
    "edf1": Participant("EDF1", "Electricite de France", "France"),
    "edf2": Participant("EDF2", "Electricite de France", "France"),
    "eure1": Participant("EURE1", "eureka!", "Spain"),
    "eure2": Participant("EURE2", "eureka!", "Spain"),
    "ife": Participant("IFE", "Institute for Energy Technology", "Norway"),
    "ifpen": Participant("IFPEN", "IFP Energies nouvelles", "France"),
    "mar": Participant("MAR", "Maritime Research Institute Netherlands", "Netherlands"),
    "nrel1": Participant("NREL1", "National Renewable Energy Laboratory", "USA"),
    "nrel2": Participant("NREL2", "National Renewable Energy Laboratory", "USA"),
    "nu": Participant("NU", "Newcastle University", "United Kingdom"),
    "on": Participant("ON", "Office National d'Etudes et de Recherches Aerospatiales", "France"),
    "poli_wind": Participant("POLI-W", "Politecnico di Milano - POLI-Wind Laboratory", "Italy"),
    "polimi": Participant("POLIMI", "Politecnico di Milano", "Italy"),
    "pri": Participant("PRI", "PRINCIPIA", "France"),
    "sjtu": Participant("SJTU", "Shanghai Jiao Tong University", "China"),
    "tecn": Participant("TECN", "Tecnalia", "Spain"),
    "tno": Participant("TNO", "Netherlands Organization for Applied Scientific Research",
                       "Netherlands"),
    "tub": Participant("TUB", "Technische Universitaet Berlin", "Germany"),
    "tud": Participant("TUD", "Delft University of Technology", "Netherlands"),
    "tuhh": Participant("TUHH", "Hamburg University of Technology", "Germany"),
    "unifi": Participant("UNIFI", "Universita degli Studi di Firenze", "Italy"),
    "uos": Participant("UoS", "University of Strathclyde", "United Kingdom"),
    "upc": Participant("UPC", "Universitat Politecnica de Catalunya", "Spain"),
    "ustutt": Participant("USTUTT", "University of Stuttgart", "Germany"),
    "vulc": Participant("VULC", "Vulcain Engineering", "France"),
    "wtek": Participant("WTEK", "WyndTek", "Netherlands"),
}

#: Simulation code per (participant, model code). Paper Table 3.
CODES: dict[tuple[str, str], str] = {
    ("bvmo", "m12"): "Opera",
    ("bvmo", "m22"): "Opera",
    ("cener", "m43"): "AeroVIEW",
    ("cener", "m54"): "OpenFOAM",
    ("cgc", "m23"): "Bladed",
    ("cssc", "m43"): "QBlade",
    ("dnv", "m23"): "Bladed",
    ("dtu1", "m23"): "HAWC2",
    ("dtu1", "m43"): "HAWC2-MIRAS",
    ("dtu1", "m53"): "HAWC2-MIRAS",
    ("dtu2", "m23"): "HAWC2",
    ("dut", "m23"): "OpenFAST",
    ("edf1", "m12"): "DIEGO",
    ("edf1", "m22"): "DIEGO",
    ("edf2", "m13"): "DIEGO",
    ("edf2", "m23"): "DIEGO",
    ("edf2", "m42"): "DIEGO",
    ("edf2", "m43"): "DIEGO",
    ("eure1", "m12"): "OpenFAST",
    ("eure1", "m22"): "OpenFAST",
    ("eure2", "m13"): "OpenFAST",
    ("eure2", "m23"): "OpenFAST",
    ("eure2", "m43"): "OpenFAST",
    ("ife", "m22"): "3DFloat",
    ("ife", "m33"): "RotorVex",
    ("ifpen", "m23"): "Aerodeep",
    ("ifpen", "m43"): "Castor",
    ("mar", "m12"): "aNySIM-XMF",
    ("nrel1", "m12"): "OpenFAST",
    ("nrel1", "m22"): "OpenFAST",
    ("nrel2", "m13"): "OpenFAST",
    ("nrel2", "m23"): "OpenFAST",
    ("nrel2", "m43"): "OpenFAST",
    ("nu", "m12"): "DARWind",
    ("on", "m42"): "PUMA",
    ("poli_wind", "m12"): "Cp-Lambda",
    ("poli_wind", "m32"): "Cp-Lambda",
    ("polimi", "m52"): "OpenFOAM",
    ("pri", "m23"): "DeepLines Wind",
    ("sjtu", "m54"): "STAR-CCM+",
    ("tecn", "m13"): "OpenFAST",
    ("tecn", "m23"): "OpenFAST",
    ("tno", "m23"): "AeroModule",
    ("tno", "m43"): "AeroModule",
    ("tub", "m43"): "QBlade",
    ("tud", "m52"): "unknown",
    ("tuhh", "m42"): "panMARE",
    ("unifi", "m23"): "OpenFAST",
    ("unifi", "m52"): "CONVERGE",
    ("uos", "m54"): "OpenFOAM",
    ("upc", "m12"): "FloaWDyn",
    ("upc", "m43"): "OpenFAST",
    ("ustutt", "m54"): "FLOWer",
    ("vulc", "m12"): "OpenFAST",
    ("vulc", "m22"): "OpenFAST",
    ("wtek", "m23"): "Ashes",
}

#: (participant, model code) pairs present in the archive but absent from paper Table 3.
NOT_IN_PAPER: set[tuple[str, str]] = {
    ("tud", "m52"),
    ("edf2", "m42"),
}

#: CFD participants that modelled the wind tunnel walls, floor and ceiling and therefore
#: used the uncorrected 4.0 m/s inflow instead of the blockage-corrected 4.19 m/s
#: (paper Sect. 4).
CFD_WITH_TUNNEL_WALLS: set[str] = {"polimi", "unifi", "ustutt", "uos"}


# --------------------------------------------------------------------------------------
# Geometry (paper Tables 1 and 2; definition document Sect. 3)
# --------------------------------------------------------------------------------------

ROTOR_DIAMETER_M = 2.38132
ROTOR_RADIUS_M = ROTOR_DIAMETER_M / 2
BLADE_LENGTH_M = 1.10166
HUB_RADIUS_M = ROTOR_RADIUS_M - BLADE_LENGTH_M  # 0.089 m
AIR_DENSITY = 1.177  # kg/m3
TILT_ANGLE_DEG = 5.0

#: Experiment 1 tower-top geometry, for projecting the measured loads onto the hub
#: (R. Bergua, pers. comm., 2026: "Projecting the measured data at Tower Top to Hub Fix
#: location for OC6 Phase III").
OVERHANG_M = 0.09467          # hub ahead of the tower top, along the shaft
TOWER_TO_SHAFT_M = 0.03667    # tower top up to the shaft axis
#: Offsets from the tower top to the hub, in ground-parallel coordinates.
HUB_OFFSET_X_M = 0.0979       # Overhang + Twr2Shft * sin(alpha)
HUB_OFFSET_Z_M = 0.0365       # Twr2Shft * cos(alpha)

#: Distance from the hub to the platform pitch axis, used to convert a prescribed pitch
#: amplitude into an equivalent surge amplitude (paper Sect. 4: "the equivalent
#: longitudinal amplitude can be approximated by multiplying the sine of the platform
#: pitch angle by the distance from the hub to the tower base").
#:
#: Taken from the Experiment 2 geometry of paper Table 2:
#: ``hypot(tower length 1.400 + tower-to-shaft 0.064, rotor overhang 0.139) = 1.4706 m``.
#: The models that ran both a surge load case and its pitch equivalent confirm it: the
#: ratio of their peak-to-peak thrusts implies 1.457-1.477 m (median 1.459 m).
#:
#: Note that the prescribed pitch amplitudes are rounded to 0.1 deg, so LC 3.5 and LC 3.7
#: are only approximately equivalent to LC 2.5 and LC 2.7 (0.0359 m vs 0.035 m, and
#: 0.0077 m vs 0.008 m).
HUB_TO_PITCH_AXIS_M = 1.4706

#: The 20 blade radial stations of definition document Table 3, root to tip. Distance is
#: measured along the pitch axis from the blade root; the radius adds the hub radius.
#: ``airfoil_id`` names the polar set (``airfoilEL_<id>`` in UNAFLOW, page ``station`` of
#: ``airfoil_data.mat``); ``output_station`` marks the nine stations whose sectional
#: quantities the participants report (Sect. 6.2).
BLADE_GEOMETRY = pd.DataFrame(
    {
        "station": list(range(1, 21)),
        "span_pct": [7.5, 12.4, 18.9, 25.8, 32.7, 39.7, 46.6, 53.3, 59.6, 65.6,
                     71.1, 76.1, 80.7, 84.7, 88.2, 91.3, 94.0, 96.3, 98.3, 100.0],
        "distance_from_root_m": [
            0.00000, 0.05817, 0.13641, 0.21766, 0.30059, 0.38379, 0.46581, 0.54530,
            0.62105, 0.69211, 0.75778, 0.81765, 0.87153, 0.91947, 0.96171, 0.99860,
            1.03056, 1.05807, 1.08162, 1.10166,
        ],
        "chord_m": [
            0.05585, 0.05678, 0.07573, 0.10620, 0.11490, 0.11044, 0.10236, 0.09272,
            0.08288, 0.07356, 0.06516, 0.05778, 0.05141, 0.04604, 0.04163, 0.03796,
            0.03440, 0.03054, 0.02541, 0.00998,
        ],
        "twist_deg": [
            17.07668, 17.04199, 15.77593, 12.30509, 9.98299, 8.65143, 7.56522, 6.38165,
            5.08008, 3.79042, 2.61685, 1.59090, 0.71754, 0.03751, -0.53510, -1.03393,
            -1.46251, -1.61172, -1.60710, -1.72236,
        ],
        "aerodynamic_center_pct_chord": [50.00, 48.76, 35.76, 29.08] + [29.00] * 16,
        "relative_thickness_pct": [100.00, 75.88, 17.91, 11.11] + [9.97] * 16,
        "airfoil_id": [2, 3] + list(range(5, 40, 2)),
    }
).assign(
    radius_m=lambda d: HUB_RADIUS_M + d.distance_from_root_m,
    output_station=lambda d: d.station.isin(range(3, 20, 2)),
)

#: The nine blade radial stations reported in the blade-aerodynamics files.
BLADE_STATIONS = BLADE_GEOMETRY[BLADE_GEOMETRY.output_station].reset_index(drop=True)

#: Reynolds numbers of the seven polar sets per station (definition document Sect. 3.2).
REYNOLDS_NUMBERS = [5.0e4, 6.0e4, 7.5e4, 1.0e5, 1.5e5, 1.7e5, 2.0e5]

#: Measurement grids (definition document Sect. 6.3 and 6.4), all relative to the hub (CS3).
HOTWIRE_ALONGWIND_X_M = [round(2.18 + 0.33 * i, 2) for i in range(11)]
HOTWIRE_ALONGWIND_Y_M = 0.9
HOTWIRE_CROSSWIND_X_M = 5.48
HOTWIRE_CROSSWIND_Y_M = [round(-1.6 + 0.1 * i, 2) for i in range(33)]
#: In Experiment 1 the crosswind probes were offset by 0.13 m in y for the steady case.
HOTWIRE_CROSSWIND_Y_EXP1_OFFSET_M = 0.13

PIV_X_RANGE_M = (0.61, 1.36)
PIV_Z_RANGE_M = (0.61, 1.40)
PIV_STEP_M = 0.005
PIV_SHAPE = (159, 151)  # (n_z, n_x) -> 24009 points


# --------------------------------------------------------------------------------------
# Load cases (definition document Tables 5-8, paper Table 4)
# --------------------------------------------------------------------------------------

_LOAD_CASES: list[dict] = [
    # id,  motion, f [Hz], amplitude, unit, U0' [m/s], rpm, pitch [deg], notes
    dict(load_case="1.1", motion="none", frequency_hz=0.0, amplitude=0.0, amplitude_unit="-",
         wind_speed_ms=4.19, wind_speed_measured_ms=4.00, rotor_speed_rpm=240.0,
         blade_pitch_deg=0.0, validation=True,
         description="Steady wind, rated 2 condition (tip-speed ratio 7.1)"),
    dict(load_case="1.2", motion="none", frequency_hz=0.0, amplitude=0.0, amplitude_unit="-",
         wind_speed_ms=6.03, wind_speed_measured_ms=6.00, rotor_speed_rpm=265.0,
         blade_pitch_deg=12.5, validation=True,
         description="Steady wind, above-rated condition"),
    dict(load_case="2.1", motion="surge", frequency_hz=0.125, amplitude=0.125,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Surge, lowest reduced frequency"),
    dict(load_case="2.5", motion="surge", frequency_hz=1.0, amplitude=0.035,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Surge, largest rotor apparent-wind variation"),
    dict(load_case="2.7", motion="surge", frequency_hz=2.0, amplitude=0.008,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Surge, highest reduced frequency"),
    dict(load_case="2.12", motion="surge", frequency_hz=2.0, amplitude=0.080,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=None,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=False,
         description="Verification only: LC 2.7 frequency, amplitude x10"),
    dict(load_case="2.16", motion="surge", frequency_hz=2.0, amplitude=0.080,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=None,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=False,
         description="Verification only: LC 2.12 plus rotor speed 240 - 36 cos(wt) rpm"),
    dict(load_case="2.17", motion="surge", frequency_hz=2.0, amplitude=0.080,
         amplitude_unit="m", wind_speed_ms=4.19, wind_speed_measured_ms=None,
         rotor_speed_rpm=240.0, blade_pitch_deg=1.5, validation=False,
         description="Verification only: LC 2.12 plus blade pitch 1.5 - 1.5 cos(wt) deg"),
    dict(load_case="3.1", motion="pitch", frequency_hz=0.125, amplitude=3.000,
         amplitude_unit="deg", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Platform pitch; robot amplitude limit prevents LC 2.1 equivalence"),
    dict(load_case="3.5", motion="pitch", frequency_hz=1.0, amplitude=1.400,
         amplitude_unit="deg", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Platform pitch equivalent to LC 2.5"),
    dict(load_case="3.7", motion="pitch", frequency_hz=2.0, amplitude=0.300,
         amplitude_unit="deg", wind_speed_ms=4.19, wind_speed_measured_ms=4.00,
         rotor_speed_rpm=240.0, blade_pitch_deg=0.0, validation=True,
         description="Platform pitch equivalent to LC 2.7"),
]


def _surge_equivalent_amplitude_m(row: dict) -> float:
    """Platform motion amplitude expressed as a hub displacement in metres."""
    if row["motion"] == "surge":
        return row["amplitude"]
    if row["motion"] == "pitch":
        return HUB_TO_PITCH_AXIS_M * math.sin(math.radians(row["amplitude"]))
    return 0.0


def load_cases() -> pd.DataFrame:
    """Load-case registry, one row per load case, indexed by ``load_case``."""
    df = pd.DataFrame(_LOAD_CASES)
    df["period_s"] = df.frequency_hz.where(df.frequency_hz > 0).rdiv(1.0)
    df["reduced_frequency"] = df.frequency_hz * ROTOR_DIAMETER_M / df.wind_speed_ms
    df["surge_equivalent_amplitude_m"] = [
        _surge_equivalent_amplitude_m(r) for r in _LOAD_CASES
    ]
    # Rotor apparent-wind amplitude, U0 - w(t), Eq. (7) of the paper.
    df["apparent_wind_amplitude_ms"] = (
        2 * math.pi * df.frequency_hz * df.surge_equivalent_amplitude_m
    )
    return df.set_index("load_case", drop=False)


LOAD_CASE_ORDER: list[str] = [r["load_case"] for r in _LOAD_CASES]


def models() -> pd.DataFrame:
    """Registry of the numerical models, one row per (participant, model code)."""
    rows = []
    for (participant, model_code), code in CODES.items():
        wake, airfoil = decode_model_code(model_code)
        p = PARTICIPANTS[participant]
        rows.append(
            dict(
                participant=participant,
                label=p.label,
                organization=p.organization,
                country=p.country,
                code=code,
                model_code=model_code,
                wake_model=wake,
                wake_model_long=WAKE_MODEL_LONG[wake],
                airfoil_model=airfoil,
                airfoil_model_long=AIRFOIL_MODEL_LONG[airfoil],
                cfd_approach=(
                    ("blade resolved" if airfoil == "Blade resolved" else "actuator line")
                    if wake == "CFD"
                    else None
                ),
                tunnel_walls_modelled=(
                    wake == "CFD" and participant in CFD_WITH_TUNNEL_WALLS
                ),
                in_paper_table3=(participant, model_code) not in NOT_IN_PAPER,
            )
        )
    df = pd.DataFrame(rows).sort_values(["label", "model_code"], ignore_index=True)
    df["wake_model"] = pd.Categorical(df.wake_model, WAKE_MODEL_ORDER, ordered=True)
    return df
