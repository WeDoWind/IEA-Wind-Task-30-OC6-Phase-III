"""Emit the mlcroissant (Croissant 1.0) JSON-LD description of the whole record.

Croissant makes the dataset machine-readable: a client can discover every file, its
checksum, and the name, type, unit and meaning of every column without opening the data.
The small reference tables - models, load cases, blade geometry, the AeroDyn airfoil
parameters, the test lists of both experiments - are embedded as inline ``data`` rather
than shipped as files.

The archive step writes ``croissant.json`` with those inline record sets; the derive step
reads them back from it and adds its own files, so the inline data never has to be
regenerated from the source archives.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import registry

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
}

DATASET_DOI = "10.5281/zenodo.23593301"
DATASET_NAME = "oc6-phase3-floating-rotor-aerodynamics-benchmark"
DATASET_TITLE = (
    "Floating Wind Turbine Rotor Aerodynamics Benchmark: Wind Tunnel Measurements and 56 "
    "Model Results for Surge and Pitch Motion (IEA Wind Task 30 OC6 Phase III)"
)
DATASET_DESCRIPTION = (
    "A ready-to-use benchmark for wind turbine aerodynamics codes on a floating platform: "
    "wind tunnel measurements of a 1:75 scaled DTU 10 MW rotor in prescribed surge and "
    "platform-pitch motion, and the results of 56 numerical models - blade element momentum, "
    "dynamic inflow, free vortex wake and CFD - for the same 11 load cases of IEA Wind TCP "
    "Task 30, OC6 Phase III. Rotor loads, sectional blade loads, hot-wire wake velocities and "
    "PIV wake planes, simulated and measured. The archive tables (oc6.phase3.archive.*) hold "
    "the source records with every value as submitted: the Wind Data Hub archive of the "
    "campaign (4,122 participant result files, the Experiment 2 measurements, the airfoil "
    "polars) and the UNAFLOW record of Experiment 1 (tower loads, hot-wire wake and PIV for "
    "every test); a source-files table gives the checksum, header and layout of each of the "
    "4,730 source files. The derived tables (oc6.phase3.derived.*) hold the paper's load "
    "metrics, the phase-averaged Experiment 1 loads and the blade-tip vortex centres. "
    "The airfoil polars of every blade station and the measured SD7032 polars are tables too; "
    "the small reference tables - models, load cases, blade geometry, the AeroDyn airfoil "
    "parameters, the test lists - are embedded below as inline data. Companion code: "
    "https://github.com/WeDoWind/IEA-Wind-Task-30."
)
CITE_AS = (
    "Marykovskiy, Y., Bergua, R. (2026). " + DATASET_TITLE + ". Zenodo. "
    f"https://doi.org/{DATASET_DOI}. Please also cite Bergua, R., et al. (2023), Wind Energy "
    "Science 8, 465-485, https://doi.org/10.5194/wes-8-465-2023, and, for Experiment 1, "
    "Fontanella, A., Bayati, I., Mikkelsen, R., Belloli, M., Zasso, A. (2021), UNAFLOW, "
    "https://doi.org/10.5281/zenodo.4740005."
)
CREATORS = [
    {"@type": "sc:Person", "givenName": "Yuriy", "familyName": "Marykovskiy",
     "identifier": "https://orcid.org/0000-0002-6570-6966",
     "sameAs": "https://orcid.org/0000-0002-6570-6966"},
    {"@type": "sc:Person", "givenName": "Roger", "familyName": "Bergua",
     "identifier": "https://orcid.org/0000-0003-4703-4729",
     "sameAs": "https://orcid.org/0000-0003-4703-4729"},
]
#: What someone looking for a benchmark would search for, then the project names.
KEYWORDS = [
    "wind energy", "wind turbine", "benchmark", "model validation", "code comparison",
    "floating wind turbine", "floating offshore wind", "rotor aerodynamics",
    "unsteady aerodynamics", "wind tunnel measurements", "surge motion",
    "platform pitch motion", "blade element momentum", "dynamic inflow", "free vortex wake",
    "CFD", "wake", "PIV", "hot-wire anemometry", "DTU 10 MW", "IEA Wind Task 30",
    "IEA Wind TCP", "OC6", "UNAFLOW",
]
#: The UNAFLOW authors, whose Experiment 1 record is converted here (CC BY 4.0).
CONTRIBUTORS = [
    {"@type": "sc:Person", "givenName": given, "familyName": family,
     "identifier": f"https://orcid.org/{orcid}", "sameAs": f"https://orcid.org/{orcid}",
     "description": "Experiment 1 (UNAFLOW) data collector"}
    for given, family, orcid in [
        ("Alessandro", "Fontanella", "0000-0002-8553-1390"),
        ("Ilmas", "Bayati", "0000-0003-3366-0454"),
        ("Robert", "Mikkelsen", "0000-0002-4095-1228"),
        ("Marco", "Belloli", "0000-0002-1780-5038"),
        ("Alberto", "Zasso", "0000-0001-8183-8312"),
    ]
]

#: What each file holds; the key is the file name without the ``oc6.phase3.`` prefix and
#: without the load-case tag of the PIV files.
FILE_DESCRIPTIONS = {
    "archive.simulations.hub_loads":
        "Hub loads (definition document Table 10) of every model and load case, every row "
        "as submitted. Steady load cases may carry several rows.",
    "archive.simulations.blade_sections":
        "Sectional blade quantities (Table 11), one row per time and blade station. Nine "
        "stations per file, except POLIMI m52 LC 1.1 with 18.",
    "archive.simulations.hotwire_wake":
        "Simulated hot-wire wake velocities (Table 12) along the along-wind (aw) and "
        "crosswind (cw) traverses.",
    "archive.simulations.piv_wake_plane":
        "Simulated PIV plane fields (Table 13) for one load case, one frame per blade "
        "azimuth (steady) or motion step (oscillating).",
    "archive.experiment2.tower_top_loads":
        "Experiment 2 (the 2021 campaign): tower-top loads and platform displacement at "
        "2 kHz, one test per load case, as published on the Wind Data Hub.",
    "archive.experiment1.tower_loads":
        "Experiment 1 (UNAFLOW): tower-top and tower-base balance and platform "
        "displacement at 2 kHz, every test of the campaign.",
    "archive.experiment1.hotwire_wake":
        "Experiment 1 (UNAFLOW): three-component hot-wire velocities at 2 kHz, every test, "
        "one row per sample and traverse point.",
    "archive.experiment1.piv_wake_plane":
        "Experiment 1 (UNAFLOW): PIV planes, averaged and single-cycle, every test.",
    "archive.simulations.airfoil_polars":
        "Lift and drag of every blade station (definition document Table 3) at 7 Reynolds "
        "numbers, from airfoil_data.mat - the polars the participants were given (Sect. 3.2). "
        "The UNAFLOW AeroDyn files hold the same polars to 4 decimals.",
    "archive.experiment1.sd7032_polars":
        "Measured 2D polars of the SD7032 blade airfoil at 7 Reynolds numbers (UNAFLOW "
        "SD7032.zip).",
    "archive.source_files":
        "One row per source file of both archives: checksum, original header, shape, number "
        "of values, layout, and every correction or inference the conversion applied.",
    "archive.simulations.extra_openfast_channels":
        "The OpenFAST output channels beyond the blade-section ones in UNIFI m23's blade files "
        "for LC 2.16 and 2.7, which are complete OpenFAST outputs: wind, platform motion, "
        "rotor speed, blade pitch and azimuth, rotor and tower loads. One row per time step, "
        "one column per channel, named as OpenFAST names it.",
    "archive.simulations.extra_hub_load_columns":
        "Columns a hub-load file holds beyond the eight of definition document Table 10. Only "
        "UNIFI m52 LC 2.7 has them: two unlabelled columns whose values lie close to the "
        "reported thrust and torque. The file does not say what they are.",
    "derived.simulations.load_metrics":
        "Per model and load case: mean, peak-to-peak amplitude and phase shift of thrust and "
        "torque (paper Sect. 5.3-5.4), and rotor loads integrated from the blade sections.",
    "derived.simulations.tip_vortex_centres":
        "Blade-tip vortex centres in every simulated steady-wind PIV plane: local maxima of "
        "Graftieaux's Gamma1, each moved onto the vorticity peak within 0.05 m of it.",
    "derived.simulations.tip_vortex_uncertainty":
        "How far each simulated tip-vortex trajectory moves across 12 reasonable detector "
        "settings (Gamma1 threshold, vorticity percentile, centring on the vorticity peak): "
        "the straight-line fit of its track at x = 0.8 and 1.2 m, lowest and highest over the "
        "settings.",
    "derived.experiment1.tip_vortex_uncertainty":
        "The same for the measured tip-vortex trajectory of Experiment 1.",
    "derived.experiment1.hub_loads":
        "Experiment 1 thrust and torque per OC6 load case, low-pass filtered at 3 Hz and "
        "phase-averaged over all motion periods.",
    "derived.experiment1.load_metrics":
        "Experiment 1 mean, peak-to-peak amplitude and phase shift per OC6 load case.",
    "derived.experiment1.hotwire_wake":
        "Experiment 1 steady-wind hot-wire velocities, time-averaged, at hub-relative "
        "positions.",
    "derived.experiment1.tip_vortex_centres":
        "Blade-tip vortex centres in the Experiment 1 steady-wind PIV planes.",
}

#: Meaning of the columns; ``table/column`` overrides the plain column name.
FIELD_DESCRIPTIONS = {
    "participant": "Participant identifier as used in the raw file names.",
    "label": "Participant label as printed in the figures of the paper.",
    "model_code": "Modelling approach code mXY: X = wake/induction model, Y = airfoil model.",
    "wake_model": "Wake/induction model: BEM, DBEM, GDW, FVW or CFD.",
    "airfoil_model": "Blade/airfoil model: Disk, Static polars, Unsteady airfoil "
                     "aerodynamics or Blade resolved.",
    "load_case": "OC6 Phase III load case, e.g. 1.1 (steady), 2.5 (surge), 3.5 (pitch).",
    "time_s": "Time, s.",
    "motion_phase_deg": "Canonical platform motion phase, deg. 0 deg = platform at "
                        "neutral moving downwind (rotor apparent wind minimum).",
    "platform_displacement": "Platform displacement as reported: m for surge load cases, "
                             "deg for platform pitch load cases.",
    "platform_displacement_m": "Platform displacement (mono-harmonic filtered laser "
                               "measurement), m.",
    "fx_n": "Aerodynamic rotor thrust at the hub, N.",
    "fy_n": "Aerodynamic rotor lateral force at the hub, N.",
    "fz_n": "Aerodynamic rotor vertical force at the hub, N.",
    "mx_nm": "Aerodynamic rotor torque at the hub, N m.",
    "my_nm": "Aerodynamic rotor moment about y at the hub, N m.",
    "mz_nm": "Aerodynamic rotor moment about z at the hub, N m.",
    "ttfx_n": "Tower-top x shear force, N.",
    "ttfy_n": "Tower-top y shear force, N.",
    "ttfz_n": "Tower-top z axial force, N.",
    "ttmx_nm": "Tower-top x moment, N m.",
    "ttmy_nm": "Tower-top y moment, N m.",
    "ttmz_nm": "Tower-top z moment, N m.",
    "tbfx_n": "Tower-base x shear force, N.",
    "tbfy_n": "Tower-base y shear force, N.",
    "tbfz_n": "Tower-base z axial force, N.",
    "tbmx_nm": "Tower-base x moment, N m.",
    "tbmy_nm": "Tower-base y moment, N m.",
    "tbmz_nm": "Tower-base z moment, N m.",
    "station": "Blade radial station of definition document Table 3 (1 = root, 20 = tip).",
    "radius_m": "Radial distance of the station from the rotor centre, m.",
    "span_pct": "Radial station position as a percentage of blade span.",
    "fn_n_per_m": "Sectional force normal to the rotor plane, N/m.",
    "ft_n_per_m": "Sectional force tangential to the rotor plane, N/m.",
    "vrel_ms": "Sectional relative wind speed, m/s.",
    "aoa_deg": "Angle of attack, deg.",
    "cl": "Lift coefficient.",
    "cd": "Drag coefficient.",
    "cm": "Moment coefficient.",
    "direction": "Hot-wire traverse: aw (along-wind) or cw (crosswind).",
    "traverse": "Hot-wire traverse: aw (along-wind, 11 points) or cw (crosswind, 33).",
    "point": "Hot-wire measurement point along the traverse, 1-based.",
    "probe_x_m": "Hot-wire probe x position relative to the hub, m.",
    "probe_y_m": "Hot-wire probe y position relative to the hub, m.",
    "u_ms": "Longitudinal (x) velocity component, m/s.",
    "v_ms": "Transverse (y) velocity component, m/s.",
    "w_ms": "Vertical (z) velocity component, m/s.",
    "x_m": "PIV point x relative to the hub, m.",
    "z_m": "PIV point z relative to the hub, m.",
    "vorticity_1_s": "Vorticity magnitude about y, 1/s.",
    "azimuth_deg": "Blade azimuth angle of the reference blade, deg; empty when the frame "
                   "is indexed by motion step.",
    "step": "Index of the instant within one platform motion period; empty when the frame "
            "is indexed by azimuth.",
    "test": "Test identifier as named in the source record.",
    "averaging": "averaged (over many rotor or motion cycles) or single_cycle.",
    "source_file": "The source file the row was read from.",
    "experiment": "Experimental campaign the row comes from.",
    "source": "Provenance of the row: 'simulation' or the experiment name.",
    "fx_mean_n": "Mean aerodynamic rotor thrust, N.",
    "mx_mean_nm": "Mean aerodynamic rotor torque, N m.",
    "fx_peak_to_peak_n": "Peak-to-peak thrust variation, twice the first-harmonic "
                         "amplitude at the platform motion frequency, N.",
    "mx_peak_to_peak_nm": "Peak-to-peak torque variation, N m.",
    "fx_phase_deg": "Phase shift of the thrust with respect to the platform motion, deg.",
    "mx_phase_deg": "Phase shift of the torque with respect to the platform motion, deg.",
    "fx_peak_to_peak_per_amplitude_n_per_m": "Thrust variation normalised by the platform "
                                             "motion amplitude in metres, N/m.",
    "mx_peak_to_peak_per_amplitude_nm_per_m": "Torque variation normalised by the platform "
                                              "motion amplitude in metres, N m/m.",
    "fx_from_blade_aero_n": "Rotor thrust integrated from the sectional normal force, N.",
    "mx_from_blade_aero_nm": "Rotor torque integrated from the sectional tangential force, "
                             "N m.",
    "blade_aero_sign_corrected": "True when the participant reported the tangential force "
                                 "with the opposite sign to the definition document and "
                                 "the integrated torque was negated.",
    "motion": "Platform motion: none, surge or pitch.",
    "frequency_hz": "Platform motion frequency, Hz.",
    "reduced_frequency": "Rotor reduced frequency k = f * D / U0.",
    "motion_amplitude_m": "Platform motion amplitude expressed as a hub displacement, m.",
    "gamma1": "Graftieaux Gamma1 scalar at the detected vortex centre.",
    "vortex_index": "Index of the vortex within the plane, ordered by increasing x.",
    "settings_found": "Number of detector settings in which the vortex track was found.",
    "settings_tried": "Number of detector settings tried.",
    "z_default_m": "Trajectory height with the default detector settings, those of the "
                   "published tip-vortex centres; empty where they found no track, m.",
    "z_min_m": "Lowest trajectory height over the detector settings, m.",
    "z_max_m": "Highest trajectory height over the detector settings, m.",
    "spread_m": "Highest minus lowest trajectory height over the detector settings, m.",
    # model registry
    "organization": "Full name of the participating organisation.",
    "country": "Country of the participating organisation.",
    "code": "Name of the simulation code used.",
    "wake_model_long": "Wake/induction model written out in full.",
    "airfoil_model_long": "Blade/airfoil model written out in full.",
    "cfd_approach": "For CFD models, whether the rotor is blade resolved or represented "
                    "by an actuator line.",
    "tunnel_walls_modelled": "True when the CFD model included the wind tunnel walls, "
                             "floor and ceiling and therefore used the uncorrected "
                             "4.0 m/s inflow.",
    "in_paper_table3": "True when the model is listed in Table 3 of the paper; the others "
                       "took part in the wake analysis only.",
    # load-case registry
    "amplitude": "Prescribed platform motion amplitude, in the unit given by amplitude_unit.",
    "amplitude_unit": "Unit of the prescribed amplitude: m for surge, deg for platform pitch.",
    "wind_speed_ms": "Inflow wind speed, m/s. Load cases: the speed used by the models, "
                     "corrected for wind tunnel blockage. UNAFLOW tests: the tunnel speed "
                     "in the test name.",
    "wind_speed_measured_ms": "Wind speed measured upstream of the turbine, m/s.",
    "rotor_speed_rpm": "Prescribed rotor speed, rpm (mean value where it varies).",
    "blade_pitch_deg": "Prescribed collective blade pitch angle, deg (mean value where it "
                       "varies).",
    "validation": "True when wind tunnel measurements exist for the load case; False for "
                  "the verification-only cases 2.12, 2.16 and 2.17.",
    "description": "Short description of the load case.",
    "period_s": "Platform motion period, s.",
    "surge_equivalent_amplitude_m": "Platform motion amplitude expressed as a hub "
                                    "displacement, m.",
    "apparent_wind_amplitude_ms": "Amplitude of the rotor apparent-wind variation, "
                                  "2*pi*f*A, m/s.",
    # blade geometry and airfoils
    "distance_from_root_m": "Distance along the pitch axis from the blade root, m.",
    "chord_m": "Chord, m.",
    "twist_deg": "Aerodynamic twist, deg (positive to feather).",
    "aerodynamic_center_pct_chord": "Aerodynamic centre, % chord.",
    "relative_thickness_pct": "Relative thickness, % chord (100 = circular root section).",
    "airfoil_id": "Airfoil ID of definition document Table 3; the number in the UNAFLOW "
                  "file name airfoilEL_<id>.",
    "output_station": "True for the nine stations whose sectional quantities the "
                      "participants report.",
    "reynolds_number": "Chord Reynolds number of the polar.",
    "stall_angle_deg": "AeroDyn: stall angle, deg.",
    "zero_lift_aoa_deg": "AeroDyn: zero-lift angle of attack, deg.",
    "cn_slope": "AeroDyn: normal-force slope at zero lift.",
    "cn_stall_positive": "AeroDyn: normal-force coefficient at stall, positive angle of "
                         "attack.",
    "cn_stall_negative": "AeroDyn: normal-force coefficient at stall, negative angle of "
                         "attack.",
    "aoa_min_cd_deg": "AeroDyn: angle of attack of minimum drag, deg.",
    "min_cd": "AeroDyn: minimum drag coefficient.",
    # experiment test lists
    "measurement": "UNAFLOW measurement: force, wake or piv.",
    "amplitude_m": "Surge amplitude in the test name, m.",
    "oc6_load_case": "OC6 Phase III load case the test corresponds to, if any.",
    "tests": "Which tests the positions apply to.",
    "x_mm": "Probe x position in the wind-tunnel frame, mm.",
    "y_mm": "Probe y position in the wind-tunnel frame, mm.",
    "z_mm": "Probe z position in the wind-tunnel frame, mm.",
    "sampling_rate_hz": "Sampling rate, Hz.",
    "motion_amplitude": "Prescribed motion amplitude, in motion_amplitude_unit.",
    "motion_amplitude_unit": "Unit of the motion amplitude: m (surge) or deg (pitch).",
    "motion_frequency_hz": "Prescribed motion frequency, Hz.",
    "platform_displacement_unit": "Unit of the platform displacement column.",
    # provenance tables
    "file_name": "Name of the source file (UNAFLOW: its path inside the zip).",
    "source_record": "DOI of the record the file was taken from.",
    "source_path": "Path of the file in the source record (Wind Data Hub path for the "
                   "OC6 files).",
    "source_received_utc": "Upload time recorded by the Wind Data Hub, UTC.",
    "sha256": "SHA-256 of the source file; equals the Wind Data Hub signature.",
    "size_bytes": "Size of the source file, bytes.",
    "encoding": "Text encoding of the source file.",
    "table": "Where the file's values went: an archive table, or croissant:<record set>.",
    "header": "The file's header lines, verbatim.",
    "n_rows": "Number of data rows in the file.",
    "n_columns": "Number of columns in the file.",
    "n_values": "Number of values in the file; all of them are stored.",
    "layout": "How the file's columns were mapped: standard, or the rule that applied.",
    "notes": "Every correction or inference the conversion applied to this file.",
}

#: Ontology of units of measure, by column-name suffix (checked in this order).
UNITS = [
    ("_n_per_m", "om:newtonPerMetre"),
    ("_1_s", "om:reciprocalSecond-Time"),
    ("_nm", "om:newtonMetre"),
    ("_ms", "om:metrePerSecond-Time"),
    ("_mm", "om:millimetre"),
    ("_hz", "om:hertz"),
    ("_deg", "om:degree"),
    ("_pct", "om:percent"),
    ("_n", "om:newton"),
    ("_m", "om:metre"),
    ("_s", "om:second-Time"),
]
DIMENSIONLESS = {"cl", "cd", "cm", "reduced_frequency"}


def unit_of(column: str) -> str | None:
    if column in DIMENSIONLESS:
        return "om:one"
    if column.endswith("_nm_per_m") or column in {"size_bytes"}:
        return None
    for suffix, unit in UNITS:
        if column.endswith(suffix):
            return unit
    return None


# --------------------------------------------------------------------------------------
# Inline record sets
# --------------------------------------------------------------------------------------

INLINE_DESCRIPTIONS = {
    "models": "One row per numerical model: participant, organisation, code, wake and "
              "airfoil model (paper Table 3; definition document Sect. 6.1).",
    "load_cases": "One row per load case: motion, frequency, amplitude, wind and rotor "
                  "speed (definition document Tables 5-8).",
    "blade_geometry": "The 20 blade stations of definition document Table 3: span, chord, "
                      "twist, thickness, airfoil ID; output_station marks the nine whose "
                      "sectional quantities the participants report.",
    "airfoil_unsteady_parameters": "Unsteady-aerodynamics parameters of every UNAFLOW "
                                   "AeroDyn airfoil file (Blade_Airfoils.zip), per airfoil "
                                   "ID and Reynolds number.",
    "experiment1_tests": "Every UNAFLOW test by measurement, with its motion, wind speed, "
                         "amplitude, frequency and matching OC6 load case.",
    "experiment1_hotwire_points": "UNAFLOW hot-wire traverse positions in the wind-tunnel "
                                  "frame, mm (companion document, WAKE).",
    "experiment2_tests": "The seven Experiment 2 tests: sampling rate, motion amplitude and "
                         "frequency as stored in the MATLAB files.",
}


def _json_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return str(value)


def frame_records(name: str, frame: pd.DataFrame) -> list[dict]:
    return [{f"{name}/{k}": _json_value(v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]


def registry_records() -> dict[str, list[dict]]:
    """The reference tables that come from the registry rather than an archive."""
    models = registry.models()
    models["wake_model"] = models.wake_model.astype(str)
    cases = registry.load_cases().reset_index(drop=True)
    return {
        "models": frame_records("models", models),
        "load_cases": frame_records("load_cases", cases),
        "blade_geometry": frame_records("blade_geometry", registry.BLADE_GEOMETRY),
    }


def _python_type(values) -> str:
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, bool) for v in present):
        return "sc:Boolean"
    if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return "sc:Integer"
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in present):
        return "sc:Float"
    return "sc:Text"


REFERENCES = {"load_case": "load_cases/load_case", "station": "blade_geometry/station"}


def _field(record_set: str, column: str, data_type: str, source: dict | None = None,
           table: str = "") -> dict:
    field = {
        "@type": "cr:Field",
        "@id": f"{record_set}/{column}",
        "name": f"{record_set}/{column}",
        "description": FIELD_DESCRIPTIONS.get(f"{table}/{column}",
                                              FIELD_DESCRIPTIONS.get(column, "")),
        "dataType": data_type,
    }
    if source is not None:
        field["source"] = source
    unit = unit_of(column)
    if unit:
        field["unit"] = {"@id": unit}
    if column in REFERENCES and record_set not in {"load_cases", "blade_geometry"}:
        field["references"] = {"field": {"@id": REFERENCES[column]}}
    return field


def inline_record_set(name: str, records: list[dict]) -> dict:
    columns = list(dict.fromkeys(k.split("/", 1)[1] for row in records for k in row))
    return {
        "@type": "cr:RecordSet",
        "@id": name,
        "name": name,
        "description": INLINE_DESCRIPTIONS.get(name, ""),
        "field": [
            _field(name, column, _python_type([r.get(f"{name}/{column}") for r in records]))
            for column in columns
        ],
        "data": records,
    }


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_type(data_type: pa.DataType) -> str:
    if pa.types.is_dictionary(data_type):
        data_type = data_type.value_type
    if pa.types.is_boolean(data_type):
        return "sc:Boolean"
    if pa.types.is_integer(data_type):
        return "sc:Integer"
    if pa.types.is_floating(data_type):
        return "sc:Float"
    if pa.types.is_timestamp(data_type) or pa.types.is_date(data_type):
        return "sc:Date"
    return "sc:Text"


def _table_key(path: Path) -> str:
    key = path.stem.removeprefix("oc6.phase3.")
    return key.rsplit(".", 1)[0] if ".piv_wake_plane.lc" in key else key


def file_entries(path: Path) -> tuple[dict, dict]:
    """The FileObject and RecordSet describing one Parquet file."""
    record_set = path.stem.removeprefix("oc6.phase3.")
    table = _table_key(path)
    description = FILE_DESCRIPTIONS.get(table, "")
    if table != record_set:
        description += f" Load case {record_set.rsplit('.lc', 1)[1]}."
    file_object = {
        "@type": "cr:FileObject",
        "@id": path.name,
        "name": path.name,
        "description": description,
        "contentUrl": path.name,
        "encodingFormat": "application/x-parquet",
        "contentSize": f"{path.stat().st_size} B",
        "sha256": _sha256(path),
    }
    schema = pq.read_schema(path)
    fields = []
    for f in schema:
        field = _field(record_set, f.name, _arrow_type(f.type),
                       {"fileObject": {"@id": path.name}, "extract": {"column": f.name}},
                       table=table.split(".")[-1])
        # columns named by the source (OpenFAST channels) carry their description in the
        # Parquet schema
        if not field["description"] and f.metadata and b"description" in f.metadata:
            field["description"] = f.metadata[b"description"].decode()
        fields.append(field)
    return file_object, {"@type": "cr:RecordSet", "@id": record_set, "name": record_set,
                         "description": description, "field": fields}


def build_croissant(out_dir: Path, inline: dict[str, list[dict]], version: str) -> dict:
    files = sorted(out_dir.glob("oc6.phase3.archive.*.parquet")) + sorted(
        out_dir.glob("oc6.phase3.derived.*.parquet"))
    distribution, record_sets = [], []
    for path in files:
        file_object, record_set = file_entries(path)
        distribution.append(file_object)
        record_sets.append(record_set)
    for name, records in inline.items():
        record_sets.append(inline_record_set(name, records))
    from datetime import date

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": DATASET_NAME,
        "title": DATASET_TITLE,
        "description": DATASET_DESCRIPTION,
        "version": version,
        "datePublished": date.today().isoformat(),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://doi.org/{DATASET_DOI}",
        "citeAs": CITE_AS,
        "creator": CREATORS,
        "contributor": CONTRIBUTORS,
        "keywords": KEYWORDS,
        "isBasedOn": [
            "https://doi.org/10.5281/zenodo.23572116",
            "https://doi.org/10.21947/1922900",
            "https://doi.org/10.5281/zenodo.4740005",
        ],
        "distribution": distribution,
        "recordSet": record_sets,
    }


def existing_inline(out_dir: Path) -> dict[str, list[dict]]:
    """The inline record sets of an existing ``croissant.json`` (empty if none).

    Only the sets listed in ``INLINE_DESCRIPTIONS`` are kept, so a set that has since moved
    to a file of its own does not linger.
    """
    path = out_dir / "croissant.json"
    if not path.is_file():
        return {}
    return {entry["@id"]: entry["data"] for entry in json.loads(path.read_text())["recordSet"]
            if "data" in entry and entry["@id"] in INLINE_DESCRIPTIONS}


def write_croissant(out_dir: Path, inline: dict[str, list[dict]] | None = None,
                    version: str = "1.0.0") -> Path:
    """Describe every archive and derived file in ``out_dir``.

    ``inline`` adds or replaces inline record sets; the ones already in
    ``out_dir/croissant.json`` are kept, so the derive step need not regenerate them.
    """
    records = existing_inline(out_dir)
    records.update(registry_records())
    records.update(inline or {})
    target = out_dir / "croissant.json"
    target.write_text(json.dumps(build_croissant(out_dir, records, version), indent=1)
                      + "\n")
    return target
