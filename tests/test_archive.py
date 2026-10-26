"""The conversion of the source archives: names, layouts, and value accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oc6_phase3.build import archive_wdh, croissant, derive
from oc6_phase3.build.textfiles import read_numeric

# --------------------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------------------


def test_piv_planes_named_by_degrees_are_read_as_azimuths():
    record = archive_wdh.parse_name("oc6.phase3.on.m42.lc12.piv_015.txt")
    assert record["kind"] == "piv"
    assert record["azimuth_deg"] == 15.0
    assert record["step"] is None


def test_edf_lc11_files_belong_to_edf2():
    record = archive_wdh.parse_name("oc6.phase3.edf.m43.lc11.piv_psi30.txt")
    assert record["participant"] == "edf2"


def test_malformed_forces_name_is_corrected():
    record = archive_wdh.parse_name("oc6.phase3.dut.m23.lc12.forces_.txt")
    assert (record["kind"], record["load_case"]) == ("forces", "1.2")


def test_matlab_files_are_not_participant_files():
    assert archive_wdh.parse_name("oc6.phase3.nrel.air.foil.data.mat") is None


# --------------------------------------------------------------------------------------
# Reading text
# --------------------------------------------------------------------------------------


def test_header_lines_are_kept_verbatim_and_values_counted():
    data = read_numeric(b"Time Fx\ns N\n0 1.5\n0.1 NaN\n0.2 Nan\n")
    assert data.header == ["Time Fx", "s N"]
    assert data.values.shape == (3, 2)
    assert np.isnan(data.values[1:, 1]).all()
    assert data.n_tokens == 6


def test_a_digit_labelled_header_is_still_a_header():
    data = read_numeric(b"05:X 07:Z\n1 2\n")
    assert data.header == ["05:X 07:Z"]
    assert data.n_rows == 1


def test_commas_separate_values():
    data = read_numeric(b"1,2,3\n4,5,6\n")
    assert data.values.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert "comma separated" in data.notes


def test_utf16_files_are_decoded():
    data = read_numeric("t x\n0 1\n".encode("utf-16"))
    assert data.encoding == "utf-16"
    assert data.values.tolist() == [[0, 1]]


def test_ragged_rows_are_padded_and_reported():
    data = read_numeric(b"1 2 3\n4 5\n")
    assert data.n_tokens == 5
    assert np.isnan(data.values[1, 2])
    assert any("ragged" in note for note in data.notes)


def test_content_fix_restores_the_truncated_exponent():
    data = read_numeric(b"0.975 2\n1.000001e 3\n", ("1.000001e ", "1.000001e+00 "))
    assert data.values[-1, 0] == pytest.approx(1.000001)


# --------------------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------------------

VULC = {"participant": "vulc", "model_code": "m12"}
POLIMI = {"participant": "polimi", "model_code": "m52"}


def test_standard_blade_file_uses_every_column_once():
    layout = archive_wdh.blade_layout(VULC, [], 55)
    layout.check(55)
    assert layout.stations == list(range(3, 20, 2))
    assert not layout.extras


def test_vulc_motion_column_is_set_aside_not_read_as_station_one():
    layout = archive_wdh.blade_layout(VULC, [], 56)
    layout.check(56)
    assert layout.standard[:2] == [0, 2]
    assert list(layout.extras) == [1]


def test_openfast_blade_channels_are_mapped_by_name():
    names = ["Time_[s]", "Wind_[m/s]"] + [
        f"B1N{i}{q}_[-]" for q in archive_wdh.OPENFAST_BLADE_CHANNELS for i in range(1, 10)
    ] + ["RtAeroFxh_[N]"]
    # shuffle the blade channels so a positional reading would be wrong
    order = [0, 1, *reversed(range(2, 56)), 56]
    header = ["\t".join(names[i] for i in order)]
    layout = archive_wdh.blade_layout(VULC, header, len(names))
    layout.check(len(names))
    mapped = [names[order[i]].split("_[")[0] for i in layout.standard]
    assert mapped[:3] == ["Time", "B1N1Fx", "B1N2Fx"]
    assert sorted(layout.extras.values()) == ["RtAeroFxh_[N]", "Wind_[m/s]"]


def test_polimi_eighteen_station_file_keeps_all_stations():
    layout = archive_wdh.blade_layout(POLIMI, [], 109)
    layout.check(109)
    assert layout.stations == list(range(1, 19))


def test_unknown_blade_layout_stops_the_conversion():
    with pytest.raises(ValueError):
        archive_wdh.blade_layout({"participant": "x", "model_code": "m11"}, [], 60)


def test_forces_beyond_eight_columns_are_kept_as_extra_columns():
    layout = archive_wdh.table_layout("hub_loads", [], 10)
    layout.check(10)
    assert list(layout.extras) == [8, 9]


def test_long_blade_frame_matches_the_wide_file():
    stations = list(range(3, 20, 2))
    row = np.concatenate([[0.5], np.arange(54, dtype=float)])
    layout = archive_wdh.Layout(list(range(55)), stations=stations)
    data = read_numeric((" ".join(map(str, row)) + "\n").encode())
    frame = archive_wdh._standard_frame("blade_sections", VULC, data, layout)
    assert len(frame) == 9
    assert frame.fn_n_per_m.tolist() == list(range(9))
    assert frame.cd.tolist() == list(range(45, 54))


# --------------------------------------------------------------------------------------
# Derived and metadata
# --------------------------------------------------------------------------------------


def test_steady_blade_rows_without_a_time_stamp_are_integrated():
    from oc6_phase3 import registry

    stations = registry.BLADE_STATIONS
    frame = pd.DataFrame({
        "participant": "p", "label": "P", "model_code": "m12", "wake_model": "BEM",
        "airfoil_model": "Static polars", "load_case": "1.1", "time_s": np.nan,
        "radius_m": stations.radius_m, "fn_n_per_m": 10.0, "ft_n_per_m": 1.0,
    })
    loads = derive.blade_aero_loads(frame)
    assert len(loads) == 1
    assert loads.fx_from_blade_aero_n.iloc[0] > 0


@pytest.mark.parametrize("column, unit", [
    ("fn_n_per_m", "om:newtonPerMetre"), ("mx_nm", "om:newtonMetre"),
    ("u_ms", "om:metrePerSecond-Time"), ("vorticity_1_s", "om:reciprocalSecond-Time"),
    ("time_s", "om:second-Time"), ("x_mm", "om:millimetre"), ("cl", "om:one"),
    ("platform_displacement", None), ("mx_peak_to_peak_per_amplitude_nm_per_m", None),
])
def test_units_follow_the_column_suffix(column, unit):
    assert croissant.unit_of(column) == unit


def test_experiment2_reduction_recovers_a_known_response():
    from oc6_phase3.build.experiments import experiment2_forces, experiment2_summary

    # LC 2.5: 1 Hz surge, the robot achieving 0.036 m; thrust lagging the motion by 90 deg
    t = np.arange(0, 20, 1 / 2000)
    displacement = 0.036 * np.sin(2 * np.pi * t)
    thrust = 34.0 - 2.0 * np.cos(2 * np.pi * t)
    frame = pd.DataFrame({
        "load_case": "2.5", "test": "synthetic", "time_s": t,
        "platform_displacement": displacement, "ttfx_n": thrust,
        "ttfy_n": 0.0, "ttfz_n": 0.0, "ttmx_nm": 3.2, "ttmy_nm": 0.0, "ttmz_nm": 0.0,
    })
    row = experiment2_summary(frame).iloc[0]
    assert row.fx_mean_n == pytest.approx(34.0, abs=1e-6)
    assert row.fx_peak_to_peak_n == pytest.approx(4.0, rel=1e-3)
    assert row.fx_phase_deg == pytest.approx(90.0, abs=0.1)
    # normalised by the achieved amplitude, not the prescribed 0.035 m
    assert row.fx_peak_to_peak_per_amplitude_n_per_m == pytest.approx(4.0 / 0.036, rel=1e-3)
    averaged = experiment2_forces(frame)
    assert averaged.fx_n.max() - averaged.fx_n.min() == pytest.approx(4.0, rel=0.02)


def test_lowpass_removes_the_cutoff_frequency_itself():
    from oc6_phase3.harmonics import lowpass_fft

    t = np.arange(0, 10, 1 / 2000)
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 3.0 * t)
    filtered = lowpass_fft(signal, 2000.0, 3.0)
    assert np.allclose(filtered, np.sin(2 * np.pi * 1.0 * t), atol=1e-9)


def test_summary_of_a_record_covering_several_periods():
    # CENER m43 submitted eight periods of LC 2.5 (1 Hz) instead of one
    t = np.arange(0, 8.0001, 0.025)
    phase = (360.0 * t + 180.0) % 360.0          # canonical surge phase
    thrust = 35.0 - 2.0 * np.cos(np.radians(phase))
    forces = pd.DataFrame({
        "participant": "p", "label": "P", "model_code": "m43", "wake_model": "FVW",
        "airfoil_model": "Unsteady airfoil aerodynamics", "load_case": "2.5",
        "time_s": t, "motion_phase_deg": phase, "fx_n": thrust, "mx_nm": 3.0,
    })
    row = derive.load_summary(forces).iloc[0]
    assert row.fx_peak_to_peak_n == pytest.approx(4.0, rel=1e-6)
    assert row.fx_phase_deg == pytest.approx(90.0, abs=1e-6)


def test_vortex_track_keeps_the_rising_path_and_drops_a_lower_cluster():
    x = np.linspace(0.7, 1.25, 30)
    track = pd.DataFrame({"x_m": x, "z_m": 1.28 + 0.08 * (x - 0.7)})
    lower = pd.DataFrame({"x_m": np.linspace(1.15, 1.30, 8), "z_m": 1.17})
    kept = derive.vortex_track(pd.concat([track, lower], ignore_index=True))
    assert len(kept) == len(track)
    assert kept.z_m.min() > 1.25


def test_vortex_centre_moves_onto_the_core_of_a_diffuse_vortex():
    from oc6_phase3 import wake

    x = np.round(np.arange(0.61, 1.3601, 0.005), 3)
    z = np.round(np.arange(0.61, 1.4001, 0.005), 3)
    xx, zz = np.meshgrid(x, z)
    dx, dz = xx - 1.0, zz - 1.25
    r2 = dx**2 + dz**2
    core = 0.03                                      # a broad, CFD-like core
    swirl = (1 - np.exp(-r2 / core**2)) / np.maximum(r2, 1e-12)
    u = 3.5 - 0.05 * dz * swirl + 0.4 * (zz - 1.0)   # convected, in a shear layer
    w = 0.05 * dx * swirl
    vorticity = np.exp(-r2 / core**2)
    found = wake.tip_vortex_centres(x, z, u, w, vorticity=vorticity, z_min_m=1.0)
    nearest = found.iloc[((found.x_m - 1.0) ** 2 + (found.z_m - 1.25) ** 2).argmin()]
    assert abs(nearest.x_m - 1.0) < 0.006 and abs(nearest.z_m - 1.25) < 0.006
    # a plane without vorticity leaves the Gamma1 positions as they are
    flat = wake.tip_vortex_centres(x, z, u, w, vorticity=np.zeros_like(u), z_min_m=1.0,
                                   vorticity_percentile=0.0)
    plain = wake.tip_vortex_centres(x, z, u, w, vorticity=np.zeros_like(u), z_min_m=1.0,
                                    vorticity_percentile=0.0, refine_px=0)
    pd.testing.assert_frame_equal(flat, plain)


def test_trajectory_sensitivity_reports_the_default_detector_and_the_range():
    x = np.round(np.arange(0.61, 1.3601, 0.005), 3)
    z = np.round(np.arange(0.61, 1.4001, 0.005), 3)
    xx, zz = np.meshgrid(x, z)
    planes = []
    for azimuth, shift in ((0.0, 0.0), (30.0, 0.03), (60.0, 0.06), (90.0, 0.09)):
        u, w, vorticity = 3.5 + 0.4 * (zz - 1.0), np.zeros_like(xx), np.zeros_like(xx)
        for x0 in (0.75 + shift, 0.9 + shift, 1.05 + shift):
            z0 = 1.25 + 0.05 * (x0 - 1.0)            # a straight, slowly rising track
            dx, dz = xx - x0, zz - z0
            r2 = dx**2 + dz**2
            swirl = (1 - np.exp(-r2 / 0.01**2)) / np.maximum(r2, 1e-12)
            u, w = u - 0.05 * dz * swirl, w + 0.05 * dx * swirl
            vorticity = vorticity + np.exp(-r2 / 0.01**2)
        planes.append(pd.DataFrame({
            "participant": "synthetic", "model_code": "m43", "azimuth_deg": azimuth,
            "x_m": xx.ravel(), "z_m": zz.ravel(), "u_ms": u.ravel(), "w_ms": w.ravel(),
            "vorticity_1_s": vorticity.ravel(),
        }))
    piv = pd.concat(planes, ignore_index=True)
    settings = [dict(threshold=0.8, vorticity_percentile=90.0, refine_px=10),
                dict(threshold=0.9, vorticity_percentile=95.0, refine_px=0)]
    result = derive.trajectory_sensitivity(piv, settings=settings, x_m=(1.0,), min_centres=5)
    row = result.iloc[0]
    assert row.settings_found == 2 and row.settings_tried == 2
    assert row.z_default_m == pytest.approx(1.25, abs=0.006)
    assert row.z_min_m <= row.z_default_m <= row.z_max_m
    # without the default among the settings there is no default height
    other = derive.trajectory_sensitivity(piv, settings=settings[1:], x_m=(1.0,),
                                          min_centres=5)
    assert other.z_default_m.isna().all()


def test_own_output_files_are_read_into_the_campaign_schema(tmp_path):
    from oc6_phase3 import contribute

    # one LC 2.5 surge period as the definition document asks: 0.025 s, surge from 0 m
    # moving upwind, so a quasi-steady thrust peaks at t = 0
    t = np.arange(0, 1.0001, 0.025)
    forces = np.column_stack([t, -0.035 * np.sin(2 * np.pi * t),
                              35.0 + 2.0 * np.cos(2 * np.pi * t), 0 * t, 0 * t,
                              3.0 + 0.5 * np.cos(2 * np.pi * t), 0 * t, 0 * t])
    np.savetxt(tmp_path / "MYCODE_M23_LC25_Forces.txt", forces,
               header="Time Surge Fx Fy Fz Mx My Mz", comments="")
    np.savetxt(tmp_path / "MYCODE_M23_LC11_BA.txt", np.arange(55.0)[None, :])
    hotwire = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(2.18, 5.48, 11),
                               np.full(11, 0.9), np.full(11, 3.5), np.zeros(11),
                               np.zeros(11)])
    np.savetxt(tmp_path / "oc6.phase3.mycode.m23.lc11.hw_aw.txt", hotwire)
    (tmp_path / "notes.md").write_text("not an output file")

    with pytest.warns(UserWarning, match="notes.md"):
        mine = contribute.read_outputs(tmp_path, label="MyCode")
    assert mine.piv is None
    contribute.check(mine.forces, "forces")
    contribute.check(mine.summary, "summary")
    row = mine.summary.set_index("load_case").loc["2.5"]
    assert row.wake_model == "DBEM" and row.label == "MyCode"
    assert row.fx_mean_n == pytest.approx(35.0, abs=1e-3)
    assert row.fx_peak_to_peak_n == pytest.approx(4.0, rel=1e-3)
    assert row.fx_phase_deg == pytest.approx(90.0, abs=0.5)
    blade = mine.blade_aero
    assert len(blade) == 9 and blade.radius_m.notna().all()
    assert blade.fn_n_per_m.tolist() == list(range(1, 10))
    assert mine.hotwire.direction.unique().tolist() == ["aw"]
