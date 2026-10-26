"""Tests for the parts that would silently give wrong physics if they broke.

These need no data files: they check the conventions and the maths against analytic cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oc6_phase3 import harmonics, registry, wake

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

def test_model_codes_decode():
    assert registry.decode_model_code("m12") == ("BEM", "Static polars")
    assert registry.decode_model_code("m43") == ("FVW", "Unsteady airfoil aerodynamics")
    assert registry.decode_model_code("m54") == ("CFD", "Blade resolved")


def test_every_registered_model_decodes():
    for _, model_code in registry.CODES:
        registry.decode_model_code(model_code)


def test_reduced_frequencies_match_the_paper():
    """Paper Sect. 4: the three motion frequencies give k = 0.071, 0.568 and 1.137."""
    cases = registry.load_cases()
    k = cases.groupby("frequency_hz").reduced_frequency.first()
    assert k[0.125] == pytest.approx(0.071, abs=5e-4)
    assert k[1.0] == pytest.approx(0.568, abs=5e-4)
    assert k[2.0] == pytest.approx(1.137, abs=5e-4)


def test_pitch_cases_are_equivalent_to_their_surge_counterparts():
    """LC 3.5 and 3.7 were prescribed to match the apparent wind of LC 2.5 and 2.7."""
    cases = registry.load_cases()
    equivalent = cases.surge_equivalent_amplitude_m
    assert equivalent["3.5"] == pytest.approx(equivalent["2.5"], rel=0.05)
    assert equivalent["3.7"] == pytest.approx(equivalent["2.7"], rel=0.05)


def test_blade_stations_span_the_documented_range():
    stations = registry.BLADE_STATIONS
    assert list(stations.station) == [3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert stations.span_pct.iloc[0] == pytest.approx(18.9)
    assert stations.span_pct.iloc[-1] == pytest.approx(98.3)
    # radius = hub radius + distance along the pitch axis, and the tip is at 98.3 % span
    ratio = stations.radius_m.iloc[-1] / registry.ROTOR_RADIUS_M
    assert ratio == pytest.approx(0.983, abs=1e-3)


# --------------------------------------------------------------------------------------
# Motion phase convention
# --------------------------------------------------------------------------------------

def test_surge_and_pitch_share_one_canonical_phase():
    """The definition document prescribes opposite starting directions for the two motion
    types, so the canonical phase must offset surge by 180 deg."""
    time = np.linspace(0, 1, 41)
    surge = harmonics.canonical_phase_deg(time, 1.0, "surge")
    pitch = harmonics.canonical_phase_deg(time, 1.0, "pitch")
    assert surge[0] == pytest.approx(180.0)
    assert pitch[0] == pytest.approx(0.0)
    assert np.allclose((surge - pitch) % 360, 180.0)


def test_steady_load_cases_have_no_phase():
    assert np.isnan(harmonics.canonical_phase_deg([0.0, 1.0], 0.0, "none")).all()


def test_quasi_steady_response_gives_a_90_degree_phase_shift():
    """Thrust follows the apparent wind, U0 - x_dot, so it lags the displacement by 90."""
    phase = np.linspace(0, 360, 73)[:-1]
    thrust = 35.0 - 2.3 * np.cos(np.radians(phase))
    assert harmonics.phase_shift_deg(thrust, phase) == pytest.approx(90.0, abs=0.1)
    assert harmonics.peak_to_peak(thrust) == pytest.approx(4.6, rel=1e-6)


def test_unsteady_airfoil_hysteresis_pushes_the_phase_past_90():
    phase = np.linspace(0, 360, 73)[:-1]
    lagged = 35.0 - 2.3 * np.cos(np.radians(phase - 5.0))
    assert harmonics.phase_shift_deg(lagged, phase) == pytest.approx(95.0, abs=0.1)


def test_phase_recovered_from_a_measured_displacement():
    sample_rate, frequency = 2000.0, 1.0
    time = np.arange(0, 10, 1 / sample_rate)
    displacement = 0.035 * np.sin(2 * np.pi * frequency * time + 0.7)
    phase = harmonics.canonical_phase_from_measurement(
        displacement, frequency, sample_rate
    )
    # the canonical phase is the argument of that sine
    assert np.allclose(0.035 * np.sin(np.radians(phase)), displacement, atol=1e-6)


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------

def test_lowpass_removes_the_1p_response_without_a_phase_lag():
    """The measured records carry a 4 Hz blade-passing response from a rotor mass
    imbalance; the 3 Hz filter must remove it and leave the 1 Hz motion untouched."""
    sample_rate = 2000.0
    time = np.arange(0, 10, 1 / sample_rate)
    motion = 2.3 * np.sin(2 * np.pi * 1.0 * time)
    imbalance = 1.5 * np.sin(2 * np.pi * 4.0 * time + 1.1)
    filtered = harmonics.lowpass_fft(motion + imbalance, sample_rate, 3.0)
    assert np.allclose(filtered, motion, atol=1e-8)


def test_phase_average_recovers_the_underlying_cycle():
    phase = np.random.default_rng(0).uniform(0, 360, 200_000)
    signal = 35.0 - 2.3 * np.cos(np.radians(phase)) + np.random.default_rng(1).normal(
        0, 0.5, phase.size
    )
    centres, averaged = harmonics.phase_average(phase, signal, n_bins=72)
    expected = 35.0 - 2.3 * np.cos(np.radians(centres))
    assert np.allclose(averaged, expected, atol=0.05)


# --------------------------------------------------------------------------------------
# Wake
# --------------------------------------------------------------------------------------

def test_average_wake_deficit_weights_by_radius():
    """Equation (4): a uniform deficit inside the rotor returns that deficit, and points
    outside the rotor radius are ignored."""
    y = np.linspace(-2.0, 2.0, 41)
    u = np.where(np.abs(y) <= registry.ROTOR_RADIUS_M, 3.0, 4.19)
    assert wake.average_wake_deficit(y, u, 4.19) == pytest.approx(-1.19, abs=1e-9)


def test_average_wake_deficit_discounts_the_hub_centre():
    """The radial weighting is what keeps the unmodelled hub nose from mattering.

    Most models return a wind speed slightly above the free stream directly behind the hub
    because they do not model the hub nose, while the experiment shows a deficit there.
    Equation (4) weights each point by its radius, so that disagreement is worth almost
    nothing - an order of magnitude less than it would be in a plain mean.
    """
    y = np.linspace(-1.0, 1.0, 21)
    base = np.full_like(y, 3.0)
    with_hub_error = base.copy()
    with_hub_error[np.abs(y) < 0.11] = 4.5  # a large error right at the centre

    weighted_shift = abs(
        wake.average_wake_deficit(y, with_hub_error, 4.19)
        - wake.average_wake_deficit(y, base, 4.19)
    )
    unweighted_shift = abs((with_hub_error - base).mean())
    assert weighted_shift < unweighted_shift / 7


def test_gamma1_finds_the_centre_of_a_synthetic_vortex():
    """A Rankine vortex convecting downwind: Gamma1 must peak at its centre, and must do
    so only when the convection velocity is removed."""
    grid = np.linspace(-0.2, 0.2, 81)
    x, z = np.meshgrid(grid, grid, indexing="xy")
    radius = np.hypot(x, z)
    core, circulation, convection = 0.02, 1.0, 4.0
    speed = np.where(
        radius < core,
        circulation * radius / (2 * np.pi * core**2),
        circulation / (2 * np.pi * np.maximum(radius, 1e-9)),
    )
    with np.errstate(invalid="ignore"):
        u = convection - speed * z / np.maximum(radius, 1e-9)
        w = speed * x / np.maximum(radius, 1e-9)

    field = np.abs(wake.gamma1(u, w, radius_px=5, galilean=True))
    peak = np.unravel_index(np.nanargmax(field), field.shape)
    assert np.hypot(x[peak], z[peak]) < 0.01
    assert np.nanmax(field) > 0.9

    convected = np.abs(wake.gamma1(u, w, radius_px=5, galilean=False))
    assert np.nanmax(convected) < np.nanmax(field)


def test_inflection_recovery_point():
    x = np.array([2.18, 2.51, 2.84, 3.17, 3.50])
    u = np.array([2.0, 1.8, 1.7, 1.75, 1.9])
    assert wake.inflection_recovery_point(x, u) == pytest.approx(2.84)


# --------------------------------------------------------------------------------------
# Tower-top to hub projection (R. Bergua, pers. comm., 2026)
# --------------------------------------------------------------------------------------

def test_hub_projection_is_identity_without_tilt_or_offset():
    from oc6_phase3.build.experiments import tower_top_to_hub

    loads = (35.0, 2.0, -3.0, 3.3, -1.7, 0.04)
    assert tower_top_to_hub(*loads, tilt_deg=0.0, offset_x_m=0.0,
                            offset_z_m=0.0) == pytest.approx(loads)


def test_hub_projection_straightens_a_shaft_aligned_force():
    """A thrust along the (horizontal) shaft reads partly vertical in the tilted frame.

    Rotating must put all of it back into x and leave nothing in z.
    """
    from oc6_phase3.build.experiments import tower_top_to_hub

    thrust, tilt = 36.0, registry.TILT_ANGLE_DEG
    angle = np.radians(tilt)
    fx_t, fz_t = thrust * np.cos(angle), -thrust * np.sin(angle)
    fx, _, fz, *_ = tower_top_to_hub(fx_t, 0.0, fz_t, 0.0, 0.0, 0.0, tilt_deg=tilt)
    assert fx == pytest.approx(thrust)
    assert fz == pytest.approx(0.0, abs=1e-12)


def test_hub_projection_moment_translation():
    """Mx picks up Fy * Z; with no lateral force the torque only rotates."""
    from oc6_phase3.build.experiments import tower_top_to_hub

    _, _, _, mx, _, _ = tower_top_to_hub(0.0, 4.0, 0.0, 3.3, 0.0, 0.0, tilt_deg=0.0)
    assert mx == pytest.approx(3.3 + 4.0 * registry.HUB_OFFSET_Z_M)


# --------------------------------------------------------------------------------------
# Bringing your own results
# --------------------------------------------------------------------------------------

def test_contributed_model_lands_in_the_packaged_schema():
    from oc6_phase3 import contribute

    case = registry.load_cases().loc["2.5"]
    time_s = np.linspace(0.0, float(case.period_s), 41)
    phase = np.radians(harmonics.canonical_phase_deg(time_s, case.frequency_hz, "surge"))
    results = contribute.model_results(
        label="Test", model_code="m43", load_case="2.5", time_s=time_s,
        fx_n=35.0 - 2.3 * np.cos(phase), mx_nm=3.0 - 0.5 * np.cos(phase),
    )
    contribute.check(results.forces, "forces")
    contribute.check(results.summary, "summary")

    row = results.summary.iloc[0]
    assert row.wake_model == "FVW"                       # decoded from m43
    assert row.airfoil_model == "Unsteady airfoil aerodynamics"
    assert row.fx_mean_n == pytest.approx(35.0, abs=0.01)
    assert row.fx_peak_to_peak_n == pytest.approx(4.6, abs=0.01)
    assert row.fx_phase_deg == pytest.approx(90.0, abs=0.2)   # quasi-steady by construction


def test_multi_period_record_is_folded_onto_one_period():
    from oc6_phase3 import contribute

    case = registry.load_cases().loc["2.5"]
    time_s = np.arange(0.0, 5 * float(case.period_s), 0.005)
    phase = np.radians(harmonics.canonical_phase_deg(time_s, case.frequency_hz, "surge"))
    results = contribute.model_results(
        label="Long", model_code="m12", load_case="2.5", time_s=time_s,
        fx_n=35.0 - 2.3 * np.cos(phase),
    )
    assert len(results.forces) <= 72
    assert results.summary.fx_peak_to_peak_n.iloc[0] == pytest.approx(4.6, abs=0.05)


def test_check_names_the_missing_columns():
    from oc6_phase3 import contribute

    with pytest.raises(ValueError, match="wake_model"):
        contribute.check(pd.DataFrame({"label": ["x"]}), "summary")
