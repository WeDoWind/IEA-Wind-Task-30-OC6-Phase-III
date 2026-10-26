"""Wake post-processing: hot-wire deficit metrics and PIV blade-tip vortex tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import ROTOR_RADIUS_M

__all__ = [
    "average_wake_deficit",
    "gamma1",
    "inflection_recovery_point",
    "piv_field",
    "tip_vortex_centres",
]


# --------------------------------------------------------------------------------------
# Hot-wire metrics
# --------------------------------------------------------------------------------------

def average_wake_deficit(y_m, u_ms, free_stream_ms: float,
                         rotor_radius_m: float = ROTOR_RADIUS_M) -> float:
    """Radius-weighted average wake deficit inside the rotor region.

    Equation (4) of the paper, the one-dimensional discrete equivalent of the polar
    integral in Eq. (2)::

        dU_avg = sum(|r_i| * v(r_i)) / sum(|r_i|),      v(r) = u_meas(r) - U0

    evaluated over the crosswind hot-wire points that fall within the rotor radius. The
    radial weighting is what keeps the hub-nose blockage near ``y = 0`` from dominating.
    """
    y = np.asarray(y_m, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    inside = (np.abs(y) <= rotor_radius_m) & np.isfinite(u)
    if not inside.any():
        return float("nan")
    r = np.abs(y[inside])
    v = u[inside] - free_stream_ms
    return float(np.sum(r * v) / np.sum(r))


def inflection_recovery_point(x_m, u_ms) -> float:
    """Along-wind station at which the wake velocity is a minimum and starts to recover.

    Returns the last measured station when the minimum sits at the downstream edge of the
    window, which is how the paper reports ">= 5.48 m" for most CFD participants.
    """
    x = np.asarray(x_m, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    order = np.argsort(x)
    x, u = x[order], u[order]
    finite = np.isfinite(u)
    if finite.sum() < 3:
        return float("nan")
    return float(x[finite][np.argmin(u[finite])])


# --------------------------------------------------------------------------------------
# PIV fields
# --------------------------------------------------------------------------------------

def piv_field(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape tidy PIV rows into ``(x, z, field)`` arrays on the measurement grid."""
    x = np.sort(frame.x_m.unique())
    z = np.sort(frame.z_m.unique())
    field = (
        frame.pivot_table(index="z_m", columns="x_m", values=column, aggfunc="mean")
        .reindex(index=z, columns=x)
        .to_numpy()
    )
    return x, z, field


def gamma1(u: np.ndarray, w: np.ndarray, radius_px: int = 5,
           galilean: bool = True) -> np.ndarray:
    """Graftieaux's scalar ``Gamma1`` on a uniform 2-D grid.

    For every point *P*, ``Gamma1`` averages ``sin(theta_M)`` over a square neighbourhood
    *S*, where ``theta_M`` is the angle between ``PM`` and the velocity at *M*
    (Graftieaux et al., 2001). It approaches +/-1 at a vortex centre, which is what makes
    its local maxima usable as vortex centres (Soto-Valle et al., 2022).

    ``galilean=True`` subtracts the neighbourhood-mean velocity before forming the angle.
    The plain ``Gamma1`` is not Galilean invariant and, in a wake that convects at ~4 m/s,
    the convection speed otherwise swamps the rotational signature.
    """
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    n_z, n_x = u.shape
    offsets = np.arange(-radius_px, radius_px + 1)
    dz_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.hypot(dx_grid, dz_grid)
    valid_offsets = distance > 0

    total = np.zeros_like(u)
    count = np.zeros_like(u)
    if galilean:
        mean_u, mean_w = _box_mean(u, radius_px), _box_mean(w, radius_px)
    else:
        mean_u = mean_w = np.zeros_like(u)

    for dz, dx, dist in zip(dz_grid[valid_offsets], dx_grid[valid_offsets],
                            distance[valid_offsets], strict=True):
        shifted_u = _shift(u, dz, dx) - mean_u
        shifted_w = _shift(w, dz, dx) - mean_w
        speed = np.hypot(shifted_u, shifted_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            contribution = (dx * shifted_w - dz * shifted_u) / (dist * speed)
        finite = np.isfinite(contribution)
        total[finite] += contribution[finite]
        count[finite] += 1
    # Trim the border, where the neighbourhood is incomplete.
    result = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    result[:radius_px, :] = np.nan
    result[-radius_px:, :] = np.nan
    result[:, :radius_px] = np.nan
    result[:, -radius_px:] = np.nan
    return result


def _shift(array: np.ndarray, dz: int, dx: int) -> np.ndarray:
    out = np.full_like(array, np.nan)
    z_src = slice(max(0, dz), array.shape[0] + min(0, dz))
    z_dst = slice(max(0, -dz), array.shape[0] + min(0, -dz))
    x_src = slice(max(0, dx), array.shape[1] + min(0, dx))
    x_dst = slice(max(0, -dx), array.shape[1] + min(0, -dx))
    out[z_dst, x_dst] = array[z_src, x_src]
    return out


def _box_mean(array: np.ndarray, radius_px: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    return uniform_filter(array, size=2 * radius_px + 1, mode="nearest")


def tip_vortex_centres(x_m: np.ndarray, z_m: np.ndarray, u: np.ndarray, w: np.ndarray,
                       vorticity: np.ndarray | None = None,
                       radius_px: int = 5, threshold: float = 0.8,
                       min_separation_px: int = 17, z_min_m: float = 1.15,
                       max_vortices: int = 4,
                       vorticity_percentile: float = 90.0,
                       refine_px: int = 10) -> pd.DataFrame:
    """Locate blade-tip vortex centres in one PIV plane.

    Local maxima of ``|Gamma1|`` above ``threshold`` are returned, ordered by increasing
    ``x`` (that is, by increasing vortex age, since the vortices convect downwind).
    ``z_min_m`` restricts the search to the blade-tip region, keeping the root vortex and
    the hub shear layer out; ``max_vortices`` keeps only the strongest peaks. When the
    ``vorticity`` field is supplied, a candidate must also sit above the
    ``vorticity_percentile`` of the plane - a tip vortex core is a strong vorticity peak,
    while the spurious candidates near the edge of the measurement window are not.

    With ``vorticity``, each centre is then moved to the vorticity peak within
    ``refine_px`` cells of it, where that peak is stronger: Gamma1 marks the vortex, the
    vorticity peak locates its core. Candidates on the edge of the search area, and planes
    without vorticity, are left as Gamma1 found them.
    For the sharp measured vortices the two coincide; for the diffuse vortices of some CFD
    models the Gamma1 maximum sits up to 0.04 m off the core (UNIFI), and the paper's
    Fig. 10 follows the core.

    The defaults reproduce the three tip vortices visible in Fig. 9b of the paper: for the
    measured plane at 30 deg azimuth they return centres at x = 0.665, 1.013 and 1.310 m.
    """
    from scipy.ndimage import maximum_filter

    field = np.abs(gamma1(u, w, radius_px=radius_px))
    filled = np.nan_to_num(field)
    # Keep well clear of the edges: a neighbourhood that is only half inside the window
    # produces a one-sided Gamma1 that can reach 1 without a vortex being there.
    border = 2 * radius_px
    filled[:border, :] = filled[-border:, :] = 0.0
    filled[:, :border] = filled[:, -border:] = 0.0
    above = filled > threshold
    above &= filled >= maximum_filter(filled, size=2 * min_separation_px + 1)
    above &= z_m[:, None] >= z_min_m
    if vorticity is not None:
        magnitude = np.abs(np.nan_to_num(vorticity))
        above &= magnitude >= np.percentile(magnitude, vorticity_percentile)
    rows, cols = np.nonzero(above)
    gamma = filled[rows, cols]
    if vorticity is not None and refine_px:
        last_row, last_col = filled.shape[0] - border - 1, filled.shape[1] - border - 1
        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            # a candidate on the edge of the search area stays where Gamma1 put it, so
            # that it can still be recognised as such; and a centre only moves to
            # vorticity that is actually stronger (some models report none at all)
            if i in (border, last_row) or j in (border, last_col):
                continue
            i0, j0 = max(i - refine_px, 0), max(j - refine_px, 0)
            window = magnitude[i0:i + refine_px + 1, j0:j + refine_px + 1]
            if window.max() <= magnitude[i, j]:
                continue
            di, dj = np.unravel_index(np.argmax(window), window.shape)
            rows[k], cols[k] = i0 + di, j0 + dj
    found = pd.DataFrame({"x_m": x_m[cols], "z_m": z_m[rows], "gamma1": gamma})
    found = found.nlargest(max_vortices, "gamma1")
    found = found.sort_values("x_m", ignore_index=True)
    found["vortex_index"] = np.arange(len(found))
    return found
