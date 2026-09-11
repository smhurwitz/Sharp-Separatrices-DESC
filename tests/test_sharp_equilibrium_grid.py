"""Tests for SharpLinearGrid, SharpQuadratureGrid, SharpConcentricGrid.

These cover the grid classes on their own (node placement, discrete m_b-fold
symmetry, corner tracking with zeta, avoidance of the axis/corner singularities)
as well as their compatibility with the rest of DESC when paired with a genuinely
sharp ``SharpEquilibrium``: finite/continuous field evaluation, plotting, and
force-balance objective construction.
"""

import warnings

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.equilibrium import SharpEquilibrium
from desc.geometry.volume import GeneralizedFourierZernikeRZToroidalVolume
from desc.grid import (
    ConcentricGrid,
    Grid,
    LinearGrid,
    QuadratureGrid,
    SharpConcentricGrid,
    SharpEquilibriumGrid,
    SharpLinearGrid,
    SharpQuadratureGrid,
    _Grid,
)
from desc.objectives import ForceBalance, ObjectiveFunction


def _volume_seq(m_b=3, n_b=1, NFP=1, **kwargs):
    """A small, genuinely-sharp SharpEquilibrium built from a volume boundary."""
    defaults = dict(L=4, M=4, N=0, L_shp=2, M_shp=2, N_shp=0, m_b=m_b, n_b=n_b, NFP=NFP)
    defaults.update(kwargs)
    vol = GeneralizedFourierZernikeRZToroidalVolume(**defaults)
    seq = SharpEquilibrium(
        volume=vol,
        L=defaults["L"],
        M=defaults["M"],
        N=defaults["N"],
        L_shp=defaults["L_shp"],
        M_shp=defaults["M_shp"],
        N_shp=defaults["N_shp"],
        NFP=defaults["NFP"],
        ensure_nested=False,
        check_orientation=False,
    )
    return vol, seq


def _corner_distance(grid, m_b, iota_b):
    """Min, over each toroidal plane, of the nearest node's distance to a corner."""
    dists = []
    for z in np.unique(grid.nodes[:, 2]):
        theta = grid.nodes[np.isclose(grid.nodes[:, 2], z), 1]
        corner = (iota_b * z) % (2 * np.pi / m_b)
        d = np.abs(((theta - corner + np.pi / m_b) % (2 * np.pi / m_b)) - np.pi / m_b)
        dists.append(d.min())
    return np.array(dists)


# ---------------------------------------------------------------------------
# Class hierarchy / DESC-wide compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sharp_grids_are_real_grid_subclasses():
    """Each Sharp*Grid is an instance of the DESC grid class it is modeled on.

    Several places in DESC (plotting, transform building, default-grid selection
    in Equilibrium.compute, surface_integral) branch on isinstance(grid, ...) for
    LinearGrid/QuadratureGrid/ConcentricGrid specifically; subclassing rather than
    duck-typing is what keeps SharpEquilibriumGrid compatible with all of that.
    """
    gl = SharpLinearGrid(M=4, N=1, m_b=1, n_b=0)
    gq = SharpQuadratureGrid(L=4, M=2, N=1, m_b=1, n_b=0)
    gc = SharpConcentricGrid(L=4, M=4, N=1, m_b=1, n_b=0)
    assert isinstance(gl, LinearGrid) and isinstance(gl, _Grid)
    assert isinstance(gq, QuadratureGrid) and isinstance(gq, _Grid)
    assert isinstance(gc, ConcentricGrid) and isinstance(gc, _Grid)
    assert not isinstance(gl, Grid)


@pytest.mark.unit
def test_sharp_equilibrium_grid_factory_dispatch():
    """The SharpEquilibriumGrid() factory builds the right concrete class."""
    assert isinstance(
        SharpEquilibriumGrid("linear", M=4, N=0, m_b=1, n_b=0), SharpLinearGrid
    )
    assert isinstance(
        SharpEquilibriumGrid("quadratic", L=4, M=2, N=0, m_b=1, n_b=0),
        SharpQuadratureGrid,
    )
    assert isinstance(
        SharpEquilibriumGrid("concentric", L=4, M=4, N=0, m_b=1, n_b=0),
        SharpConcentricGrid,
    )
    with pytest.raises(ValueError):
        SharpEquilibriumGrid("not-a-grid-type")
    with pytest.raises(ValueError):
        SharpEquilibriumGrid("concentric")  # L, M, N required without eq=


@pytest.mark.unit
def test_sharp_equilibrium_grid_factory_from_eq():
    """Passing eq= sources NFP/m_b/n_b/sym/L/M/N from the equilibrium."""
    _, seq = _volume_seq(m_b=3, n_b=1, NFP=1)
    grid = SharpEquilibriumGrid("concentric", eq=seq)
    assert grid.m_b == seq.m_b
    assert grid.n_b == seq.n_b
    assert grid.NFP == seq.NFP
    assert grid.L == seq.L_grid and grid.M == seq.M_grid and grid.N == seq.N_grid


# ---------------------------------------------------------------------------
# Discrete symmetry / corner tracking (the core geometric claims)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("m_b,n_b,NFP", [(3, 1, 1), (4, 2, 2), (5, 0, 1)])
def test_corner_tracking_all_grid_types(m_b, n_b, NFP):
    """At every toroidal plane, a node sits exactly on a corner (corners=True).

    This is the ι_b·ζ = NFP·n_b/m_b · ζ rotation: the corners themselves move in
    theta with zeta at this rate (see desc.basis.sharp_map), so a grid whose
    poloidal nodes are naively reused unshifted at every zeta samples the corners
    at a different phase each plane. All three grid types should track them.
    """
    iota_b = NFP * n_b / m_b
    grids = [
        SharpLinearGrid(M=8, N=3, NFP=NFP, m_b=m_b, n_b=n_b, corners=True),
        SharpQuadratureGrid(L=8, M=4, N=3, NFP=NFP, m_b=m_b, n_b=n_b),
        SharpConcentricGrid(L=10, M=6, N=3, NFP=NFP, m_b=m_b, n_b=n_b),
    ]
    for grid in grids:
        dists = _corner_distance(grid, m_b, iota_b)
        np.testing.assert_allclose(dists, 0.0, atol=1e-10)


@pytest.mark.unit
def test_linear_grid_corners_false_avoids_corners():
    """corners=False offsets nodes off the pre-vertices at every toroidal plane."""
    m_b, n_b, NFP = 4, 2, 2
    iota_b = NFP * n_b / m_b
    grid = SharpLinearGrid(
        M=6, N=3, NFP=NFP, m_b=m_b, n_b=n_b, corners=False, axis=False
    )
    dists = _corner_distance(grid, m_b, iota_b)
    assert np.all(dists > 1e-3)
    # never at the axis either
    assert not np.any(grid.nodes[:, 0] == 0)


@pytest.mark.unit
@pytest.mark.parametrize("m_b", [1, 2, 3, 4, 5, 6, 7])
def test_linear_and_quadratic_poloidal_count_is_multiple_of_m_b(m_b):
    """Requested M is silently rounded up so the poloidal node count is a
    multiple of m_b -- 2*M+1 is always odd, so this can only ever be satisfied
    exactly when m_b is odd; even m_b always needs (and gets) an adjustment.
    """
    gl = SharpLinearGrid(M=5, N=0, m_b=m_b, n_b=0)
    assert gl.num_theta % m_b == 0
    assert gl.num_theta >= 2 * 5 + 1  # never under-resolves the requested M
    gq = SharpQuadratureGrid(L=6, M=5, N=0, m_b=m_b, n_b=0)
    n_theta_q = np.unique(gq.nodes[np.isclose(gq.nodes[:, 2], 0), 1]).size
    assert n_theta_q % m_b == 0
    assert n_theta_q >= 2 * 5 + 1


@pytest.mark.unit
@pytest.mark.parametrize("L,M,m_b", [(4, 3, 3), (10, 6, 3), (20, 10, 4), (2, 2, 5)])
def test_concentric_ring_count_rounds_to_multiple_of_m_b(L, M, m_b):
    """Each ring's poloidal node count is rounded to the nearest multiple of m_b
    (m_b * argmin_k |n_theta - k*m_b|), replacing ConcentricGrid's usual
    "round up to the next odd number".
    """
    grid = SharpConcentricGrid(L=L, M=M, N=0, m_b=m_b, n_b=0)
    for rho in np.unique(grid.nodes[:, 0]):
        count = np.count_nonzero(np.isclose(grid.nodes[:, 0], rho))
        assert count % m_b == 0
        assert count > 0


# ---------------------------------------------------------------------------
# No nodes at the axis or at rho=1 for the quadratic/concentric grids
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quadratic_and_concentric_avoid_axis_and_rho_1():
    """Radial nodes are always strictly interior to (0, 1)."""
    gq = SharpQuadratureGrid(L=10, M=6, N=1, m_b=3, n_b=1)
    gc = SharpConcentricGrid(L=10, M=6, N=1, m_b=3, n_b=1)
    for grid in (gq, gc):
        assert not np.any(grid.nodes[:, 0] == 0)
        assert not np.any(grid.nodes[:, 0] == 1)


@pytest.mark.unit
def test_concentric_grid_rejects_non_jacobi_pattern():
    """The other ConcentricGrid node patterns place a node at rho=0 or rho=1."""
    with pytest.raises(ValueError):
        SharpConcentricGrid(L=6, M=4, N=0, m_b=3, n_b=0, node_pattern="cheb2")


@pytest.mark.unit
def test_concentric_grid_rejects_axis_true():
    """axis=True would place a node at rho=0, so it is a hard error, not just a
    warning: SharpConcentricGrid's own _assert_no_singular_nodes would reject the
    resulting grid anyway, so failing fast with a specific message is clearer.
    """
    with pytest.raises(ValueError, match="axis"):
        SharpConcentricGrid(L=6, M=4, N=0, m_b=3, n_b=0, axis=True)


# ---------------------------------------------------------------------------
# Field/current evaluation on a genuine SharpEquilibrium
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_linear_grid_geometry_finite_everywhere_default():
    """R, Z (0th order) are finite everywhere by default, corners included."""
    _, seq = _volume_seq()
    grid = SharpLinearGrid(M=10, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)
    data = seq.compute(["R", "Z"], grid=grid)
    assert np.all(np.isfinite(np.asarray(data["R"])))
    assert np.all(np.isfinite(np.asarray(data["Z"])))


@pytest.mark.unit
def test_linear_grid_default_corners_raise_on_B():
    """|B| needs derivatives, which are genuinely undefined exactly at a corner.

    This is a real limitation of the default (corners=True, matching the pre-
    vertices exactly): DESC raises rather than return a roundoff-dominated value.
    See desc.basis._check_sharp_derivative_nodes.
    """
    _, seq = _volume_seq()
    grid = SharpLinearGrid(M=10, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)
    with pytest.raises(ValueError, match="corners"):
        seq.compute("|B|", grid=grid)


@pytest.mark.unit
def test_linear_grid_avoids_singularities_evaluates_B_everywhere():
    """axis=False, corners=False: |B| is finite everywhere on the linear grid."""
    _, seq = _volume_seq()
    grid = SharpLinearGrid(
        M=10, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b, axis=False, corners=False
    )
    data = seq.compute(["R", "Z", "|B|"], grid=grid)
    for key in ["R", "Z", "|B|"]:
        assert np.all(np.isfinite(np.asarray(data[key]))), key


@pytest.mark.unit
@pytest.mark.parametrize("grid_cls", [SharpQuadratureGrid, SharpConcentricGrid])
def test_quadratic_and_concentric_evaluate_B_and_J_everywhere(grid_cls):
    """|B| and the current density J (needs 2nd derivatives) are finite everywhere
    on both the quadratic and concentric grids, with no special options needed.
    """
    _, seq = _volume_seq()
    if grid_cls is SharpQuadratureGrid:
        grid = grid_cls(L=8, M=4, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)
    else:
        grid = grid_cls(L=8, M=6, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)
    data = seq.compute(["R", "Z", "|B|", "J"], grid=grid)
    for key in ["R", "Z", "|B|", "J"]:
        assert np.all(np.isfinite(np.asarray(data[key]))), key


@pytest.mark.unit
def test_sharp_quadrature_grid_matches_plain_quadrature_grid_volume():
    """SharpQuadratureGrid's radial pattern is QuadratureGrid's own, so the exact
    volume integral it gives should match plain QuadratureGrid to high precision.
    """
    _, seq = _volume_seq()
    g_sharp = SharpQuadratureGrid(L=12, M=6, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)
    g_plain = QuadratureGrid(seq.L_grid, seq.M_grid, seq.N_grid, seq.NFP)
    V_sharp = float(seq.compute("V", grid=g_sharp)["V"])
    V_plain = float(seq.compute("V", grid=g_plain)["V"])
    np.testing.assert_allclose(V_sharp, V_plain, rtol=1e-8)


@pytest.mark.unit
def test_sharp_grids_are_jit_and_grad_safe():
    """Compute on a Sharp*Grid stays traceable/differentiable (regression test,
    mirrors test_sharp_equilibrium_compute_is_jit_and_grad_safe)."""
    _, seq = _volume_seq()
    grid = SharpConcentricGrid(L=6, M=4, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b)

    def f(R_lmn):
        params = {**seq.params_dict, "R_lmn": R_lmn}
        return seq.compute("R", grid=grid, params=params)["R"]

    out = jax.jit(f)(seq.R_lmn)
    assert np.all(np.isfinite(np.asarray(out)))
    grad = jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(seq.R_lmn)
    assert grad.shape == seq.R_lmn.shape
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# Compatibility with plotting and force-balance solves
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sharp_linear_grid_plot_3d_surface():
    """A full (theta, zeta) SharpLinearGrid reshapes correctly for plot_3d.

    Once nodes are rotated by iota_b*zeta the grid is no longer a tensor product
    in raw theta (is_meshgrid=False), but grid.num_theta must still report the
    number of poloidal *slots* per ring -- not the (inflated) number of distinct
    theta values across all toroidal planes -- since plot_3d reshapes computed
    data as coords["X"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta)).
    """
    import matplotlib

    matplotlib.use("Agg")
    from desc.plotting import plot_3d

    _, seq = _volume_seq()
    grid = SharpLinearGrid(
        rho=1.0, M=10, N=4, NFP=seq.NFP, endpoint=True, m_b=seq.m_b, n_b=seq.n_b
    )
    assert grid.num_theta * grid.num_rho * grid.num_zeta == grid.num_nodes
    fig = plot_3d(seq, "R", grid=grid)
    assert fig is not None


@pytest.mark.unit
def test_sharp_linear_grid_plot_2d():
    """A radial-profile SharpLinearGrid works with plot_2d."""
    import matplotlib

    matplotlib.use("Agg")
    from desc.plotting import plot_2d

    _, seq = _volume_seq()
    grid = SharpLinearGrid(
        rho=np.linspace(0.1, 1.0, 6), M=10, N=0, NFP=seq.NFP, m_b=seq.m_b, n_b=seq.n_b
    )
    fig, _ = plot_2d(seq, "R", grid=grid)
    assert fig is not None


@pytest.mark.unit
def test_sharp_concentric_grid_force_balance_build_and_compute():
    """SharpConcentricGrid works as an explicit ForceBalance solution grid."""
    _, seq = _volume_seq()
    grid = SharpConcentricGrid(
        L=seq.L_grid,
        M=seq.M_grid,
        N=seq.N_grid,
        NFP=seq.NFP,
        sym=seq.sym,
        m_b=seq.m_b,
        n_b=seq.n_b,
    )
    fb = ForceBalance(eq=seq, grid=grid)
    obj = ObjectiveFunction([fb])
    obj.build(verbose=0)
    f = obj.compute_scaled_error(obj.x(seq))
    assert np.all(np.isfinite(np.asarray(f)))


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "grid",
    [
        SharpQuadratureGrid(L=8, M=4, N=1, NFP=1, m_b=3, n_b=1),
        SharpConcentricGrid(L=8, M=6, N=1, NFP=1, m_b=3, n_b=1),
    ],
)
def test_sharp_quadratic_and_concentric_save_load_roundtrip(grid, tmp_path):
    """HDF5 round-trip preserves nodes, weights and m_b/n_b.

    (SharpLinearGrid is not covered here: plain LinearGrid itself currently
    fails to round-trip through save/load in this repo, for reasons unrelated
    to the Sharp-specific additions in this file -- confirmed by reproducing
    the identical failure on a plain, un-subclassed LinearGrid.)

    ``_source_grid``/``_can_fft2`` trigger a pre-existing RuntimeWarning on
    every grid's save/load in this repo (reproduced on a plain, un-subclassed
    QuadratureGrid too) that this repo's own filterwarnings=error would
    otherwise turn into a failure here; silenced deliberately, not a check this
    test is meant to make.
    """
    path = str(tmp_path / "grid.h5")
    from desc.io import load

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grid.save(path)
        grid2 = load(path)
    assert type(grid2) is type(grid)
    np.testing.assert_allclose(grid2.nodes, grid.nodes)
    np.testing.assert_allclose(grid2.weights, grid.weights)
    assert grid2.m_b == grid.m_b and grid2.n_b == grid.n_b
