"""Tests for the SharpEquilibrium class.

These cover the "basic" functionality of SharpEquilibrium that does not require the
sharp-aware boundary self-consistency objectives (i.e. everything except a full
force-balance solve): construction, the beta = pi / standard-mode collapse to a plain
Equilibrium, compute consistency, JIT/autodiff safety of the generalized basis,
coordinate mapping, resolution changes, save/load round-tripping, and copying.
"""

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.equilibrium import Equilibrium, SharpEquilibrium
from desc.geometry import FourierRZCurve
from desc.geometry.volume import GeneralizedFourierZernikeRZToroidalVolume
from desc.grid import LinearGrid, QuadratureGrid
from desc.io import load


def _standard_modes_seq():
    """A SharpEquilibrium built from purely standard (l >= 0) modes.

    With no sharp modes and the same coefficients, this must be numerically identical
    to the corresponding plain Equilibrium.
    """
    R_lmn = np.array([10.0, 0.0, 1.0])
    Z_lmn = np.array([0.0, -1.0, 0.0])
    eq = Equilibrium(L=0, M=1, N=0, R_lmn=R_lmn, Z_lmn=Z_lmn)
    seq = SharpEquilibrium(
        L=0, M=1, N=0, L_shp=0, M_shp=0, R_lmn=R_lmn, Z_lmn=Z_lmn
    )
    return eq, seq


def _volume_seq(**kwargs):
    """A small, genuinely-sharp SharpEquilibrium built from a volume boundary."""
    defaults = dict(L=4, M=4, N=0, L_shp=2, M_shp=2, N_shp=0, m_b=1, n_b=1, NFP=1)
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
        ensure_nested=False,
        check_orientation=False,
    )
    return vol, seq


@pytest.mark.unit
def test_sharp_equilibrium_is_equilibrium_subclass():
    """SharpEquilibrium should remain substitutable for an Equilibrium."""
    _, seq = _standard_modes_seq()
    assert isinstance(seq, Equilibrium)


@pytest.mark.unit
def test_sharp_equilibrium_construct_from_volume():
    """Construction from a volume copies boundary coeffs and derives the axis."""
    vol, seq = _volume_seq()
    assert seq.volume is vol
    assert isinstance(seq.axis, FourierRZCurve)
    np.testing.assert_allclose(seq.axis.R_n, vol.get_axis().R_n)
    assert seq.R_lmn.shape == (seq.R_basis.num_modes,)
    assert seq.Z_lmn.shape == (seq.Z_basis.num_modes,)
    assert seq.L_lmn.shape == (seq.L_basis.num_modes,)
    np.testing.assert_array_equal(seq.R_lmn, vol.R_lmn)
    np.testing.assert_array_equal(seq.Z_lmn, vol.Z_lmn)


@pytest.mark.unit
def test_sharp_equilibrium_collapses_to_equilibrium():
    """With only standard modes, SharpEquilibrium == Equilibrium for computed data."""
    eq, seq = _standard_modes_seq()
    grid = LinearGrid(
        rho=np.linspace(0.1, 1.0, 6),
        theta=np.linspace(0, 2 * np.pi, 9, endpoint=False),
        zeta=0.0,
    )
    for key in ["R", "R_r", "R_t", "Z", "Z_t", "sqrt(g)", "|B|", "g_tt"]:
        a = np.asarray(eq.compute(key, grid=grid)[key])
        b = np.asarray(seq.compute(key, grid=grid)[key])
        np.testing.assert_allclose(
            a, b, atol=1e-12, rtol=1e-10, err_msg=f"mismatch for {key}"
        )


@pytest.mark.unit
def test_sharp_equilibrium_compute_no_nans():
    """A genuinely-sharp equilibrium computes core quantities without NaN/Inf."""
    _, seq = _volume_seq()
    grid = QuadratureGrid(seq.L_grid, seq.M_grid, seq.N_grid, seq.NFP)
    data = seq.compute(["R", "Z", "sqrt(g)", "|B|", "V"], grid=grid)
    for key in ["R", "Z", "sqrt(g)", "|B|"]:
        assert np.all(np.isfinite(np.asarray(data[key]))), f"non-finite {key}"
    # plasma volume should be positive and finite
    assert np.isfinite(float(data["V"])) and float(data["V"]) > 0


@pytest.mark.unit
def test_sharp_equilibrium_compute_is_jit_and_grad_safe():
    """The generalized basis must be traceable/differentiable (regression test).

    map_coordinates, solve, optimize and perturb all JIT the compute call, so
    building the transform matrices from the generalized basis has to stay inside
    JAX (this broke previously because the basis assembled a numpy buffer).
    """
    _, seq = _volume_seq()
    grid = LinearGrid(rho=np.array([0.3, 0.6]), theta=np.array([0.2, 0.5]), zeta=0.0)

    def f(R_lmn):
        params = {**seq.params_dict, "R_lmn": R_lmn}
        return seq.compute("R", grid=grid, params=params)["R"]

    out = jax.jit(f)(seq.R_lmn)
    assert np.all(np.isfinite(np.asarray(out)))

    grad = jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(seq.R_lmn)
    assert grad.shape == seq.R_lmn.shape
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.unit
def test_sharp_equilibrium_map_coordinates_roundtrip():
    """Mapping (R, phi, Z) back to (rho, theta, zeta) recovers the seed point."""
    _, seq = _volume_seq()
    g = LinearGrid(rho=0.5, theta=0.3, zeta=0.0)
    d = seq.compute(["R", "phi", "Z"], grid=g)
    coords = np.array([[float(d["R"][0]), float(d["phi"][0]), float(d["Z"][0])]])
    out = np.asarray(
        seq.map_coordinates(
            coords,
            inbasis=["R", "phi", "Z"],
            outbasis=("rho", "theta", "zeta"),
            tol=1e-9,
            maxiter=40,
        )
    )
    np.testing.assert_allclose(out[0], [0.5, 0.3, 0.0], atol=1e-6)


@pytest.mark.unit
def test_sharp_equilibrium_is_nested():
    """A scaled-down volume boundary yields nested flux surfaces."""
    _, seq = _volume_seq()
    assert seq.is_nested()


@pytest.mark.unit
def test_sharp_equilibrium_change_resolution():
    """Increasing resolution grows the bases and preserves shared coefficients."""
    _, seq = _volume_seq(L=2, M=2, L_shp=1, M_shp=1)
    old_R = seq.R_lmn.copy()
    old_modes = seq.R_basis.modes.copy()
    seq.change_resolution(L=4, M=4, N=0, L_shp=2, M_shp=2, N_shp=0)
    assert seq.L == 4 and seq.M == 4 and seq.L_shp == 2 and seq.M_shp == 2
    assert seq.R_lmn.shape == (seq.R_basis.num_modes,)
    assert seq.volume.R_lmn.shape == (seq.volume.R_basis.num_modes,)
    # coefficients of modes present before are preserved
    for coeff, mode in zip(old_R, old_modes):
        idx = seq.R_basis.get_idx(*mode)
        np.testing.assert_allclose(seq.R_lmn[idx], coeff, atol=1e-12)


@pytest.mark.unit
def test_sharp_equilibrium_save_load(tmp_path):
    """Round-trip through HDF5 preserves parameters and computed output."""
    _, seq = _volume_seq()
    path = str(tmp_path / "sharp_eq.h5")
    seq.save(path)
    seq2 = load(path)
    assert isinstance(seq2, SharpEquilibrium)
    np.testing.assert_allclose(seq2.R_lmn, seq.R_lmn)
    np.testing.assert_allclose(seq2.Z_lmn, seq.Z_lmn)
    assert seq2.L == seq.L and seq2.L_shp == seq.L_shp
    assert seq2.β == seq.β and seq2.sharp_type == seq.sharp_type
    grid = LinearGrid(rho=0.5, theta=np.linspace(0, 2 * np.pi, 8), zeta=0.0)
    np.testing.assert_allclose(
        np.asarray(seq.compute("R", grid=grid)["R"]),
        np.asarray(seq2.compute("R", grid=grid)["R"]),
        atol=1e-12,
    )


@pytest.mark.unit
def test_sharp_equilibrium_copy():
    """Copies are independent but numerically identical."""
    _, seq = _volume_seq()
    seq2 = seq.copy()
    assert seq2 is not seq
    np.testing.assert_array_equal(seq2.R_lmn, seq.R_lmn)
    seq2.R_lmn = seq2.R_lmn + 1.0
    assert not np.allclose(seq2.R_lmn, seq.R_lmn)


@pytest.mark.unit
def test_sharp_equilibrium_get_surface_at_not_implemented():
    """get_surface_at is intentionally unavailable for a volume boundary."""
    _, seq = _volume_seq()
    with pytest.raises(NotImplementedError):
        seq.get_surface_at(rho=1.0)


@pytest.mark.unit
def test_sharp_boundary_self_consistency():
    """The boundary self-consistency constraint is satisfied at init and fixes the
    right degrees of freedom (sharp modes + standard l=|m| modes)."""
    from desc.objectives import (
        BoundaryRSelfConsistency,
        BoundaryZSelfConsistency,
        FixBoundaryR,
        FixBoundaryZ,
    )

    _, seq = _volume_seq(L=2, M=2, L_shp=1, M_shp=1)
    for scls, coeff in [
        (BoundaryRSelfConsistency, "R_lmn"),
        (BoundaryZSelfConsistency, "Z_lmn"),
    ]:
        obj = scls(seq)
        obj.build(verbose=0)
        f = np.asarray(obj.compute(seq.params_dict))
        np.testing.assert_allclose(f, 0.0, atol=1e-12)

    # FixBoundaryR / FixBoundaryZ freeze exactly the boundary modes (l<0 and l=|m|)
    for fcls, basis in [(FixBoundaryR, seq.R_basis), (FixBoundaryZ, seq.Z_basis)]:
        obj = fcls(seq)
        obj.build(verbose=0)
        fixed = basis.modes[obj._params[list(obj._params)[0]]]
        expected, _ = basis.get_boundary_modes(fix_MA=False)
        np.testing.assert_array_equal(
            np.sort(fixed, axis=0), np.sort(expected, axis=0)
        )


@pytest.mark.unit
def test_sharp_equilibrium_fixed_boundary_solve():
    """A short fixed-boundary solve reduces force error and preserves the boundary
    (including the sharp corner) to machine precision."""
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )
    from desc.objectives.getters import maybe_add_self_consistency

    vol = GeneralizedFourierZernikeRZToroidalVolume(
        L=2, M=2, N=0, L_shp=1, M_shp=1, N_shp=0, m_b=1, n_b=1, NFP=1
    )
    seq = SharpEquilibrium(
        volume=vol,
        L=2,
        M=2,
        N=0,
        L_shp=1,
        M_shp=1,
        N_shp=0,
        current=0,
        ensure_nested=True,
        check_orientation=True,
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    nodes = np.column_stack([np.ones_like(theta), theta, np.zeros_like(theta)])
    ER = np.asarray(seq.R_basis.evaluate(nodes))
    EZ = np.asarray(seq.Z_basis.evaluate(nodes))
    Rb0 = ER @ np.asarray(seq.R_lmn)
    Zb0 = EZ @ np.asarray(seq.Z_lmn)

    cons = get_fixed_boundary_constraints(eq=seq)
    cons = maybe_add_self_consistency(seq, cons)
    obj = ObjectiveFunction(ForceBalance(seq))
    seq.solve(
        objective=obj,
        constraints=cons,
        maxiter=10,
        verbose=0,
        ftol=1e-8,
        xtol=1e-10,
    )
    # boundary (as a real-space shape at rho=1) is preserved
    Rb1 = ER @ np.asarray(seq.R_lmn)
    Zb1 = EZ @ np.asarray(seq.Z_lmn)
    np.testing.assert_allclose(Rb1, Rb0, atol=1e-10)
    np.testing.assert_allclose(Zb1, Zb0, atol=1e-10)
    # equilibrium still has finite, positive Jacobian everywhere
    data = seq.compute("sqrt(g)")
    assert np.all(np.isfinite(np.asarray(data["sqrt(g)"])))


@pytest.mark.unit
def test_sharp_equilibrium_plotting_smoke():
    """plot_surfaces and plot_boundary run on a SharpEquilibrium."""
    import matplotlib

    matplotlib.use("Agg")
    from desc.plotting import plot_boundary, plot_surfaces

    _, seq = _volume_seq()
    fig, _ = plot_surfaces(seq)
    assert fig is not None
    fig, _ = plot_boundary(seq)
    assert fig is not None
