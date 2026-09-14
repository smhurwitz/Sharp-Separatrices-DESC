"""Tests for the X-line tangency constraint on lambda for SharpEquilibrium.

The m_b corners of a sharp boundary are magnetic X-lines and must be field lines,
which (with iota(1) = iota_b = NFP*n_b/m_b) requires lambda to be constant along each
of them. These tests cover the constraint matrix (``xline_constraint_matrix``), the
linear objective (``FixXLine``), its automatic inclusion via
``maybe_add_self_consistency``, and the guarantee that a SharpEquilibrium respects
the condition as soon as it is initialized.
"""

import warnings

import numpy as np
import pytest

from desc.basis import (
    FourierZernikeBasis,
    GeneralizedFourierZernikeBasis,
    SharpFourierZernikeBasis,
    fourier,
)
from desc.equilibrium import SharpEquilibrium
from desc.geometry.volume import GeneralizedFourierZernikeRZToroidalVolume
from desc.objectives import FixLambdaGauge, FixXLine, xline_constraint_matrix
from desc.objectives.getters import maybe_add_self_consistency
from desc.profiles import PowerSeriesProfile


def _lambda_basis(M, N, NFP, m_b, n_b, sym=False, L_shp=2, M_shp=2):
    std = FourierZernikeBasis(L=M, M=M, N=N, NFP=NFP, sym=sym)
    shp = SharpFourierZernikeBasis(
        L=L_shp,
        M=M_shp,
        N=N,
        NFP=NFP,
        m_b=m_b,
        n_b=n_b,
        β=0.75 * np.pi,
        sharp_type="lens",
        sym=sym,
        quasiconformal=True,
    )
    return GeneralizedFourierZernikeBasis(std, shp)


def _lambda_on_ridges(basis, L_lmn, nz=97):
    """lambda(1, alpha_k + iota_b*zeta, zeta) for each ridge k, shape(m_b, nz)."""
    m_b, n_b, NFP = basis.shp_basis.m_b, basis.shp_basis.n_b, basis.NFP
    iota_b = NFP * n_b / m_b
    zeta = np.linspace(0, 2 * np.pi * m_b / NFP, nz, endpoint=False) + 0.0371
    out = []
    for k in range(m_b):
        nodes = np.column_stack(
            [np.ones_like(zeta), 2 * np.pi * k / m_b + iota_b * zeta, zeta]
        )
        with warnings.catch_warnings():  # rho=1 is the corner; values are fine there
            warnings.simplefilter("ignore")
            out.append(np.asarray(basis.evaluate(nodes)) @ L_lmn)
    return np.array(out)


def _xline_matrix_nodes(basis, oversample=1.0):
    """Independent node-sampled construction of the same constraints.

    Evaluates the along-ridge derivative iota_b*f_m'(theta)*g_n(zeta) +
    f_m(theta)*g_n'(zeta) of each mode's ridge trace at 2*(M*n_b + N*m_b) + 1 nodes
    per ridge (the aliasing minimum, over one ridge period 2*pi*m_b/NFP), then
    SVD-compresses. Cross-checks the closed-form rows in ``xline_constraint_matrix``.
    """
    modes = np.asarray(basis.modes)
    m_b, n_b, NFP = basis.shp_basis.m_b, basis.shp_basis.n_b, basis.NFP
    iota_b = NFP * n_b / m_b
    M, N = np.abs(modes[:, 1]).max(), np.abs(modes[:, 2]).max()
    nz = int(np.ceil(oversample * (2 * (M * n_b + N * m_b) + 1)))
    zeta = np.arange(nz) * 2 * np.pi * m_b / (NFP * nz)
    m, n = modes[:, 1], modes[:, 2]
    rows = []
    for k in range(m_b):
        th = (2 * np.pi * k / m_b + iota_b * zeta)[:, None]
        ze = zeta[:, None]
        rows.append(
            iota_b
            * np.asarray(fourier(th, m, 1, 1))
            * np.asarray(fourier(ze, n, NFP, 0))
            + np.asarray(fourier(th, m, 1, 0)) * np.asarray(fourier(ze, n, NFP, 1))
        )
    D = np.vstack(rows)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[: int(np.sum(sv > 1e-10 * sv[0]))]


@pytest.mark.unit
@pytest.mark.parametrize(
    "M, N, NFP, m_b, n_b, sym, rank",
    [
        (7, 12, 5, 5, 1, False, 134),
        (5, 6, 5, 5, 1, False, 70),
        (4, 4, 3, 3, 1, False, 32),
        (4, 6, 2, 4, 2, False, 60),
        (6, 6, 2, 4, 2, "sin", 34),
    ],
)
def test_xline_matrix_rank_and_rowspace(M, N, NFP, m_b, n_b, sym, rank):
    """Rank and row space of the closed-form constraint matrix.

    The rank matches the independent node-sampled construction (and the derivation
    notes' table), rows are orthonormal, and the two row spaces coincide.
    """
    basis = _lambda_basis(M, N, NFP, m_b, n_b, sym=sym)
    A = xline_constraint_matrix(basis)
    S = _xline_matrix_nodes(basis)
    assert A.shape == (rank, basis.num_modes)
    assert S.shape[0] == rank
    np.testing.assert_allclose(A @ A.T, np.eye(rank), atol=1e-12)
    I = np.eye(basis.num_modes)
    assert np.abs(A @ (I - S.T @ S)).max() < 1e-10
    assert np.abs(S @ (I - A.T @ A)).max() < 1e-10


@pytest.mark.unit
@pytest.mark.parametrize(
    "M, N, NFP, m_b, n_b, sym",
    [(4, 4, 3, 3, 1, False), (4, 6, 2, 4, 2, False), (6, 6, 1, 3, 3, "sin")],
)
def test_xline_projection_makes_lambda_constant_on_ridges(M, N, NFP, m_b, n_b, sym):
    """Projection onto the constraint null space makes lambda constant on ridges.

    Lambda is evaluated through the actual generalized basis, sharp modes included,
    and the resonant (m*n_b == n*m_b) helicities are left untouched.
    """
    basis = _lambda_basis(M, N, NFP, m_b, n_b, sym=sym)
    A = xline_constraint_matrix(basis)
    x = np.random.default_rng(0).standard_normal(basis.num_modes)
    xp = x - A.T @ (A @ x)

    lam = _lambda_on_ridges(basis, xp)
    lam0 = _lambda_on_ridges(basis, x)
    assert np.ptp(lam0, axis=1).min() > 1.0  # the unprojected one really does vary
    assert np.ptp(lam, axis=1).max() < 1e-12

    # the m = n = 0 (gauge) modes are untouched ...
    modes = np.asarray(basis.modes)
    gauge = (modes[:, 1] == 0) & (modes[:, 2] == 0)
    if gauge.any():  # absent with stellarator symmetry
        assert np.abs(A[:, gauge]).max() < 1e-13
        np.testing.assert_allclose(xp[gauge], x[gauge], atol=1e-13)
    # ... and so are the resonant helicities m*n_b == n*m_b. In the real product
    # basis a single mode f_m(theta) g_n(zeta) carries both helicities
    # |m|*iota_b +- |n|*NFP, so what is preserved is the resonant combination
    # cos(m theta - n NFP zeta) = cos cos + sin sin (and sin cos - cos sin).
    idx = {tuple(mode): i for i, mode in enumerate(modes)}
    checked = 0
    for (l, m, n), i in idx.items():
        if m <= 0 or n == 0 or m * n_b != abs(n) * m_b:
            continue
        # partner of (m, n) is (-m, -n): cos*cos + sin*sin or cos*sin - sin*cos
        j = idx.get((l, -m, -n))
        if j is None:
            continue
        sgn = 1 if n > 0 else -1
        np.testing.assert_allclose(xp[i] + sgn * xp[j], x[i] + sgn * x[j], atol=1e-12)
        checked += 1
    assert checked > 0


def _seq(m_b=2, n_b=1, NFP=1, N=0, iota=True, **kwargs):
    vol = GeneralizedFourierZernikeRZToroidalVolume(
        L=4, M=4, N=N, L_shp=2, M_shp=2, N_shp=N, m_b=m_b, n_b=n_b, NFP=NFP
    )
    iota_b = NFP * n_b / m_b
    if iota is True:  # a profile consistent with the boundary, iota(1) = iota_b
        profiles = dict(iota=PowerSeriesProfile([iota_b / 2, 0, iota_b / 2]))
    else:
        profiles = dict(iota=iota)
    return SharpEquilibrium(
        volume=vol,
        L=4,
        M=4,
        N=N,
        L_shp=2,
        M_shp=2,
        N_shp=N,
        ensure_nested=False,
        check_orientation=False,
        **profiles,
        **kwargs,
    )


@pytest.mark.unit
def test_sharp_equilibrium_default_init_respects_xline():
    """The default initial guess (lambda = 0) satisfies the X-line condition.

    Nothing is projected and no warning is raised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seq = _seq()
    assert not np.any(np.asarray(seq.L_lmn))
    assert seq.xline_lambda_error().size > 0
    np.testing.assert_allclose(seq.xline_lambda_error(), 0.0)
    assert seq.project_lambda_xline() == 0.0
    assert seq.iota_b == 0.5


@pytest.mark.unit
def test_sharp_equilibrium_supplied_lambda_is_projected():
    """A user-supplied L_lmn that varies along the X-lines is projected at init.

    A warning is raised, and copying lambda from another equilibrium through
    ``set_initial_guess`` goes through the same projection.
    """
    ref = _seq()
    rng = np.random.default_rng(3)
    L_random = rng.standard_normal(ref.L_basis.num_modes)
    A = xline_constraint_matrix(ref.L_basis)
    assert np.linalg.norm(A @ L_random) > 1.0

    with pytest.warns(UserWarning, match="not constant along the X-lines"):
        seq = _seq(R_lmn=ref.R_lmn, Z_lmn=ref.Z_lmn, L_lmn=L_random)
    np.testing.assert_allclose(seq.xline_lambda_error(), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(seq.L_lmn), L_random - A.T @ (A @ L_random), atol=1e-12
    )
    # (the default volume uses the conformal lens map, whose (1 +- z)^gamma
    # evaluation near the corner carries ~1e-11 of roundoff)
    lam = _lambda_on_ridges(seq.L_basis, np.asarray(seq.L_lmn))
    assert np.ptp(lam, axis=1).max() < 1e-9

    # copying the flux surfaces (and lambda) from another equilibrium goes through
    # the same projection
    other = _seq()
    other._L_lmn = L_random  # bypass the projection to plant a bad lambda
    seq2 = _seq()
    with pytest.warns(UserWarning, match="not constant along the X-lines"):
        seq2.set_initial_guess(other, ensure_nested=False)
    np.testing.assert_allclose(seq2.xline_lambda_error(), 0.0, atol=1e-12)


@pytest.mark.unit
def test_sharp_equilibrium_change_resolution_keeps_xline():
    """Resolution changes keep lambda constant along the X-lines.

    Raising resolution preserves the ridge trace exactly (no projection needed);
    lowering it can change the trace, and the result is projected back.
    """
    seq = _seq()
    rng = np.random.default_rng(5)
    x = rng.standard_normal(seq.L_basis.num_modes)
    A = xline_constraint_matrix(seq.L_basis)
    seq.L_lmn = x - A.T @ (A @ x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seq.change_resolution(L=6, M=6)
    np.testing.assert_allclose(seq.xline_lambda_error(), 0.0, atol=1e-12)
    with pytest.warns(UserWarning):  # DESC's usual "reducing L" warning
        seq.change_resolution(L=2, M=2)
    np.testing.assert_allclose(seq.xline_lambda_error(), 0.0, atol=1e-12)


@pytest.mark.unit
def test_fix_xline_auto_added_and_satisfied_after_solve():
    """``FixXLine`` is added automatically whenever L_lmn is free.

    A fixed-boundary solve then leaves lambda constant along the X-lines.
    """
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )

    seq = _seq()
    cons = maybe_add_self_consistency(seq, get_fixed_boundary_constraints(seq))
    assert any(isinstance(c, FixXLine) for c in cons)
    assert any(isinstance(c, FixLambdaGauge) for c in cons)
    # not duplicated if the user already supplied one
    cons2 = maybe_add_self_consistency(seq, cons)
    assert sum(isinstance(c, FixXLine) for c in cons2) == 1

    con = FixXLine(seq)
    con.build(verbose=0)
    assert con.dim_f == xline_constraint_matrix(seq.L_basis).shape[0]
    np.testing.assert_allclose(con.compute(seq.params_dict), 0.0)

    # default constraints in solve() go through maybe_add_self_consistency too
    seq.solve(
        objective=ObjectiveFunction(ForceBalance(seq)),
        maxiter=3,
        verbose=0,
        ftol=1e-8,
        xtol=1e-10,
    )
    assert np.linalg.norm(np.asarray(seq.L_lmn)) > 0  # lambda actually moved
    np.testing.assert_allclose(seq.xline_lambda_error(), 0.0, atol=1e-10)
    lam = _lambda_on_ridges(seq.L_basis, np.asarray(seq.L_lmn))
    assert np.ptp(lam, axis=1).max() < 1e-9


@pytest.mark.unit
def test_fix_xline_iota_guard():
    """The condition is only the X-line condition when iota(1) == iota_b.

    A mismatched iota profile is rejected at init and at FixXLine.build. A
    fixed-current solve (where iota(1) is an uncontrolled output) is not
    reachable at all, since SharpEquilibrium does not accept a current profile.
    """
    with pytest.raises(ValueError, match="does not match iota_b"):
        _seq(iota=PowerSeriesProfile([1.0, 0, 0.5]))  # iota(1) = 1.5 != 0.5

    seq = _seq()
    seq._iota = PowerSeriesProfile([0.5, 0, 0.5])  # bypass the init check
    with pytest.raises(ValueError, match="does not match iota_b"):
        FixXLine(seq).build(verbose=0)

    with pytest.raises(ValueError, match="always solves at fixed iota"):
        _seq(current=0)
