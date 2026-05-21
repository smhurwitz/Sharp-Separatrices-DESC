import warnings
import numpy as np
from .core import VolumeRegion as Volume
from desc.backend import (
    execute_on_cpu,
    jnp,
    put,
    sign,
)
from desc.basis import FourierZernikeBasis, SharpFourierZernikeBasis, GeneralizedFourierZernikeBasis
from desc.geometry import FourierRZCurve 
from desc.optimizable import optimizable_parameter
from desc.utils import (
    check_nonnegint,
    check_posint,
    copy_coeffs,
    setdefault,
)

class FourierZernikeRZToroidalVolume(Volume):
    """Toroidal volume represented by Fourier-Zernike polynomials.
    
    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    L, M, N: int or None
        Maximum radial, poloidal, and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this volume
    check_orientation : bool
        ensure that this volume has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the volume).
    
    """

    _io_attrs_ = Volume._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "_NFP",
    ]
    @execute_on_cpu
    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        L=None,
        M=None,
        N=None,
        name="",
        check_orientation=True,
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0, 0], [1, 1, 0]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
            modes_Z = np.array([[0, 0, 0], [1, -1, 0]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert (
            R_lmn.size == modes_R.shape[0]
        ), "R_lmn size and modes_R.shape[0] must be the same size!"
        assert (
            Z_lmn.size == modes_Z.shape[0]
        ), "Z_lmn size and modes_Z.shape[0] must be the same size!"

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        LR = np.max(np.abs(modes_R[:, 0]-np.abs(modes_R[:, 1])))
        MR = np.max(abs(modes_R[:, 1]))
        NR = np.max(abs(modes_R[:, 2]))
        LZ = np.max(np.abs(modes_Z[:, 0]-np.abs(modes_Z[:, 1])))
        MZ = np.max(abs(modes_Z[:, 1]))
        NZ = np.max(abs(modes_Z[:, 2]))

        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        NFP = check_posint(NFP, "NFP", False)
        self._L = setdefault(L, max(LR, LZ))
        self._M = setdefault(M, max(MR, MZ))
        self._N = setdefault(N, max(NR, NZ))
        self._NFP = NFP

        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N, NFP=NFP, sym="sin" if sym else False
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes)
        self._sym = bool(sym)

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0. You may also need to switch the sign of iota or"
                + " current profiles."
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        self.name = name

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def R_basis(self):
        """Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z."""
        return self._Z_basis
    
    @property
    def L(self):
        """int: Maximum radial mode number."""
        return self._L
    
    @property
    def M(self):
        """int: Maximum poloidal mode number."""
        return self._M
    
    @property
    def N(self):
        """int: Maximum toroidal mode number."""
        return self._N

    @execute_on_cpu
    def change_resolution(self, *args, **kwargs):
        """Change the maximum poloidal and toroidal resolution."""
        raise NotImplementedError("Will implement later.")
    
    @optimizable_parameter
    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Z_basis.num_modes} modes."
            )
        
    def get_coeffs(self, l, m, n=0):
        """Get Fourier-Zernike coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)
        l = np.atleast_1d(l).astype(int)

        l, m, n = np.broadcast_arrays(l, m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        lmn = np.array([l,m, n]).T
        idxR = np.where(
            (lmn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, :]).all(axis=-1)
        )
        idxZ = np.where(
            (lmn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, :]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z
    
    def set_coeffs(self, l, m, n=0, R=None, Z=None):
        """Set specific Fourier coefficients."""
        l, m, n, R, Z = (
            np.atleast_1d(l),
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        l, m, n, R, Z = np.broadcast_arrays(l, m, n, R, Z)
        for ll, mm, nn, RR, ZZ in zip(l, m, n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(ll, mm, nn)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(ll, mm, nn)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    def get_axis(self):
        """Get the magnetic axis of the volume."""

        modes_R_vol = self.R_basis.modes
        modes_Z_vol = self.Z_basis.modes
        coeffs_R = self.R_lmn
        coeffs_Z = self.Z_lmn

        R_m_zero_mask = modes_R_vol[:, 1] == 0
        Z_m_zero_mask = modes_Z_vol[:, 1] == 0
        R_modes_m0 = modes_R_vol[R_m_zero_mask]
        Z_modes_m0 = modes_Z_vol[Z_m_zero_mask]
        coeffs_R_m0 = coeffs_R[R_m_zero_mask]
        coeffs_Z_m0 = coeffs_Z[Z_m_zero_mask]
        func_R = coeffs_R_m0 * (-1.0) ** (np.abs(R_modes_m0[:, 0]) / 2.0)
        func_Z = coeffs_Z_m0 * (-1.0) ** (np.abs(Z_modes_m0[:, 0]) / 2.0)

        modes_R, R_inverse_indices = np.unique(R_modes_m0[:, 2], return_inverse=True)
        modes_Z, Z_inverse_indices = np.unique(Z_modes_m0[:, 2], return_inverse=True)
        R_n = np.zeros(len(modes_R))
        Z_n = np.zeros(len(modes_Z))
        np.add.at(R_n, R_inverse_indices, func_R)
        np.add.at(Z_n, Z_inverse_indices, func_Z)

        curve = FourierRZCurve(R_n, Z_n, modes_R, modes_Z)
        return curve
    
    @execute_on_cpu
    def change_resolution(self, *args, **kwargs):
        """Change the maximum radial, poloidal, and toroidal resolution."""
        assert (
            ((len(args) in [3]) and len(kwargs) == 0)
            or ((len(args) in [3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        NFP = kwargs.pop("NFP", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"

        if len(args) == 3:
            L, M, N = args

        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        NFP = check_posint(NFP, "NFP")
        self._NFP = int(NFP if NFP is not None else self.NFP)

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or ((L is not None) and (L != self.L))
            or (NFP is not None)
            or ((sym is not None) and (sym != self.sym))
        ):
            self._sym = sym if sym is not None else self.sym
            L = int(L if L is not None else self.L)
            M = int(M if M is not None else self.M)
            N = int(N if N is not None else self.N)
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                L=L, M=M, N=N, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                L=L, M=M, N=N, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M
            self._N = N
    
class GeneralizedFourierZernikeRZToroidalVolume(Volume):
    """Toroidal volume represented by Generalized Fourier-Zernike polynomials.
    
    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    m_b : int
        poloidal mode number associated with the boundary of the volume
    n_b : int
        toroidal mode number associated with the boundary of the volume
    β : float
        Angle of corners for lens mapping method.
    sharp_type : str
        Method for sharp mapping, either "lens" or "hypergeometric".
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    L, M, N: int or None
        Maximum radial, poloidal, and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this volume
    check_orientation : bool
        ensure that this volume has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the volume).
    
    """

    _io_attrs_ = Volume._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "_NFP",
    ]
    @execute_on_cpu
    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        m_b=1,
        n_b=1,
        β=0.5*np.pi,
        sharp_type="lens",
        sym="auto",
        L=None,
        M=None,
        N=None,
        L_shp=None,
        M_shp=None,
        N_shp=None,
        name="",
        check_orientation=True,
    ):
        
        def safe_max(a):
            return np.max(a) if a.size else 0

        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0, 0], [1, 1, 0]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
            modes_Z = np.array([[0, 0, 0], [1, -1, 0]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert (
            R_lmn.size == modes_R.shape[0]
        ), "R_lmn size and modes_R.shape[0] must be the same size!"
        assert (
            Z_lmn.size == modes_Z.shape[0]
        ), "Z_lmn size and modes_Z.shape[0] must be the same size!"

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        modes_R_std = modes_R[modes_R[:, 0] >= 0]
        modes_Z_std = modes_Z[modes_Z[:, 0] >= 0]
        modes_R_sharp = modes_R[modes_R[:, 0] < 0]
        modes_Z_sharp = modes_Z[modes_Z[:, 0] < 0]

        LR = np.max(np.abs(modes_R_std[:, 0]-np.abs(modes_R_std[:, 1])))
        MR = np.max(abs(modes_R_std[:, 1]))
        NR = np.max(abs(modes_R_std[:, 2]))
        LZ = np.max(np.abs(modes_Z_std[:, 0]-np.abs(modes_Z_std[:, 1])))
        MZ = np.max(abs(modes_Z_std[:, 1]))
        NZ = np.max(abs(modes_Z_std[:, 2]))

        LR_sharp = safe_max(np.abs(modes_R_sharp[:, 0]-np.abs(modes_R_sharp[:, 1])))
        MR_sharp = safe_max(abs(modes_R_sharp[:, 1]))
        NR_sharp = safe_max(abs(modes_R_sharp[:, 2]))
        LZ_sharp = safe_max(np.abs(modes_Z_sharp[:, 0]-np.abs(modes_Z_sharp[:, 1])))
        MZ_sharp = safe_max(abs(modes_Z_sharp[:, 1]))
        NZ_sharp = safe_max(abs(modes_Z_sharp[:, 2]))

        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        L_shp = check_nonnegint(L_shp, "Lsh")
        M_shp = check_nonnegint(M_shp, "Msh")
        N_shp = check_nonnegint(N_shp, "Nsh")
        NFP = check_posint(NFP, "NFP", False)
        m_b = check_posint(m_b, "m_b", False)
        n_b = check_posint(n_b, "m_b", False)
        self._L = setdefault(L, max(LR, LZ))
        self._M = setdefault(M, max(MR, MZ))
        self._N = setdefault(N, max(NR, NZ, NR_sharp, NZ_sharp))
        self._L_shp = setdefault(L_shp, max(LR_sharp, LZ_sharp))
        self._M_shp = setdefault(M_shp, max(MR_sharp, MZ_sharp))
        self._N_shp = self._N
        self._NFP = NFP
        self._m_b = m_b
        self._n_b = n_b

        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        R_basis_std = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N, NFP=NFP, sym="cos" if sym else False
        )
        R_basis_sharp = SharpFourierZernikeBasis(
            L=self._L_shp, M=self._M_shp, N=self._N_shp, NFP=NFP, m_b=m_b, n_b=n_b, β=β, sharp_type=sharp_type, sym="cos" if sym else False
        )
        Z_basis_std = FourierZernikeBasis(
            L=self._L, M=self._M, N=self._N, NFP=NFP, sym="sin" if sym else False
        )
        Z_basis_sharp = SharpFourierZernikeBasis(
            L=self._L_shp, M=self._M_shp, N=self._N_shp, NFP=NFP, m_b=m_b, n_b=n_b, β=β, sharp_type=sharp_type, sym="sin" if sym else False
        )

        self._R_basis = GeneralizedFourierZernikeBasis(
            std_basis = R_basis_std, shp_basis = R_basis_sharp
        )
        self._Z_basis = GeneralizedFourierZernikeBasis(
            std_basis = Z_basis_std, shp_basis = Z_basis_sharp
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes)
        self._sym = bool(sym)

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0. You may also need to switch the sign of iota or"
                + " current profiles."
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        self.name = name

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP
    
    @property
    def m_b(self):
        """Poloidal mode number associated with the boundary of the volume."""
        return self._m_b
    
    @property
    def n_b(self):
        """Toroidal mode number associated with the boundary of the volume."""
        return self._n_b

    @property
    def R_basis(self):
        """Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z."""
        return self._Z_basis
    
    @property
    def L(self):
        """int: Maximum radial mode number."""
        return self._L
    
    @property
    def M(self):
        """int: Maximum poloidal mode number."""
        return self._M
    
    @property
    def N(self):
        """int: Maximum toroidal mode number."""
        return self._N
    
    @property
    def L_shp(self):
        """int: Maximum radial mode number of sharp piece."""
        return self._L_shp
    
    @property
    def M_shp(self):
        """int: Maximum poloidal mode number of sharp piece."""
        return self._M_shp
    
    @property
    def N_shp(self):
        """int: Maximum toroidal mode number of sharp piece."""
        return self._N_shp

    @execute_on_cpu
    def change_resolution(self, *args, **kwargs):
        """Change the maximum poloidal and toroidal resolution."""
        raise NotImplementedError("Will implement later.")
    
    @optimizable_parameter
    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Z_basis.num_modes} modes."
            )
        
    def get_coeffs(self, l, m, n=0):
        """Get Fourier-Zernike coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)
        l = np.atleast_1d(l).astype(int)

        l, m, n = np.broadcast_arrays(l, m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        lmn = np.array([l,m, n]).T
        idxR = np.where(
            (lmn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, :]).all(axis=-1)
        )
        idxZ = np.where(
            (lmn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, :]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z
    
    def set_coeffs(self, l, m, n=0, R=None, Z=None):
        """Set specific Fourier coefficients."""
        l, m, n, R, Z = (
            np.atleast_1d(l),
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        l, m, n, R, Z = np.broadcast_arrays(l, m, n, R, Z)
        for ll, mm, nn, RR, ZZ in zip(l, m, n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(ll, mm, nn)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(ll, mm, nn)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    def get_axis(self):
        """Get the magnetic axis of the volume."""

        modes_R_vol = self.R_basis.modes
        modes_Z_vol = self.Z_basis.modes
        coeffs_R = self.R_lmn
        coeffs_Z = self.Z_lmn

        R_m_zero_mask = modes_R_vol[:, 1] == 0
        Z_m_zero_mask = modes_Z_vol[:, 1] == 0
        R_modes_m0 = modes_R_vol[R_m_zero_mask]
        Z_modes_m0 = modes_Z_vol[Z_m_zero_mask]
        coeffs_R_m0 = coeffs_R[R_m_zero_mask]
        coeffs_Z_m0 = coeffs_Z[Z_m_zero_mask]
        func_R = coeffs_R_m0 * (-1.0) ** (np.abs(R_modes_m0[:, 0]) / 2.0)
        func_Z = coeffs_Z_m0 * (-1.0) ** (np.abs(Z_modes_m0[:, 0]) / 2.0)

        modes_R, R_inverse_indices = np.unique(R_modes_m0[:, 2], return_inverse=True)
        modes_Z, Z_inverse_indices = np.unique(Z_modes_m0[:, 2], return_inverse=True)
        R_n = np.zeros(len(modes_R))
        Z_n = np.zeros(len(modes_Z))
        np.add.at(R_n, R_inverse_indices, func_R)
        np.add.at(Z_n, Z_inverse_indices, func_Z)

        curve = FourierRZCurve(R_n, Z_n, modes_R, modes_Z, NFP=self.NFP)
        return curve
    
    @execute_on_cpu
    def change_resolution(self, *args, **kwargs):
        """Change the maximum radial, poloidal, and toroidal resolution."""
        assert (
            ((len(args) in [6]) and len(kwargs) == 0)
            or ((len(args) in [6]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with (L,M,N,L_shp,M_shp,N_shp) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        L_shp = kwargs.pop("L_shp", None)
        M_shp = kwargs.pop("M_shp", None)
        N_shp = kwargs.pop("N_shp", None)
        NFP = kwargs.pop("NFP", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"

        if len(args) == 6:
            L, M, N, L_shp, M_shp, N_shp = args

        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        L_shp = check_nonnegint(L_shp, "L_shp")
        M_shp = check_nonnegint(M_shp, "M_shp")
        N_shp = check_nonnegint(N_shp, "N_shp")
        NFP = check_posint(NFP, "NFP")
        self._NFP = int(NFP if NFP is not None else self.NFP)

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or ((L is not None) and (L != self.L))
            or ((N_shp is not None) and (N_shp != self.N_shp))
            or ((M_shp is not None) and (M_shp != self.M_shp))
            or ((L_shp is not None) and (L_shp != self.L_shp))
            or (NFP is not None)
            or ((sym is not None) and (sym != self.sym))
        ):
            self._sym = sym if sym is not None else self.sym
            L = int(L if L is not None else self.L)
            M = int(M if M is not None else self.M)
            N = int(N if N is not None else self.N)
            L_shp = int(L_shp if L_shp is not None else self.L_shp)
            M_shp = int(M_shp if M_shp is not None else self.M_shp)
            N_shp = int(N_shp if N_shp is not None else self.N_shp)
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                L=L, M=M, N=N, L_shp=L_shp, M_shp=M_shp, N_shp=N_shp, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                L=L, M=M, N=N, L_shp=L_shp, M_shp=M_shp, N_shp=N_shp, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M
            self._N = N
            self._L_shp = L_shp
            self._M_shp = M_shp
            self._N_shp = N_shp
    