"""Tests for custom volume classes in desc_mods.geometry."""
import numpy as np
import pytest

from desc.basis import FourierZernikeBasis, SharpFourierZernikeBasis
from desc.geometry.volume import FourierZernikeRZToroidalVolume, GeneralizedFourierZernikeRZToroidalVolume
from desc.grid import Grid
from desc.transform import Transform


class TestFourierZernikeRZToroidalVolume:
    """Test FourierZernikeRZToroidalVolume class."""

    @pytest.mark.unit
    def test_initialization_defaults(self):
        """Test basic initialization with default parameters."""
        vol = FourierZernikeRZToroidalVolume()
        assert vol.NFP == 1
        assert vol.name == ""
        # Check default coefficients
        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [10])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [1])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [-1])

    @pytest.mark.unit
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        R_lmn = np.array([5, 2])
        Z_lmn = np.array([0, -3])
        modes_R = np.array([[0, 0, 0], [1, 1, 0]])
        modes_Z = np.array([[0, 0, 0], [1, -1, 0]])
        vol = FourierZernikeRZToroidalVolume(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, NFP=2, name="test"
        )
        assert vol.NFP == 2
        assert vol.name == "test"
        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [5])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [2])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [-3])

    @pytest.mark.unit
    def test_get_coeffs_single_mode(self):
        """Test get_coeffs for a single mode."""
        vol = FourierZernikeRZToroidalVolume()
        R, Z = vol.get_coeffs(0, 0, 0)
        assert len(R) == 1
        assert len(Z) == 1
        np.testing.assert_allclose(R, [10])
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_get_coeffs_multiple_modes(self):
        """Test get_coeffs for multiple modes."""
        vol = FourierZernikeRZToroidalVolume()
        R, Z = vol.get_coeffs([0, 1, 1], [0, 1, -1], [0, 0, 0])
        assert len(R) == 3
        assert len(Z) == 3
        np.testing.assert_allclose(R, [10, 1, 0])
        np.testing.assert_allclose(Z, [0, 0, -1])

    @pytest.mark.unit
    def test_get_coeffs_nonexistent_mode(self):
        """Test get_coeffs for a mode not in the basis returns zero."""
        vol = FourierZernikeRZToroidalVolume()
        R, Z = vol.get_coeffs(2, 0, 0)  # l=2 not in basis
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_set_coeffs_single_mode_R(self):
        """Test set_coeffs for a single R coefficient."""
        vol = FourierZernikeRZToroidalVolume()
        vol.set_coeffs(1, 1, 0, R=15.0)
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [15.0])
        # Z unchanged
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_set_coeffs_single_mode_Z(self):
        """Test set_coeffs for a single Z coefficient."""
        vol = FourierZernikeRZToroidalVolume()
        vol.set_coeffs(1, -1, 0, Z=-2.0)
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(Z, [-2.0])
        # R unchanged
        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [10])

    @pytest.mark.unit
    def test_set_coeffs_multiple_modes(self):
        """Test set_coeffs for multiple modes."""
        vol = FourierZernikeRZToroidalVolume()
        vol.set_coeffs([1, 0], [1, 0], [0, 0], R=[20.0, 3.0])
        R, Z = vol.get_coeffs([1, 0], [1, 0], [0, 0])
        np.testing.assert_allclose(R, [20.0, 3.0])

    @pytest.mark.unit
    def test_set_coeffs_both_R_and_Z(self):
        """Test set_coeffs setting both R and Z for the same mode."""
        vol = FourierZernikeRZToroidalVolume(sym=False)
        vol.set_coeffs(1, 1, 0, R=25.0, Z=5.0)
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [25.0])
        np.testing.assert_allclose(Z, [5.0])

    @pytest.mark.unit
    def test_get_axis(self):

        vol = FourierZernikeRZToroidalVolume(R_lmn=[10, 1, 0.5], 
                                     Z_lmn=[0.5, 0.5, -1], 
                                     modes_R=[[0, 0, 2], [1, 1, 0], [2, 0, 2]], 
                                     modes_Z=[[0, 0, 0], [1, -1, 0], [2, 0, 2]])
        axis = vol.get_axis()

        ζs = np.linspace(0, 2 * np.pi, 100)
        ρs = np.zeros_like(ζs)
        θs = np.zeros_like(ζs)
        nodes = np.array([ρs, θs, ζs]).T
        grid = Grid(nodes)

        Rvol_trans = Transform(grid, vol.R_basis)
        Zvol_trans = Transform(grid, vol.Z_basis)
        Rax_trans = Transform(grid, axis.R_basis)
        Zax_trans = Transform(grid, axis.Z_basis)

        R_vol = Rvol_trans.transform(vol.R_lmn)
        Z_vol = Zvol_trans.transform(vol.Z_lmn)
        R_axis = Rax_trans.transform(axis.R_n)
        Z_axis = Zax_trans.transform(axis.Z_n)

        np.testing.assert_allclose(R_vol, R_axis)
        np.testing.assert_allclose(Z_vol, Z_axis)


class TestGeneralizedZernikeRZToroidalVolume:
    """Test GeneralizedZernikeRZToroidalVolume class."""

    @pytest.mark.unit
    def test_initialization_defaults(self):
        """Test basic initialization with default parameters."""
        vol = GeneralizedFourierZernikeRZToroidalVolume()
        assert vol.NFP == 1
        assert vol.name == ""
        # Check default coefficients
        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [10])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [1])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [-1])

    @pytest.mark.unit
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        R_lmn = np.array([5, 2, 1, 0.3])
        Z_lmn = np.array([0, -3, 0.1, 0.1])
        modes_R = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [-2, 0, 2]])
        modes_Z = np.array([[0, 0, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 2]])
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, 
            NFP=2, m_b=2, n_b=2, name="test"
        )
        assert vol.NFP == 2
        assert vol.name == "test"

        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [5])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, 1, 0)
        np.testing.assert_allclose(R, [2])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [-3])
        R, Z = vol.get_coeffs(-1, -1, 0)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [0.1])
        R, Z = vol.get_coeffs(-2, 0, 2)
        np.testing.assert_allclose(R, [0.3])
        np.testing.assert_allclose(Z, [0])
        R, Z = vol.get_coeffs(0, 0, 2)
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [0.1])

    @pytest.mark.unit
    def test_resolution_properties(self):
        """Test resolution getters for generalized toroidal volume."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            NFP=3,
            L=2,
            M=3,
            N=1,
            L_shp=4,
            M_shp=5,
            N_shp=1,
            sym=True,
            m_b=3,
            n_b=3
        )

        assert vol.NFP == 3
        assert vol.L == 2
        assert vol.M == 3
        assert vol.N == 1
        assert vol.L_shp == 4
        assert vol.M_shp == 5
        assert vol.N_shp == 1

    @pytest.mark.unit
    def test_change_resolution(self):
        """Test changing the resolution of the generalized toroidal volume."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            NFP=1,
            L=1,
            M=1,
            N=0,
            L_shp=1,
            M_shp=1,
            N_shp=0,
            sym=False,
        )

        vol.change_resolution(
            L=2,
            M=3,
            N=0,
            L_shp=2,
            M_shp=3,
            N_shp=0,
            NFP=1,
            sym=False,
        )

        assert vol.NFP == 1
        assert vol.L == 2
        assert vol.M == 3
        assert vol.N == 0
        assert vol.L_shp == 2
        assert vol.M_shp == 3
        assert vol.N_shp == 0

        expected_std = FourierZernikeBasis(L=2, M=3, N=0, NFP=2, sym=False)
        expected_sharp = SharpFourierZernikeBasis(L=2, M=3, N=0, NFP=2, m_b=2, n_b=2, sharp_type="lens", sym=False)
        expected_sharp_modes = expected_sharp.modes.copy()
        expected_sharp_modes[:, 0] = -expected_sharp_modes[:, 0]
        expected_sharp_modes = expected_sharp_modes[expected_sharp_modes[:, 0] != 0]
        expected_modes = np.vstack((expected_sharp_modes, expected_std.modes))
        expected_modes = expected_modes[np.lexsort((expected_modes[:, 1], expected_modes[:, 0], expected_modes[:, 2]))]

        np.testing.assert_array_equal(vol.R_basis.modes, expected_modes)
        np.testing.assert_array_equal(vol.Z_basis.modes, expected_modes)

    @pytest.mark.unit
    def test_get_coeffs_single_mode_std(self):
        """Test get_coeffs for a single mode."""
        R_lmn = np.array([5, 2, 1, 0.3])
        Z_lmn = np.array([0, -3, 0.1, 0.1])
        modes_R = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [-2, 0, 2]])
        modes_Z = np.array([[0, 0, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 2]])
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, 
            NFP=2, m_b=2, n_b=2, name="test"
        )
        R, Z = vol.get_coeffs(0, 0, 0)
        assert len(R) == 1
        assert len(Z) == 1
        np.testing.assert_allclose(R, [5])
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_get_coeffs_single_mode_sharp(self):
        """Test get_coeffs for a single mode."""
        R_lmn = np.array([5, 2, 1, 0.3])
        Z_lmn = np.array([0, -3, 0.1, 0.1])
        modes_R = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [-2, 0, 2]])
        modes_Z = np.array([[0, 0, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 2]])
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, 
            NFP=2, m_b=2, n_b=2, name="test"
        )
        R, Z = vol.get_coeffs(-1, 1, 0)
        assert len(R) == 1
        assert len(Z) == 1
        np.testing.assert_allclose(R, [1])
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_get_coeffs_multiple_modes(self):
        """Test get_coeffs for multiple modes."""
        R_lmn = np.array([5, 2, 1, 0.3])
        Z_lmn = np.array([0, -3, 0.1, 0.1])
        modes_R = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [-2, 0, 2]])
        modes_Z = np.array([[0, 0, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 2]])
        vol = GeneralizedFourierZernikeRZToroidalVolume(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, 
            NFP=2, m_b=2, n_b=2, name="test"
        )
        R, Z = vol.get_coeffs([0, 1, 1], [0, 1, -1], [0, 0, 0])
        assert len(R) == 3
        assert len(Z) == 3
        np.testing.assert_allclose(R, [5, 2, 0])
        np.testing.assert_allclose(Z, [0, 0, -3])

    @pytest.mark.unit
    def test_get_coeffs_nonexistent_mode(self):
        """Test get_coeffs for a mode not in the basis returns zero."""
        vol = GeneralizedFourierZernikeRZToroidalVolume()
        R, Z = vol.get_coeffs(2, 0, 0)  # l=2 not in basis
        np.testing.assert_allclose(R, [0])
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_set_coeffs_single_mode_R(self):
        """Test set_coeffs for a single R coefficient."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(M_shp=1)
        vol.set_coeffs(-1, 1, 0, R=15.0)
        R, Z = vol.get_coeffs(-1, 1, 0)
        np.testing.assert_allclose(R, [15.0])
        # Z unchanged
        R, Z = vol.get_coeffs(-1, 1, 0)
        np.testing.assert_allclose(Z, [0])

    @pytest.mark.unit
    def test_set_coeffs_single_mode_Z(self):
        """Test set_coeffs for a single Z coefficient."""
        vol = GeneralizedFourierZernikeRZToroidalVolume()
        vol.set_coeffs(1, -1, 0, Z=-2.0)
        R, Z = vol.get_coeffs(1, -1, 0)
        np.testing.assert_allclose(Z, [-2.0])
        # R unchanged
        R, Z = vol.get_coeffs(0, 0, 0)
        np.testing.assert_allclose(R, [10])

    @pytest.mark.unit
    def test_set_coeffs_multiple_modes(self):
        """Test set_coeffs for multiple modes."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(M_shp=1)
        vol.set_coeffs([-1, 0], [1, 0], [0, 0], R=[20.0, 3.0])
        R, Z = vol.get_coeffs([-1, 0], [1, 0], [0, 0])
        np.testing.assert_allclose(R, [20.0, 3.0])

    @pytest.mark.unit
    def test_set_coeffs_both_R_and_Z(self):
        """Test set_coeffs setting both R and Z for the same mode."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(sym=False, M_shp=1)
        vol.set_coeffs(-1, 1, 0, R=25.0, Z=5.0)
        R, Z = vol.get_coeffs(-1, 1, 0)
        np.testing.assert_allclose(R, [25.0])
        np.testing.assert_allclose(Z, [5.0])

    @pytest.mark.unit
    def test_get_axis(self):
        """Test get_axis returns correct magnetic axis."""
        vol = GeneralizedFourierZernikeRZToroidalVolume(
                    R_lmn=[1, 0.5, 0.1],
                    Z_lmn=[0.5, -1],
                    modes_R=[[1, 1, 0], [2, 0, 2], [-1, 1, 0]],
                    modes_Z=[[1, -1, 0], [2, 0, 2]]
                )
        axis = vol.get_axis()

        ζs = np.linspace(0, 2 * np.pi, 100)
        ρs = np.zeros_like(ζs)
        θs = np.zeros_like(ζs)
        nodes = np.array([ρs, θs, ζs]).T
        grid = Grid(nodes)

        Rvol_trans_std = Transform(grid, vol.R_basis.std_basis)
        Zvol_trans_std = Transform(grid, vol.Z_basis.std_basis)
        Rvol_trans_sharp = Transform(grid, vol.R_basis.shp_basis, method="jitable")
        Zvol_trans_sharp = Transform(grid, vol.Z_basis.shp_basis, method="jitable")
        Rax_trans = Transform(grid, axis.R_basis)
        Zax_trans = Transform(grid, axis.Z_basis)

        R_vol_std = Rvol_trans_std.transform(vol.R_lmn[vol.R_basis.modes[:, 0] >= 0])
        Z_vol_std = Zvol_trans_std.transform(vol.Z_lmn[vol.Z_basis.modes[:, 0] >= 0])
        R_vol_sharp = Rvol_trans_sharp.transform(vol.R_lmn[vol.R_basis.modes[:, 0] <= 0])
        Z_vol_sharp = Zvol_trans_sharp.transform(vol.Z_lmn[vol.Z_basis.modes[:, 0] <= 0])
        R_vol = R_vol_std + R_vol_sharp
        Z_vol = Z_vol_std + Z_vol_sharp

        R_axis = Rax_trans.transform(axis.R_n)
        Z_axis = Zax_trans.transform(axis.Z_n)

        np.testing.assert_allclose(R_vol, R_axis)
        np.testing.assert_allclose(Z_vol, Z_axis)