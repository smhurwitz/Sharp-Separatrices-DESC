import os
import warnings

import numpy as np
import pytest
from qic import Qic

from desc.__main__ import main
from desc.backend import sign
from desc.compute.utils import get_transforms
from desc.equilibrium import EquilibriaFamily, Equilibrium, SharpEquilibrium
from desc.equilibrium.coords import _map_poloidal_coordinates
from desc.examples import get
from desc.geometry import FourierRZCurve
from desc.geometry.volume import GeneralizedFourierZernikeRZToroidalVolume
from desc.grid import Grid, LinearGrid
from desc.io import InputReader, load
from desc.objectives import ForceBalance, ObjectiveFunction, get_equilibrium_objective
from desc.objectives.normalization import compute_scaling_factors
from desc.profiles import PowerSeriesProfile

i=1

vol = GeneralizedFourierZernikeRZToroidalVolume(L=4, M=4, N=4, L_shp=1)
eq = SharpEquilibrium(volume=vol, ensure_nested=False, check_orientation=True)
assert eq.R_lmn.shape == (eq.R_basis.num_modes,)
assert eq.Z_lmn.shape == (eq.Z_basis.num_modes,)
assert eq.L_lmn.shape == (eq.L_basis.num_modes,)
vol = eq.volume
print(vol.L, vol.M, vol.N, vol.L_shp, vol.M_shp, vol.N_shp)
print(f"{i} passed!")
i+=1

vol = GeneralizedFourierZernikeRZToroidalVolume(L=4, M=4, N=4, L_shp=2)
eq = SharpEquilibrium(volume=vol, ensure_nested=False, check_orientation=True)
assert eq.R_lmn.shape == (eq.R_basis.num_modes,)
assert eq.Z_lmn.shape == (eq.Z_basis.num_modes,)
assert eq.L_lmn.shape == (eq.L_basis.num_modes,)
vol = eq.volume
print(vol.L, vol.M, vol.N, vol.L_shp, vol.M_shp, vol.N_shp)
print(f"{i} passed!")
i+=1
