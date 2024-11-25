import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FLRW, Planck18
from astropy import units

from typing import Union


class Cosmology:
    def __init__(self, cosmo: FLRW, z_min: float, z_max: float, n_z: int, unit: Union[str, units.Unit] = 'Gpc') -> None:
        self.cosmo = cosmo
        self.unit = units.Unit(unit)
        z_array = np.linspace(z_min, z_max, n_z)
        chi_array = cosmo.comoving_distance(z_array).to_value(self.unit)
        dL_array = chi_array * (1+z_array)
        self.z2chi_spl = InterpolatedUnivariateSpline(z_array, chi_array)
        self.dL2z_spl = InterpolatedUnivariateSpline(dL_array, z_array)

    def chi_from_z(self, z: np.ndarray):
        return self.z2chi_spl(z)

    def dL_from_z(self, z: np.ndarray):
        return self.chi_from_z(z) * (1+z)

    def dVc_from_z(self, z: np.ndarray):
        chi = self.chi_from_z(z)
        return self.cosmo._hubble_distance.to_value(self.unit) * (chi**2.0) / self.cosmo.efunc(z)

    def z_from_dL(self, dL: np.ndarray):
        return self.dL2z_spl(dL)

    def ddL_dz(self, z: np.ndarray):
        return (1+z) * self.cosmo._hubble_distance.to_value(self.unit) * self.cosmo.inv_efunc(z) + self.chi_from_z(z)

    @property
    def H0_si(self):
        return self.cosmo.H0.to_value(units.s**-1)


planck = Cosmology(Planck18, 0, 6, 1000)
