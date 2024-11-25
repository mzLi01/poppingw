import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import norm

from .cosmology import Cosmology

from typing import List, Dict, Tuple, Callable, Optional


class RedshiftPrior:
    def __init__(self, cosmo: Cosmology, parameter_keys: List, merger_rate: Callable[[np.ndarray], np.ndarray], 
                 z_range: Tuple[float, float], interp_array_len: int = 1000) -> None:
        self.cosmo = cosmo
        self.parameter_keys = parameter_keys
        self.merger_rate_func = merger_rate

        self.z_range = z_range
        self.z_array = np.linspace(*z_range, interp_array_len)

        self.parameters_fixed = False

        self._parameters = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Dict):
        if self.parameters_fixed:
            raise RuntimeError("Parameters should not be updated for a fixed parameter instance.")
        self._parameters = {key: parameters[key] for key in self.parameter_keys}
        self._update_pz_interp()

    def fix_parameters(self):
        if self._parameters is None:
            raise RuntimeError("Parameters have not been set.")
        self.parameters_fixed = True

    def __call__(self, z: np.ndarray) -> np.ndarray:
        if self._parameters is None:
            raise RuntimeError("Parameters have not been set.")
        return self.interp(z)

    def merger_rate(self, z: np.ndarray) -> np.ndarray:
        return self.merger_rate_func(z, **self.parameters)

    # from doi: 10.1103/PhysRevD.107.123036
    def z_prior(self, z: np.ndarray) -> np.ndarray:
        d_Vc = self.cosmo.dVc_from_z(z)
        pz = d_Vc * self.merger_rate(z) / (1+z)
        return pz

    def _update_pz_interp(self, normalize: bool = True):
        pz_array = self.z_prior(self.z_array)
        if normalize:
            pz_array /= np.trapz(pz_array, self.z_array)
        self.interp = interp1d(self.z_array, pz_array, bounds_error=False, fill_value=0)


def merger_rate_exp_pow(z, n, a, b):
    return (1+z)**n * np.exp(-z**a/b)


def get_pz_exp_pow(cosmo: Cosmology, z_range: Tuple[float, float] = (0.001, 6),
                   parameters: Optional[Dict] = None, fix_parameters: bool = False) -> RedshiftPrior:
    pz = RedshiftPrior(cosmo, parameter_keys=['n', 'a', 'b'], merger_rate=merger_rate_exp_pow, z_range=z_range)
    if parameters is not None:
        pz.parameters = parameters
    if fix_parameters:
        pz.fix_parameters()
    return pz


class RedshiftPriorSpline(RedshiftPrior):
    def __init__(self, cosmo: Cosmology, z_control: np.ndarray, z_range: Tuple[float, float], k_interp: int = 3) -> None:
        self.z_control = z_control
        parameter_keys = [f'pz{i}' for i in range(len(z_control))]

        self.k_interp = k_interp

        super().__init__(cosmo, parameter_keys, merger_rate=None, z_range=z_range)

    def _update_pz_interp(self, normalize: bool = True):
        self.pz_control = np.array([self.parameters[key] for key in self.parameter_keys])
        self.interp = InterpolatedUnivariateSpline(self.z_control, self.pz_control, k=self.k_interp, ext='zeros')
        if normalize:
            self.norm = self.interp.integral(*self.z_range)
        else:
            self.norm = 1

    def __call__(self, z: np.ndarray) -> np.ndarray:
        pz_values = self.interp(z) / self.norm
        pz_values[pz_values<0] = 0
        return pz_values
