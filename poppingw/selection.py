import numpy as np

from numba import njit

from .cosmology import Cosmology, planck

from typing import Optional
from .typing import InterpFunc


class BNSSelection:
    def __init__(self, z_samples: np.ndarray, snrs: np.ndarray, sample_pz: InterpFunc,
                 snr_threshold: float = 10, snr_sigma: float = 1, cosmo: Cosmology = planck,
                 use_numba: bool = False) -> None:
        self.z_samples = z_samples
        self.snrs = snrs
        self.N_total = len(z_samples)

        self.sample_pz = sample_pz
        self.sample_weights = self.sample_pz(self.z_samples)
        self.cosmo = cosmo
        self.update_snr_selection(snr_threshold, snr_sigma)

        if use_numba:
            self.selection = self.selection_numba
        else:
            self.selection = self.selection_numpy

    def update_snr_selection(self, snr_threshold: float, snr_sigma: Optional[float] = None):
        if snr_sigma is not None:
            self.snr_sigma = snr_sigma
        self.snr_threshold = snr_threshold
        mask = self.snrs > self.snr_threshold
        self.z_samples = self.z_samples[mask]
        self.snrs = self.snrs[mask]
        self.sample_weights = self.sample_weights[mask]

    def selection_numpy(self, pz: InterpFunc, calc_Neff: bool = False) -> float:
        weights_calc = pz(self.z_samples)/self.sample_weights
        alpha = np.sum(weights_calc) / self.N_total
        if calc_Neff:
            var = np.sum(weights_calc**2)/self.N_total**2-alpha**2/self.N_total
            return alpha, alpha**2/var
        else:
            return alpha

    def selection_numba(self, pz: InterpFunc) -> float:
        return selection_numba(pz(self.z_samples), self.sample_weights, self.N_total)


@njit
def selection_numba(pz_samples, weights, N_total):
    weights_calc = pz_samples / weights
    alpha = np.sum(weights_calc) / N_total
    return alpha


class NoSelect(BNSSelection):
    def __init__(self) -> None:
        self.selection = self.selection_numpy

    def selection_numpy(self, pz: InterpFunc) -> float:
        return 1.
