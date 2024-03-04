import numpy as np
import pandas as pd
from scipy.stats import norm

from .cosmology import Cosmology, planck
from .redshift_prior import RedshiftPrior

from typing import Callable, Optional
from .typing import InterpFunc


class BNSSelection:
    def __init__(self, z_samples: np.ndarray, snrs: np.ndarray, sample_pz: InterpFunc,
                 snr_threshold: float = 10, snr_sigma: float = 1, cosmo: Cosmology = planck) -> None:
        self.z_samples = z_samples
        self.snrs = snrs
        self.N_total = len(z_samples)

        self.sample_pz = sample_pz
        self.cosmo = cosmo
        self.update_snr_selection(snr_threshold, snr_sigma)
    
    def update_snr_selection(self, snr_threshold: float, snr_sigma: Optional[float] = None):
        if snr_sigma is not None:
            self.snr_sigma = snr_sigma
        self.snr_threshold = snr_threshold
        snr_weights = 1 - norm.cdf(self.snr_threshold, loc=self.snrs, scale=self.snr_sigma)
        self.events_weights = snr_weights / self.sample_pz(self.z_samples)

    def selection(self, pz: InterpFunc) -> float:
        weights = pz(self.z_samples)
        return np.sum(weights*self.events_weights) / self.N_total
