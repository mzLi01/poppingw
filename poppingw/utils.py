import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from .redshift_prior import RedshiftPrior

from typing import Callable, Optional, Tuple, Dict


def load_catalog(path) -> pd.DataFrame:
    with h5py.File(path, 'r') as f:
        catalog = {key: np.array(f[key]) for key in f.keys()}
    return pd.DataFrame(catalog)


def plot_nsigma_pz(posterior: pd.DataFrame, z_array: np.ndarray, event_zs: np.ndarray,
                   pz: RedshiftPrior, true_params: Dict, bestfit_params: Dict, Ntot_key: str = 'N_total',
                   nsigma: float = 1, hist_normalize: bool = False, nbins: int = 50,
                   save_path: Optional[str] = None, **kwargs) -> plt.Axes:
    fig, ax = plt.subplots(**kwargs)
    fig: plt.Figure
    ax: plt.Axes

    pz.parameters = true_params
    true_pz = pz(z_array)
    N_total_true = true_params[Ntot_key]
    pz.parameters = bestfit_params
    fit_pz = pz(z_array)

    bin_count, bin_edge, _ = plt.hist(
        event_zs, bins=nbins, histtype='step',
        density=hist_normalize, label='all events')
    if not hist_normalize:
        normalization = np.trapz(bin_count, (bin_edge[:-1]+bin_edge[1:])/2)
    else:
        normalization = 1.

    Npost = posterior.shape[0]
    Nplot_fit = int(Npost * (norm.cdf(nsigma)-norm.cdf(-nsigma)))
    for i in range(Npost-Nplot_fit, Npost):
        pz.parameters = posterior.iloc[i].to_dict()
        Ntot_norm = posterior.iloc[i][Ntot_key] / N_total_true
        ax.plot(z_array, pz(z_array)*normalization*Ntot_norm, color='gray', alpha=0.1, zorder=0)
    ax.plot(z_array, true_pz*normalization, label='true pz')
    ax.plot(z_array, fit_pz*normalization * bestfit_params[Ntot_key]/N_total_true, label='best fit')

    ax.legend()
    ax.set_xlabel('z')
    ax.set_xlim(z_array[0], z_array[-1])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return ax


def create_white_noise(f, psd):
    norm = ((f[-1]-f[0])*8)**-0.5
    re, im = np.random.normal(0, norm, size=(2, len(f)))
    white_noise = (re + 1j*im) * psd**0.5
    return white_noise
