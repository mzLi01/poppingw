import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from .redshift_prior import RedshiftPriorSpline

from typing import Optional, Union, Dict


def load_catalog(path) -> pd.DataFrame:
    with h5py.File(path, 'r') as f:
        catalog = {key: np.array(f[key]) for key in f.keys()}
    return pd.DataFrame(catalog)


def plot_nsigma_pz(posterior: pd.DataFrame, z_array: np.ndarray, event_zs: np.ndarray,
                   pz: RedshiftPriorSpline, true_params: Dict, nsigma: float = 1,
                   hist_normalize: bool = False, bins: Union[int, np.ndarray] = 50, plot_pz_true: bool = True,
                   Ntot_key: str = 'N_total', save_path: Optional[str] = None, ax: Optional[plt.Axes] = None,
                   **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(bins, int):
        bin_edges = np.linspace(z_array[0], z_array[-1], bins+1)
    else:
        bin_edges = bins
    ax.hist(
        event_zs, bins=bin_edges, histtype='step',
        density=hist_normalize, label=kwargs.get('label_hist', 'events'))
    
    if hist_normalize:
        def plot_bins_norm(Ntot):
            return 1
    else:
        def plot_bins_norm(Ntot):
            return Ntot*(bin_edges[1] - bin_edges[0])

    Npost = posterior.shape[0]
    Nplot_fit = int(Npost * (norm.cdf(nsigma)-norm.cdf(-nsigma)))
    for i in range(Npost-Nplot_fit, Npost):
        params = posterior.iloc[i].to_dict()
        pz.parameters = params
        ax.plot(z_array, pz(z_array)*plot_bins_norm(params[Ntot_key]),
                color=kwargs.get('color_samples', 'gray'), alpha=kwargs.get('alpha_samples', 0.1), zorder=0)

    if plot_pz_true:
        pz.parameters = true_params
        ax.plot(z_array, pz(z_array)*plot_bins_norm(true_params[Ntot_key]),
                label=kwargs.get('label_pztrue', 'true distribution'))

    params_ml = posterior.median().to_dict()
    norm_ml = plot_bins_norm(params_ml[Ntot_key])
    pz.parameters = params_ml
    ax.plot(z_array, pz(z_array)*norm_ml, label=kwargs.get('label_bestfit', 'best fit'))
    ax.scatter(pz.z_control, pz(pz.z_control)*norm_ml, color='red', s=3, zorder=100)

    ax.set_xlim(*(kwargs.get('xlim', (z_array[0], z_array[-1]))))
    ax.set_ylim(*(kwargs.get('ylim', (0, None))))
    ax.set_xlabel(kwargs.get('xlabel', 'z'))
    ax.set_ylabel(kwargs.get('ylabel', r'N(z)'))

    if save_path is not None:
        ax.legend()
        fig.savefig(save_path, bbox_inches='tight')
    return ax


def create_gaussian_noise(f, psd, scale=0.5):
    re, im = np.random.normal(0, scale, size=(2, len(f)))
    white_noise = (re + 1j*im) * psd**0.5
    return white_noise


def Omega_GW(f, hf, H0, timespan):
    return 4*np.pi**2/(3*H0**2) * f**3 * hf/timespan
