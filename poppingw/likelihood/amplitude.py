import numpy as np
import pandas as pd

from numba import njit
from scipy.integrate import simps

from . import BNSPopulation
from ..cosmology import Cosmology, planck
from ..selection import BNSSelection
from ..redshift_prior import RedshiftPriorSpline


class BNSAmplitudePopulation(BNSPopulation):
    def __init__(self, log_kappa_array: np.ndarray, p_log_kappa: np.ndarray, selection: BNSSelection,
                 log_kappa_prior: RedshiftPriorSpline, cosmo: Cosmology = planck, use_numba: bool = False):
        super(BNSPopulation, self).__init__(parameters={key: None for key in ['N_total']+log_kappa_prior.parameter_keys})
        self.log_kappa_array = log_kappa_array
        self.p_log_kappa = p_log_kappa
        self.N_obs = log_kappa_array.shape[1]

        self.selection = selection
        self.log_kappa_prior = log_kappa_prior
        self.cosmo = cosmo

        # handling infinite likelihood: happens when kappa posterior completely out of range
        # change p_pop: p_pop_new = (1-N_inf/N_tot)*p_pop_old + 1/N_tot * Sum[Uniform(kappa_posterior_range)]
        # for new p_pop, contribution of each out range events to log likelihood is log(1/(N_tot*(kappa_posterior_range_length))) --> doesn't really matter, same for all population parameters
        outrange_mask = (self.log_kappa_array[-1] <= self.log_kappa_prior.z_range[0]) |\
                        (self.log_kappa_array[0] >= self.log_kappa_prior.z_range[1])
        self.outrange_id = np.where(outrange_mask)[0]
        self.outrange_log_likelihood = -len(self.outrange_id) * np.log(self.N_obs) -\
            np.sum(np.log(self.log_kappa_array[-1, self.outrange_id]-self.log_kappa_array[0, self.outrange_id]))
        self.log_kappa_array = self.log_kappa_array[:, ~outrange_mask]
        self.p_log_kappa = self.p_log_kappa[:, ~outrange_mask]
        self.N_obs = self.log_kappa_array.shape[1]

        # in current implementation, numba is slower than numpy
        # TODO: optimize performance of numba implementation
        if use_numba:
            self.log_likelihood = self.log_likelihood_numba
        else:
            self.log_likelihood = self.log_likelihood_numpy

    def log_likelihood_event_pop(self):
        # p(d|\lambda) = \int p(d|K) p(K|\lambda) dK, here K=log(kappa)
        p_log_kappa_pop = self.log_kappa_prior(self.log_kappa_array)
        return np.log(simps(self.p_log_kappa*p_log_kappa_pop, self.log_kappa_array, axis=0))

    def log_likelihood_numpy(self):
        self.log_kappa_prior.parameters = self.parameters
        roots = self.log_kappa_prior.interp.roots()
        if np.any((roots > self.log_kappa_prior.z_range[0]) & (roots < self.log_kappa_prior.z_range[1])):
            return -np.inf

        Ntot = self.parameters['N_total']
        logp_events = self.log_likelihood_event_pop()
        logp = np.sum(logp_events) + self.outrange_log_likelihood

        alpha = self.selection.selection_numpy(self.log_kappa_prior)
        logp += -Ntot*alpha + self.N_obs*np.log(Ntot)
        return logp

    def log_likelihood_numba(self):
        # self.log_kappa_prior.parameters = self.parameters
        # if len(self.log_kappa_prior.interp.roots())>2:
        #     return -np.inf

        # p_log_kappa_pop = self.log_kappa_prior(self.log_kappa_array)
        # alpha = self.selection.selection(self.log_kappa_prior)
        # return log_likelihood_numba(self.p_log_kappa, p_log_kappa_pop, self.log_kappa_array, alpha, self.parameters['N_total'], self.N_obs, self.outrange_log_likelihood)
        raise NotImplementedError


@njit
def log_likelihood_numba(p_log_kappa, p_log_kappa_pop, log_kappa_array, alpha, Ntot, Nobs, outrange_log_likelihood=0):
    p_lk_int = np.zeros(p_log_kappa.shape[1])
    for i in range(p_log_kappa.shape[1]):
        p_lk_int[i] = np.trapz(p_log_kappa[:, i]*p_log_kappa_pop[:, i], log_kappa_array[:, i])
    logp = np.sum(np.log(p_lk_int)) + outrange_log_likelihood
    logp += -Ntot*alpha + Nobs*np.log(Ntot)
    return logp
