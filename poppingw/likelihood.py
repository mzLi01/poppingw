import numpy as np
import pandas as pd
from bilby import Likelihood

from .cosmology import Cosmology, planck
from .selection import BNSSelection
from .redshift_prior import RedshiftPrior


class BNSPopulation(Likelihood):
    def __init__(self, log_dL_array: np.ndarray, pdL_value: np.ndarray, selection: BNSSelection, z_prior: RedshiftPrior, cosmo: Cosmology = planck):
        super().__init__(parameters={key: None for key in ['N_total']+z_prior.parameter_keys})
        self.log_dL_array = log_dL_array
        self.pdL_value = pdL_value
        self.N_obs = log_dL_array.shape[1]

        self.selection = selection
        self.z_prior = z_prior
        self.cosmo = cosmo

        self.z_array = self.cosmo.z_from_dL(np.exp(self.log_dL_array))
        self.jacobian = np.exp(self.log_dL_array) / self.cosmo.ddL_dz(self.z_array)

    def log_likelihood_event_pop(self):
        # p(d|\lambda) = \int p(d|z) p(z|\lambda) dz
        pz_value = self.z_prior(self.z_array)
        return np.log(np.trapz(self.pdL_value*pz_value*self.jacobian, self.log_dL_array, axis=0))

    def log_likelihood(self):
        self.z_prior.parameters = self.parameters
        logp_events = self.log_likelihood_event_pop()
        logp = np.sum(logp_events)

        alpha = self.selection.selection(self.z_prior)
        N_det = self.parameters['N_total'] * alpha
        logp -= self.N_obs * np.log(alpha)
        logp += -N_det + self.N_obs*np.log(N_det)
        return logp


class BNSGaussianPosterior(BNSPopulation):
    def __init__(self, catalog: pd.DataFrame, selection: BNSSelection, cosmo: Cosmology = planck, z_range: tuple = (0.001, 6)):
        super().__init__(parameters={
            key: None for key in self.PARAMETER_NAMES})
        self.catalog = catalog
        self.N_obs = catalog.shape[0]
        self.log_dL_obs = catalog['log_dL'].to_numpy()
        self.sigma_log_dL = catalog['sigma_log_dL'].to_numpy()

        self.selection = selection
        self.cosmo = cosmo
        self.z_range = z_range
        self.log_dL_range = tuple(np.log(self.cosmo.dL_from_z(np.array(self.z_range))))

        self.log_dL_array, self.pdL_value = self.precalc_log_likelihood_event(
            self.log_dL_obs, self.sigma_log_dL)
        self.z_array = self.cosmo.z_from_dL(np.exp(self.log_dL_array))
        self.jacobian = np.exp(self.log_dL_array) / self.cosmo.ddL_dz(self.z_array)

    def precalc_log_likelihood_event(self, log_dL_obs: np.ndarray, sigma_log_dL: np.ndarray, nint_per_event: int = 1000):
        log_dL_array = np.linspace(
            np.max([log_dL_obs-3*sigma_log_dL, self.log_dL_range[0]*np.ones_like(log_dL_obs)], axis=0),
            np.min([log_dL_obs+3*sigma_log_dL, self.log_dL_range[1]*np.ones_like(log_dL_obs)], axis=0),
            nint_per_event)  # shape: (nint_per_event, N_obs)

        total_out_cat_i = np.where(log_dL_array[0,:]>log_dL_array[-1,:])[0]
        for i in total_out_cat_i:
            log_dL_array[:, i] = np.linspace(self.log_dL_range[0], self.log_dL_range[1], nint_per_event)
        if len(total_out_cat_i) > 0:
            print(f'event {total_out_cat_i} are totally out of range, resetting log dL integration array')

        pdL_value = np.exp(-0.5 * (log_dL_array - log_dL_obs)**2 / sigma_log_dL**2) / np.sqrt(2*np.pi) / sigma_log_dL
        return log_dL_array, pdL_value


class BNSAmplitudePopulation(BNSPopulation):
    def __init__(self, log_kappa_array: np.ndarray, p_log_kappa: np.ndarray, selection: BNSSelection, log_kappa_prior: RedshiftPrior, cosmo: Cosmology = planck):
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

    def log_likelihood_event_pop(self):
        # p(d|\lambda) = \int p(d|K) p(K|\lambda) dK, here K=log(kappa)
        p_log_kappa_pop = self.log_kappa_prior(self.log_kappa_array)
        return np.log(np.trapz(self.p_log_kappa*p_log_kappa_pop, self.log_kappa_array, axis=0))

    def log_likelihood(self):
        self.log_kappa_prior.parameters = self.parameters
        logp_events = self.log_likelihood_event_pop()
        logp = np.sum(logp_events) + self.outrange_log_likelihood

        alpha = self.selection.selection(self.log_kappa_prior)
        N_det = self.parameters['N_total'] * alpha
        logp -= self.N_obs * np.log(alpha)
        logp += -N_det + self.N_obs*np.log(N_det)
        return logp
