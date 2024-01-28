import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from bilby import run_sampler
from bilby.core.prior import Uniform, PriorDict

from poppingw import planck, get_pz_exp_pow, RedshiftPriorSpline, BNSPopulation, BNSSelection
from poppingw.utils import load_catalog, plot_nsigma_pz


parser = argparse.ArgumentParser()
parser.add_argument('--posterior-path', type=str)
parser.add_argument('--events-path', type=str)
parser.add_argument('--fisher-path', type=str)
parser.add_argument('--N-zcontrol', type=int, default=10)
parser.add_argument('--outdir', type=str)
parser.add_argument('--Ntotal', type=int, default=-1)
parser.add_argument('--label', type=str, default='pop')
parser.add_argument('--npool', type=int, default=1)
parser.add_argument('--nlive', type=int, default=100)
parser.add_argument('--plot-pz-nbins', type=int, default=20)
parser.add_argument('-plot-pz-normalize', action='store_true', default=False)
parser.add_argument('--plot-pz-postfrac', type=int, default=0.68)
args = parser.parse_args()

posterior_data = np.load(args.posterior_path)
log_dL_array = posterior_data['log_dL']
log_dL_posterior = posterior_data['p_log_dL']
events = load_catalog(args.events_path)
snrs = np.load(args.fisher_path)['snr']

N_total = args.Ntotal
if N_total == -1:
    N_total = events.shape[0]

np.random.seed(42)
snrs_obs = snrs[:N_total] + np.random.normal(0, 1, N_total)
select = snrs_obs > 10
print(f'{np.sum(select)} of {N_total} observed')

pz_analytical = get_pz_exp_pow(planck, parameters={'n': 2.9, 'a': 2, 'b': 3}, fix_parameters=True)
z_control = np.linspace(*pz_analytical.z_range, args.N_zcontrol+1)[:-1]  # keep control point at z_min to prevent p(z)<0 near z=0
fit_pz = RedshiftPriorSpline(planck, z_control, z_range=pz_analytical.z_range)
print('control points:\n', z_control)

pz_control_true = pz_analytical(z_control)
true_parameters = {f'pz{i}': pzi for i, pzi in enumerate(pz_control_true)}
true_parameters['N_total'] = N_total

selection = BNSSelection(events, snrs, sample_pz=pz_analytical)
likelihood = BNSPopulation(log_dL_array[:N_total][select].T, log_dL_posterior[:N_total][select].T, selection, fit_pz)

N_estimate = likelihood.N_obs / selection.selection(pz_analytical)

priors = {
    'N_total': Uniform(N_estimate/10, N_estimate*10, 'N_total', latex_label=r'$N_{\rm total}$')
}
for i in range(args.N_zcontrol):
    key = f'pz{i}'
    priors[key] = Uniform(true_parameters[key]/10, true_parameters[key]*10, latex_label=r'$p(z_'+f'{i}'+')$')
priors = PriorDict(priors)

outdir = args.outdir
result = run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=args.nlive, npool=args.npool,
    outdir=outdir, label=args.label, injection_parameters=true_parameters, save=True)

bestfit_params = {key: result.get_one_dimensional_median_and_error_bar(key).median for key in true_parameters.keys()}

for nsigma in range(1, 4):
    quantiles_left = norm.cdf(-nsigma, loc=0, scale=1)
    result.plot_corner(quantiles=[quantiles_left, 1-quantiles_left], save=True, filename=os.path.join(outdir, f'corner_{nsigma:d}sigma.png'))
    plot_nsigma_pz(
        result.posterior, z_array=np.linspace(0,6,1000), event_zs=events.loc[:N_total, 'z'].to_numpy(),
        pz=fit_pz, true_params=true_parameters.copy(), bestfit_params=bestfit_params,
        nsigma=nsigma, hist_normalize=args.plot_pz_normalize, nbins=args.plot_pz_nbins,
        save_path=os.path.join(outdir, f'pz_{nsigma:d}sigma.png'))
