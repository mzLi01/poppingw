import os
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

from bilby import run_sampler
from bilby.core.prior import Uniform, PriorDict

from poppingw import planck, RedshiftPriorSpline, BNSAmplitudePopulation, BNSSelection
from poppingw.selection import NoSelect
from poppingw.utils import load_catalog, plot_nsigma_pz


parser = argparse.ArgumentParser()
parser.add_argument('--likelihood-path', type=str)
parser.add_argument('--true-pop-path', type=str)
parser.add_argument('--events-infer-path', type=str)
parser.add_argument('--events-select-path', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--Ntotal', type=int, default=-1)
parser.add_argument('--Nselect', type=int, default=-1)
parser.add_argument('-disable-select', action='store_true', default=False)
parser.add_argument('--snr-threshold', type=float, default=10)
parser.add_argument('--label', type=str, default='pop')
parser.add_argument('--npool', type=int, default=1)
parser.add_argument('--nlive', type=int, default=100)
parser.add_argument('--seed', type=int)
parser.add_argument('--plot-pz-nbins', type=int, default=20)
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

events_infer = load_catalog(args.events_infer_path)
events_select = load_catalog(args.events_select_path)

N_total = args.Ntotal
if N_total == -1:
    N_total = events_infer.shape[0]
events_infer = events_infer.iloc[:N_total]

# load BNS events and kappa posterior
likelihood_data = np.load(args.likelihood_path)
log_kappa_array = likelihood_data['log_kappa'][:N_total]
log_kappa_likelihood = likelihood_data['p_log_kappa'][:N_total]

# load true population, control points and values
true_pop_data = np.load(args.true_pop_path)
log_kappa_control = true_pop_data['control']
p_log_kappa_control = true_pop_data['p_control']
log_kappa_range = true_pop_data['range']

fit_p_log_kappa = RedshiftPriorSpline(planck, log_kappa_control, z_range=log_kappa_range)
p_log_kappa_true = RedshiftPriorSpline(planck, log_kappa_control, z_range=log_kappa_range)
p_log_kappa_true.parameters = {f'pz{i}': pki for i, pki in enumerate(p_log_kappa_control)}
p_log_kappa_true.fix_parameters()

# set selection and likelihood for population inference
true_parameters = p_log_kappa_true.parameters.copy()
true_parameters['N_total'] = N_total

if args.disable_select:
    selection = NoSelect()
    select = np.ones(N_total, dtype=bool)
else:
    log_kappa_select = events_select['log_kappa'].to_numpy()[:args.Nselect]
    log_kappa_density = gaussian_kde(log_kappa_select)
    selection = BNSSelection(log_kappa_select, events_select['snr_obs'].to_numpy()[:args.Nselect], sample_pz=log_kappa_density)

    select = events_infer['snr_obs'] > args.snr_threshold

likelihood = BNSAmplitudePopulation(log_kappa_array[select].T, log_kappa_likelihood[select].T, selection, fit_p_log_kappa)

# set prior dict and run sampler
N_estimate = likelihood.N_obs / selection.selection(p_log_kappa_true)

priors = {
    'N_total': Uniform(N_estimate/10, N_estimate*10, 'N_total', latex_label=r'$N_{\rm total}$')
}
for i in range(1, len(log_kappa_control)-1):
    key = f'pz{i}'
    priors[key] = Uniform(true_parameters[key]/10, true_parameters[key]*10, latex_label=r'$p(\log\kappa_{'+f'{i}'+'})$')
priors['pz0'] = p_log_kappa_control[0]  # fix boundary values (should be zero)
priors[f'pz{len(log_kappa_control)-1}'] = p_log_kappa_control[-1]
priors = PriorDict(priors)

outdir = Path(args.outdir)
result = run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=args.nlive, npool=args.npool,
    outdir=outdir, label=args.label, injection_parameters=true_parameters, save=True)

# post process (plot posterior corner plot and p(kappa) uncertainty range)
bestfit_params = {key: result.get_one_dimensional_median_and_error_bar(key).median for key in true_parameters.keys()}

result.plot_corner(save=True, filename=outdir/'corner.png')
for nsigma in range(1, 4):
    plot_nsigma_pz(
        result.posterior, z_array=np.linspace(*log_kappa_range, 1000), event_zs=events_infer['log_kappa'].to_numpy()[:N_total],
        pz=fit_p_log_kappa, true_params=true_parameters.copy(),
        nsigma=nsigma, hist_normalize=False, nbins=args.plot_pz_nbins,
        xlabel=r'$\log \kappa$', ylabel=r'$N(\log \kappa)$', save_path=outdir/f'pk_{nsigma:d}sigma.png')
