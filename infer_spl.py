import os
import argparse

import numpy as np
from scipy.stats import norm, gaussian_kde

from bilby import run_sampler
from bilby.core.prior import Uniform, PriorDict

from poppingw import planck, RedshiftPriorSpline, BNSAmplitudePopulation, BNSSelection
from poppingw.utils import load_catalog, plot_nsigma_pz


parser = argparse.ArgumentParser()
parser.add_argument('--posterior-path', type=str)
parser.add_argument('--true-pop-path', type=str)
parser.add_argument('--events-path', type=str)
parser.add_argument('--fisher-path', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--Ntotal', type=int, default=-1)
parser.add_argument('--label', type=str, default='pop')
parser.add_argument('--npool', type=int, default=1)
parser.add_argument('--nlive', type=int, default=100)
parser.add_argument('--seed', type=int)
parser.add_argument('--plot-pz-nbins', type=int, default=20)
parser.add_argument('-plot-pz-normalize', action='store_true', default=False)
parser.add_argument('--plot-pz-postfrac', type=int, default=0.68)
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

# load BNS events and kappa posterior
posterior_data = np.load(args.posterior_path)
log_kappa_array = posterior_data['log_kappa']
log_kappa_posterior = posterior_data['p_log_kappa']
events = load_catalog(args.events_path)
snrs = np.load(args.fisher_path)['snr']

N_total = args.Ntotal
if N_total == -1:
    N_total = events.shape[0]

# calculate kappa
Mz = events['Mc'].to_numpy()
dL = events['dL'].to_numpy()
log_kappa_samples = np.log((Mz)**(5/6)/dL)
log_kappa_density = gaussian_kde(log_kappa_samples)

# observed SNR have +-1 deviation from true SNR
snrs_obs = snrs[:N_total] + np.random.normal(0, 1, N_total)
select = snrs_obs > 10
print(f'{np.sum(select)} of {N_total} observed')

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

selection = BNSSelection(log_kappa_samples, snrs, sample_pz=log_kappa_density)
likelihood = BNSAmplitudePopulation(log_kappa_array[:N_total][select].T, log_kappa_posterior[:N_total][select].T, selection, fit_p_log_kappa)

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

outdir = args.outdir
result = run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=args.nlive, npool=args.npool,
    outdir=outdir, label=args.label, injection_parameters=true_parameters, save=True)

np.save(os.path.join(outdir, 'select.npy'), select)

# post process (plot posterior corner plot and p(kappa) uncertainty range)
bestfit_params = {key: result.get_one_dimensional_median_and_error_bar(key).median for key in true_parameters.keys()}

result.plot_corner(save=True, filename=os.path.join(outdir, f'corner.png'))
for nsigma in range(1, 4):
    plot_nsigma_pz(
        result.posterior, z_array=np.linspace(*log_kappa_range, 1000), event_zs=log_kappa_samples[:N_total],
        pz=fit_p_log_kappa, true_params=true_parameters.copy(), bestfit_params=bestfit_params,
        nsigma=nsigma, hist_normalize=args.plot_pz_normalize, nbins=args.plot_pz_nbins,
        save_path=os.path.join(outdir, f'pk_{nsigma:d}sigma.png'))
