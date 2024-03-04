import sys
import json
import argparse
from datetime import datetime
from schwimmbad import MPIPool

import numpy as np
from scipy.integrate import simps

from gwfast.waveforms import IMRPhenomD

from poppingw import planck
from poppingw.gwfast import GWSignalExtended, DetNetExtended
from poppingw.utils import load_catalog


parser = argparse.ArgumentParser()
parser.add_argument("--events-path", type=str)
parser.add_argument("--fisher-path", type=str)
parser.add_argument("--netfile", type=str)
parser.add_argument("--Ncalc", default=-1, type=int)
parser.add_argument("--logl-path", type=str)
parser.add_argument("--posterior-path", type=str)
parser.add_argument("--fmin", default=2., type=float)
parser.add_argument("--dL-shape", default=100, type=int)
parser.add_argument("--iota-shape", default=100, type=int)
args = parser.parse_args()

events = load_catalog(args.events_path)
fisher_data = np.load(args.fisher_path)

snrs = fisher_data['snr']
fishers = fisher_data['fisher']

Ncalc: int = args.Ncalc
if Ncalc == -1:
    Ncalc = events.shape[0]


signal = IMRPhenomD()
id2para = {v: k for k, v in signal.ParNums.items()}
para = [id2para[i] for i in range(len(id2para))]

para_nospin = [i for i in para if i not in ['chi1z', 'chi2z']]
index_nospin = [signal.ParNums[p] for p in para_nospin]
fishers = fishers[index_nospin, :, :][:, index_nospin, :]
para2id = {para_i: i for i, para_i in enumerate(para_nospin)}

# convert iota to cos iota
iota_i = para2id['iota']
mult = -1/np.sin(events['iota'].to_numpy())
fishers[iota_i, :, :] *= mult
fishers[:, iota_i, :] *= mult

# convert dL to log dL:
dL = events['dL'].to_numpy()
dL_i = para2id['dL']
fishers[dL_i, :, :] *= dL
fishers[:, dL_i, :] *= dL

covs = np.linalg.inv(fishers.astype(np.float64).transpose(2, 0, 1))


with open(args.netfile, 'r') as f:
    detectors = json.load(f)

signals = {
    name: GWSignalExtended(
        IMRPhenomD(), psd_path=det['psd_path'], detector_shape=det['shape'],
        det_lat=det['lat'], det_long=det['long'], det_xax=det['xax'],
        verbose=False, useEarthMotion=True, fmin=args.fmin, IntTablePath=None)
    for name, det in detectors.items()
}
network = DetNetExtended(signals, verbose=False)

log_dL_range = tuple(np.log(planck.dL_from_z(np.array([0.001, 6]))))


def wrapper(i):
    params1 = events.iloc[i].to_dict()
    for para in ['z', 'M1', 'M2']:
        params1.pop(para)

    covi = covs[i, :, :]
    sigma_log_dL = covi[signal.ParNums['dL'], signal.ParNums['dL']]**0.5

    log_dL_real = np.log(params1['dL'])
    log_dL_array = np.linspace(
        max(log_dL_real-3*sigma_log_dL, log_dL_range[0]),
        min(log_dL_real+3*sigma_log_dL, log_dL_range[1]),
        args.dL_shape)
    cos_iota_array = np.linspace(-1, 1, args.iota_shape)
    dL_grid, iota_grid = np.meshgrid(np.exp(log_dL_array), np.arccos(cos_iota_array))
    params2 = {k: np.array([v]*args.dL_shape*args.iota_shape) for k, v in params1.items()}
    params2['dL'] = dL_grid.flatten()
    params2['iota'] = iota_grid.flatten()

    params1 = {k: np.array([v]) for k, v in params1.items()}

    log_l = network.log_likelihood(params1, params2)
    return dL_grid, iota_grid, log_l.reshape(dL_grid.shape)

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

start = datetime.now()
results = pool.map(wrapper, range(Ncalc))
end = datetime.now()
print('Elapsed time: ', end-start)

pool.close()

all_dL = np.array([res[0] for res in results])
all_iota = np.array([res[1] for res in results])
all_logl = np.array([res[2] for res in results])

np.savez(args.logl_path, dL=all_dL, iota=all_iota, logl=all_logl)

# calculate log dL posterior
p_log_dL = []
log_dL_array = []
cos_iota_grid_all = np.cos(all_iota)
log_dL_grid_all = np.log(all_dL)
for i in range(Ncalc):
    log_l_grid = all_logl[i]
    log_l_grid -= np.max(log_l_grid)
    log_dL_array_i = log_dL_grid_all[i, 0, :]
    cos_iota_array = cos_iota_grid_all[i, :, 0]

    p_log_dL_i = simps(np.exp(log_l_grid), cos_iota_array, axis=0)
    p_log_dL_i /= simps(p_log_dL_i, log_dL_array_i)
    p_log_dL.append(p_log_dL_i)
    log_dL_array.append(log_dL_array_i)

p_log_dL = np.array(p_log_dL)
log_dL_array = np.array(log_dL_array)

Mz = events['Mc'].to_numpy()
Mz_2d = Mz[:,None] * np.ones_like(log_dL_array)
log_kappa_array = 5/6*np.log(Mz_2d) - log_dL_array
log_kappa_array = log_kappa_array[:, ::-1]

log_kappa_posterior = p_log_dL[:, ::-1]
norm = np.array([simps(log_kappa_posterior[i], log_kappa_array[i]) for i in range(Ncalc)])
log_kappa_posterior = log_kappa_posterior / norm[:, None]

np.savez(args.posterior_path, p_log_dL=p_log_dL, log_dL=log_dL_array, log_kappa=log_kappa_array, p_log_kappa=log_kappa_posterior)