import sys
import json
import argparse
from datetime import datetime
from schwimmbad import MPIPool

import numpy as np
from scipy.integrate import simps

from gwfast.waveforms import IMRPhenomD, TaylorF2_RestrictedPN

from poppingw.gwfast import GWSignalExtended, DetNetExtended
from poppingw.utils import load_catalog


parser = argparse.ArgumentParser()
parser.add_argument("--events-path", type=str)
parser.add_argument("--range-path", type=str, default='')
parser.add_argument("--grid-path", type=str, default='')
parser.add_argument("--netfile", type=str)
parser.add_argument("--Nstart", default=0, type=int)
parser.add_argument("--Ncalc", default=-1, type=int)
parser.add_argument("--logl-path", type=str, default='')
parser.add_argument("--likelihood-path", type=str)
parser.add_argument("--noise-scale-sqr", default=0.25, type=float)
parser.add_argument("--fmin", default=2., type=float)
parser.add_argument("--dL-shape", default=100, type=int)
parser.add_argument("--iota-shape", default=100, type=int)
args = parser.parse_args()

events = load_catalog(args.events_path)

Nstart: int = args.Nstart
Ncalc: int = args.Ncalc
if Ncalc == -1:
    Ncalc = events.shape[0] - Nstart
Nend = Nstart + Ncalc

events = events.iloc[Nstart:Nend]

signal = TaylorF2_RestrictedPN()
id2para = {v: k for k, v in signal.ParNums.items()}
para = [id2para[i] for i in range(len(id2para))]


if args.grid_path:
    grid_data = np.load(args.grid_path)
    log_kappa_array_total = grid_data['log_kappa'][Nstart:Nend]
    cos_iota_array_total = grid_data['cos_iota'][Nstart:Nend]
else:
    if not args.range_path:
        raise ValueError('range-path is required if grid-path is not provided') 
    range_data = np.load(args.range_path)
    log_kappa_range = range_data['log_kappa']
    cos_iota_range = range_data['cos_iota']

    log_kappa_array_total = np.array([np.linspace(*log_kappa_range[i], args.dL_shape) for i in range(Ncalc)])
    cos_iota_array_total = np.array([np.linspace(*cos_iota_range[i], args.iota_shape) for i in range(Ncalc)])

Mc = events['Mc'].to_numpy()
log_dL_array_total = (5/6)*np.log(Mc[:,np.newaxis]) - log_kappa_array_total


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


def wrapper(i):
    params1 = events.iloc[i].to_dict()
    for para in ['z', 'M1', 'M2']:
        params1.pop(para)

    dL_grid, iota_grid = np.meshgrid(np.exp(log_dL_array_total[i]), np.arccos(cos_iota_array_total[i]))
    params2 = {k: np.array([v]*args.dL_shape*args.iota_shape) for k, v in params1.items()}

    params1 = {k: np.array([v]) for k, v in params1.items()}

    params2['dL'] = dL_grid.flatten()
    params2['iota'] = iota_grid.flatten()

    log_l = network.log_likelihood(params1, params2, noise_scale=np.sqrt(args.noise_scale_sqr))
    return {key: logli.reshape(dL_grid.shape) for key, logli in log_l.items()}

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

start = datetime.now()
results = pool.map(wrapper, range(Ncalc))
# results = [wrapper(i) for i in range(Ncalc)]
end = datetime.now()
print('Elapsed time: ', end-start)

pool.close()

all_logl = np.array([np.sum(list(res.values()), axis=0) for res in results])

if args.logl_path:
    save_logl = {f'logl_{key}': [res[key] for res in results] for key in network.signals.keys()}
    np.savez(args.logl_path, log_kappa=log_kappa_array_total, cos_iota=cos_iota_array_total, logl=all_logl, **save_logl)

# calculate log dL likelihood
p_log_dL = []
for i in range(Ncalc):
    log_l_grid = all_logl[i]
    log_l_grid -= np.max(log_l_grid)
    log_dL_array_i = log_dL_array_total[i, ::-1]
    cos_iota_array = cos_iota_array_total[i]

    p_log_dL_i = simps(np.exp(log_l_grid[:, ::-1]), cos_iota_array, axis=0)
    p_log_dL_i /= simps(p_log_dL_i, log_dL_array_i)
    p_log_dL.append(p_log_dL_i)

p_log_dL = np.array(p_log_dL)

p_log_kappa = p_log_dL[:, ::-1]
norm = np.array([simps(p_log_kappa[i], log_kappa_array_total[i]) for i in range(Ncalc)])
p_log_kappa = p_log_kappa / norm[:, None]

np.savez(args.likelihood_path, p_log_dL=p_log_dL, log_dL=log_dL_array_total, log_kappa=log_kappa_array_total, p_log_kappa=p_log_kappa)