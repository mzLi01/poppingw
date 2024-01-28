import sys
import json
import argparse
import pathlib
from datetime import datetime
from schwimmbad import MPIPool

import numpy as np
import pandas as pd

from gwfast.waveforms import IMRPhenomD_NRTidalv2, IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast.gwfastUtils import load_population


parser = argparse.ArgumentParser()
parser.add_argument("--events_path", default='', type=str, required=True,
                    help='Name of the file containing the catalog, without the extension ``h5``.')
parser.add_argument("--outpath", default='', type=str, required=True,
                    help='Path to output folder, which has to exist before the script is launched.')
parser.add_argument("--batch_size", default=1, type=int, required=False,
                    help='Size of the batch to be computed in vectorized form on each process.')
parser.add_argument("--snr_th", default=10, type=float, required=False,
                    help='Threshold value for the SNR to consider the event detectable. FIMs are computed only for events with SNR exceeding this value.')
parser.add_argument("--fmin", default=2., type=float,
                    required=False, help='Minimum frequency of the grid, in Hz.')
parser.add_argument("--netfile", default=None, type=str, required=False,
                    help='``json`` file containing the detector configuration, alternative to **--net** and **--psds**.')
args = parser.parse_args()

events_path: str = args.events_path
outpath: str = args.outpath
batch_size: int = args.batch_size
snr_th: float = args.snr_th
fmin: float = args.fmin
netfile: str = args.netfile

pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)

with open(netfile, 'r') as f:
    detectors = json.loads(f.read())

signals = {
    name: GWSignal(
        IMRPhenomD(), psd_path=det['psd_path'], detector_shape=det['shape'],
        det_lat=det['lat'], det_long=det['long'], det_xax=det['xax'],
        verbose=False, useEarthMotion=True, fmin=fmin, IntTablePath=None)
    for name, det in detectors.items()
}
network = DetNet(signals, verbose=False)


def df_to_events_dict(df: pd.DataFrame, exclusive_cols=None):
    if exclusive_cols is None:
        exclusive_cols = []
    cols = set(df.columns) - set(exclusive_cols)
    return {para: df[para].to_numpy() for para in cols}


def fisher_wrapper(events: pd.DataFrame):
    events_dict = df_to_events_dict(events, ['z'])
    snr = network.SNR(events_dict)
    detected = events.index[snr > snr_th]
    events_detected_dict = df_to_events_dict(events.loc[detected, :], ['z'])
    fisher = network.FisherMatr(events_detected_dict, computeDerivFinDiff=True)
    return snr, detected, fisher


pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

events = pd.DataFrame(load_population(events_path))
nevents = events.shape[0]
nbatch = int(np.ceil(nevents/batch_size))
batches = [
    events.loc[i*batch_size: min((i+1)*batch_size, nevents)-1, :] for i in range(nbatch)]

start = datetime.now()
results = pool.map(fisher_wrapper, batches)
end = datetime.now()
print('Elapsed time: ', end-start)

pool.close()

all_snr = np.concatenate([res[0] for res in results])
all_detect_index = np.concatenate([res[1] for res in results])
all_fisher = np.concatenate([res[2] for res in results], axis=2)

np.savez(outpath, snr=all_snr, detect_index=all_detect_index, fisher=all_fisher)
