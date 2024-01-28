import argparse

import numpy as np
from gwfast.gwfastUtils import save_data
from bilby.core.prior import PriorDict, Uniform, Interped, Sine
from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_symmetric_mass_ratio

from poppingw import get_pz_exp_pow, planck

parser = argparse.ArgumentParser()
parser.add_argument('--nevents', type=float, default=1e6, help='number of events to generate')
parser.add_argument('--zmin', type=float, default=0.001, help='minimum redshift')
parser.add_argument('--zmax', type=float, default=6, help='maximum redshift')
parser.add_argument('--outpath', type=str, help='events catalog output path')
args = parser.parse_args()

nevents = int(args.nevents)
z_min = args.zmin
z_max = args.zmax
outpath = args.outpath

z_array = np.linspace(z_min, z_max, 1000)  # z=0.001 ~ dL=4Mpc
pz = get_pz_exp_pow(planck, z_range=(z_min, z_max), parameters = {'n': 2.9, 'a': 2, 'b': 3}, fix_parameters=True)
pz_array = pz(z_array)

priors = PriorDict({
    'm1': Uniform(minimum=1.2, maximum=2.5, name='mass_1'),
    'm2': Uniform(minimum=1.2, maximum=2.5, name='mass_2'),
    'z': Interped(z_array, pz_array, minimum=z_min, maximum=z_max, name='redshift'),
    'theta': Sine(name='theta'),
    'phi': Uniform(minimum=0, maximum=2*np.pi, name='phi'),
    'iota': Sine(name='iota'),
    'psi': Uniform(minimum=0, maximum=2*np.pi, name='psi'),
    'tcoal': Uniform(minimum=0, maximum=1, name='geocent_time'),  # unit: day
    'Phicoal': Uniform(minimum=0, maximum=2*np.pi, name='phase'),
    'chi1z': Uniform(minimum=-.05, maximum=.05, name='chi_1'),
    'chi2z': Uniform(minimum=-.05, maximum=.05, name='chi_2'),
})

events = priors.sample(nevents)
m1 = events.pop('m1')
m2 = events.pop('m2')
z = events['z']

events['Mc'] = component_masses_to_chirp_mass(m1, m2) * (1+z)
events['eta'] = component_masses_to_symmetric_mass_ratio(m1, m2)
events['dL'] = planck.dL_from_z(z)

save_data(outpath, events)
