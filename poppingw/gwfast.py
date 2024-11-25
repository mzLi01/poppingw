import numpy as onp
import jax.numpy as np
from scipy.integrate import simps

from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast import gwfastUtils as utils

from .utils import create_gaussian_noise

from typing import Dict, Union


ParamsTypeInput = Dict[str, Union[float, onp.ndarray]]
ParamsType = Dict[str, onp.ndarray]


class GWSignalExtended(GWSignal):
    def _convert_parameters_default(self, evParams: ParamsTypeInput) -> Dict[str, onp.ndarray]:
        utils.check_evparams(evParams)

        evParams = evParams.copy()
        if isinstance(evParams['Mc'], float):
            for key, value in evParams.items():
                evParams[key] = onp.array([value])
        else:
            for key, value in evParams.items():
                evParams[key] = value.astype('complex128')

        if self.wf_model.is_Precessing:
            if 'chi1x' not in evParams:
                if self.verbose:
                    print('Adding cartesian components of the spins from angular variables')
                spin_angles = ['thetaJN', 'phiJL', 'tilt1', 'tilt2', 'phi12', 'chi1', 'chi2', 'Mc', 'eta']
                spin_angles_dict = {}
                for key in spin_angles:
                    if key in evParams:
                        spin_angles_dict[key] = evParams[key]
                    else:
                        raise ValueError('Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided.')
                evParams['iota'], evParams['chi1x'], evParams['chi1y'], evParams['chi1z'], evParams['chi2x'], evParams['chi2y'], evParams['chi2z'] = utils.TransformPrecessing_angles2comp(
                    **spin_angles_dict, fRef=self.fmin, phiRef=0.)
        else:
            if 'chi1z' not in evParams:
                if self.verbose:
                    print('Adding chi1z, chi2z from chiS, chiA')
                if 'chiS' not in evParams or 'chiA' not in evParams:
                    raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')
                evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
                evParams['chi2z'] = evParams['chiS'] - evParams['chiA']

        if self.wf_model.is_tidal:
            if 'Lambda1' not in evParams or 'Lambda2' not in evParams:
                if 'LambdaTilde' not in evParams or 'deltaLambda' not in evParams:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
                evParams['Lambda1'], evParams['Lambda2'] = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'], evParams['deltaLambda'], evParams['eta'])

        if self.wf_model.is_eccentric:
            if 'ecc' not in evParams:
                raise ValueError('Eccentricity has to be provided.')

        return evParams

    def _convert_parameters_strain(self, evParams: ParamsTypeInput,
                                   use_m1m2: bool, use_chi1chi2: bool, use_prec_ang: bool) -> Dict[str, onp.ndarray]:
        evParams = self._convert_parameters_default(evParams)

        McOr, dL, theta, phi = evParams['Mc'], evParams['dL'], evParams['theta'], evParams['phi']
        iota, psi, tcoal, etaOr, Phicoal = evParams['iota'], evParams['psi'], evParams['tcoal'], evParams['eta'], evParams['Phicoal']
        chi1z, chi2z = evParams['chi1z'], evParams['chi2z']

        if use_m1m2:
            # In this case Mc represents m1 and eta represents m2
            Mc, eta  = utils.m1m2_from_Mceta(McOr, etaOr)
        else:
            Mc, eta  = McOr, etaOr

        if self.wf_model.is_Precessing:
            if not use_prec_ang:
                chiS, chiA = chi1z, chi2z
                chi1x, chi2x, chi1y, chi2y = evParams['chi1x'], evParams['chi2x'], evParams['chi1y'], evParams['chi2y']
            else:
                # In this case iota=thetaJN, chi1y=phiJL, chi1x=tilt1, chi2x=tilt2, chi2y=phi12, chiS=chi1, chiA=chi2
                iota, chi1y, chi1x, chi2x, chi2y, chiS, chiA = utils.TransformPrecessing_comp2angles(iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, McOr, etaOr, fRef=self.fmin, phiRef=0.)
        else:
            if use_chi1chi2:
            # In this case chiS represents chi1z and chiA represents chi2z
                chiS, chiA = chi1z, chi2z
            else:
                chiS, chiA = 0.5*(chi1z + chi2z), 0.5*(chi1z - chi2z)
            chi1x, chi2x, chi1y, chi2y = onp.zeros(Mc.shape), onp.zeros(Mc.shape), onp.zeros(Mc.shape), onp.zeros(Mc.shape)
        
        if self.wf_model.is_tidal:
            Lambda1, Lambda2 = evParams['Lambda1'], evParams['Lambda2']
            LambdaTilde, deltaLambda = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, etaOr)
        else:
            Lambda1, Lambda2, LambdaTilde, deltaLambda = np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape)
        
        if self.wf_model.is_eccentric:
            ecc = evParams['ecc']
        else:
            ecc = np.zeros(Mc.shape)

        strain_params = {
            'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal,
            'chiS': chiS, 'chiA': chiA, 'chi1x': chi1x, 'chi1y': chi1y, 'chi2x': chi2x, 'chi2y': chi2y,
            'LambdaTilde': LambdaTilde, 'deltaLambda': deltaLambda, 'ecc': ecc,
        }
        return strain_params

    def _get_f_psd_array(self, evParams: ParamsType, res: int, return_psd: bool = True):
        fcut = self.wf_model.fcut(**evParams)

        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)

        fminarr = np.full(fcut.shape, self.fmin)
        fgrids = np.geomspace(fminarr, fcut, num=int(res))
        if not return_psd:
            return fgrids
        else:
            # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
            strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)
            return fgrids, strainGrids

    def log_likelihood_from_strain(self, f, strain1, strain2, noise, psd_sqrt) -> np.ndarray:
        ifo_strain = np.array([strain1[:, 0]+noise]).T
        integrand = np.abs((ifo_strain/psd_sqrt) - (strain2/psd_sqrt))**2
        return -2 * simps(integrand, f, axis=0)

    def waveform_log_likelihood(self, params1: ParamsTypeInput, params2: ParamsTypeInput, res: int = 1000,
                                noise_scale: float = 0.5, use_m1m2: bool = False, use_chi1chi2: bool = True, use_prec_ang: bool = True) -> np.ndarray:
        fgrids, psd = self._get_f_psd_array(params1, res)
        psd_sqrt = psd ** 0.5

        params1 = self._convert_parameters_strain(params1, use_m1m2, use_chi1chi2, use_prec_ang)
        params2 = self._convert_parameters_strain(params2, use_m1m2, use_chi1chi2, use_prec_ang)

        if self.detector_shape == 'L':
            strain1 = self.GWstrain(fgrids, **params1, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
            strain2 = self.GWstrain(fgrids, **params2, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
            noise = create_gaussian_noise(fgrids[:, 0], psd[:, 0], scale=noise_scale)
            return self.log_likelihood_from_strain(fgrids, strain1, strain2, noise, psd_sqrt)
        else:
            log_l = []
            strain1 = []
            strain2 = []
            noises = []
            for i in range(2):
                strain1_i = self.GWstrain(fgrids, **params1, rot=i*60., is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                strain2_i = self.GWstrain(fgrids, **params2, rot=i*60., is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                noise_i = create_gaussian_noise(fgrids[:, 0], psd[:, 0], scale=noise_scale)
                log_l.append(self.log_likelihood_from_strain(fgrids, strain1_i, strain2_i, noise_i, psd_sqrt))
                strain1.append(strain1_i)
                strain2.append(strain2_i)
                noises.append(noise_i)

            if not self.compute2arms:
                strain1_3 = self.GWstrain(fgrids, **params1, rot=120., is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                strain2_3 = self.GWstrain(fgrids, **params2, rot=120., is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
            else:
                strain1_3 = -onp.sum(strain1, axis=0)
                strain2_3 = -onp.sum(strain2, axis=0)
            noise_3 = -onp.sum(noises, axis=0)
            log_l.append(self.log_likelihood_from_strain(fgrids, strain1_3, strain2_3, noise_3, psd_sqrt))
            return onp.sum(log_l, axis=0)

    def energy(self, evParams: ParamsTypeInput, res: int = 1000) -> np.ndarray:
        params = self._convert_parameters_default(evParams)
        fgrids = self._get_f_psd_array(evParams, res, return_psd=False)

        iota = params['iota']
        amplitude = self.wf_model.Ampl(fgrids, **params)
        energy = amplitude**2 * ((0.5*(1+np.cos(iota)**2))**2 + np.cos(iota)**2)
        return fgrids, energy.real


class DetNetExtended(DetNet):
    def log_likelihood(self, params1: ParamsTypeInput, params2: ParamsTypeInput, res: int = 1000, noise_scale: float = 0.5) -> np.ndarray:
        log_l = {}
        for name, signal in self.signals.items():
            log_l[name] = signal.waveform_log_likelihood(params1, params2, res, noise_scale=noise_scale)
        return log_l
