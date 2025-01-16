# PoppinGW
**Pop**ulation **In**ference for **GW** detection

## Installation

```
pip install poppingw
```

## Usage

The whole workflow includes generating mock GW events, calculating likelihood $\mathcal{L}(\kappa,\iota|d)$ for each events $\{d_i\}$, and perform population inference on events with SNR larger than a certain threshold value $\rho_{\rm thr}$. An example script [total_workflow.sh](example/total_workflow.sh) is provided to perform this whole procedure.

### Preparation

Before run `total_workflow.sh`, you need to prepare the following `npz` files (generate them with `numpy.savez`):
- `result/pop.npz`: defines parameterization for $\log\kappa$ distribution. We fit $p(\log\kappa)$ with spline interpolation funtion, the corresponding configurations should be stored in this file containing following keys:
  - `control`: $\log\kappa$ values on control points of interpolation, size $(L,)$ ($L$ is defined by user).
  - `p_control`: truth values of $p(\log\kappa)$ on control points, size $(L,)$.
  - `range`: lower and upper limit of $\log\kappa$, size $(2,)$.
- `result/likelihood/range.npz`: defines limits of $\log\kappa$ and $\cos\iota$ grid for calculating likelihood. Should contain following keys:
  - `log_kappa`: limits of $\log\kappa$, size $(N,2)$, where $N$ is number of total events to calculate likelihood for.
  - `cos_iota`: limits of $\cos\iota$, size $(N,2)$.

### Generate events

See [sample_events.py](/sample_events.py). After running this script, one `h5` file will be generated, which is the generated GW catalog. 

Two independent catalogs are needed for population inference, one catalog "infer" as mocked observations, the other catalog "select" used for calculating detection rate during the inference. The two catalogs should have all parameters except $\kappa$ follwing the same distribution. Number of the "select" catalog should be at least four times of the "infer" catalog.

### Calculate Likelihood

See [match_filter_likelihood.py](/match_filter_likelihood.py). Provide the "infer" catalog and configurations of the detector network (see [detectors/ET_2CE_gwfast.json](detectors/ET_2CE_gwfast.json) for an example) and $\log\kappa$ - $\cos\iota$ 2D grid, this script will calculate and store 2D log-likelihood $\log\mathcal{L}(\kappa,\iota|d)$ and 1D likelihood $\mathcal{L}(\kappa|d)$ for each event.

If `range.npz` is provided with argument `--range-path`, the log-likelihood is calculated on uniformly separated $\log\kappa,\cos\iota$ 2D grid. The grid size is defined by values of the arguments `--dL-shape` and `--iota-shape`, default to be $100\times 100$. Insteadly, you can also provide `grid.npz` through argument `--grid-path`, which defines $P\times Q$ grid for the events. `grid.npz` also contains keys `log_kappa` and `cos_iota`, shape os both are $(N,P,Q)$. If provided, the arguments `--range-path`, `--dL-shape` and `--iota-shape` will be ignored.

### Population Inference

See [infer_spl.py](/infer_spl.py). "Infer" and "select" catalog, `pop.npz`, 1D kappa likelihood are needed.

You can pass the argument `-disable-select` to disable selection effect, letting all of the events to be included in the poopulation inference and the detection rate in the inference becomes constant unity.

## Citation

Cite our work:

**Improving the detection sensitivity to primordial stochastic gravitational waves with reduced astrophysical foregrounds. II. Subthreshold binary neutron stars**, [Arxiv:2403.01846](https://arxiv.org/abs/2403.01846) [PhysRevD.111.023009](https://doi.org/10.1103/PhysRevD.111.023009)

if you find this code useful to you.