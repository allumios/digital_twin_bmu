# Open-Source Digital Twin for Structural Dynamics

Python-based digital twin for a 3-DOF shear frame structure.
Calibrates storey stiffness using Bayesian Model Updating (TMCMC)
and validates against an independent experimental dataset.

**Author:** Osman Mukuk, University of Strathclyde  
**Supervisor:** Dr Marco De Angelis  
**Academic Year:** 2025-2026

## What it does

Given accelerometer data from shaking table tests, this tool:

1. **Identifies** natural frequencies, mode shapes, and damping ratios (FFT + Hilbert envelope)
2. **Builds** a physics-based forward model (generalised eigenvalue problem)
3. **Calibrates** the storey stiffness using Bayesian inference (TMCMC sampler)
4. **Validates** the calibrated model against an independent test session

## Quick start

```bash
pip install numpy scipy matplotlib openpyxl
python run_digital_twin.py
```

Results are saved to the `figures/` directory.

## For your own structure

1. Place your experimental Excel files in `data/`
2. Edit `config.py` with your masses, file paths, sheet names, and sampling rate
3. Run `python run_digital_twin.py`

## Repository structure

```
├── config.py                # all user settings (edit this)
├── run_digital_twin.py      # main entry point — runs M1-M4
├── signal_processing.py     # Module 1: FFT, damping estimation
├── forward_model.py         # Module 2: eigenvalue solver
├── bayesian_updating.py     # Module 3: TMCMC sampler
├── data/                    # experimental Excel files (not tracked)
├── figures/                 # output plots (not tracked)
├── requirements.txt
└── README.md
```

## Method

- **System identification:** FFT with Hanning window, peak picking, Hilbert envelope for damping
- **Forward model:** generalised eigenvalue problem via `scipy.linalg.eig`
- **Bayesian updating:** TMCMC (Ching & Chen, 2007) with uniform prior and Gaussian likelihood
- **Validation:** posterior-predictive check against independent Session 2 frequencies

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib, openpyxl

## References

- Lye, A., Cicirello, A. and Patelli, E. (2021) 'Sampling methods for solving Bayesian model updating problems: A tutorial', *MSSP*, 159, 107760.
- Ching, J. and Chen, Y. (2007) 'Transitional Markov Chain Monte Carlo method for Bayesian model updating', *J. Eng. Mech.*, 133(7), 816-832.
- Wagg, D.J. et al. (2022) 'Development of a digital twin operational platform using Python Flask', *Data-Centric Engineering*, 3, e1.
- Chopra, A.K. (2012) *Dynamics of Structures*. 4th edn. Pearson.

## Licence

MIT
