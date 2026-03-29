"""
run_digital_twin.py — runs the full digital twin pipeline.
  M1: system identification (FFT + damping from Session 1)
  M2: forward model (eigenvalue problem, initial k)
  M3: Bayesian model updating (TMCMC on Session 1 frequencies)
  M4: independent validation (predict Session 2 frequencies from posterior)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from config import (
    MASSES, F_MEASURED, SAMPLING_RATE,
    SESSION_1_FILE, SESSION_2_FILE,
    IMPACT_SHEET_S1, IMPACT_SHEET_S2,
    K_PRIOR_LOW, K_PRIOR_HIGH, SIGMA_FRACTION,
    NSAMPLES, TMCMC_BETA, RANDOM_SEED
)
from forward_model import compute_natural_frequencies, compute_initial_k
from signal_processing import (
    load_sheet, amplitude_spectrum, find_spectral_peaks,
    estimate_damping, plot_fft_three_floors,
    plot_damping_three_modes, plot_mode_shapes
)
from bayesian_updating import (
    log_likelihood, tmcmc_sampler,
    plot_prior_posterior, plot_frequency_comparison, plot_convergence
)

os.makedirs('figures', exist_ok=True)

# plot defaults (same across all modules)
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3,
})

MASSES_ARR = np.array(MASSES)
F_MEAS_ARR = np.array(F_MEASURED)


# ==================================================================
# MODULE 1: system identification
# ==================================================================
def run_module_1():
    print("\n" + "=" * 65)
    print("  MODULE 1: SYSTEM IDENTIFICATION")
    print("=" * 65)

    t, ch1, ch2, ch3 = load_sheet(SESSION_1_FILE, IMPACT_SHEET_S1)
    fs = SAMPLING_RATE
    print(f"  Loaded {len(t)} pts from {SESSION_1_FILE}")

    # FFT — identify natural frequencies per floor
    print("\n  Identified frequencies:")
    for lbl, ch in [('Fl1', ch1), ('Fl2', ch2), ('Fl3', ch3)]:
        pf, _ = find_spectral_peaks(*amplitude_spectrum(ch, fs), n_peaks=3)
        print(f"    {lbl}: {[f'{x:.1f}' for x in pf]} Hz")

    plot_fft_three_floors(
        [ch1, ch2, ch3], fs,
        'Amplitude spectrum — Impact test (Session 1)',
        'fig_5_2_fft_impact_session1.png')

    # damping from post-impact decay
    print("\n  Damping estimation:")
    ch3_zm = ch3 - np.mean(ch3)
    ch1_zm = ch1 - np.mean(ch1)
    thr = 5 * np.std(ch3_zm[:int(0.5 * fs)])
    imp_idx = np.argmax(np.abs(ch3_zm) > thr)

    modes_cfg = [
        ('Mode 1', 7.2, 5.0, 9.5, ch3_zm[imp_idx:]),
        ('Mode 2', 21.0, 17.5, 24.0, ch1_zm[imp_idx:]),
        ('Mode 3', 30.5, 27.0, 34.0, ch3_zm[imp_idx:]),
    ]
    damp_data = []
    damping = {}
    for ml, fn, fl, fh, sig in modes_cfg:
        try:
            z, a, A0, td, es = estimate_damping(sig, fs, fn, fl, fh)
            damping[ml] = z
            damp_data.append((td, es, z, a, A0, fn, ml))
            print(f"    {ml}: zeta = {z:.4f} ({z*100:.2f}%)")
        except Exception as e:
            print(f"    {ml}: FAILED — {e}")

    if damp_data:
        plot_damping_three_modes(damp_data, 'fig_5_5_damping_estimation.png')
    return damping


# ==================================================================
# MODULE 2: forward model
# ==================================================================
def run_module_2():
    print("\n" + "=" * 65)
    print("  MODULE 2: FORWARD MODEL")
    print("=" * 65)

    k_init = compute_initial_k(MASSES, F_MEASURED)
    f_pred, modes = compute_natural_frequencies(k_init, MASSES)

    print(f"  k_init = {k_init:.0f} N/m")
    print(f"  f_pred = [{f_pred[0]:.2f}, {f_pred[1]:.2f}, {f_pred[2]:.2f}] Hz")
    print(f"  f_meas = {F_MEASURED} Hz")
    err = np.abs(f_pred - F_MEAS_ARR) / F_MEAS_ARR * 100
    print(f"  error  = [{err[0]:.1f}%, {err[1]:.1f}%, {err[2]:.1f}%]")

    plot_mode_shapes(f_pred, modes, k_init, 'fig_5_7_mode_shapes.png')
    return k_init, f_pred, modes


# ==================================================================
# MODULE 3: Bayesian model updating (calibration on Session 1)
# ==================================================================
def run_module_3():
    print("\n" + "=" * 65)
    print("  MODULE 3: BAYESIAN MODEL UPDATING")
    print("=" * 65)

    sigma = SIGMA_FRACTION * F_MEAS_ARR
    k_prior_mean = (K_PRIOR_LOW + K_PRIOR_HIGH) / 2.0
    prior_std = (K_PRIOR_HIGH - K_PRIOR_LOW) / np.sqrt(12)

    print(f"  Prior: U[{K_PRIOR_LOW:.0f}, {K_PRIOR_HIGH:.0f}] N/m")
    print(f"  Sigma: {sigma} Hz")

    def logl(k):
        return log_likelihood(k, MASSES_ARR, F_MEAS_ARR, sigma)

    print("\n  Running TMCMC...")
    samples, log_ev, stages = tmcmc_sampler(
        logl, K_PRIOR_LOW, K_PRIOR_HIGH,
        nsamples=NSAMPLES, beta=TMCMC_BETA, seed=RANDOM_SEED)

    km = np.mean(samples)
    ks = np.std(samples)
    ci = np.percentile(samples, [2.5, 97.5])
    red = (1 - ks / prior_std) * 100

    f_prior, _ = compute_natural_frequencies(k_prior_mean, MASSES_ARR)
    f_post, _ = compute_natural_frequencies(km, MASSES_ARR)

    print(f"\n  Posterior: k = {km:.0f} ± {ks:.0f} N/m")
    print(f"  95% CI:   [{ci[0]:.0f}, {ci[1]:.0f}] N/m")
    print(f"  Uncertainty reduction: {red:.0f}%")

    # per-mode comparison table
    print(f"\n  {'Mode':<6} {'Meas':>8} {'Prior':>8} {'Post':>8} {'Err_pr':>8} {'Err_po':>8}")
    for i in range(3):
        ep = abs(f_prior[i] - F_MEAS_ARR[i]) / F_MEAS_ARR[i] * 100
        epo = abs(f_post[i] - F_MEAS_ARR[i]) / F_MEAS_ARR[i] * 100
        print(f"  {i+1:<6} {F_MEAS_ARR[i]:>8.2f} {f_prior[i]:>8.2f} "
              f"{f_post[i]:>8.2f} {ep:>7.1f}% {epo:>7.1f}%")

    # figures
    plot_prior_posterior(samples, K_PRIOR_LOW, K_PRIOR_HIGH,
                        'fig_5_8_prior_posterior.png')
    plot_frequency_comparison(F_MEAS_ARR, f_prior, f_post,
                             'fig_5_9_frequency_comparison.png')
    plot_convergence(stages, km, 'fig_5_10_tmcmc_convergence.png')

    return samples, km, ks, red


# ==================================================================
# MODULE 4: independent validation (Session 2)
# ==================================================================
def run_module_4(posterior_samples):
    """
    Validate posterior k against an independent dataset (Session 2).
    - extract frequencies from Session 2 impact test via FFT
    - propagate posterior samples through eigenvalue model
    - check whether Session 2 freqs fall within 95% credible interval
    """
    print("\n" + "=" * 65)
    print("  MODULE 4: INDEPENDENT VALIDATION (Session 2)")
    print("=" * 65)

    # --- Session 2 FFT ---
    t, ch1, ch2, ch3 = load_sheet(SESSION_2_FILE, IMPACT_SHEET_S2)
    fs = SAMPLING_RATE
    print(f"  Loaded {len(t)} pts from {SESSION_2_FILE}")

    plot_fft_three_floors(
        [ch1, ch2, ch3], fs,
        'Amplitude spectrum — Impact test (Session 2)',
        'fig_5_4_fft_impact_session2.png')

    # identify Session 2 frequencies (average across floors)
    all_pf = []
    for lbl, ch in [('Fl1', ch1), ('Fl2', ch2), ('Fl3', ch3)]:
        pf, _ = find_spectral_peaks(*amplitude_spectrum(ch, fs), n_peaks=3)
        print(f"    {lbl}: {[f'{x:.1f}' for x in pf]} Hz")
        if len(pf) == 3:
            all_pf.append(pf)

    f_s2 = np.mean(all_pf, axis=0)
    print(f"\n  Session 2 mean freqs: [{f_s2[0]:.2f}, {f_s2[1]:.2f}, {f_s2[2]:.2f}] Hz")
    print(f"  Session 1 freqs:     {F_MEASURED} Hz")

    # --- predictive distribution from posterior ---
    n_post = len(posterior_samples)
    f_pred_all = np.zeros((n_post, 3))
    for i, k in enumerate(posterior_samples):
        f_pred_all[i, :], _ = compute_natural_frequencies(k, MASSES_ARR)

    # posterior-predictive statistics
    f_pred_mean = np.mean(f_pred_all, axis=0)
    f_pred_lo = np.percentile(f_pred_all, 2.5, axis=0)
    f_pred_hi = np.percentile(f_pred_all, 97.5, axis=0)

    print(f"\n  Posterior-predictive (from Session 1 calibration):")
    print(f"  {'Mode':<6} {'S2 meas':>8} {'Pred mean':>10} {'95% CI lo':>10} {'95% CI hi':>10} {'In CI?':>8}")
    validated = True
    for i in range(3):
        in_ci = f_pred_lo[i] <= f_s2[i] <= f_pred_hi[i]
        flag = 'YES' if in_ci else 'NO'
        if not in_ci:
            validated = False
        print(f"  {i+1:<6} {f_s2[i]:>8.2f} {f_pred_mean[i]:>10.2f} "
              f"{f_pred_lo[i]:>10.2f} {f_pred_hi[i]:>10.2f} {flag:>8}")

    if validated:
        print("\n  >> VALIDATION PASSED: all Session 2 freqs within 95% CI")
    else:
        print("\n  >> VALIDATION PARTIAL: not all freqs within 95% CI (see discussion)")

    # --- validation figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    mode_labels = [r'Mode 1 ($f_1$)', r'Mode 2 ($f_2$)', r'Mode 3 ($f_3$)']

    # posterior mean with 95% CI as error bars
    ci_err = np.array([f_pred_mean - f_pred_lo, f_pred_hi - f_pred_mean])
    ax.errorbar(x, f_pred_mean, yerr=ci_err, fmt='o', color='red',
                markersize=8, capsize=6, capthick=1.5, elinewidth=1.5,
                zorder=5, label=f'Posterior mean ± 95% CI (k={np.mean(posterior_samples):.0f} N/m)')

    # Session 1 measured (calibration)
    ax.plot(x - 0.08, F_MEAS_ARR, 'x', color='blue', markersize=10,
            markeredgewidth=2, zorder=5, label='Session 1 (calibration)')

    # Session 2 measured (validation)
    ax.plot(x + 0.08, f_s2, 'x', color='green', markersize=10,
            markeredgewidth=2, zorder=5, label='Session 2 (validation)')

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel('Natural frequency [Hz]')
    ax.set_title('Validation — posterior prediction vs independent Session 2 data')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig_5_11_validation.png', dpi=300)
    plt.close()
    print("  Saved: figures/fig_5_11_validation.png")

    return f_s2, f_pred_mean, f_pred_lo, f_pred_hi, validated


# ==================================================================
# MAIN
# ==================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  OPEN-SOURCE DIGITAL TWIN FOR STRUCTURAL DYNAMICS")
    print("  University of Strathclyde — Osman Mukuk, 2025-26")
    print("=" * 65)

    # M1: identify modal properties from Session 1
    damping = run_module_1()

    # M2: eigenvalue model, initial k from freq matching
    k_init, f_init, modes = run_module_2()

    # M3: calibrate k via TMCMC on Session 1
    samples, k_post, k_std, reduction = run_module_3()

    # M4: validate posterior against independent Session 2 data
    f_s2, f_pred, ci_lo, ci_hi, valid = run_module_4(samples)

    # --- final summary ---
    f_final, _ = compute_natural_frequencies(k_post, MASSES)
    print("\n" + "=" * 65)
    print("  DIGITAL TWIN COMPLETE")
    print("=" * 65)
    print(f"  Calibrated k:   {k_post:.0f} ± {k_std:.0f} N/m")
    print(f"  Uncertainty red: {reduction:.0f}%")
    print(f"  Final freqs:     [{f_final[0]:.2f}, {f_final[1]:.2f}, {f_final[2]:.2f}] Hz")
    print(f"  Validation:      {'PASSED' if valid else 'PARTIAL'}")
    print(f"\n  Figures saved in ./figures/")
    print("=" * 65)
