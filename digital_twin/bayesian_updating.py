"""
bayesian_updating.py — Module 3: TMCMC sampler for stiffness calibration.
Python port of the MATLAB code from Lye, Cicirello & Patelli (2021), MSSP 159.
Ref: Ching & Chen (2007), J. Eng. Mech. 133(7).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from forward_model import compute_natural_frequencies

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3,
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log_likelihood(k_global, masses, measured_freqs, sigma):
    """Gaussian log-likelihood: predicted vs measured natural frequencies."""
    predicted, _ = compute_natural_frequencies(k_global, masses)
    residuals = (measured_freqs - predicted) / sigma
    return -0.5 * np.sum(residuals ** 2)


def tmcmc_sampler(log_likelihood_fn, prior_low, prior_high,
                  nsamples=1000, beta=0.2, seed=42):
    """
    TMCMC for 1D parameter. Adaptive tempering, importance resampling, MH chains.
    Returns posterior samples, log evidence, and per-stage diagnostics.
    """
    rng = np.random.default_rng(seed)

    # draw initial samples from uniform prior
    theta = rng.uniform(prior_low, prior_high, size=nsamples)
    p_j = 0.0
    log_S = []
    stages = []
    it = 0

    while p_j < 1.0:
        it += 1
        log_L = np.array([log_likelihood_fn(t) for t in theta])

        if np.any(np.isinf(log_L)):
            raise ValueError("Likelihood returned -inf; check prior bounds.")

        # adaptive tempering step
        p_j1 = _find_next_p(log_L, p_j)

        # importance weights
        log_w = (p_j1 - p_j) * log_L
        w = np.exp(log_w - np.max(log_w))
        w_norm = w / np.sum(w)
        log_S.append(np.log(np.mean(np.exp(log_w))))

        # proposal covariance from weighted samples
        mu = np.sum(w_norm * theta)
        cov = np.sum(w_norm * (theta - mu) ** 2)
        prop_std = beta * np.sqrt(cov)

        # resample + MH perturbation (3 sub-steps per sample)
        idx = rng.choice(nsamples, size=nsamples, replace=True, p=w_norm)
        theta_new = np.empty(nsamples)
        acc = 0

        for i in range(nsamples):
            cur = theta[idx[i]]
            lp_cur = _log_prior(cur, prior_low, prior_high) + p_j1 * log_likelihood_fn(cur)

            for _ in range(3):
                cand = cur + prop_std * rng.standard_normal()
                lp_cand_prior = _log_prior(cand, prior_low, prior_high)
                if np.isfinite(lp_cand_prior):
                    lp_cand = lp_cand_prior + p_j1 * log_likelihood_fn(cand)
                    if np.log(rng.random()) < (lp_cand - lp_cur):
                        cur = cand
                        lp_cur = lp_cand
                        acc += 1
            theta_new[i] = cur

        acc_rate = acc / (nsamples * 3)
        stages.append({
            'iteration': it, 'p': p_j1,
            'mean': np.mean(theta_new), 'std': np.std(theta_new),
            'acceptance_rate': acc_rate,
        })
        print(f"  Stage {it}: p={p_j1:.4f}, k={np.mean(theta_new):.1f}"
              f"±{np.std(theta_new):.1f}, acc={acc_rate:.1%}")

        theta = theta_new
        p_j = p_j1

    return theta, np.sum(log_S), stages


def _log_prior(k, low, high):
    """Uniform prior in log-space. Returns -inf outside bounds."""
    if low <= k <= high:
        return -np.log(high - low)
    return -np.inf


def _find_next_p(log_L, p_current, cov_target=1.0):
    """Bisection for next tempering parameter: CoV(weights) <= target."""
    def cov_w(dp):
        lw = dp * log_L - np.max(dp * log_L)
        w = np.exp(lw)
        return np.std(w) / np.mean(w) - cov_target

    lo, hi = 0.0, 1.0 - p_current
    if cov_w(hi) <= 0:
        return 1.0   # jump straight to posterior
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if cov_w(mid) > 0:
            hi = mid
        else:
            lo = mid
    return p_current + lo


# --- plotting ---

def plot_prior_posterior(posterior_samples, prior_low, prior_high, filename):
    """Prior (uniform fill) vs posterior (histogram)."""
    k_mean = np.mean(posterior_samples)
    k_prior_mean = (prior_low + prior_high) / 2.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    k_range = np.linspace(prior_low - 5000, prior_high + 5000, 500)
    prior_v = np.where((k_range >= prior_low) & (k_range <= prior_high),
                       1.0 / (prior_high - prior_low), 0)
    ax.fill_between(k_range, prior_v * 1e3, alpha=0.25, color='C0', label='Prior')
    ax.hist(posterior_samples, bins=40, density=True, alpha=0.6, color='C3',
            edgecolor='0.3', linewidth=0.5, label='Posterior')
    ax.axvline(k_mean, color='C3', ls='--', lw=1.2,
               label=f'Posterior mean: {k_mean:.0f} N/m')
    ax.axvline(k_prior_mean, color='C0', ls='--', lw=1.2,
               label=f'Prior mean: {k_prior_mean:.0f} N/m')
    ax.set_xlabel('Storey stiffness k [N/m]')
    ax.set_ylabel('Probability density')
    ax.set_title('Prior vs posterior distribution of k')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"  Saved: figures/{filename}")


def plot_frequency_comparison(f_meas, f_prior, f_post, filename):
    """Grouped bar chart: measured / prior-predicted / posterior-predicted."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    w = 0.25
    modes = [r'Mode 1 ($f_1$)', r'Mode 2 ($f_2$)', r'Mode 3 ($f_3$)']
    for dx, vals, lbl, col in [(-w, f_meas, 'Measured', '0.3'),
                                (0, f_prior, 'Prior model', 'C0'),
                                (w, f_post, 'Posterior model', 'C3')]:
        bars = ax.bar(x + dx, vals, w, label=lbl, color=col, alpha=0.8,
                      edgecolor='0.1')
        for b in bars:
            ax.annotate(f'{b.get_height():.1f}',
                        xy=(b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel('Natural frequency [Hz]')
    ax.set_title('Frequency comparison — measured vs model predictions')
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"  Saved: figures/{filename}")


def plot_convergence(stages, k_post_mean, filename):
    """Tempering parameter + k estimate vs TMCMC stage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    iters = [s['iteration'] for s in stages]

    ax1.plot(iters, [s['p'] for s in stages], 'o-', color='C0', ms=5)
    ax1.set(xlabel='TMCMC stage', ylabel='Tempering parameter p',
            title='Convergence of p', ylim=[-0.05, 1.1])

    ax2.errorbar(iters, [s['mean'] for s in stages],
                 yerr=[s['std'] for s in stages],
                 fmt='o-', color='C3', ms=5, capsize=3)
    ax2.axhline(k_post_mean, color='C3', ls='--', alpha=0.5)
    ax2.set(xlabel='TMCMC stage', ylabel='k [N/m]',
            title='Stiffness estimate per stage')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"  Saved: figures/{filename}")


# --- standalone ---

if __name__ == '__main__':
    from config import (MASSES, F_MEASURED, K_PRIOR_LOW, K_PRIOR_HIGH,
                        SIGMA_FRACTION, NSAMPLES, TMCMC_BETA, RANDOM_SEED)

    M = np.array(MASSES)
    F = np.array(F_MEASURED)
    S = SIGMA_FRACTION * F

    print("=" * 65)
    print("  MODULE 3: BAYESIAN MODEL UPDATING")
    print("=" * 65)

    def logl(k):
        return log_likelihood(k, M, F, S)

    samples, log_ev, stages = tmcmc_sampler(
        logl, K_PRIOR_LOW, K_PRIOR_HIGH,
        nsamples=NSAMPLES, beta=TMCMC_BETA, seed=RANDOM_SEED)

    km = np.mean(samples)
    ks = np.std(samples)
    print(f"\n  Posterior: k = {km:.0f} ± {ks:.0f} N/m")

    f_prior, _ = compute_natural_frequencies((K_PRIOR_LOW+K_PRIOR_HIGH)/2, M)
    f_post, _ = compute_natural_frequencies(km, M)

    plot_prior_posterior(samples, K_PRIOR_LOW, K_PRIOR_HIGH, 'fig_5_8_prior_posterior.png')
    plot_frequency_comparison(F, f_prior, f_post, 'fig_5_9_frequency_comparison.png')
    plot_convergence(stages, km, 'fig_5_10_tmcmc_convergence.png')
