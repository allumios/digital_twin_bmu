"""
signal_processing.py — Module 1: system identification from accelerometer data.
FFT for natural frequencies, Hilbert envelope for damping ratios.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.optimize import curve_fit
import openpyxl
import os
import gc

from forward_model import compute_natural_frequencies, compute_initial_k

# --- plot defaults: serif, 300dpi, light grid ---
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'lines.linewidth': 0.8, 'axes.grid': True, 'grid.alpha': 0.3,
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- data loading ---

def load_sheet(filepath, sheet_name):
    """Read time + 3 accel channels from Excel sheet. Cols: D=time, F/G/H=ch1-3."""
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    data = np.array(rows, dtype=float)
    return data[:, 3], data[:, 5], data[:, 6], data[:, 7]


# --- FFT ---

def amplitude_spectrum(signal, fs, f_max=40.0):
    """Single-sided amplitude spectrum, Hanning window, properly scaled."""
    N = len(signal)
    window = np.hanning(N)
    windowed = (signal - np.mean(signal)) * window
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    amp = 2.0 * np.abs(fft_vals) / np.sum(window)
    mask = freqs <= f_max
    return freqs[mask], amp[mask]


def find_spectral_peaks(freqs, amplitude, n_peaks=3, min_freq=5.0,
                        min_distance_hz=4.0):
    """Top N peaks above min_freq, sorted by frequency."""
    freq_res = freqs[1] - freqs[0]
    min_dist = max(1, int(min_distance_hz / freq_res))
    mask = freqs >= min_freq
    f_sub, a_sub = freqs[mask], amplitude[mask]
    idx, _ = find_peaks(a_sub, distance=min_dist)
    if len(idx) == 0:
        return np.array([]), np.array([])
    # pick strongest, then sort by freq
    top = idx[np.argsort(a_sub[idx])[::-1][:n_peaks]]
    top = top[np.argsort(f_sub[top])]
    return f_sub[top], a_sub[top]


# --- filtering ---

def bandpass_filter(signal, f_low, f_high, fs, order=3):
    """Zero-phase Butterworth bandpass."""
    nyq = 0.5 * fs
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype='band')
    return filtfilt(b, a, signal)


# --- damping ---

def estimate_damping(signal, fs, f_n, f_band_low, f_band_high,
                     t_start_offset=0.05, t_window=12.0):
    """
    Damping ratio from Hilbert envelope of bandpass-filtered decay.
    Fits A0*exp(-alpha*t), then zeta = alpha / (2*pi*f_n).
    """
    filtered = bandpass_filter(signal, f_band_low, f_band_high, fs)
    envelope = np.abs(hilbert(filtered))

    i0 = int(t_start_offset * fs)
    i1 = min(len(envelope), i0 + int(t_window * fs))
    t_decay = np.arange(i1 - i0) / fs
    env_decay = envelope[i0:i1]

    # smooth over 2 vibration periods
    win = max(1, int(2 * fs / f_n))
    env_smooth = np.convolve(env_decay, np.ones(win) / win, mode='same')

    # subsample + threshold for fitting
    step = max(1, len(t_decay) // 500)
    ts, es = t_decay[::step], env_smooth[::step]
    keep = es > np.max(es) * 0.01
    ts, es = ts[keep], es[keep]

    popt, _ = curve_fit(lambda t, A0, a: A0 * np.exp(-a * t),
                        ts, es, p0=[es[0], 1.0], maxfev=10000)
    A0, alpha = popt
    zeta = alpha / (2 * np.pi * f_n)
    return zeta, alpha, A0, t_decay, env_smooth


# --- plotting ---

def plot_fft_three_floors(channels_list, fs, test_name, filename, f_max=40.0):
    """3-panel FFT plot with adaptive peak labels."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    floor_labels = ['Floor 1', 'Floor 2', 'Floor 3']

    for ax, ch, label in zip(axes, channels_list, floor_labels):
        freqs, amp = amplitude_spectrum(ch, fs, f_max)
        ax.plot(freqs, amp, color='0.15', linewidth=0.6)
        ax.set_ylabel('Acceleration\namplitude [a.u.]')
        ax.set_xlim([3, 38])

        # floor label box
        ax.text(0.97, 0.90, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))

        # annotate top 3 peaks
        pf, pa = find_spectral_peaks(freqs, amp, n_peaks=3, min_freq=5.0)
        x_lo, x_hi = 3, 38
        right_margin = x_hi - 0.15 * (x_hi - x_lo)
        close_thr = 0.08 * (x_hi - x_lo)
        prev = []

        for f_pk, a_pk in zip(pf, pa):
            ax.plot(f_pk, a_pk, 'o', color='C3', markersize=4, zorder=5)
            x_off, ha_a = (-8, 'right') if f_pk > right_margin else (8, 'left')
            y_off = -2
            for pf_prev, py_prev in prev:
                if abs(f_pk - pf_prev) < close_thr:
                    y_off = py_prev - 12
            prev.append((f_pk, y_off))
            ax.annotate(f'{f_pk:.1f} Hz', (f_pk, a_pk),
                        textcoords='offset points', xytext=(x_off, y_off),
                        fontsize=9, color='C3', ha=ha_a, va='top')

    axes[2].set_xlabel('Frequency [Hz]')
    axes[0].set_title(test_name, fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_mode_shapes(frequencies, mode_shapes, k_val, filename):
    """Mode shape plot for 3-DOF shear frame."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)
    floors = np.array([0, 1, 2, 3])
    for i, ax in enumerate(axes):
        shape = np.concatenate(([0.0], mode_shapes[:, i]))
        ax.plot(shape, floors, 'o-', color='C0', markersize=7, linewidth=1.8)
        ax.axvline(0, color='0.6', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Normalised displacement')
        ax.set_title(f'Mode {i+1}\nf = {frequencies[i]:.2f} Hz')
        ax.set_xlim([-1.3, 1.3])
        ax.set_yticks(floors)
        ax.set_yticklabels(['Base', 'Floor 1', 'Floor 2', 'Floor 3'])
    fig.suptitle(f'Predicted mode shapes (k = {k_val:.0f} N/m)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_damping_three_modes(axes_data, filename):
    """Envelope + exp fit for 3 modes."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for ax, (t_dec, env_s, zeta, alpha, A0, fn, ml) in zip(axes, axes_data):
        ax.plot(t_dec, env_s, color='0.3', linewidth=0.6, label='Smoothed envelope')
        t_fit = np.linspace(0, t_dec[-1], 500)
        ax.plot(t_fit, A0 * np.exp(-alpha * t_fit), '--', color='C3', linewidth=1.5,
                label=fr'Exp. fit: $\zeta$ = {zeta:.4f} ({zeta*100:.2f}%)')
        ax.set_title(f'{ml} (f = {fn:.1f} Hz)')
        ax.set_ylabel('Envelope [a.u.]')
        ax.legend(loc='upper right', framealpha=0.9)
    axes[2].set_xlabel('Time after impact [s]')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# --- standalone ---

if __name__ == '__main__':
    from config import (MASSES, SAMPLING_RATE, F_MEASURED,
                        SESSION_1_FILE, SESSION_2_FILE,
                        IMPACT_SHEET_S1, IMPACT_SHEET_S2)

    FS = SAMPLING_RATE
    F_MEAS = np.array(F_MEASURED)

    print("=" * 65)
    print("  MODULE 1: SYSTEM IDENTIFICATION")
    print("=" * 65)

    # Session 1 FFT
    print("\n--- FFT: Impact test, Session 1 ---")
    t, ch1, ch2, ch3 = load_sheet(SESSION_1_FILE, IMPACT_SHEET_S1)
    print(f"  {len(t)} samples ({t[-1]:.1f} s)")
    plot_fft_three_floors([ch1, ch2, ch3], FS,
        'Amplitude spectrum — Impact test (Session 1)',
        'fig_5_2_fft_impact_session1.png')
    for lbl, ch in [('Fl1', ch1), ('Fl2', ch2), ('Fl3', ch3)]:
        pf, _ = find_spectral_peaks(*amplitude_spectrum(ch, FS), n_peaks=3)
        print(f"  {lbl}: {[f'{x:.1f}' for x in pf]} Hz")
    del t, ch1, ch2, ch3; gc.collect()

    # Session 2 FFT
    print("\n--- FFT: Impact test, Session 2 ---")
    t, ch1, ch2, ch3 = load_sheet(SESSION_2_FILE, IMPACT_SHEET_S2)
    print(f"  {len(t)} samples ({t[-1]:.1f} s)")
    plot_fft_three_floors([ch1, ch2, ch3], FS,
        'Amplitude spectrum — Impact test (Session 2)',
        'fig_5_4_fft_impact_session2.png')
    del t, ch1, ch2, ch3; gc.collect()

    # initial stiffness
    print("\n--- Initial stiffness (frequency matching) ---")
    K_INIT = compute_initial_k(MASSES, F_MEASURED)
    F_PRED, MODES = compute_natural_frequencies(K_INIT, MASSES)
    print(f"  k_init = {K_INIT:.0f} N/m")
    print(f"  f_pred = [{F_PRED[0]:.2f}, {F_PRED[1]:.2f}, {F_PRED[2]:.2f}] Hz")
    plot_mode_shapes(F_PRED, MODES, K_INIT, 'fig_5_7_mode_shapes.png')

    # damping
    print("\n--- Damping (Impact, Session 1) ---")
    t, ch1, ch2, ch3 = load_sheet(SESSION_1_FILE, IMPACT_SHEET_S1)
    ch3_zm = ch3 - np.mean(ch3)
    ch1_zm = ch1 - np.mean(ch1)
    thr = 5 * np.std(ch3_zm[:int(0.5 * FS)])
    imp_idx = np.argmax(np.abs(ch3_zm) > thr)

    modes_cfg = [
        ('Mode 1', 7.2, 5.0, 9.5, ch3_zm[imp_idx:]),
        ('Mode 2', 21.0, 17.5, 24.0, ch1_zm[imp_idx:]),
        ('Mode 3', 30.5, 27.0, 34.0, ch3_zm[imp_idx:]),
    ]
    damp_data = []
    for ml, fn, fl, fh, sig in modes_cfg:
        try:
            z, a, A0, td, es = estimate_damping(sig, FS, fn, fl, fh)
            print(f"  {ml}: zeta = {z:.4f} ({z*100:.2f}%)")
            damp_data.append((td, es, z, a, A0, fn, ml))
        except Exception as e:
            print(f"  {ml}: FAILED — {e}")
    if damp_data:
        plot_damping_three_modes(damp_data, 'fig_5_5_damping_estimation.png')
