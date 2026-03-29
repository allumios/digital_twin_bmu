"""
forward_model.py — eigenvalue solver for N-DOF shear frame.
Solves K*phi = omega^2 * M * phi.
Single source of truth for stiffness matrix and freq computation.
"""

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar


def build_stiffness_matrix(k_vector):
    """Assemble tridiagonal stiffness matrix for shear frame."""
    d = len(k_vector)
    k = np.asarray(k_vector, dtype=float)
    K = np.zeros((d, d))
    for i in range(d):
        K[i, i] = k[i]
        if i + 1 < d:
            K[i, i] += k[i + 1]
            K[i, i + 1] = -k[i + 1]
            K[i + 1, i] = -k[i + 1]
    return K


def compute_natural_frequencies(k_global, masses):
    """
    Generalised eigenvalue problem, uniform k across storeys.
    Returns sorted frequencies [Hz] and mode shapes (cols, max-normalised).
    """
    m = np.asarray(masses, dtype=float)
    d = len(m)
    M = np.diag(m)
    K = build_stiffness_matrix([k_global] * d)

    eigenvalues, eigenvectors = eig(K, M)
    eigenvalues = np.real(eigenvalues)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = np.real(eigenvectors[:, idx])

    # normalise each mode so max component = 1
    for i in range(d):
        eigenvectors[:, i] /= np.max(np.abs(eigenvectors[:, i]))

    frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
    return frequencies, eigenvectors


def analytical_storey_stiffness(E, b, d, L, n_cols=4):
    """Fixed-fixed column lateral stiffness: k = n * 12EI/L^3."""
    I = b * d**3 / 12.0
    return n_cols * 12.0 * E * I / L**3


def compute_initial_k(masses, measured_freqs):
    """Least-squares fit of uniform k to measured frequencies."""
    measured = np.asarray(measured_freqs)

    def residual(k):
        predicted, _ = compute_natural_frequencies(k, masses)
        return np.sum((predicted - measured) ** 2)

    result = minimize_scalar(residual, bounds=(1000, 500000), method='bounded')
    return result.x


if __name__ == '__main__':
    from config import (MASSES, F_MEASURED, E_MATERIAL, COLUMN_WIDTH,
                        COLUMN_DEPTH, STOREY_HEIGHTS, N_COLUMNS)

    print("=" * 55)
    print("  FORWARD MODEL — Module 2")
    print("=" * 55)

    # analytical k per storey from measured geometry
    print("\n  Analytical storey stiffness:")
    for i, L in enumerate(STOREY_HEIGHTS):
        k_an = analytical_storey_stiffness(E_MATERIAL, COLUMN_WIDTH,
                                           COLUMN_DEPTH, L, N_COLUMNS)
        print(f"    Storey {i+1}: L = {L*1000:.1f} mm, k = {k_an:.0f} N/m")

    # best-fit global k
    k_init = compute_initial_k(MASSES, F_MEASURED)
    f_pred, modes = compute_natural_frequencies(k_init, MASSES)
    print(f"\n  Best-fit k = {k_init:.0f} N/m")
    print(f"  Predicted: [{f_pred[0]:.2f}, {f_pred[1]:.2f}, {f_pred[2]:.2f}] Hz")
    print(f"  Measured:  {F_MEASURED} Hz")

    print("\n  Mode shapes:")
    for i in range(len(MASSES)):
        ms = modes[:, i]
        print(f"    Mode {i+1}: [{ms[0]:.3f}, {ms[1]:.3f}, {ms[2]:.3f}]")
