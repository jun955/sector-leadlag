"""
Regularized PCA for stable factor extraction.

Shrinks the sample correlation matrix toward a prior target C0 and extracts
the top K eigenvectors as the factor loading matrix.
"""

from __future__ import annotations

import numpy as np


def rolling_standardize(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-scores using a trailing window.

    For each date t the z-score is computed using the mean and std of
    [t-window, ..., t-1] (strictly past data — date t is NOT included
    in its own statistics to avoid lookahead bias).

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Raw daily returns matrix.
    window : int
        Trailing window length L.

    Returns
    -------
    z_scores : np.ndarray, shape (T, N)
        Rolling z-scores.  Rows 0..window-1 will be NaN because there
        is insufficient history to compute statistics.
    """
    T, N = returns.shape
    z_scores = np.full((T, N), np.nan)

    # Pre-compute cumulative sums for O(1) per-step statistics
    # Pad with a leading zero row so cumsum[i] = sum of returns[0..i-1]
    cumsum = np.zeros((T + 1, N))
    cumsum2 = np.zeros((T + 1, N))
    cumsum[1:] = np.cumsum(returns, axis=0)
    cumsum2[1:] = np.cumsum(returns ** 2, axis=0)

    for t in range(window, T):
        # Statistics from returns[t-window : t]  (window observations, excluding t)
        s = cumsum[t] - cumsum[t - window]       # sum of x
        s2 = cumsum2[t] - cumsum2[t - window]    # sum of x^2
        mu = s / window
        var = s2 / window - mu ** 2
        var_corrected = var * window / (window - 1)
        std = np.sqrt(np.maximum(var_corrected, 1e-16))
        z_scores[t] = (returns[t] - mu) / std

    return z_scores


def regularized_pca(
    returns_window: np.ndarray,
    C0: np.ndarray,
    lam: float = 0.9,
    K: int = 3,
) -> np.ndarray:
    """Extract top-K eigenvectors from a regularized correlation matrix.

    Parameters
    ----------
    returns_window : np.ndarray, shape (L, N)
        Standardized returns in the rolling window.
    C0 : np.ndarray, shape (N, N)
        Prior target correlation matrix.
    lam : float
        Regularization parameter in [0, 1].  Higher values shrink more
        toward C0.
    K : int
        Number of factors (eigenvectors) to extract.

    Returns
    -------
    V_K : np.ndarray, shape (N, K)
        Top K eigenvectors of the regularized correlation matrix,
        columns ordered by descending eigenvalue.
    """
    L, N = returns_window.shape
    if C0.shape != (N, N):
        raise ValueError(
            f"C0 shape {C0.shape} does not match returns width {N}"
        )

    # --- Sample correlation from the window ---
    # Standardize each column within the window (zero mean, unit variance)
    mu = returns_window.mean(axis=0, keepdims=True)
    std = returns_window.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    Z = (returns_window - mu) / std
    Ct = Z.T @ Z / (L - 1)  # (N, N) sample correlation

    # --- Regularize ---
    Creg = (1.0 - lam) * Ct + lam * C0

    # --- Eigendecomposition ---
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(Creg)

    # Take the last K columns (largest eigenvalues) and reverse to
    # get descending order
    V_K = eigenvectors[:, -K:][:, ::-1]  # (N, K)

    return V_K


if __name__ == "__main__":
    np.random.seed(42)

    N, L, K = 28, 60, 3

    # Synthetic returns and prior
    returns_win = np.random.randn(L, N) * 0.01
    C0 = np.eye(N)

    V_K = regularized_pca(returns_win, C0, lam=0.9, K=K)
    print(f"V_K shape: {V_K.shape}")
    print(f"Columns orthonormal? (V^T V ~ I):\n{np.round(V_K.T @ V_K, 6)}")

    # Verify rolling_standardize
    T = 120
    raw = np.random.randn(T, 5) * 0.02 + 0.001
    zscores = rolling_standardize(raw, window=60)
    print(f"\nRolling z-scores shape: {zscores.shape}")
    print(f"NaN rows (first 60): {np.isnan(zscores[:60]).all()}")
    print(f"Sample z-score mean (rows 60+): {np.nanmean(zscores[60:]):.4f}")
    print(f"Sample z-score std  (rows 60+): {np.nanstd(zscores[60:]):.4f}")
