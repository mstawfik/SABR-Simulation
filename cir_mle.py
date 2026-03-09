# generated using Claude
#
import numpy as np
from scipy.optimize import minimize
from scipy.special import iv  # Modified Bessel function of the first kind

def cir_log_likelihood(params, X, dt):
    """
    Log-likelihood for CIR process: dX = kappa*(theta - X)*dt + sigma*sqrt(X)*dW
    
    Parameters:
        params : [kappa, theta, sigma]
        X      : observed time series (array)
        dt     : time step
    """
    kappa, theta, sigma = params

    # Parameter constraints
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return np.inf
    # Feller condition (ensures process stays positive)
    if 2 * kappa * theta <= sigma**2:
        return np.inf

    X0 = X[:-1]  # X_t
    X1 = X[1:]   # X_{t+1}

    # Transition density of CIR (non-central chi-squared)
    c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
    q = 2 * kappa * theta / sigma**2 - 1
    u = c * np.exp(-kappa * dt) * X0
    v = c * X1

    # Log of the non-central chi-squared transition density
    log_c = np.log(c)
    log_density = (
        log_c
        + (-u - v)
        + (q / 2) * (np.log(v) - np.log(u))
        + np.log(iv(q, 2 * np.sqrt(u * v)))  # Bessel function
    )

    # Guard against numerical issues
    if np.any(~np.isfinite(log_density)):
        return np.inf

    return -np.sum(log_density)  # Negative log-likelihood


def fit_cir(X, dt, n_restarts=5):
    """
    Fit CIR model via MLE with multiple random restarts.
    
    Returns dict of {kappa, theta, sigma} estimates.
    """
    best_result = None
    best_nll = np.inf

    # Use sample stats to anchor initial guesses
    mu_x = np.mean(X)
    std_x = np.std(X)

    for _ in range(n_restarts):
        # Randomised initial parameters
        init_params = [
            np.random.uniform(0.1, 5.0),   # kappa
            np.random.uniform(mu_x * 0.5, mu_x * 1.5),  # theta ≈ long-run mean
            np.random.uniform(0.01, std_x)  # sigma
        ]

        result = minimize(
            cir_log_likelihood,
            init_params,
            args=(X, dt),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8}
        )

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

    kappa, theta, sigma = best_result.x
    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "neg_log_likelihood": best_nll
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # True parameters
    kappa_true, theta_true, sigma_true = 2.0, 0.05, 0.1
    dt, N = 1/252, 1000  # daily steps, ~4 years

    # Simulate CIR path (Euler–Maruyama)
    X = np.zeros(N)
    X[0] = theta_true
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.randn()
        X[t] = max(X[t-1] + kappa_true*(theta_true - X[t-1])*dt
                   + sigma_true * np.sqrt(max(X[t-1], 0)) * dW, 1e-8)

    # Fit
    params = fit_cir(X, dt, n_restarts=5)

    print("True   → kappa=%.4f  theta=%.4f  sigma=%.4f" % (kappa_true, theta_true, sigma_true))
    print("MLE    → kappa=%.4f  theta=%.4f  sigma=%.4f" % (params["kappa"], params["theta"], params["sigma"]))
    print("Neg LL →", round(params["neg_log_likelihood"], 4))