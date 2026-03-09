import numpy as np

def cir_milstein(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
    seed: int = None
) -> np.ndarray:
    """
    Simulate CIR process paths using the explicit Milstein scheme.
    
    Parameters
    ----------
    n_paths : int
        Number of simulated paths
    n_steps : int
        Number of time steps
    T : float
        Time horizon (years)
    kappa : float
        Speed of mean reversion (> 0)
    theta : float
        Long-term mean (> 0)
    sigma : float
        Volatility parameter (> 0)
    x0 : float
        Initial value (>= 0)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    paths : np.ndarray
        Shape (n_paths, n_steps + 1) array of simulated paths
        paths[:, 0] = x0 for all paths
    
    The Milstein update is:
        X_{t+Δt} = X_t + κ(θ - X_t)Δt + σ√X_t ΔW + (σ²/4)(ΔW² - Δt)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0
    
    # Precompute constants
    drift_const = kappa * theta * dt
    mean_reversion = kappa * dt
    diffusion_coeff = sigma * np.sqrt(dt)
    milstein_coeff = 0.25 * sigma**2 * dt   # = σ²/4 * Δt
    
    for i in range(n_steps):
        X = paths[:, i]
        
        # Generate normal increments
        dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
        
        # Diffusion term
        sqrtX = np.sqrt(np.maximum(X, 0.0))   # prevent sqrt(negative)
        diffusion = sigma * sqrtX * dW
        
        # Milstein correction term
        milstein = (sigma**2 / 4.0) * (dW**2 - dt)
        
        # Drift term
        drift = kappa * (theta - X) * dt
        
        # Full update
        X_next = X + drift + diffusion + milstein
        
        # Basic reflection / truncation to avoid negative values
        # (you can change this to absorption: X_next = np.maximum(X_next, 0))
        paths[:, i + 1] = np.maximum(X_next, 0.0)
    
    return paths


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Typical short-rate parameters
    params = {
        "kappa": 3.0,      # fast mean reversion
        "theta": 0.05,     # long-run mean 5%
        "sigma": 0.12,     # volatility
        "x0":    0.03,     # starting rate 3%
        "T":     5.0,      # 5 years
    }
    
    paths = cir_milstein(
        n_paths=2000,
        n_steps=500,
        T=params["T"],
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        x0=params["x0"],
        seed=42
    )
    
    # Quick inspection
    print("Shape of paths:", paths.shape)
    print("Mean at maturity:", paths[:, -1].mean())
    print("Min at maturity:", paths[:, -1].min())
    
    # Optional: plot a few paths (requires matplotlib)
    # import matplotlib.pyplot as plt
    # t = np.linspace(0, params["T"], paths.shape[1])
    # plt.plot(t, paths[:30].T, lw=0.8, alpha=0.6)
    # plt.title("CIR paths – Milstein scheme")
    # plt.xlabel("Time (years)")
    # plt.ylabel("Interest rate")
    # plt.grid(True, alpha=0.3)
    # plt.show()