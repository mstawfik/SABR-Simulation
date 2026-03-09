# Grok
# 
# Assuming your dataframes are called df_base and df_exponent
result = df_base ** df_exponent

# or equivalently:
result = df_base.pow(df_exponent)

import pandas as pd
import numpy as np

def perform_svd_on_first_differences(time_series):
    """
    Performs SVD on the matrix formed by the first differences of the time series.
    
    Assumptions:
    - time_series is a list of pandas DataFrames, each of the same shape (r rows x c columns).
    - The DataFrames contain numeric data that can be subtracted element-wise.
    - First differences are computed as time_series[i] - time_series[i-1] for i >= 1.
    - Each difference DataFrame is converted to a NumPy array and flattened to a vector.
    - These vectors are stacked into a matrix of shape (num_differences, r * c).
    - SVD is then performed on this stacked matrix.
    
    Returns:
    - U, S, Vh from np.linalg.svd (economy SVD).
    
    Raises:
    - ValueError if not all elements are DataFrames or if shapes differ.
    """
    if not time_series or len(time_series) < 2:
        raise ValueError("Time series must have at least two DataFrames.")
    
    if not all(isinstance(df, pd.DataFrame) for df in time_series):
        raise ValueError("All elements in time_series must be pandas DataFrames.")
    
    # Check shapes are consistent
    shape = time_series[0].shape
    for df in time_series[1:]:
        if df.shape != shape:
            raise ValueError("All DataFrames must have the same shape.")
    
    # Compute first differences
    diffs = [time_series[i] - time_series[i-1] for i in range(1, len(time_series))]
    
    # Flatten each difference and stack into a matrix
    num_diffs = len(diffs)
    flat_size = shape[0] * shape[1]
    diff_matrix = np.zeros((num_diffs, flat_size))
    
    for j, diff in enumerate(diffs):
        diff_np = diff.to_numpy(dtype=float)  # Ensure numeric
        diff_flat = diff_np.flatten()
        diff_matrix[j, :] = diff_flat
    
    # Perform SVD
    U, S, Vh = np.linalg.svd(diff_matrix, full_matrices=False)
    
    return U, S, Vh

# Example usage (commented out):
# time_series = [pd.DataFrame(...), pd.DataFrame(...), ...]
# U, S, Vh = perform_svd_on_first_differences(time_series)
# print("Singular values:", S)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # optional

def pca_on_first_differences(time_series, center=True, scale=False):
    """
    Performs PCA on the flattened first differences of the time series using SVD.
    
    Parameters:
    - time_series: list of pd.DataFrame, all with same shape
    - center: whether to center the difference matrix (strongly recommended)
    - scale: whether to standardize each feature to unit variance (less common here)
    
    Returns:
    - principal_components: array (n_components, n_features) – the PCs (Vh)
    - explained_variance: array – variance explained by each component
    - explained_variance_ratio: array – proportion of variance explained
    - U, S, Vh: full SVD results
    - diff_matrix: the (n_diffs × flattened_size) matrix used
    """
    if len(time_series) < 2:
        raise ValueError("Need at least 2 time points to compute differences.")
    
    # Check consistency
    shape = time_series[0].shape
    if not all(df.shape == shape for df in time_series):
        raise ValueError("All DataFrames must have the same shape.")
    
    # Compute first differences
    diffs = [time_series[i] - time_series[i-1] for i in range(1, len(time_series))]
    n_diffs = len(diffs)
    
    # Build data matrix: rows = time steps (differences), columns = flattened spatial features
    flat_size = shape[0] * shape[1]
    X = np.zeros((n_diffs, flat_size))
    for i, diff_df in enumerate(diffs):
        X[i] = diff_df.to_numpy(dtype=float).ravel()
    
    # Optional: center (and scale)
    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        X = scaler.fit_transform(X)
    
    # SVD (economy mode)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Explained variance
    variance = S ** 2 / (n_diffs - 1 if center else n_diffs)   # match sklearn behavior
    total_var = variance.sum()
    explained_variance_ratio = variance / total_var
    
    return {
        'principal_components': Vh,                # shape: (n_components, n_features)
        'singular_values': S,
        'explained_variance': variance,
        'explained_variance_ratio': explained_variance_ratio,
        'U': U,
        'diff_matrix_original': X if not (center or scale) else scaler.inverse_transform(X),
        'n_differences': n_diffs,
        'feature_shape': shape
    }


# Example usage:
# results = pca_on_first_differences(your_time_series_list, center=True, scale=False)

# Quick inspection:
# print("Explained variance ratio:", results['explained_variance_ratio'][:10])
# print("Cumulative:", np.cumsum(results['explained_variance_ratio'])[:10])