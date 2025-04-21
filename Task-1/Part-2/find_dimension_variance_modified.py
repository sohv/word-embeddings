'''
This script finds the optimal SVD dimension using the explained variance + noise threshold method. Here, the approach used is slightly different from the find_dimension_variance.py file.
We use Frobenius norm to calculate the total energy of the matrix instead of full SVD as the generated matrix is very sparse and this method is computationally more efficient than full SVD.
The ideal dimension obtained here matches with the set variance threshold and also returns an ideal Spearman correlation during word embedding evaluation.
'''

import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import pandas as pd

def find_optimal_dimension(s, variance_threshold=0.85, min_sv_ratio=0.01):
    total_variance = np.sum(s**2)
    cumulative_var = np.cumsum(s**2) / total_variance

    dims_above_threshold = np.where(cumulative_var >= variance_threshold)[0]
    min_dim_for_threshold = dims_above_threshold[0] + 1 if len(dims_above_threshold) > 0 else len(s)

    noise_threshold = min_sv_ratio * s[0]
    dims_above_noise = np.where(s >= noise_threshold)[0]
    max_dim_before_noise = len(dims_above_noise)

    s_norm = s / s[0]
    derivatives = np.diff(s_norm)
    second_deriv = np.diff(derivatives)

    if len(second_deriv) > 0:
        elbow_idx1 = np.argmax(np.abs(second_deriv)) + 1
    else:
        elbow_idx1 = len(s) - 1

    ratios = s[:-1] / s[1:]
    if len(ratios) > 0:
        elbow_idx2 = np.argmax(ratios) + 1
    else:
        elbow_idx2 = len(s) - 1

    elbow_dim = int(np.ceil((elbow_idx1 + elbow_idx2) / 2))

    optimal_dim = min(min_dim_for_threshold, max_dim_before_noise, elbow_dim)
    optimal_dim = max(optimal_dim, 10)

    return optimal_dim, cumulative_var

def test_svd_dimensions(cooc_matrix, dimensions=range(100, 2001, 100)):
    results = {}
    cooc_matrix_log = cooc_matrix.copy()
    cooc_matrix_log.data = np.log1p(cooc_matrix_log.data)

    print("Testing SVD dimensions for explained variance...")

    total_matrix_energy = np.sum(cooc_matrix_log.data**2)
    print(f"Total energy from Frobenius norm: {total_matrix_energy:.4f}")
    energy_source = "Frobenius norm"

    for d in tqdm(dimensions, desc="Testing SVD dimensions"):
        try:
            u, s, vt = svds(cooc_matrix_log, k=d, random_state=42, maxiter=2000, tol=1e-6)
            explained_var = (s**2).sum() / total_matrix_energy
            results[d] = {
                'explained_variance': explained_var,
                'singular_values': s
            }

            print(f"\nDimension {d}:")
            print(f"Explained variance: {explained_var:.4f}")

        except Exception as e:
            print(f"Error processing dimension {d}: {str(e)}")

    max_dim = max(dimensions)
    try:
        u, s, vt = svds(cooc_matrix_log, k=max_dim, random_state=42, maxiter=2000, tol=1e-6)
        optimal_dim, cumulative_var = find_optimal_dimension(s)
        print(f"Found optimal dimension: {optimal_dim} (based on max SVD dimension {max_dim})")
    except Exception as e:
        print(f"Error finding optimal dimension: {str(e)}")
        largest_dim = max(results.keys())
        s = results[largest_dim]['singular_values']
        optimal_dim, cumulative_var = find_optimal_dimension(s)
        print(f"Found optimal dimension: {optimal_dim} (based on available dimension {largest_dim})")

    if optimal_dim not in results:
        try:
            u, s, vt = svds(cooc_matrix_log, k=optimal_dim, random_state=42, maxiter=2000, tol=1e-6)
            explained_var = (s**2).sum() / total_matrix_energy
            results[optimal_dim] = {
                'explained_variance': explained_var,
                'singular_values': s
            }
            print(f"\nCalculated explained variance for optimal dimension {optimal_dim}: {explained_var:.4f}")
        except Exception as e:
            print(f"Error computing optimal dimension {optimal_dim}: {str(e)}")
            avail_dims = np.array(list(results.keys()))
            nearest_idx = np.argmin(np.abs(avail_dims - optimal_dim))
            optimal_dim = avail_dims[nearest_idx]
            print(f"Using nearest available dimension: {optimal_dim}")

    plt.figure(figsize=(12, 8))

    dims = sorted(results.keys())
    explained_vars = [results[d]['explained_variance'] for d in dims]

    plt.plot(dims, explained_vars, 'ro-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_dim, color='b', linestyle='--', label=f'Optimal Dimension: {optimal_dim}')
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel(f'Explained Variance Ratio (relative to {energy_source})')
    plt.title('SVD Dimension Impact on Explained Variance')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/code/plots/svd_dimension_analysis_changed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    if optimal_dim in results:
        s_opt = results[optimal_dim]['singular_values']
        cum_var_opt = np.cumsum(s_opt**2) / np.sum(s_opt**2)
        plt.plot(range(1, len(cum_var_opt) + 1), cum_var_opt, 'b-', linewidth=2)

        plt.figure(figsize=(10, 6))
        s_norm = s_opt / s_opt[0]
        plt.plot(range(1, len(s_norm) + 1), s_norm, 'g-', linewidth=2)
        plt.axvline(x=optimal_dim, color='r', linestyle='--', label=f'Optimal Dimension: {optimal_dim}')
        plt.xlabel('Dimension')
        plt.ylabel('Normalized Singular Value')
        plt.title('Singular Value Decay and Elbow Point')
        plt.grid(True)
        plt.legend()
        plt.savefig('/content/drive/MyDrive/Colab Notebooks/code/plots/svd_singular_value_decay_changed.png', dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'b-', linewidth=2)
    plt.axhline(y=0.85, color='r', linestyle='--', label='85% Variance Threshold')
    plt.axvline(x=optimal_dim, color='g', linestyle='--', label=f'Optimal Dimension: {optimal_dim}')
    plt.xlabel('Number of dimensions')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Dimensions')
    plt.grid(True)
    plt.legend()
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/code/plots/svd_cumulative_variance_changed.png', dpi=300, bbox_inches='tight')
    plt.close()

    results_df = pd.DataFrame({
        'Dimension': dims,
        'Explained Variance': explained_vars
    })

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=results_df.round(4).values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.savefig('/content/drive/MyDrive/Colab Notebooks/code/plots/svd_dimension_results_table_changed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nOptimal SVD Results:")
    print(f"Optimal dimension: {optimal_dim}")
    print(f"Explained variance at optimal dim: {results[optimal_dim]['explained_variance']:.4f}")
    print(f"Total energy calculation method: {energy_source}")

    return results, optimal_dim

def main():
    try:
        matrix_path = '/content/drive/MyDrive/Colab Notebooks/code/models/cooc_matrix_w5.pkl'
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            cooc_matrix = data['matrix']
    except FileNotFoundError:
        print(f"Error: Could not find co-occurrence matrix at {matrix_path}")
        return

    dimensions = range(100, 2001, 100)
    results, optimal_dim = test_svd_dimensions(cooc_matrix, dimensions)

    with open('/content/drive/MyDrive/Colab Notebooks/code/models/svd_analysis_results_changed.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'optimal_dimension': optimal_dim,
            'window_size': 5,
            'dimensions_tested': list(dimensions)
        }, f)

if __name__ == "__main__":
    main()


