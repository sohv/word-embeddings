'''
This script finds the optimal SVD dimension for the co-occurrence matrix using the explained variance method. The approach used is explained variance + noise threshold, where the explained variance cutoff is 85% and the minimum singular value ratio is 1% of the total singular values. 
An ideal dimension should retain 85% of the original matrix's information and must also balance the trade-off between noise and information loss. The dimensions are tested from 100 to 2000 with a step of 100. The results are stored in the plots/Task-1/Part-2/ folder.

NOTE: The ideal dimension obtained by running this code is incorrect as it does not match with the set variance threshold and it's most likely that noise threshold is prioritized over variance threshold. The modified code is in the file find_dimension_variance_modified.py
'''

import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import pandas as pd

def find_optimal_dimension(s, variance_threshold=0.85, min_sv_ratio=0.01):
    s_sorted = np.sort(s)[::-1]
    total_variance = np.sum(s_sorted**2)
    cumulative_var = np.cumsum(s_sorted**2) / total_variance

    # Find dimensions meeting variance threshold
    dims_above_threshold = np.where(cumulative_var >= variance_threshold)[0]
    if len(dims_above_threshold) == 0:
        min_dim_for_threshold = len(s_sorted)
    else:
        min_dim_for_threshold = dims_above_threshold[0] + 1

    noise_threshold = min_sv_ratio * s_sorted[0]
    dims_above_noise = np.where(s_sorted >= noise_threshold)[0]
    max_dim_before_noise = len(dims_above_noise)
    optimal_dim = min(min_dim_for_threshold, max_dim_before_noise)

    return optimal_dim, cumulative_var

def test_svd_dimensions(cooc_matrix, dimensions=range(100, 2001, 100)):
    results = {}
    cooc_matrix_log = cooc_matrix.copy()
    cooc_matrix_log.data = np.log1p(cooc_matrix_log.data)

    print("Testing SVD dimensions for explained variance...")

    for d in tqdm(dimensions, desc="Testing SVD dimensions"):
        try:
            u, s, vt = svds(cooc_matrix_log, k=d, random_state=42, maxiter=2000, tol=1e-6)
            explained_var = (s**2).sum() / (cooc_matrix_log.data**2).sum()
            results[d] = {
                'explained_variance': explained_var,
                'singular_values': s
            }

            print(f"\nDimension {d}:")
            print(f"Explained variance: {explained_var:.4f}")

        except Exception as e:
            print(f"Error processing dimension {d}: {str(e)}")

    # Find optimal dimension using the new function
    max_dim = max(dimensions)
    u, s, vt = svds(cooc_matrix_log, k=max_dim, random_state=42, maxiter=2000, tol=1e-6)
    optimal_dim, cumulative_var = find_optimal_dimension(s)

    if optimal_dim not in results:
        u, s, vt = svds(cooc_matrix_log, k=optimal_dim, random_state=42, maxiter=2000, tol=1e-6)
        explained_var = (s**2).sum() / (cooc_matrix_log.data**2).sum()
        results[optimal_dim] = {
            'explained_variance': explained_var,
            'singular_values': s
        }
        print(f"\nCalculated explained variance for optimal dimension {optimal_dim}: {explained_var:.4f}")

    plt.figure(figsize=(12, 8))

    dims = list(results.keys())
    explained_vars = [results[d]['explained_variance'] for d in dims]

    plt.plot(dims, explained_vars, 'ro-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_dim, color='b', linestyle='--', label=f'Optimal Dimension: {optimal_dim}')
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Explained Variance Ratio')
    plt.title('SVD Dimension Impact on Explained Variance')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/svd_dimension_analysis_full.png', dpi=300, bbox_inches='tight')
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

    plt.tight_layout()
    plt.savefig('plots/svd_cumulative_variance.png', dpi=300, bbox_inches='tight')
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

    plt.savefig('plots/svd_dimension_results_table_full.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nOptimal SVD Results:")
    print(f"Optimal dimension: {optimal_dim}")
    print(f"Explained variance at optimal dim: {results[optimal_dim]['explained_variance']:.4f}")

    return results, optimal_dim

def main():
    try:
        with open('models/co-occurrence/cooc_matrix_w5.pkl', 'rb') as f:
            data = pickle.load(f)
            cooc_matrix = data['matrix']
    except FileNotFoundError:
        print("Error: Could not find co-occurrence matrix for window size 5")
        return

    dimensions = range(100, 2001, 100)
    results, optimal_dim = test_svd_dimensions(cooc_matrix, dimensions)

    with open('svd_analysis_results_full.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'optimal_dimension': optimal_dim,
            'window_size': 5,
            'dimensions_tested': list(dimensions)
        }, f)

if __name__ == "__main__":
    main()