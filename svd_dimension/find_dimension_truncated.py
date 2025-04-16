import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix # Ensure this is imported if using sparse matrix operations

def test_truncated_svd_dimensions(sparse_matrix, dimensions):
    """
    Test different dimensions using sklearn's TruncatedSVD on a 
    log-scaled sparse matrix.
    """
    results = {}
    
    print("Applying log(1+x) scaling to the sparse matrix...")
    # Apply log scaling directly to the sparse matrix data
    log_scaled_matrix = sparse_matrix.copy()
    log_scaled_matrix.data = np.log1p(log_scaled_matrix.data)
    
    print("Testing TruncatedSVD dimensions...")
    # Store explained variance for each dimension
    explained_variances = []
    cumulative_explained_variance = {}

    # Fit TruncatedSVD once with the maximum dimension to get all singular values efficiently
    max_d = max(dimensions)
    try:
        print(f"Fitting TruncatedSVD with n_components={max_d}...")
        # n_iter can be increased if convergence is slow
        svd = TruncatedSVD(n_components=max_d, n_iter=7, random_state=42) 
        svd.fit(log_scaled_matrix)
        
        # Calculate cumulative explained variance for each requested dimension
        for d in dimensions:
            if d <= max_d:
                explained_variance_ratio = svd.explained_variance_ratio_[:d].sum()
                cumulative_explained_variance[d] = explained_variance_ratio
                print(f"Dimension {d}: Cumulative Explained Variance = {explained_variance_ratio:.4f}")
            else:
                print(f"Dimension {d} > max tested ({max_d}). Skipping.")
        
        # Store singular values from the largest fit
        results['singular_values_max_d'] = svd.singular_values_
        results['explained_variance_ratios'] = svd.explained_variance_ratio_
        results['cumulative_explained_variance'] = cumulative_explained_variance

    except Exception as e:
        print(f"Error during TruncatedSVD fitting: {str(e)}")
        return None # Indicate failure

    return results

def plot_truncated_svd_results(results, dimensions):
    """Plot the results of TruncatedSVD analysis"""
    
    if not results or 'cumulative_explained_variance' not in results:
        print("No valid TruncatedSVD results to plot.")
        return
    
    cumulative_variances = results['cumulative_explained_variance']
    valid_dimensions = sorted([d for d in dimensions if d in cumulative_variances])

    if not valid_dimensions:
        print("No valid dimensions found in results.")
        return

    plt.figure(figsize=(15, 6)) 
    
    # Plot 1: Cumulative Explained Variance
    plt.subplot(1, 2, 1)
    exp_var_values = [cumulative_variances[d] for d in valid_dimensions]
    plt.plot(valid_dimensions, exp_var_values, 'bo-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs Dimensions (TruncatedSVD)')
    plt.grid(True)
    
    # Plot 2: Singular Values Distribution (from max_d fit)
    plt.subplot(1, 2, 2)
    if 'singular_values_max_d' in results:
        singular_values = results['singular_values_max_d']
        plt.plot(range(1, len(singular_values) + 1), singular_values, 'go-')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.title(f'Singular Values (TruncatedSVD, d={len(singular_values)})')
        plt.yscale('log') 
        plt.grid(True)
    else:
        plt.title("Singular Values Not Available")

    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    # Use a different filename 
    plt.savefig('plots/svd_truncated_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define paths
    matrix_path = 'models/cooc_matrix_w5.pkl' 
    results_path = 'svd_truncated_analysis_results.pkl' 

    try:
        print(f"Loading co-occurrence matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            # Ensure it's loaded as CSR for TruncatedSVD compatibility if needed
            cooc_matrix = data['matrix']
            if not isinstance(cooc_matrix, csr_matrix):
                 cooc_matrix = csr_matrix(cooc_matrix)

    except FileNotFoundError:
        print(f"Error: Could not find co-occurrence matrix at {matrix_path}")
        return
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    print(f"Matrix shape: {cooc_matrix.shape}")
    print(f"Matrix non-zero elements: {cooc_matrix.nnz}")
    
    # Define dimensions to test
    dimensions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Test different SVD dimensions using TruncatedSVD
    results = test_truncated_svd_dimensions(cooc_matrix, dimensions)
    
    if results is None:
        print("TruncatedSVD analysis failed.")
        return

    # Plot results
    plot_truncated_svd_results(results, dimensions)
    
    # Find optimal dimension based on explained variance threshold
    threshold = 0.8
    optimal_d = None
    cumulative_variances = results.get('cumulative_explained_variance', {})
    sorted_valid_dimensions = sorted([d for d in dimensions if d in cumulative_variances])

    for d in sorted_valid_dimensions:
        if cumulative_variances[d] >= threshold:
            optimal_d = d
            break # Stop at the first dimension exceeding the threshold
    
    print("\nTruncatedSVD Analysis Results (Sparse Log-Scaled Matrix):")
    if optimal_d is not None:
        print(f"Recommended dimension (for >= {threshold*100}% cumulative explained variance): {optimal_d}")
        print(f"Explained variance at d={optimal_d}: {cumulative_variances[optimal_d]:.4f}")
    else:
        max_explained = max(cumulative_variances.values()) if cumulative_variances else 0
        print(f"Could not reach {threshold*100}% explained variance with tested dimensions.")
        print(f"Max cumulative explained variance achieved: {max_explained:.4f} at d={max(sorted_valid_dimensions) if sorted_valid_dimensions else 'N/A'}")
        # Optionally recommend the dimension with the highest variance if threshold not met
        if sorted_valid_dimensions:
             optimal_d_fallback = max(sorted_valid_dimensions, key=lambda d: cumulative_variances[d])
             print(f"Consider using dimension {optimal_d_fallback} (highest explained variance found).")
             optimal_d = optimal_d_fallback # Assign fallback if threshold wasn't met

    # Save results
    print(f"Saving results to {results_path}...")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results, # Contains cumulative variances and potentially singular values
            'dimensions_tested': dimensions,
            'optimal_dimension': optimal_d, # Will be the threshold-based or fallback
            'threshold': threshold,
            'method': 'truncated_svd_log_scaled' # Indicate the method used
        }, f)
    print("Analysis complete.")

if __name__ == "__main__":
    main()