import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

def test_svd_dimensions_sparse(sparse_matrix, dimensions):
    """
    Test different SVD dimensions on a log-scaled sparse matrix.
    Avoids dense matrix conversion for memory efficiency.
    """
    results = {}
    
    print("Applying log(1+x) scaling to the sparse matrix...")
    # Apply log scaling directly to the sparse matrix data
    log_scaled_matrix = sparse_matrix.copy()
    log_scaled_matrix.data = np.log1p(log_scaled_matrix.data)
    
    # Calculate total variance (squared Frobenius norm) from sparse data
    total_variance = np.sum(np.square(log_scaled_matrix.data))
    
    print("Testing SVD dimensions on sparse matrix...")
    for d in tqdm(dimensions):
        try:
            # Compute SVD directly on the log-scaled sparse matrix
            # k should be less than min(matrix.shape) - 1
            max_k = min(log_scaled_matrix.shape) - 2
            if d >= max_k:
               print(f"Warning: Dimension {d} is too large for matrix shape {log_scaled_matrix.shape}. Skipping.")
               continue

            U, s, Vt = svds(log_scaled_matrix, k=d, which='LM') # Use 'LM' for largest magnitude singular values

            # svds returns s in ascending order, reverse it
            s = s[::-1]
            
            # Calculate explained variance directly from singular values
            explained_variance = np.sum(np.square(s)) / total_variance
            
            results[d] = {
                'singular_values': s,
                'explained_variance': explained_variance,
                # 'reconstruction_error': reconstruction_error, # Removed for memory efficiency
                # 'relative_error': relative_error            # Removed for memory efficiency
            }
            
        except Exception as e:
            print(f"Error processing dimension {d}: {str(e)}")
            # Add resilience for SVD convergence issues
            results[d] = {
                'singular_values': None,
                'explained_variance': 0,
            }
    
    return results

def plot_svd_results_sparse(results, dimensions):
    """Plot the results of SVD analysis (modified for sparse approach)"""
    
    # Filter out dimensions that might have failed
    valid_dimensions = [d for d in dimensions if d in results and results[d]['singular_values'] is not None]
    if not valid_dimensions:
        print("No valid SVD results to plot.")
        return

    plt.figure(figsize=(15, 6)) # Adjusted size as we have fewer plots
    
    # Plot 1: Explained Variance
    plt.subplot(1, 2, 1)
    explained_var = [results[d]['explained_variance'] for d in valid_dimensions]
    plt.plot(valid_dimensions, explained_var, 'bo-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance vs Dimensions (Log-Scaled Sparse Matrix)')
    plt.grid(True)
    
    # Plot 2: Singular Values Distribution
    plt.subplot(1, 2, 2)
    max_d = max(valid_dimensions)
    plt.plot(range(1, max_d + 1), results[max_d]['singular_values'], 'go-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Singular Values Distribution (d={max_d})')
    plt.yscale('log') # Keep log scale for singular values
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    # Use a different filename to avoid overwriting previous results
    plt.savefig('plots/svd_sparse_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the co-occurrence matrix for window size 5
    matrix_path = 'models/cooc_matrix_w5.pkl' # Define path
    results_path = 'svd_sparse_analysis_results.pkl' # Define output path

    try:
        print(f"Loading co-occurrence matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            cooc_matrix = data['matrix']
            # word2idx = data['word2idx'] # Not needed for this version of SVD analysis
    except FileNotFoundError:
        print(f"Error: Could not find co-occurrence matrix at {matrix_path}")
        return
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    print(f"Matrix shape: {cooc_matrix.shape}")
    
    # Define dimensions to test
    dimensions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Test different SVD dimensions using the memory-efficient sparse method
    results = test_svd_dimensions_sparse(cooc_matrix, dimensions)
    
    # Filter dimensions based on actual results obtained
    valid_dimensions = [d for d in dimensions if d in results]

    if not valid_dimensions:
       print("SVD analysis failed for all dimensions.")
       return

    # Plot results
    plot_svd_results_sparse(results, valid_dimensions)
    
    # Find optimal dimension based on explained variance threshold
    threshold = 0.8
    optimal_d = None
    # Iterate through dimensions in increasing order
    sorted_valid_dimensions = sorted(valid_dimensions)
    for d in sorted_valid_dimensions:
        if results[d]['explained_variance'] >= threshold:
            optimal_d = d
            break # Stop at the first dimension exceeding the threshold
    
    print("\nSVD Analysis Results (Sparse Log-Scaled Matrix):")
    if optimal_d is not None:
        print(f"Recommended dimension (for >= {threshold*100}% explained variance): {optimal_d}")
    else:
        max_explained = max(results[d]['explained_variance'] for d in valid_dimensions) if valid_dimensions else 0
        print(f"Could not reach {threshold*100}% explained variance. Max achieved: {max_explained:.4f}")
        # Optionally recommend the dimension with the highest variance if threshold not met
        if valid_dimensions:
             optimal_d = max(valid_dimensions, key=lambda d: results[d]['explained_variance'])
             print(f"Consider using dimension {optimal_d} (highest explained variance found).")


    print("\nDetailed results for each dimension:")
    for d in sorted_valid_dimensions:
         # Check if results for d exist and are valid
         if d in results and results[d]['singular_values'] is not None:
             print(f"Dimension {d}: Explained variance = {results[d]['explained_variance']:.4f}")
         else:
             print(f"Dimension {d}: SVD computation failed or skipped.")

    # Save results
    print(f"Saving results to {results_path}...")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'dimensions_tested': dimensions,
            'optimal_dimension': optimal_d,
            'threshold': threshold,
            'method': 'sparse_log_scaled' # Indicate the method used
        }, f)
    print("Analysis complete.")

if __name__ == "__main__":
    main()