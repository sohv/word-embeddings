import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr # Import spearmanr
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix

def test_truncated_svd_dimensions(sparse_matrix, dimensions, word2idx, simlex_pairs):
    """
    Test different dimensions using sklearn's TruncatedSVD on a 
    log-scaled sparse matrix and evaluate using Spearman correlation.
    """
    results = {}
    
    print("Applying log(1+x) scaling to the sparse matrix...")
    log_scaled_matrix = sparse_matrix.copy()
    log_scaled_matrix.data = np.log1p(log_scaled_matrix.data)
    
    # Prepare SimLex data
    valid_pairs = [(w1, w2, score) for w1, w2, score in simlex_pairs 
                  if w1 in word2idx and w2 in word2idx]
    if not valid_pairs:
        print("Error: No valid SimLex pairs found in vocabulary.")
        return None
    print(f"Evaluating with {len(valid_pairs)} valid SimLex pairs.")
    human_scores = [score for _, _, score in valid_pairs]

    print("Testing TruncatedSVD dimensions and calculating Spearman correlation...")
    
    for d in tqdm(dimensions, desc="Testing dimensions"):
        try:
            # Instantiate and fit TruncatedSVD for dimension d
            # Using fit_transform directly gives the transformed data (approx U*S)
            svd = TruncatedSVD(n_components=d, n_iter=7, random_state=42)
            # U_d * S_d
            embeddings = svd.fit_transform(log_scaled_matrix) 
            
            # Calculate similarities for SimLex pairs using these embeddings
            sim_scores = []
            for w1, w2, _ in valid_pairs:
                idx1 = word2idx[w1]
                idx2 = word2idx[w2]
                
                # Ensure indices are within the bounds of computed embeddings
                if idx1 < embeddings.shape[0] and idx2 < embeddings.shape[0]:
                    vec1 = embeddings[idx1]
                    vec2 = embeddings[idx2]
                    
                    # Normalize vectors before cosine similarity
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(vec1, vec2) / (norm1 * norm2)
                        sim_scores.append(sim)
                    else:
                        sim_scores.append(0) # Handle zero vectors if they occur
                else:
                    # This case shouldn't happen if word2idx is correct, but as a safeguard:
                    sim_scores.append(0) 
            
            # Calculate Spearman correlation
            if len(sim_scores) == len(human_scores):
                correlation, p_value = spearmanr(sim_scores, human_scores)
            else:
                 print(f"Warning: Mismatch in score list lengths for dim {d}. Skipping correlation.")
                 correlation = 0.0
                 p_value = 1.0

            # Get cumulative explained variance
            explained_variance = svd.explained_variance_ratio_.sum()
            
            results[d] = {
                'correlation': correlation,
                'p_value': p_value,
                'cumulative_explained_variance': explained_variance,
                'singular_values': svd.singular_values_ # Store singular values for this d
            }
            
            print(f"  Dim {d}: Corr={correlation:.4f}, ExpVar={explained_variance:.4f}")

        except IndexError as ie:
             print(f"IndexError for dimension {d}: {ie}. Ensure word indices are valid.")
             results[d] = {'correlation': 0, 'cumulative_explained_variance': 0, 'error': str(ie)}
        except Exception as e:
            print(f"Error processing dimension {d}: {str(e)}")
            results[d] = {'correlation': 0, 'cumulative_explained_variance': 0, 'error': str(e)}

    return results

def plot_truncated_svd_results(results, dimensions):
    """Plot the results of TruncatedSVD analysis including Spearman correlation"""
    
    valid_dimensions = sorted([d for d in dimensions if d in results and 'error' not in results[d]])
    if not valid_dimensions:
        print("No valid dimensions found in results to plot.")
        return

    plt.figure(figsize=(15, 10)) # Adjusted size for 3 plots

    # Plot 1: Cumulative Explained Variance
    plt.subplot(2, 2, 1)
    exp_var_values = [results[d]['cumulative_explained_variance'] for d in valid_dimensions]
    plt.plot(valid_dimensions, exp_var_values, 'bo-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs Dimensions (TruncatedSVD)')
    plt.grid(True)
    
    # Plot 2: Spearman Correlation
    plt.subplot(2, 2, 2)
    corr_values = [results[d]['correlation'] for d in valid_dimensions]
    plt.plot(valid_dimensions, corr_values, 'ro-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Spearman Correlation (SimLex Subset)')
    plt.title('Semantic Correlation vs Dimensions (TruncatedSVD)')
    plt.grid(True)

    # Plot 3: Singular Values Distribution (from max valid d fit)
    plt.subplot(2, 2, 3)
    max_d = max(valid_dimensions)
    if 'singular_values' in results[max_d]:
        singular_values = results[max_d]['singular_values']
        plt.plot(range(1, len(singular_values) + 1), singular_values, 'go-')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.title(f'Singular Values (TruncatedSVD, d={len(singular_values)})')
        plt.yscale('log') 
        plt.grid(True)
    else:
        plt.title("Singular Values Not Available")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/svd_truncated_semantic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define paths
    matrix_path = 'models/cooc_matrix_w5.pkl' 
    simlex_path = 'data/simlex_subset.txt' # Path to SimLex subset
    results_path = 'svd_truncated_semantic_results.pkl' 

    # Load Matrix
    try:
        print(f"Loading co-occurrence matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            cooc_matrix = data['matrix']
            word2idx = data['word2idx'] # Need word2idx now
            if not isinstance(cooc_matrix, csr_matrix):
                 cooc_matrix = csr_matrix(cooc_matrix)
    except FileNotFoundError:
        print(f"Error: Could not find matrix at {matrix_path}")
        return
    except KeyError:
        print(f"Error: 'word2idx' not found in {matrix_path}")
        return
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    # Load SimLex subset
    simlex_pairs = []
    try:
        print(f"Loading SimLex subset from {simlex_path}...")
        with open(simlex_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    w1, w2, score = parts[0], parts[1], float(parts[2])
                    simlex_pairs.append((w1, w2, score))
        if not simlex_pairs:
             print("Error: SimLex subset file is empty or invalid.")
             return
    except FileNotFoundError:
        print(f"Error: Could not find SimLex subset at {simlex_path}")
        return
    except Exception as e:
        print(f"Error loading SimLex subset: {e}")
        return


    print(f"Matrix shape: {cooc_matrix.shape}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Matrix non-zero elements: {cooc_matrix.nnz}")
    
    # Define dimensions to test
    dimensions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Test different SVD dimensions using TruncatedSVD and Spearman evaluation
    results = test_truncated_svd_dimensions(cooc_matrix, dimensions, word2idx, simlex_pairs)
    
    if results is None:
        print("TruncatedSVD semantic analysis failed.")
        return

    # Plot results
    plot_truncated_svd_results(results, dimensions)
    
    # Find optimal dimension based on MAX Spearman correlation
    valid_results = {d: r for d, r in results.items() if 'error' not in r}
    if not valid_results:
         print("No valid results found to determine optimal dimension.")
         optimal_d = None
         ideal_threshold = None
    else:
        optimal_d = max(valid_results, key=lambda d: valid_results[d]['correlation'])
        ideal_threshold = valid_results[optimal_d]['cumulative_explained_variance']
        best_corr = valid_results[optimal_d]['correlation']
        print("\nTruncatedSVD Semantic Analysis Results:")
        print(f"Optimal dimension based on max Spearman correlation: {optimal_d}")
        print(f"  Best Spearman Correlation: {best_corr:.4f}")
        print(f"  Empirically found ideal threshold (Explained Variance at d={optimal_d}): {ideal_threshold:.4f}")


    # Save results
    print(f"Saving results to {results_path}...")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results, 
            'dimensions_tested': dimensions,
            'optimal_dimension_semantic': optimal_d, 
            'ideal_threshold_semantic': ideal_threshold,
            'method': 'truncated_svd_semantic' 
        }, f)
    print("Analysis complete.")

if __name__ == "__main__":
    main()