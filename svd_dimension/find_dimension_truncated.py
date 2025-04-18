'''
This script is used to find the optimal SVD dimension for a subset of SimLex-999 with 20 pairs. We will use truncated SVD (Singular Value Decomposition) to find the optimal dimension.
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix
import pandas as pd

def test_truncated_svd_dimensions(sparse_matrix, dimensions, word2idx, simlex_pairs):
    results = {}
    
    print("Applying log(1+x) scaling to the sparse matrix...")
    log_scaled_matrix = sparse_matrix.copy()
    log_scaled_matrix.data = np.log1p(log_scaled_matrix.data)
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
            svd = TruncatedSVD(n_components=d, n_iter=7, random_state=42)
            embeddings = svd.fit_transform(log_scaled_matrix) 
            
            sim_scores = []
            for w1, w2, _ in valid_pairs:
                idx1 = word2idx[w1]
                idx2 = word2idx[w2]
                
                if idx1 < embeddings.shape[0] and idx2 < embeddings.shape[0]:
                    vec1 = embeddings[idx1]
                    vec2 = embeddings[idx2]                    
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(vec1, vec2) / (norm1 * norm2)
                        sim_scores.append(sim)
                    else:
                        sim_scores.append(0) # handle zero vectors if they occur
                else:
                    sim_scores.append(0) 
            
            # calculate Spearman correlation
            if len(sim_scores) == len(human_scores):
                correlation, p_value = spearmanr(sim_scores, human_scores)
            else:
                 print(f"Warning: Mismatch in score list lengths for dim {d}. Skipping correlation.")
                 correlation = 0.0
                 p_value = 1.0

            explained_variance = svd.explained_variance_ratio_.sum()
            
            results[d] = {
                'correlation': correlation,
                'p_value': p_value,
                'cumulative_explained_variance': explained_variance,
                'singular_values': svd.singular_values_
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
    valid_dimensions = sorted([d for d in dimensions if d in results and 'error' not in results[d]])
    if not valid_dimensions:
        print("No valid dimensions found in results to plot.")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    exp_var_values = [results[d]['cumulative_explained_variance'] for d in valid_dimensions]
    plt.plot(valid_dimensions, exp_var_values, 'bo-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs Dimensions (TruncatedSVD)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    corr_values = [results[d]['correlation'] for d in valid_dimensions]
    plt.plot(valid_dimensions, corr_values, 'ro-', linewidth=2)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Spearman Correlation (SimLex Subset)')
    plt.title('Semantic Correlation vs Dimensions (TruncatedSVD)')
    plt.grid(True)

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
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/svd_truncated_analysis_20.png', dpi=300, bbox_inches='tight')
    plt.close()

    # create and save results table
    valid_results = {d: r for d, r in results.items() if 'error' not in r}
    if valid_results:
        results_df = pd.DataFrame({
            'Dimension': list(valid_results.keys()),
            'Spearman Correlation': [valid_results[d]['correlation'] for d in valid_results],
            'Explained Variance': [valid_results[d]['cumulative_explained_variance'] for d in valid_results]
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
        plt.savefig('plots/svd_truncated_results_table_20.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    matrix_path = 'models/cooc_matrix_w5.pkl' 
    simlex_path = 'data/simlex_subset.txt'
    results_path = 'svd_truncated_semantic_results_20.pkl' 

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
    
    dimensions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]    
    results = test_truncated_svd_dimensions(cooc_matrix, dimensions, word2idx, simlex_pairs)
    
    if results is None:
        print("TruncatedSVD semantic analysis failed.")
        return

    plot_truncated_svd_results(results, dimensions)
    
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