'''
This script is used to find the optimal SVD dimension for a subset of SimLex-999 with 50 pairs. We will use full SVD (Singular Value Decomposition) to find the optimal dimension.
'''

import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import pandas as pd

def test_svd_dimensions(cooc_matrix, word2idx, simlex_pairs, dimensions=[50, 100, 200, 300, 400, 500]):
    results = {}
    cooc_matrix_log = cooc_matrix.copy()
    cooc_matrix_log.data = np.log1p(cooc_matrix_log.data)
    
    valid_pairs = [(w1, w2, score) for w1, w2, score in simlex_pairs 
                  if w1 in word2idx and w2 in word2idx]
    
    print(f"Testing SVD dimensions with {len(valid_pairs)} valid SimLex pairs")
    
    for d in tqdm(dimensions, desc="Testing SVD dimensions"):
        try:
            u, s, vt = svds(cooc_matrix_log, k=d, random_state=42, maxiter=2000, tol=1e-6)
            embeddings = u * np.sqrt(s)
            sim_scores = []
            human_scores = []
            
            for w1, w2, score in valid_pairs:
                vec1 = embeddings[word2idx[w1]]
                vec2 = embeddings[word2idx[w2]]
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)   
                sim = np.dot(vec1, vec2)
                sim_scores.append(sim)
                human_scores.append(score)
            
            correlation, p_value = spearmanr(sim_scores, human_scores)
            explained_var = (s**2).sum() / (cooc_matrix_log.data**2).sum()
            results[d] = {
                'correlation': correlation,
                'p_value': p_value,
                'explained_variance': explained_var,
                'singular_values': s
            }
            
            print(f"\nDimension {d}:")
            print(f"Spearman correlation: {correlation:.4f}")
            print(f"Explained variance: {explained_var:.4f}")
            
        except Exception as e:
            print(f"Error processing dimension {d}: {str(e)}")
    
    plt.figure(figsize=(12, 8))
    
    dims = list(results.keys())
    correlations = [results[d]['correlation'] for d in dims]
    explained_vars = [results[d]['explained_variance'] for d in dims]
    
    plt.subplot(2, 1, 1)
    plt.plot(dims, correlations, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Spearman Correlation')
    plt.title('SVD Dimension Impact on Semantic Similarity')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(dims, explained_vars, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Explained Variance Ratio')
    plt.title('SVD Dimension Impact on Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/svd_dimension_analysis_50.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results_df = pd.DataFrame({
        'Dimension': dims,
        'Spearman Correlation': correlations,
        'Explained Variance': explained_vars
    })
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # create the results table
    table = ax.table(cellText=results_df.round(4).values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.savefig('plots/svd_dimension_results_table_50.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    best_dim = max(results.keys(), key=lambda d: results[d]['correlation'])
    print("\nOptimal SVD Results:")
    print(f"Best dimension: {best_dim}")
    print(f"Best correlation: {results[best_dim]['correlation']:.4f}")
    print(f"Explained variance at best dim: {results[best_dim]['explained_variance']:.4f}")
    
    return results, best_dim

def main():
    try:
        with open('models/cooc_matrix_w5.pkl', 'rb') as f:
            data = pickle.load(f)
            cooc_matrix = data['matrix']
            word2idx = data['word2idx']
    except FileNotFoundError:
        print("Error: Could not find co-occurrence matrix for window size 5")
        return

    simlex_pairs = []
    try:
        with open('data/simlex_subset_50.txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    w1, w2, score = parts[0], parts[1], float(parts[2])
                    simlex_pairs.append((w1, w2, score))
    except FileNotFoundError:
        print("Error: Could not find SimLex subset file")
        return
    
    dimensions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results, best_dim = test_svd_dimensions(cooc_matrix, word2idx, simlex_pairs, dimensions)
    
    with open('svd_analysis_results_50.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'best_dimension': best_dim,
            'window_size': 5,
            'dimensions_tested': dimensions
        }, f)

if __name__ == "__main__":
    main()