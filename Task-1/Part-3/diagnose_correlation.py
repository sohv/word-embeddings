import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import random

def load_vocab(vocab_path):
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            words = line.strip().split()
            if words:
                word = words[0]
                vocab.append(word)
    
    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for idx, word in enumerate(vocab)}
    print(f"Loaded vocabulary with {len(vocab)} words")
    return vocab, word2id, id2word

def load_co_occurrence(matrix_path):
    print(f"Loading co-occurrence matrix from {matrix_path}...")
    with open(matrix_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        print("Co-occurrence data is in dictionary format")
        if 'matrix' in data:
            matrix = data['matrix']
            print(f"Matrix shape: {matrix.shape}")
        else:
            print("No 'matrix' key found in co-occurrence data")
            matrix = None
    else:
        matrix = data
        print(f"Matrix shape: {matrix.shape}")
    
    return matrix

def analyze_co_occurrence_matrix(matrix, word2id, id2word, sample_size=10, top_n=5):
    print(f"Co-occurrence matrix shape: {matrix.shape}")
    
    if sparse.issparse(matrix):
        print(f"Matrix is sparse with {matrix.nnz} non-zero entries")
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        print(f"Matrix density: {density:.6f} ({density*100:.4f}%)")
        
        row_sums = matrix.sum(axis=1).A1
        col_sums = matrix.sum(axis=0).A1
    else:
        print("Matrix is dense")
        density = np.count_nonzero(matrix) / (matrix.shape[0] * matrix.shape[1])
        print(f"Matrix density: {density:.6f} ({density*100:.4f}%)")
        
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
    
    zero_rows = np.where(row_sums == 0)[0]
    zero_cols = np.where(col_sums == 0)[0]
    
    print(f"Number of rows with all zeros: {len(zero_rows)} ({len(zero_rows)/matrix.shape[0]*100:.2f}%)")
    print(f"Number of columns with all zeros: {len(zero_cols)} ({len(zero_cols)/matrix.shape[1]*100:.2f}%)")
    
    if sparse.issparse(matrix):
        values = matrix.data
    else:
        values = matrix.flatten()
        values = values[values > 0] 
        
    print("\nValue statistics:")
    print(f"  Min value: {values.min()}")
    print(f"  Max value: {values.max()}")
    print(f"  Mean value: {values.mean():.4f}")
    print(f"  Median value: {np.median(values):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, log=True)
    plt.title('Histogram of Non-zero Co-occurrence Values (log scale)')
    plt.xlabel('Co-occurrence Count')
    plt.ylabel('Frequency (log scale)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/cooc_value_histogram.png', dpi=300, bbox_inches='tight')
    print("Saved value histogram to plots/cooc_value_histogram.png")
    
    common_words = ["the", "and", "of", "to", "in", "is", "it", "that", "was", "for"]
    found_common_words = [word for word in common_words if word in word2id]
    
    if found_common_words:
        print(f"\nAnalyzing co-occurrences for common words: {', '.join(found_common_words)}")
        analyze_words(found_common_words, matrix, word2id, id2word, top_n)
    
    vocab_size = len(word2id)
    if sample_size > vocab_size:
        sample_size = vocab_size
        
    valid_indices = np.where(row_sums > 0)[0]
    if len(valid_indices) < sample_size:
        sample_indices = valid_indices
    else:
        sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)
    
    sample_words = [id2word[idx] for idx in sample_indices]
    print(f"\nAnalyzing co-occurrences for {len(sample_words)} random words")
    analyze_words(sample_words, matrix, word2id, id2word, top_n)
    
    if matrix.shape[0] == matrix.shape[1]:
        print("\nChecking matrix symmetry...")
        if sparse.issparse(matrix):
            diff = matrix - matrix.T
            symmetry = diff.nnz == 0
        else:
            symmetry = np.allclose(matrix, matrix.T)
        
        print(f"Matrix is {'symmetric' if symmetry else 'not symmetric'}")
    
    return {
        'shape': matrix.shape,
        'density': density,
        'zero_rows': len(zero_rows),
        'zero_cols': len(zero_cols),
        'value_stats': {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'median': np.median(values)
        }
    }

def analyze_words(words, matrix, word2id, id2word, top_n=5):
    for word in words:
        if word not in word2id:
            print(f"Word '{word}' not found in vocabulary")
            continue
            
        word_idx = word2id[word]
        
        if sparse.issparse(matrix):
            co_occ = matrix[word_idx].toarray().flatten()
        else:
            co_occ = matrix[word_idx]
        
        top_indices = np.argsort(co_occ)[::-1][:top_n+1]  
        
        print(f"\nWord: '{word}'")
        print(f"  Total co-occurrences: {co_occ.sum():.0f}")
        print(f"  Non-zero co-occurrences: {np.count_nonzero(co_occ)}")
        print(f"  Top co-occurring words:")
        
        for idx in top_indices:
            if idx != word_idx:  
                co_word = id2word[idx]
                count = co_occ[idx]
                print(f"    '{co_word}': {count:.0f}")

def check_co_occurrence_matrix():
    vocab_path = 'data/vocab_file.txt'
    vocab, word2id, id2word = load_vocab(vocab_path)
    
    matrix_path = 'cooc_matrix_w5.pkl'
    cooc_matrix = load_co_occurrence(matrix_path)
    
    analysis_results = analyze_co_occurrence_matrix(cooc_matrix, word2id, id2word)
    
    if analysis_results['zero_rows'] > 0.1 * cooc_matrix.shape[0]:  
        print("\nWARNING: Many words have no co-occurrences. This will affect embedding quality.")
    
    if len(vocab) != cooc_matrix.shape[0]:
        print(f"\nWARNING: Vocabulary size ({len(vocab)}) doesn't match matrix rows ({cooc_matrix.shape[0]})")
    
    if analysis_results['density'] < 0.0001:  
        print("\nWARNING: Matrix is extremely sparse. May not contain enough information.")
    
    return analysis_results

if __name__ == "__main__":
    check_co_occurrence_matrix()