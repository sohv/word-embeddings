'''
This script builds a co-occurrence matrix for different context window sizes, evaluates the matrix statistics and finds the optimal window size.
'''
import numpy as np
from collections import defaultdict, Counter
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pickle
from scipy.signal import find_peaks
import os

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_vocabulary(file_path, min_freq=5):
    word_freq = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting words"):
            text = line.strip().split('\t')[-1]
            words = preprocess_text(text)
            word_freq.update(words)
    
    vocabulary = {word for word, freq in word_freq.items() if freq >= min_freq}
    word2idx = {word: idx for idx, word in enumerate(sorted(vocabulary))}
    
    return word2idx

def create_cooccurrence_matrix(file_path, word2idx, window_size):
    vocab_size = len(word2idx)
    cooc_dict = defaultdict(float)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Building co-occurrence matrix (window={window_size})"):
            text = line.strip().split('\t')[-1]
            words = preprocess_text(text)
            word_indices = [word2idx[word] for word in words if word in word2idx]
            
            for i, center_word_idx in enumerate(word_indices):
                window_start = max(0, i - window_size)
                window_end = min(len(word_indices), i + window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j:
                        context_word_idx = word_indices[j]
                        cooc_dict[(center_word_idx, context_word_idx)] += 1.0 / abs(i - j)
    
    rows, cols, data = zip(*[(i, j, v) for (i, j), v in cooc_dict.items()])
    cooc_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    
    return cooc_matrix

def evaluate_matrix_statistics(cooc_matrix):
    results = {}
    
    # Calculate coverage and sparsity
    total_possible = cooc_matrix.shape[0] * cooc_matrix.shape[1]
    nonzero = cooc_matrix.nnz
    sparsity = 1 - (nonzero / total_possible)
    results['sparsity'] = sparsity
    results['coverage'] = 1 - sparsity
    results['nonzero'] = nonzero
    
    # apply log(1+x) scaling to co-occurrence values
    data = cooc_matrix.data
    log_scaled_data = np.log1p(data)

    results['mean_cooc'] = np.mean(log_scaled_data)
    results['std_cooc'] = np.std(log_scaled_data)
    results['max_cooc'] = np.max(log_scaled_data)
    results['raw_mean_cooc'] = np.mean(data)
    results['raw_std_cooc'] = np.std(data)
    
    return results

def find_inflection_points(x, y):
    dy = np.gradient(y)
    peaks, _ = find_peaks(dy)
    valleys, _ = find_peaks(-dy)
    
    inflection_points = sorted(list(peaks) + list(valleys))
    return [x[i] for i in inflection_points]

def plot_evaluation_results(results):
    window_sizes = [r['window_size'] for r in results]
    metrics = {
        'coverage': ([r['coverage'] for r in results], 'Coverage', 'b', 'Coverage Rate'),
        'mean_cooc': ([r['mean_cooc'] for r in results], 'Log-scaled Mean Co-occurrence', 'r', 'Log(1+x) Average'),
        'std_cooc': ([r['std_cooc'] for r in results], 'Log-scaled Std Co-occurrence', 'm', 'Log(1+x) Standard Deviation'),
        'sparsity': ([r['sparsity'] for r in results], 'Sparsity', 'g', 'Matrix Sparsity'),
        'raw_mean_cooc': ([r['raw_mean_cooc'] for r in results], 'Raw Mean Co-occurrence', 'c', 'Raw Average'),
        'raw_std_cooc': ([r['raw_std_cooc'] for r in results], 'Raw Std Co-occurrence', 'y', 'Raw Standard Deviation')
    }
    
    show_inflection_points = {
        'coverage': True,
        'sparsity': True,
        'mean_cooc': False, 
        'std_cooc': False,  
        'raw_mean_cooc': False, 
        'raw_std_cooc': False
    }
    
    for metric_name, (values, title, color, ylabel) in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(window_sizes, values, f'{color}o-', label=title, linewidth=2, markersize=8)
        if metric_name in ['mean_cooc', 'std_cooc']:
            if metric_name == 'mean_cooc':
                plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Expected lower bound')
                plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Expected upper bound')
            else:  # std_cooc
                plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Expected lower bound')
                plt.axhline(y=2.0, color='k', linestyle='--', alpha=0.3, label='Expected upper bound')
        
        if show_inflection_points.get(metric_name, False):
            try:
                inflection_points = find_inflection_points(window_sizes, values)
                if inflection_points:
                    y_values = np.interp(inflection_points, window_sizes, values)
                    plt.plot(inflection_points, y_values, 'k^', 
                            label='Inflection Points', 
                            markersize=10)
                    
                    for x, y in zip(inflection_points, y_values):
                        plt.annotate(f'w={x}', 
                                   (x, y),
                                   xytext=(10, 10),
                                   textcoords='offset points',
                                   bbox=dict(facecolor='white', 
                                           edgecolor='black', 
                                           alpha=0.7))
            except Exception as e:
                print(f"Warning: Could not plot inflection points for {metric_name}: {str(e)}")
        
        plt.xlabel('Window Size', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{title} vs Window Size', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(window_sizes)
        
        plt.tight_layout()
        try:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{metric_name}_analysis.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Could not save plot for {metric_name}: {str(e)}")
        finally:
            plt.close()

    plt.figure(figsize=(12, 8))
    summary_metrics = {k: v for k, v in metrics.items() 
                      if k in ['coverage', 'mean_cooc', 'std_cooc', 'sparsity']}
    
    for metric_name, (values, title, color, _) in summary_metrics.items():
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        plt.plot(window_sizes, normalized_values, f'{color}o-', 
                label=title, linewidth=2, markersize=8)
    
    plt.xlabel('Window Size', fontsize=12)
    plt.ylabel('Normalized Value', fontsize=12)
    plt.title('Comparison of Metrics (Log-scaled)', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(window_sizes)
    
    try:
        plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save comparison plot: {str(e)}")
    finally:
        plt.close()

def main():
    file_path = 'data/eng_news_2024_300K-sentences.txt'
    window_sizes = [2, 3, 5, 7, 10, 12, 15, 18, 20]
    
    print("Building vocabulary...")
    word2idx = build_vocabulary(file_path)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    results = []
    for window_size in window_sizes:
        print(f"\nProcessing window size: {window_size}")
        
        # create and evaluate matrix
        cooc_matrix = create_cooccurrence_matrix(file_path, word2idx, window_size)
        eval_results = evaluate_matrix_statistics(cooc_matrix)
        eval_results['window_size'] = window_size
        results.append(eval_results)

        with open(f'cooc_matrix_w{window_size}.pkl', 'wb') as f:
            pickle.dump({
                'matrix': cooc_matrix,
                'word2idx': word2idx,
                'window_size': window_size,
                'evaluation': eval_results
            }, f)
        
        print(f"Window size: {window_size}")
        print(f"Matrix shape: {cooc_matrix.shape}")
        print(f"Coverage: {eval_results['coverage']:.4f}")
        print(f"Sparsity: {eval_results['sparsity']:.4f}")
        print(f"Non-zero elements: {eval_results['nonzero']}")
        print(f"Mean co-occurrence: {eval_results['mean_cooc']:.4f}")
        print(f"Std co-occurrence: {eval_results['std_cooc']:.4f}")
    
    plot_evaluation_results(results)
    
    coverage_values = [r['coverage'] for r in results]
    coverage_inflections = find_inflection_points(window_sizes, coverage_values)
    
    if coverage_inflections:
        recommended_window = min(coverage_inflections, key=lambda x: abs(x - 5))
        print("\nWindow Size Analysis:")
        print(f"Inflection points found at window sizes: {coverage_inflections}")
        print(f"Recommended window size: {recommended_window}")

if __name__ == "__main__":
    main()