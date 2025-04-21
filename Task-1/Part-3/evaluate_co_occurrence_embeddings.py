'''
This script loads the generated co-occurrence matrix, extracts word embeddings and evaluates them according to word similarity on SimLex-999 dataset.
'''
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import os

def load_cooc_matrix(filename):
    print(f"Loading co-occurrence matrix from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    matrix = data['matrix']
    word2idx = data['word2idx']
    idx2word = {idx: word for word, idx in word2idx.items()}
    evaluation = data.get('evaluation', {})  
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(word2idx)}")
    symmetry_diff = (matrix - matrix.T).sum()
    print(f"Matrix symmetry check - difference: {symmetry_diff}")
    
    if abs(symmetry_diff) > 1e-10:
        print("WARNING: Matrix is not perfectly symmetric!")
    
    return matrix, word2idx, idx2word, evaluation

def train_embeddings(matrix, dimensions=100, random_state=42):
    print(f"Training {dimensions}-dimensional embeddings using SVD...")
    X = matrix.copy()
    X.data = np.log1p(X.data)
    svd = TruncatedSVD(n_components=dimensions, random_state=random_state)
    embeddings = svd.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance:.4f}")
    return embeddings, svd

def normalize_embeddings(embeddings):
    norms = np.sqrt((embeddings ** 2).sum(axis=1, keepdims=True))
    return embeddings / norms

def load_word_similarity_dataset(filename):
    word_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word1, word2, score = parts[0], parts[1], float(parts[2])
            word_pairs.append((word1, word2, score))
    
    return word_pairs

def evaluate_word_similarity(embeddings, word2idx, similarity_dataset):
    results = {'coverage': 0, 'pairs_found': 0, 'total_pairs': len(similarity_dataset)}   
    if not similarity_dataset:
        print("No word pairs found in similarity dataset.")
        return results
    
    human_scores = []
    model_scores = []
    missing_words = set()
    
    for word1, word2, score in similarity_dataset:
        if word1 in word2idx and word2 in word2idx:
            idx1, idx2 = word2idx[word1], word2idx[word2]
            vec1, vec2 = embeddings[idx1], embeddings[idx2]
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            
            human_scores.append(score)
            model_scores.append(similarity)
            results['pairs_found'] += 1
        else:
            if word1 not in word2idx:
                missing_words.add(word1)
            if word2 not in word2idx:
                missing_words.add(word2)
    
    results['coverage'] = results['pairs_found'] / results['total_pairs'] if results['total_pairs'] > 0 else 0
    
    if human_scores and model_scores:
        from scipy.stats import spearmanr, pearsonr
        spearman_corr, _ = spearmanr(human_scores, model_scores)
        pearson_corr, _ = pearsonr(human_scores, model_scores)
        
        results['spearman'] = spearman_corr
        results['pearson'] = pearson_corr
        
        print(f"Found {results['pairs_found']} word pairs out of {results['total_pairs']} ({results['coverage']:.2%})")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print(f"Pearson correlation: {pearson_corr:.4f}")
        
        if missing_words:
            print(f"Missing {len(missing_words)} unique words from vocabulary")
            if len(missing_words) <= 10:
                print(f"Missing words: {', '.join(missing_words)}")
    else:
        print("Not enough matching word pairs found for evaluation")
    
    return results

def visualize_embeddings(embeddings, idx2word, num_words=100, perplexity=30):    
    vocab_subset = list(idx2word.keys())[:num_words]
    embeddings_subset = embeddings[vocab_subset]
    
    print(f"Running t-SNE on {len(vocab_subset)} words...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_subset)
    
    plt.figure(figsize=(16, 16))
    for i, word_idx in enumerate(vocab_subset):
        x, y = reduced_embeddings[i, :]
        word = idx2word[word_idx]
        plt.scatter(x, y, alpha=0)
        plt.annotate(word, xy=(x, y), alpha=0.7)
    
    plt.title(f"t-SNE visualization of {num_words} embeddings")
    plt.grid(False)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/embedding_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def find_nearest_neighbors(query_word, word2idx, idx2word, embeddings, top_n=10):
    if query_word not in word2idx:
        print(f"Word '{query_word}' not found in vocabulary")
        return []
    
    query_idx = word2idx[query_word]
    query_vec = embeddings[query_idx]
    
    similarities = cosine_similarity([query_vec], embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_n+1]
    
    neighbors = []
    for idx in top_indices:
        if idx != query_idx:
            neighbors.append((idx2word[idx], similarities[idx]))
    
    return neighbors

def run_analogy_test(embeddings, word2idx, analogy_file):
    if not os.path.exists(analogy_file):
        print(f"Analogy file {analogy_file} not found")
        return {}
    
    normed_embeddings = normalize_embeddings(embeddings)
    results = {'semantic': {'correct': 0, 'total': 0},
              'syntactic': {'correct': 0, 'total': 0},
              'total': {'correct': 0, 'total': 0}}
    
    current_section = None
    
    with open(analogy_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Testing analogies"):
            if line.startswith(':'):
                section = line[1:].strip().lower()
                if 'gram' in section:
                    current_section = 'syntactic'
                else:
                    current_section = 'semantic'
                continue
            
            words = line.strip().lower().split()
            if len(words) != 4:
                continue
                
            if not all(word in word2idx for word in words):
                continue
                
            results[current_section]['total'] += 1
            results['total']['total'] += 1
            
            idx_a, idx_b, idx_c, idx_d = [word2idx[word] for word in words]
            vec_a = normed_embeddings[idx_a]
            vec_b = normed_embeddings[idx_b]
            vec_c = normed_embeddings[idx_c]
            
            target_vec = vec_b - vec_a + vec_c
            target_vec = target_vec / np.linalg.norm(target_vec)
            
            sims = np.dot(normed_embeddings, target_vec)
            sims[idx_a] = -np.inf
            sims[idx_b] = -np.inf
            sims[idx_c] = -np.inf
            
            predicted_idx = np.argmax(sims)
            
            if predicted_idx == idx_d:
                results[current_section]['correct'] += 1
                results['total']['correct'] += 1
    
    for section in results:
        if results[section]['total'] > 0:
            accuracy = results[section]['correct'] / results[section]['total']
            results[section]['accuracy'] = accuracy
            print(f"{section.capitalize()} accuracy: {accuracy:.4f} ({results[section]['correct']}/{results[section]['total']})")
    
    return results

def main():
    matrix_file = 'models/co-occurrence-symmetry/cooc_matrix_w20.pkl'
    embedding_dim = 1490
    matrix, word2idx, idx2word, evaluation = load_cooc_matrix(matrix_file)
    embeddings, svd_model = train_embeddings(matrix, dimensions=embedding_dim)
    normed_embeddings = normalize_embeddings(embeddings)
    
    print("Saving embeddings...")
    with open(f"word_embeddings_{embedding_dim}d.pkl", 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'word2idx': word2idx,
            'idx2word': idx2word,
            'dimensions': embedding_dim
        }, f)
    
    example_words = ['king', 'computer', 'good', 'day']
    for word in example_words:
        if word in word2idx:
            print(f"\nNearest neighbors for '{word}':")
            neighbors = find_nearest_neighbors(word, word2idx, idx2word, normed_embeddings)
            for neighbor, similarity in neighbors:
                print(f"{neighbor}: {similarity:.4f}")
    
    visualize_embeddings(embeddings, idx2word, num_words=100)
    
    similarity_datasets = {
        'simlex999': 'data/simlex_full.txt'
    }
    
    similarity_results = {}
    for name, path in similarity_datasets.items():
        if os.path.exists(path):
            print(f"\nEvaluating on {name}...")
            dataset = load_word_similarity_dataset(path)
            result = evaluate_word_similarity(normed_embeddings, word2idx, dataset)
            similarity_results[name] = result
        else:
            print(f"Dataset {name} not found at {path}")


if __name__ == "__main__":
    main()