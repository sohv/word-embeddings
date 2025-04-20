import numpy as np
import pickle
from scipy.stats import spearmanr
import pandas as pd
from scipy.sparse.linalg import svds

# Load reduced co-occurrence embeddings (300d)
with open('/content/drive/MyDrive/Colab Notebooks/code/models/cooc_matrix_300d.pkl', 'rb') as f:
    cooc_data = pickle.load(f)
    cooc_embeddings = cooc_data['matrix']
    cooc_vocab = cooc_data['vocabulary']  # Assuming vocabulary is saved in the pickle file

# Load GloVe embeddings
glove_embeddings = {}  # Your loaded GloVe embeddings
glove_vocab = list(glove_embeddings.keys())

word2vec_embeddings = {}  
word2vec_vocab = list(word2vec_embeddings.keys())

simlex_data = []
with open('data/simlex_full.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            word1, word2 = parts[0], parts[1]
            score = float(parts[-1])
            simlex_data.append((word1, word2, score))

simlex_df = pd.DataFrame(simlex_data, columns=['word1', 'word2', 'score'])

def calculate_similarities(embeddings_dict, vocab, word1, word2):
    if word1 in vocab and word2 in vocab:
        vec1 = embeddings_dict[word1]
        vec2 = embeddings_dict[word2]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return None

def evaluate_embeddings(embeddings, vocab, name):
    similarities = []
    human_scores = []
    
    for _, row in simlex_df.iterrows():
        word1, word2 = row['word1'], row['word2']
        sim = calculate_similarities(embeddings, vocab, word1, word2)
        if sim is not None:
            similarities.append(sim)
            human_scores.append(row['score'])
    
    correlation, p_value = spearmanr(similarities, human_scores)
    print(f"\n{name} Results:")
    print(f"Spearman Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Valid word pairs: {len(similarities)} out of {len(simlex_df)}")
    
    return correlation

cooc_dict = {word: cooc_embeddings[i] for i, word in enumerate(cooc_vocab)}

print("Evaluating embeddings against SimLex-999...")
cooc_corr = evaluate_embeddings(cooc_dict, cooc_vocab, "Co-occurrence (300d)")
glove_corr = evaluate_embeddings(glove_embeddings, glove_vocab, "GloVe")
word2vec_corr = evaluate_embeddings(word2vec_embeddings, word2vec_vocab, "Word2Vec")

print("\nFinal Comparison:")
print(f"Co-occurrence (300d): {cooc_corr:.4f}")
print(f"GloVe: {glove_corr:.4f}")
print(f"Word2Vec: {word2vec_corr:.4f}")

results = {
    "Co-occurrence": cooc_corr,
    "GloVe": glove_corr,
    "Word2Vec": word2vec_corr
}

best_model = max(results.items(), key=lambda x: x[1])
print(f"\nBest performing model: {best_model[0]} with correlation {best_model[1]:.4f}")