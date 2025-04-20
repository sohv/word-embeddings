import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy import sparse

class EmbeddingEvaluator:
    def __init__(self):
        self.vocab = None
        self.word2id = None
        self.id2word = None
        self.co_occurrence_matrix = None
        self.co_occurrence_dict = None
        self.embeddings = {}
    
    def load_vocab(self, vocab_path):
        """Load vocabulary from file"""
        print(f"Loading vocabulary from {vocab_path}...")
        vocab = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if '\t' in line: 
                    word = line.strip().split('\t')[0]
                else:
                    word = line.strip()
                vocab.append(word)
        
        self.vocab = vocab
        self.word2id = {word: idx for idx, word in enumerate(vocab)}
        self.id2word = {idx: word for idx, word in enumerate(vocab)}
        print(f"Loaded vocabulary with {len(vocab)} words")
        return self
    
    def load_co_occurrence(self, matrix_path):
        print(f"Loading co-occurrence matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            print("Co-occurrence data is in dictionary format")
            self.co_occurrence_dict = data
            
            if self.vocab:
                vocab_size = len(self.vocab)
                print(f"Will convert to sparse matrix with size {vocab_size}x{vocab_size} when needed")
        else:
            print(f"Co-occurrence data format: {type(data)}")
            self.co_occurrence_matrix = data
            if hasattr(data, 'shape'):
                print(f"Co-occurrence matrix shape: {data.shape}")
                
                if self.vocab and data.shape[0] != len(self.vocab):
                    print(f"WARNING: Matrix dimensions ({data.shape[0]}) don't match vocabulary size ({len(self.vocab)})")
        
        return self
    
    def convert_dict_to_matrix(self):
        """Convert co-occurrence dictionary to sparse matrix if needed"""
        if self.co_occurrence_matrix is not None:
            return
            
        if self.co_occurrence_dict is None:
            print("No co-occurrence data loaded")
            return
            
        if not self.vocab:
            print("Vocabulary not loaded, cannot determine matrix dimensions")
            return
            
        print("Converting co-occurrence dictionary to sparse matrix...")
        vocab_size = len(self.vocab)
        
        rows = []
        cols = []
        data = []
        
        for key, value in self.co_occurrence_dict.items():
            if isinstance(key, tuple) and len(key) == 2:
                i, j = key
                rows.append(i)
                cols.append(j)
                data.append(value)
            else:
                print(f"WARNING: Unexpected key format in co-occurrence dictionary: {key}")
        
        # Create sparse matrix
        self.co_occurrence_matrix = sparse.csr_matrix((data, (rows, cols)), 
                                                      shape=(vocab_size, vocab_size))
        print(f"Created sparse matrix with shape {self.co_occurrence_matrix.shape}")
    
    def load_embeddings(self, embedding_path, name):
        print(f"Loading {name} embeddings from {embedding_path}...")
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        self.embeddings[name] = embeddings
        
        if isinstance(embeddings, dict):
            print(f"{name} embeddings are in dictionary format with {len(embeddings)} entries")
            dims = [len(vec) for vec in embeddings.values() if hasattr(vec, '__len__')]
            if dims:
                print(f"Embedding dimension: {dims[0]}")
        elif hasattr(embeddings, 'shape'):
            print(f"{name} embeddings shape: {embeddings.shape}")
            if self.vocab and embeddings.shape[0] != len(self.vocab):
                print(f"WARNING: Embedding dimensions ({embeddings.shape[0]}) don't match vocabulary size ({len(self.vocab)})")
        else:
            print(f"Embeddings format not recognized: {type(embeddings)}")
            
        return self
    
    def extract_embeddings_from_co_occurrence(self, name="co_occurrence_svd", dim=300):
        self.convert_dict_to_matrix()
        
        if self.co_occurrence_matrix is None:
            print("Co-occurrence matrix not available")
            return self
            
        print(f"Creating {name} embeddings with dimension {dim} using SVD...")
        matrix = self.co_occurrence_matrix.toarray() if hasattr(self.co_occurrence_matrix, 'toarray') else self.co_occurrence_matrix
        log_matrix = np.log(1 + matrix)
        
        U, s, Vh = np.linalg.svd(log_matrix, full_matrices=False)      
        self.embeddings[name] = U[:, :dim] * np.sqrt(s[:dim])
        print(f"Created {name} embeddings with shape: {self.embeddings[name].shape}")
        return self
    
    def find_similar_words(self, word, embedding_name, n=10):
        """Find most similar words to the given word"""
        if embedding_name not in self.embeddings:
            print(f"Embedding {embedding_name} not found")
            return []
        
        if word not in self.word2id:
            print(f"Word '{word}' not in vocabulary")
            return []
            
        embeddings = self.embeddings[embedding_name]
        
        # Handle dictionary format embeddings
        if isinstance(embeddings, dict):
            if word not in embeddings:
                print(f"Word '{word}' not in embeddings")
                return []
                
            word_vector = embeddings[word]
            # Calculate similarities with all words
            similarities = []
            for w, vec in embeddings.items():
                if w != word:  # Skip the query word
                    sim = cosine_similarity([word_vector], [vec])[0][0]
                    similarities.append((w, sim))
            
            # Sort by similarity and take top n
            most_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
            return most_similar
        else:
            # Get the word's embedding vector
            word_id = self.word2id[word]
            word_vector = embeddings[word_id].reshape(1, -1)
            
            # Calculate cosine similarity with all other words
            similarities = cosine_similarity(word_vector, embeddings)[0]
            
            # Get top N similar words (excluding the word itself)
            most_similar = [(self.id2word[i], similarities[i]) 
                            for i in similarities.argsort()[::-1]
                            if i != word_id][:n]
            
            return most_similar
    
    def evaluate_word_pairs(self, pairs_path, embedding_name):
        """Evaluate embeddings on word similarity task"""
        if embedding_name not in self.embeddings:
            print(f"Embedding {embedding_name} not found")
            return {}
        
        # Check if pairs file exists
        if not os.path.exists(pairs_path):
            print(f"Word pairs file not found: {pairs_path}")
            return {}
        
        # Load word pairs and human similarity scores
        word_pairs = []
        human_scores = []
        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                word1, word2, score = parts[0], parts[1], float(parts[2])
                word_pairs.append((word1, word2))
                human_scores.append(score)
        
        # Calculate cosine similarity for each word pair
        embedding_scores = []
        valid_pairs = []
        valid_human_scores = []
        embeddings = self.embeddings[embedding_name]
        is_dict_format = isinstance(embeddings, dict)
        
        for (word1, word2), human_score in zip(word_pairs, human_scores):
            # Handle dictionary format embeddings
            if is_dict_format:
                if word1 in embeddings and word2 in embeddings:
                    vector1 = embeddings[word1]
                    vector2 = embeddings[word2]
                    similarity = cosine_similarity([vector1], [vector2])[0][0]
                    embedding_scores.append(similarity)
                    valid_pairs.append((word1, word2))
                    valid_human_scores.append(human_score)
            else:
                if word1 in self.word2id and word2 in self.word2id:
                    vector1 = embeddings[self.word2id[word1]]
                    vector2 = embeddings[self.word2id[word2]]
                    similarity = cosine_similarity([vector1], [vector2])[0][0]
                    embedding_scores.append(similarity)
                    valid_pairs.append((word1, word2))
                    valid_human_scores.append(human_score)
        
        # Calculate Spearman correlation
        if len(embedding_scores) > 0:
            correlation, p_value = stats.spearmanr(embedding_scores, valid_human_scores)
            result = {
                'correlation': correlation,
                'p_value': p_value,
                'num_pairs': len(embedding_scores),
                'missing_words': len(word_pairs) - len(embedding_scores)
            }
            return result
        else:
            return {'error': 'No valid word pairs found'}
    
    def analogy_task(self, embedding_name, a, b, c, n=5):
        """Solve word analogy task: a is to b as c is to ?"""
        if embedding_name not in self.embeddings:
            print(f"Embedding {embedding_name} not found")
            return []
            
        embeddings = self.embeddings[embedding_name]
        is_dict_format = isinstance(embeddings, dict)
        
        # Check if words are in vocabulary
        if is_dict_format:
            missing_words = []
            for w in [a, b, c]:
                if w not in embeddings:
                    missing_words.append(w)
        else:
            missing_words = []
            for w in [a, b, c]:
                if w not in self.word2id:
                    missing_words.append(w)
        
        if missing_words:
            print(f"Words not in vocabulary: {', '.join(missing_words)}")
            return []
        
        # Get embeddings for words
        if is_dict_format:
            a_vec = embeddings[a]
            b_vec = embeddings[b]
            c_vec = embeddings[c]
            
            # Calculate target vector: b - a + c
            target = np.array(b_vec) - np.array(a_vec) + np.array(c_vec)
            
            # Normalize for better results
            target = target / np.linalg.norm(target)
            
            # Calculate similarities
            results = []
            for word, vec in embeddings.items():
                if word not in [a, b, c]:
                    # Normalize vector
                    norm_vec = np.array(vec) / np.linalg.norm(vec)
                    # Calculate cosine similarity
                    similarity = np.dot(norm_vec, target)
                    results.append((word, similarity))
            
            # Sort by similarity
            results = sorted(results, key=lambda x: x[1], reverse=True)[:n]
        else:
            # Get embeddings for words
            a_vec = embeddings[self.word2id[a]]
            b_vec = embeddings[self.word2id[b]]
            c_vec = embeddings[self.word2id[c]]
            
            # Calculate target vector: b - a + c
            target = b_vec - a_vec + c_vec
            
            # Normalize for better results
            target = target / np.linalg.norm(target)
            
            # Get normalized embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # Calculate cosine similarity with all words
            similarities = np.dot(normalized_embeddings, target)
            
            # Get top N similar words (excluding input words)
            word_ids = [self.word2id[w] for w in [a, b, c]]
            results = [(self.id2word[i], similarities[i]) 
                        for i in similarities.argsort()[::-1]
                        if i not in word_ids][:n]
        
        return results
    
    def visualize_embeddings(self, embedding_name, words=None, n=50, method='pca'):
        """Visualize embeddings in 2D using PCA or t-SNE"""
        if embedding_name not in self.embeddings:
            print(f"Embedding {embedding_name} not found")
            return
            
        embeddings = self.embeddings[embedding_name]
        is_dict_format = isinstance(embeddings, dict)
        
        # If no specific words provided, use available words
        if words is None:
            if is_dict_format:
                available_words = list(embeddings.keys())
                words = available_words[:n]
            else:
                words = list(self.id2word.values())[:n]
        else:
            # Filter words that are available
            if is_dict_format:
                words = [w for w in words if w in embeddings]
            else:
                words = [w for w in words if w in self.word2id]
            
            # If we still need more words, add some common ones
            if len(words) < n:
                if is_dict_format:
                    common_words = [w for w in list(embeddings.keys())[:n*2] if w not in words]
                    words = words + common_words[:n-len(words)]
                else:
                    common_words = [w for w in list(self.id2word.values())[:n*2] if w not in words]
                    words = words + common_words[:n-len(words)]
        
        # Get embeddings for selected words
        if is_dict_format:
            word_vectors = np.array([embeddings[w] for w in words])
        else:
            word_ids = [self.word2id[w] for w in words]
            word_vectors = embeddings[word_ids]
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:  # Default to PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            
        reduced_vecs = reducer.fit_transform(word_vectors)
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], alpha=0.5)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, (reduced_vecs[i, 0], reduced_vecs[i, 1]), fontsize=9)
            
        plt.title(f'2D visualization of {embedding_name} embeddings using {method.upper()}')
        plt.tight_layout()
        plt.show()
        
    def compare_embeddings(self, word, topn=10):
        """Compare different embeddings for a specific word"""
        results = {}
        
        for name, embeddings in self.embeddings.items():
            is_dict_format = isinstance(embeddings, dict)
            
            if is_dict_format:
                if word in embeddings:
                    similar = self.find_similar_words(word, name, topn)
                    results[name] = similar
                else:
                    print(f"Word '{word}' not found in {name} embeddings")
                    results[name] = []
            else:
                if word in self.word2id:
                    similar = self.find_similar_words(word, name, topn)
                    results[name] = similar
                else:
                    print(f"Word '{word}' not in vocabulary")
                    results[name] = []
            
        # Display results
        for name, similar_words in results.items():
            if similar_words:
                print(f"\n{name} - Most similar to '{word}':")
                for w, sim in similar_words:
                    print(f"  {w}: {sim:.4f}")
            else:
                print(f"\n{name} - No results for '{word}'")
                
        return results

    def inspect_co_occurrence_dict(self, n=10):
        """Inspect the co-occurrence dictionary structure"""
        if self.co_occurrence_dict is None:
            print("No co-occurrence dictionary loaded")
            return
            
        print("\nCo-occurrence dictionary structure:")
        print(f"Total entries: {len(self.co_occurrence_dict)}")
        
        # Get a sample of the dictionary
        sample_items = list(self.co_occurrence_dict.items())[:n]
        print(f"\nSample entries (first {len(sample_items)}):")
        
        for key, value in sample_items:
            # For tuples of word indices, convert to words if possible
            if isinstance(key, tuple) and len(key) == 2:
                i, j = key
                word_i = self.id2word.get(i, f"id_{i}") if self.id2word else f"id_{i}"
                word_j = self.id2word.get(j, f"id_{j}") if self.id2word else f"id_{j}"
                print(f"({word_i}, {word_j}) -> {value}")
            else:
                print(f"{key} -> {value}")

def main():
    # Example usage
    evaluator = EmbeddingEvaluator()
    
    # Load vocabulary
    evaluator.load_vocab('models/vocab_file.txt')
    
    # Load co-occurrence matrix
    evaluator.load_co_occurrence('models/cooc_matrix_300d.pkl')
    
    # Load pre-trained embeddings
    evaluator.load_embeddings('embeddings/word2vec_embeddings.pkl', 'word2vec')
    evaluator.load_embeddings('embeddings/glove_6B_300d.pkl', 'glove')
    
    # Extract embeddings from co-occurrence matrix
    evaluator.extract_embeddings_from_co_occurrence('svd', dim=100)
    
    # Sample evaluation
    print("\nSample similar words:")
    word = 'technology'
    for model in evaluator.embeddings:
        similar = evaluator.find_similar_words(word, model, 5)
        print(f"\n{model} - Similar to '{word}':")
        for w, sim in similar:
            print(f"  {w}: {sim:.4f}")
    
    # Sample analogy
    print("\nSample analogy: 'man is to woman as king is to ?'")
    for model in evaluator.embeddings:
        results = evaluator.analogy_task(model, 'man', 'woman', 'king')
        print(f"\n{model} results:")
        for w, sim in results:
            print(f"  {w}: {sim:.4f}")
    
    # Visualize sample words
    sample_words = ['technology', 'computer', 'data', 'internet', 'business', 
                    'money', 'health', 'science', 'research', 'education']
    for model in evaluator.embeddings:
        evaluator.visualize_embeddings(model, sample_words)

if __name__ == "__main__":
    main()