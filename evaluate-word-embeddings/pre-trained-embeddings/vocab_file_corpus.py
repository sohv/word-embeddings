'''
This script builds a vocabulary file from the English text corpus used to generate the co-occurrence matrix. THe minimum frequency of the words is set to 5 to match the dimension of the co-occurrence matrix. The vocabulary is saved in the data/ folder.
'''

import re
import os
from collections import Counter
import pickle

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_vocabulary(corpus_path, output_path, target_size=36308, min_count=1):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    word_counts = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = preprocess_text(line)
            word_counts.update(words)
    
    # Filter by minimum count and take top target_size
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:target_size]
    
    word2id = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Vocabulary from {os.path.basename(corpus_path)}\n")
        f.write(f"# Total unique words: {len(sorted_vocab)}\n")
        f.write("word\tcount\n")
        for word, count in sorted_vocab:
            f.write(f"{word}\t{count}\n")
    
    with open('data/word2id.pkl', 'wb') as f:
        pickle.dump(word2id, f)
    
    print(f"Vocabulary with {len(sorted_vocab)} entries saved to {output_path}")
    return len(sorted_vocab)

def main():
    corpus_path = "data/eng_news_2024_300K-sentences.txt"
    output_path = "data/vocab-eng-news-2024.txt"
    try:
        vocab_size = build_vocabulary(corpus_path, output_path, target_size=36308)
        with open('models/cooc_matrix_w5.pkl', 'rb') as f:
            data = pickle.load(f)
            matrix_shape = data['matrix'].shape[0]
        print(f"Matrix vocabulary size: {matrix_shape}")
        if vocab_size != matrix_shape:
            print(f"Warning: Vocabulary size ({vocab_size}) does not match matrix size ({matrix_shape})")
        else:
            print("Vocabulary size matches matrix size.")
    except Exception as e:
        print(f"Error building vocabulary: {e}")

if __name__ == "__main__":
    main()