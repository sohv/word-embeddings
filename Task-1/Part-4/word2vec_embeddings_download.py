import numpy as np
import struct
import os
import pickle
import zipfile

def load_word2vec_binary(file_path, limit=100000, save_path=None):
    embeddings = {}
    
    print(f"Loading embeddings from {file_path}")
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').strip().split()
        vocab_size = int(header[0])
        vector_size = int(header[1])
        
        print(f"Vocabulary size: {vocab_size}, Vector dimension: {vector_size}")
        max_vectors = min(vocab_size, limit) if limit else vocab_size
        
        for i in range(max_vectors):
            if i % 10000 == 0:
                print(f"Loaded {i}/{max_vectors} word vectors")
                
            word = b''
            char = f.read(1)
            while char != b' ' and char != b'\n':
                word += char
                char = f.read(1)
            
            word = word.decode('utf-8', errors='ignore')
            vector = np.array(struct.unpack(f'{vector_size}f', f.read(vector_size * 4)), dtype=np.float32)
            embeddings[word] = vector
    
    print(f"Successfully loaded {len(embeddings)} word vectors")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving embeddings to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved successfully")
    
    return embeddings

def extract_word2vec_from_zip(zip_path, extract_dir=None):
    if extract_dir is None:
        extract_dir = os.path.dirname(zip_path)
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Files in archive: {file_list}")
        
        bin_files = [f for f in file_list if f.endswith('.bin')]
        if not bin_files:
            raise ValueError("No .bin file found in the zip archive")
        
        bin_file = bin_files[0]
        print(f"Extracting {bin_file}...")
        
        zip_ref.extract(bin_file, extract_dir)
        
    bin_path = os.path.join(extract_dir, bin_file)
    print(f"Extracted to {bin_path}")
    
    return bin_path

zip_path = "GoogleNews-vectors-negative300.bin.zip"
embeddings_folder = "embeddings"
os.makedirs(embeddings_folder, exist_ok=True)

bin_path = extract_word2vec_from_zip(zip_path)

save_path = os.path.join(embeddings_folder, "word2vec_embeddings.pkl")
word2vec_embeddings = load_word2vec_binary(bin_path, limit=100000, save_path=save_path)

print(f"Loaded and saved {len(word2vec_embeddings)} Word2Vec embeddings")