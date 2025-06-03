'''
This script reduces the dimension of the generated co-occurrence matrix of window size 5 to 300 using SVD. The reduction is done to facilitate easier evaluation of the embeddings with pre-trained embeddings of word2vec and glove. 
The reduced embeddings are saved in the models/ folder. Create the folder models/ in the root directory before running the script.
'''

import numpy as np
import pickle
from scipy.sparse.linalg import svds

with open('models/cooc_matrix_w5.pkl', 'rb') as f:
    data = pickle.load(f)
    cooc_matrix = data['matrix']

print(f"Original matrix shape: {cooc_matrix.shape}")

print("Computing SVD to reduce to 300 dimensions...")
u, s, vt = svds(cooc_matrix, k=300, random_state=42)

reduced_embeddings = u @ np.diag(s)

reduced_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)

print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

output_data = {
    'matrix': reduced_embeddings,
    'original_shape': cooc_matrix.shape
}

with open('models/cooc_matrix_300d.pkl', 'wb') as f:
    pickle.dump(output_data, f)

print("Reduced embeddings saved to cooc_matrix_300d.pkl")