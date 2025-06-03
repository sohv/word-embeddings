'''
This script downloads the GloVe embeddings from the Stanford NLP group and saves the top 100000 embeddings of dimension 300 in the embeddings/ folder.
Since embeddings/ folder does not exist, create the folder in the root directory before running the script.
'''

import numpy as np
import requests
import zipfile
import os
import pickle 

def download_glove(dim=300, save_dir='embeddings', top_n=100000):
    if dim not in [50, 100, 200, 300]:
        raise ValueError("Embedding dimension must be one of: 50, 100, 200, 300")

    os.makedirs(save_dir, exist_ok=True)

    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(save_dir, "glove.6B.zip")

    if not os.path.exists(zip_path):  
        print(f"Downloading GloVe embeddings from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {zip_path}")
    else:
        print(f"File already exists at {zip_path}")

    embed_file = f"glove.6B.{dim}d.txt"
    extract_path = os.path.join(save_dir, embed_file)

    if not os.path.exists(extract_path):
        print(f"Extracting {embed_file}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            matching_files = [f for f in zip_ref.namelist() if f.endswith(f"{dim}d.txt")]
            if not matching_files:
                raise ValueError(f"No embedding file found with dimension {dim}")
            
            zip_ref.extract(matching_files[0], save_dir)
            extracted_file = os.path.join(save_dir, matching_files[0])
            if extracted_file != extract_path:
                os.rename(extracted_file, extract_path)
        print(f"Extracted to {extract_path}")
    else:
        print(f"Embeddings already extracted at {extract_path}")

    embeddings = {}
    with open(extract_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break

            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector

            if (i + 1) % 10000 == 0:
                print(f"Loaded {i + 1} embeddings")

    print(f"Successfully loaded {len(embeddings)} embeddings.")

    pkl_path = os.path.join(save_dir, f"glove_6B_{dim}d_top{top_n}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {pkl_path}")

    return embeddings

if __name__ == "__main__":
    glove_embeddings = download_glove(dim=300, top_n=100000)