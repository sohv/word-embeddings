# Word Embedding Analysis Project

## Overview
This project implements a comprehensive framework for analyzing and evaluating word embeddings derived from co-occurrence matrices, with comparison to pre-trained embeddings (GloVe and Word2Vec). The analysis focuses on understanding how different dimensionality reduction techniques affect embedding quality and semantic relationships between words.

## Key Features
- **Co-occurrence Matrix Processing**: Efficient handling and analysis of word co-occurrence data
- **Dimensionality Reduction**: Implementation of SVD-based dimensionality reduction with configurable dimensions
- **Embedding Evaluation**: Quantitative assessment using SimLex dataset and qualitative analysis through word similarity tasks
- **Visualization Tools**: PCA and t-SNE based visualization of high-dimensional word embeddings
- **Comparative Analysis**: Benchmarking against established embedding methods (GloVe, Word2Vec)

## Project Structure

```
.
├── data/                  # Data files including SimLex and other evaluation datasets
├── models/               # Saved models and embeddings
├── embeddings/           # Generated word embeddings
├── Task-1/              # Main analysis tasks
│   ├── Part-1/          # Initial data processing
│   ├── Part-2/          # Intermediate analysis
│   └── Part-4/          # Final evaluation and visualization
│       ├── evaluate.py                    # Main evaluation script
│       ├── evaluate_embeddings.py         # Embedding evaluation utilities
│       ├── glove_embeddings_download.py   # GloVe embedding downloader
│       ├── word2vec_embeddings_download.py # Word2Vec embedding downloader
│       ├── reduce_dimension_cooccurrence_matrix.py # Dimensionality reduction
│       └── vocab_file_corpus.py          # Vocabulary processing
├── plots/               # Generated plots and visualizations
├── images/             # Project images and diagrams
├── vocab_file.txt      # Main vocabulary file
└── requirements.txt    # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required data:
- Place vocabulary file in root directory
- Place co-occurrence matrix in models directory
- Place SimLex data in data directory

## Key Components

### 1. Embedding Evaluation (`evaluate.py`)
The main evaluation script that provides functionality for:
- Loading and processing vocabulary
- Loading co-occurrence matrices
- Extracting embeddings using SVD
- Evaluating word similarities
- Performing analogy tasks
- Visualizing embeddings

### 2. Embedding Downloaders
- `glove_embeddings_download.py`: Downloads and processes GloVe embeddings
- `word2vec_embeddings_download.py`: Downloads and processes Word2Vec embeddings

### 3. Dimensionality Reduction
- `reduce_dimension_cooccurrence_matrix.py`: Reduces dimensionality of co-occurrence matrices using SVD

## Results

The project evaluates word embeddings through several metrics:
1. Word Similarity: Using SimLex dataset to measure correlation between predicted and human similarity scores
2. Dimensionality Analysis: Testing different dimensions (50-2000) for optimal performance
3. Visualization: PCA and t-SNE plots for embedding visualization

## Conclusion

The project provides a comprehensive framework for:
- Comparing different word embedding methods
- Evaluating embedding quality through various metrics
- Visualizing and analyzing word relationships
- Understanding the impact of dimensionality on embedding quality

## Dependencies

- numpy
- scipy
- matplotlib
- nltk
- scikit-learn
- pandas
- tqdm

