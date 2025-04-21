# Language Representation Through Word Embeddings

## Table of Contents
1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
   - [Neo4j](#neo4j-for-windows)

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
├── README.md
├── requirements.txt
├── data/
│   ├── eng_news_2024_300K-sentences.txt
│   ├── SimLex-999.txt
│   ├── simlex_full.txt
│   ├── simlex_subset_100.txt
│   ├── simlex_subset_200.txt
│   ├── simlex_subset_50.txt
│   ├── simlex_subset_500.txt
│   ├── vocab-eng-news-2024.txt
│   ├── vocab_file.txt
│   └── word2id.pkl
├── images/
├── plots/
│   ├── Task-1/
│   │   ├── Part-1/
│   │   ├── Part-2/
│   │   ├── Part-3/
│   │   └── Part-4/
│   │       ├── modified-results/
│   │       └── original/
│   └── Task-3/
│       ├── gender_bias/
│       └── racial_bias/
├── Task-1/
│   ├── Part-1/
│   │   ├── analyse_matrix.py
│   │   ├── check_spearman.py
│   │   ├── check_spearman_fixed.py
│   │   ├── co_occurence_matrix.py
│   │   ├── co_occurrence_fixed.py
│   │   ├── visualise_cooccurrence.py
│   ├── Part-2/
│   │   ├── find_dimension_spearman.py
│   │   ├── find_dimension_variance.py
│   │   ├── find_dimension_variance_modified.py
│   │   ├── generate_simlex_subset.py
│   │   └── preprocess_simlex.py
│   ├── Part-3/
│   │   └── evaluate_co_occurrence_embeddings.py
│   └── Part-4/
│       ├── diagnose_negative_correlation.py
│       ├── evaluate_embeddings.py
│       ├── glove_embeddings_download.py
│       ├── reduce_dimension_cooccurrence_matrix.py
│       ├── vocab_file_corpus.py
│       └── word2vec_embeddings_download.py
├── Task-2/
│   └── cross_lingual_alignment.ipynb
└── Task-3/
    └── evaluation_harmful_association.ipynb
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/sohv/word-embeddings.git
cd word-embeddings
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

