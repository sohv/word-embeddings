# Language Representation Through Word Embeddings

## Table of Contents
1. [Overview](#Overview)
   - [Key Features](#key-features)
2. [Codebase](#codebase)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Results](#results)
6. [Dependencies](#dependencies)

## Overview
In natural language processing (NLP), word embeddings are numerical representations of words, that capture the syntactic and semantic relationship between words. These embeddings have become the foundation for various NLP applications like similarity matching and machine translation, enabling computers to understand and interpret human language.

This project focuses on word embeddings - creation, reduction and evaluation and cross-alignment of bilingual word embeddings (English and Hindi) using Procrustes alignment. Additionally, the evaluation of harmful word associations in pre-trained word embeddings, particularly in gender and racial bias is discussed. The word embeddings derived from co-occurrence matrices are analyzed and evaluated with the pre-trained embeddings (GloVe and Word2Vec). The analysis also focuses on understanding how different dimensionality reduction techniques affect embedding quality and semantic relationships between words. Access the project report [here](https://drive.google.com/file/d/1arvF6EAzD143_SfyVrheGsgBo-Djt71P/view?usp=sharing).

### Key Features
- **Co-occurrence Matrix Processing**: Efficient handling and analysis of word co-occurrence data
- **Dimensionality Reduction**: Implementation of SVD-based dimensionality reduction with configurable dimensions
- **Embedding Evaluation**: Quantitative assessment using SimLex dataset and qualitative analysis through word similarity tasks
- **Visualization Tools**: PCA and t-SNE based visualization of high-dimensional word embeddings and pre-trained embeddings.
- **Comparative Analysis**: Benchmarking against established embedding methods (GloVe, Word2Vec)
- **Multilingual Alignment**: Effective bilingual alignment of word embeddings using techniques like Procrustes, Canonical Correlation Analysis (CCA).
- **Robust Bias Evaluation**: Multi-step approach to evaluate biased word associations in word embeddings.

## Codebase

This section discusses the breakdown of the project repository's files and folders and gives a brief overview of the code functionality.

- README.md page provides comprehensive information about the repository, including its structure, functionality and usage.
- requirements.txt contains the dependencies to be installed during the initial set-up before running the project.
- data/ folder holds the dataset files used in this project, including subsets of Simlex999 and the English text corpus.
- images/ folder contains the images to enhance the understanding of the project, including images of visualised co-occurrence matrix.
- plots/ folder contains all the visualizations and results that provide detailed insights into each task performed.
   - Task-1/ folder contains the outputs, including results and plots, generated during the first task. This folder is further divided into Part-1, Part-2, Part-3, Part-4 which contain the outputs of first, second, third and fourth parts of Task-1 respectively.
   - Task-3/ contains the outputs generated during the third task. This folder is further divided into gender_bias and racial_bias, which contain the outputs and results pertaining to gender bias and racial bias evaluation respectively.
- Task-1/ folder contains all the code files related to the first task. This folder is further divided into many parts :
   - Part-1/ focuses on creation and analysis of the co-occurrence matrix. An optimal window size is decided by running the `co_occurrence_matrix.py` file and the matrix is visualised by running the `visualise_cooccurrence.py` file.
   - Part-2/ contains files related to the dimensionality reduction of the co-occurrence matrix using full SVD. This includes analyses based on both Spearman correlation and explained variance threshold methods.
   - Part-3/ focuses on the evaluation of the word embeddings generated from the constructed co-occurrence matrix using Spearman's rank correlation and Pearson's rank correlation.
   - Part-4/ focuses on the evaluation of the generated word-embeddings with the pre-trained embeddings of Word2Vec and GLoVe. The files for downloading both the pre-trained embeddings are also included in this folder.
- Task-2/ contains the jupyter notebook that focuses on cross-lingual alignment of English and Hindi embeddings downloaded from Facebook's fastText. The aligned embeddings are evaluated on various metrics like p@1, p@5 and ablation studies is also conducted to better understand these embeddings.
- Task-3/ focuses on harmful word associations in pre-trained embeddings of Word2Vec and GLoVe, specifically in gender and racial bias.

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
├── images/
├── plots/
│   ├── Task-1/
│   │   ├── Part-1/
│   │   ├── Part-2/
│   │   ├── Part-3/
│   │   └── Part-4/
│   └── Task-3/
│       ├── gender_bias/
│       └── racial_bias/
├── Task-1/
│   ├── Part-1/
│   │   ├── analyse_matrix.py
│   │   ├── check_spearman.py
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

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create new folders
```bash
mkdir models  //to store the model files
mkdir embeddings //to store the pre-trained embeddings
```
 
5. Download required data:
- Place vocabulary file in data directory
- Place co-occurrence matrix in models directory
- Place SimLex data in data directory
- The instructions to download GLoVe and Word2Vec embeddings are given in the notebook `evaluation_harmful_association.ipynb` under Task-3 folder.


## Results

The project evaluates word embeddings through several metrics:
1. Word Similarity: Using SimLex dataset to measure correlation between predicted and human similarity scores
2. Dimensionality Analysis: Testing different dimensions (50-2000) for optimal performance
3. Visualization: PCA and t-SNE plots for embedding visualization.

## Dependencies

- numpy
- scipy
- matplotlib
- nltk
- scikit-learn
- pandas
- tqdm

