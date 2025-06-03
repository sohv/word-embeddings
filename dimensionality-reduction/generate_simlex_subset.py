'''
This script is used to generate a subset of SimLex-999 with 20, 50, 100, 200, 500 pairs.
'''

import pandas as pd
import random

def generate_simlex_subset(input_file='data/SimLex-999.txt', output_file='data/simlex_subset_500.txt', num_pairs=500):
    df = pd.read_csv(input_file, sep='\t')
    
    subset = df.sample(n=num_pairs, random_state=42)
    
    with open(output_file, 'w') as f:
        for _, row in subset.iterrows():
            f.write(f"{row['word1']} {row['word2']} {row['SimLex999']}\n")
    
    print(f"Generated subset of {num_pairs} word pairs saved to {output_file}")

if __name__ == "__main__":
    generate_simlex_subset() 