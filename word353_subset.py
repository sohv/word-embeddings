import os
import random
import numpy as np

def create_wordsim_subset(input_txt_path, output_txt_path, num_pairs=20, seed=42):
    """
    Create a subset of WordSim-353 from TXT format (tab-separated)
    and save to TXT format (space-separated) similar to SimLex format.
    """
    pairs = []
    
    # Read the original WordSim TXT file
    with open(input_txt_path, 'r', encoding='utf-8') as txtfile:
        # Skip header (first row)
        next(txtfile)
        
        # Read all pairs
        for line in txtfile:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word1 = parts[0].strip()
                word2 = parts[1].strip()
                score = float(parts[2].strip())
                pairs.append((word1, word2, score))
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Sort pairs by similarity score
    pairs.sort(key=lambda x: x[2])
    
    # Strategy: Select pairs across the similarity spectrum
    total_pairs = len(pairs)
    pairs_per_quintile = num_pairs // 5
    remainder = num_pairs % 5
    
    # Distribute the remainder across quintiles
    quintile_counts = [pairs_per_quintile] * 5
    for i in range(remainder):
        quintile_counts[i] += 1
    
    # Select pairs from each quintile
    selected_pairs = []
    for i in range(5):
        start_idx = i * (total_pairs // 5)
        end_idx = (i + 1) * (total_pairs // 5) if i < 4 else total_pairs
        quintile_pairs = pairs[start_idx:end_idx]
        
        # Randomly select pairs from this quintile
        selected_from_quintile = random.sample(quintile_pairs, quintile_counts[i])
        selected_pairs.extend(selected_from_quintile)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    # Write the subset to file in TXT format (space-separated)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for word1, word2, score in selected_pairs:
            f.write(f"{word1} {word2} {score}\n")
    
    print(f"Created WordSim subset with {len(selected_pairs)} pairs at {output_txt_path}")
    
    # Print the pairs for reference
    print("\nSelected Word Pairs:")
    for word1, word2, score in selected_pairs:
        print(f"{word1:<15} {word2:<15} {score}")
    
    return selected_pairs

if __name__ == "__main__":
    # Input and output paths
    input_txt_path = "data/word353.txt"
    output_txt_path = "data/wordsim_subset.txt"
    
    # Create the subset
    create_wordsim_subset(input_txt_path, output_txt_path, num_pairs=20)
    
    print("\nDone! The WordSim subset is ready for use.")