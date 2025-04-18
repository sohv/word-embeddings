'''
This script creates a subset of SimLex-999 containing only word pairs and their similarity scores.
'''

def create_simlex_subset():
    input_file = 'data/SimLex-999.txt'
    output_file = 'data/simlex_full.txt'
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            # Skip header line
            next(f_in)
            
            for line in f_in:
                # Split by tab and get first three columns
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # Ensure we have at least 4 columns
                    word1 = parts[0]
                    word2 = parts[1]
                    score = parts[3]  # SimLex-999 score is in the 4th column
                    
                    # Write to output file
                    f_out.write(f"{word1} {word2} {score}\n")
        
        print(f"Successfully created subset file: {output_file}")
        print("Format: word1 word2 similarity_score")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    create_simlex_subset()