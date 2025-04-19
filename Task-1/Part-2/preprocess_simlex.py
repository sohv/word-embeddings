'''
This script creates a processed SimLex-999 file containing only word pairs and their similarity scores. The processed file is saved in data/simlex_full.txt and is used for SVD dimension analysis and evaluation.
'''

def create_simlex_subset():
    input_file = 'data/SimLex-999.txt'
    output_file = 'data/simlex_full.txt'
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            next(f_in)
            
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    word1 = parts[0]
                    word2 = parts[1]
                    score = parts[3]
                    f_out.write(f"{word1} {word2} {score}\n")
        
        print(f"Successfully created subset file: {output_file}")
        print("Format: word1 word2 similarity_score")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    create_simlex_subset() 