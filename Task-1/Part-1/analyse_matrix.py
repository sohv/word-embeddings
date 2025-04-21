'''
This script analyses the structure of the generated co-occurrence matrix for the window size 5. The results are stored in the images/matrix-structure.png file. The structure is analysd to better diagnose the SVD dimension analysis issues.
'''
import pickle

PICKLE_PATH = 'cooc_matrix_w5.pkl'

def inspect_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    print("\n=== Pickle File Structure ===")
    print(f"Type of loaded data: {type(data)}")
    
    if isinstance(data, dict):
        print("\nDictionary keys:")
        for key in data.keys():
            print(f"- {key}: {type(data[key])}")
            if isinstance(data[key], (list, dict)):
                print(f"  Size/shape: {len(data[key])}")
            elif hasattr(data[key], 'shape'):
                print(f"  Shape: {data[key].shape}")
    else:
        print("\nData structure:")
        print(data)

if __name__ == "__main__":
    inspect_pickle(PICKLE_PATH)