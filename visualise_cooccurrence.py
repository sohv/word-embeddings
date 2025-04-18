import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the co-occurrence matrix from the .pkl file
def load_matrix(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # Add after loading the matrix
    # Debug the structure
    print(f"Type of loaded data: {type(data)}")
    print(f"Keys in loaded data: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

    # If data is a dict with a 'matrix' key, access it properly:
    if isinstance(data, dict) and 'matrix' in data:
        cooc_matrix = data['matrix']
        print(f"Matrix shape: {cooc_matrix.shape}")
        print(f"Matrix type: {type(cooc_matrix)}")
        return cooc_matrix
    else:
        print("Data structure doesn't contain the expected 'matrix' key")
        return None

# Step 2: Visualize the matrix as a heatmap
def visualize_matrix(matrix):
    plt.figure(figsize=(10, 8))
    subset_matrix = matrix[:5, :5].toarray()
    sns.heatmap(subset_matrix, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Co-occurrence Matrix Heatmap')
    plt.show()

# Example usage
file_path = 'models/cooc_matrix_w5.pkl'  # Replace with your actual file path
matrix = load_matrix(file_path)
if matrix is not None:
    visualize_matrix(matrix)
