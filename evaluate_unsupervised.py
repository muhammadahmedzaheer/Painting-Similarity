import numpy as np

# File paths
TOP_SIMILAR_FILE = "top_10_similar.npy"
SIMILARITY_FILE = "similarity_scores.npy"
NAMES_FILE = "image_names.npy"

# Load top 10 similar paintings
top_similar = np.load(TOP_SIMILAR_FILE, allow_pickle=True)

# Load image names and create a mapping {filename: index}
image_names = np.load(NAMES_FILE)
image_name_to_index = {name: i for i, name in enumerate(image_names)}

# Load similarity matrix
similarity_scores = np.memmap(SIMILARITY_FILE, dtype='float32', mode='r', shape=(len(image_names), len(image_names)))

def intra_similarity(query_index):
    """Compute average similarity of top 10 retrieved images."""
    top_indices = [image_name_to_index[img[0]] for img in top_similar[query_index]]  # Convert filenames to indices
    return np.mean([similarity_scores[query_index, idx] for idx in top_indices])

def inter_similarity(query_index):
    """Compute average similarity to 10 randomly chosen images (should be lower)."""
    num_images = len(image_names)
    random_indices = np.random.choice(num_images, 10, replace=False)  # Random images
    return np.mean([similarity_scores[query_index, idx] for idx in random_indices])

# Compute for a few test images
test_indices = np.random.choice(len(image_names), 5, replace=False)
intra_similarities = [intra_similarity(i) for i in test_indices]
inter_similarities = [inter_similarity(i) for i in test_indices]

# Print results
print(f"✅ Average Intra-Painting Similarity (Higher is better): {np.mean(intra_similarities):.4f}")
print(f"✅ Average Inter-Painting Similarity (Lower is better): {np.mean(inter_similarities):.4f}")

if np.mean(intra_similarities) > np.mean(inter_similarities):
    print("✅ Model is working: Similar paintings have higher similarity scores!")
else:
    print("❌ Model might not be distinguishing well.")
