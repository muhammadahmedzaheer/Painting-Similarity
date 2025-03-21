import numpy as np
import os
from tqdm import tqdm

# File paths
SIMILARITY_FILE = "similarity_scores.npy"
NAMES_FILE = "image_names.npy"
TOP_SIMILAR_FILE = "top_10_similar.npy"

# Load image names
image_names = np.load(NAMES_FILE)

# Get total number of images
num_images = len(image_names)
top_k = 10  # We only need the top 10 most similar images

# Use NumPy memmap to handle large files efficiently
similarity_scores = np.memmap(SIMILARITY_FILE, dtype='float32', mode='r', shape=(num_images, num_images))

# Check if progress exists
if os.path.exists(TOP_SIMILAR_FILE):
    print("🔄 Resuming from previous progress...")
    top_similar = np.load(TOP_SIMILAR_FILE, allow_pickle=True)
else:
    print("📝 Creating new top-10 similarity file...")
    top_similar = np.empty(num_images, dtype=object)

# Find where we left off
start_idx = np.count_nonzero([x is not None for x in top_similar])  # Count non-empty rows

print(f"🖼️ Total images: {num_images}, Already processed: {start_idx}, Remaining: {num_images - start_idx}")

# Process remaining images
for i in tqdm(range(start_idx, num_images), desc="Extracting Top-10 Similarity"):
    similarities = similarity_scores[i]  # Load similarity scores for image i
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Get top 10 (excluding self)
    top_similar[i] = [(image_names[j], similarities[j]) for j in top_indices]  # Store names & scores

    # Save progress every 1000 rows
    if i % 1000 == 0:
        np.save(TOP_SIMILAR_FILE, top_similar)
        print(f"💾 Progress saved! Processed {i}/{num_images} images.")

# Final save
np.save(TOP_SIMILAR_FILE, top_similar)
print("✅ Extraction complete! Saved top 10 similar images per painting.")
