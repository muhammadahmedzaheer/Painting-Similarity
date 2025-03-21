import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load feature vectors & image names
FEATURES_FILE = "features.npy"
NAMES_FILE = "image_names.npy"
TOP_SIMILAR_FILE = "top_10_similar.npy"

# Load data
image_features = np.load(FEATURES_FILE)
image_names = np.load(NAMES_FILE)

# Total number of images
num_images = len(image_features)
top_k = 10  # Store only top 10 most similar images

# If interrupted before, load existing progress
if os.path.exists(TOP_SIMILAR_FILE):
    print("🔄 Resuming from previous progress...")
    top_similar = np.load(TOP_SIMILAR_FILE, allow_pickle=True)
else:
    print("📝 Creating new similarity file...")
    top_similar = np.empty(num_images, dtype=object)

# Find where we left off
start_idx = np.count_nonzero([x is not None for x in top_similar])  # Count non-empty rows

print(f"🖼️ Total images: {num_images}, Already processed: {start_idx}, Remaining: {num_images - start_idx}")

# Compute top-10 similarity row by row
for i in tqdm(range(start_idx, num_images), desc="Computing Top-10 Similarity"):
    similarities = cosine_similarity([image_features[i]], image_features)[0]  # Compute similarity
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Get top 10 (excluding self)
    top_similar[i] = [(image_names[j], similarities[j]) for j in top_indices]  # Store names & scores

    # Save progress every 1000 rows
    if i % 1000 == 0:
        np.save(TOP_SIMILAR_FILE, top_similar)
        print(f"💾 Progress saved! Processed {i}/{num_images} images.")

# Final save
np.save(TOP_SIMILAR_FILE, top_similar)
print("✅ Optimized similarity computation complete! Saved top 10 similar images per painting.")
