import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load feature vectors & image names
FEATURES_FILE = "features.npy"
NAMES_FILE = "image_names.npy"
SIMILARITY_FILE = "similarity_scores.npy"

# Load data
image_features = np.load(FEATURES_FILE)
image_names = np.load(NAMES_FILE)

# Total number of images
num_images = len(image_features)

# Use NumPy memmap (saves directly to disk, avoids RAM overload)
if os.path.exists(SIMILARITY_FILE):
    print("🔄 Resuming from previous progress...")
    similarity_scores = np.memmap(SIMILARITY_FILE, dtype='float32', mode='r+', shape=(num_images, num_images))
else:
    print("📝 Creating new similarity file...")
    similarity_scores = np.memmap(SIMILARITY_FILE, dtype='float32', mode='w+', shape=(num_images, num_images))

# Find where we left off
start_idx = np.count_nonzero(similarity_scores.sum(axis=1))  # Find first row with all zeros

print(f"🖼️ Total images: {num_images}, Already processed: {start_idx}, Remaining: {num_images - start_idx}")

# Compute similarity row by row
for i in tqdm(range(start_idx, num_images), desc="Computing Similarity"):
    similarity_scores[i, :] = cosine_similarity([image_features[i]], image_features)[0]

    # Save progress every 1000 rows
    if i % 1000 == 0:
        similarity_scores.flush()  # Flush to disk
        print(f"💾 Progress saved! Processed {i}/{num_images} images.")

# Final save
similarity_scores.flush()
print("✅ Similarity computation complete! Scores saved.")
