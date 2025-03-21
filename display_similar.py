import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load data
IMAGE_FOLDER = "images"
IMAGE_NAMES_FILE = "image_names.npy"
TOP_SIMILAR_FILE = "top_10_similar.npy"

# Load image names and similarity data
image_names = np.load(IMAGE_NAMES_FILE)
top_similar = np.load(TOP_SIMILAR_FILE, allow_pickle=True)

# Pick a random image
random_index = random.randint(0, len(image_names) - 1)
random_image_name = image_names[random_index]

# Get its top 10 similar images
similar_images = top_similar[random_index]

# Load and display the original image
fig, axes = plt.subplots(1, 11, figsize=(20, 5))

original_path = os.path.join(IMAGE_FOLDER, random_image_name)
original_img = Image.open(original_path)
axes[0].imshow(original_img)
axes[0].set_title("Original")
axes[0].axis("off")

# Load and display similar images
for i, (sim_name, sim_score) in enumerate(similar_images):
    img_path = os.path.join(IMAGE_FOLDER, sim_name)
    img = Image.open(img_path)
    
    axes[i + 1].imshow(img)
    axes[i + 1].set_title(f"Sim {i+1}\nScore: {sim_score:.2f}")
    axes[i + 1].axis("off")

plt.show()
