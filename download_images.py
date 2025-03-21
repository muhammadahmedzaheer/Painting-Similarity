import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load dataset
df_images = pd.read_csv("published_images.csv")

# Create folder to save images
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)

# Get all image URLs
image_urls = df_images['iiifthumburl'].dropna()

# Function to check if an image already exists
def is_downloaded(idx):
    img_path = os.path.join(save_dir, f"image_{idx}.jpg")
    return os.path.exists(img_path)

# Function to download and save an image
def download_image(idx, url):
    if is_downloaded(idx):
        return True  # Skip if already downloaded

    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))

        # Save image
        img_path = os.path.join(save_dir, f"image_{idx}.jpg")
        img.save(img_path)

        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

# Get indices of missing images
missing_indices = [idx for idx in range(len(image_urls)) if not is_downloaded(idx)]
missing_urls = [image_urls.iloc[idx] for idx in missing_indices]

# Use ThreadPoolExecutor for parallel downloads
num_threads = 10  # Number of parallel downloads
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    list(tqdm(executor.map(download_image, missing_indices, missing_urls), total=len(missing_urls)))

print("✅ Download complete (or resumed from last stop)!")
