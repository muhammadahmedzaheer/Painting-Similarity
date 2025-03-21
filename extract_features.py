import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar

# Path to images folder
IMAGE_FOLDER = "images"
FEATURES_FILE = "features.npy"
NAMES_FILE = "image_names.npy"

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load Pretrained ResNet50 Model
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.eval()  # Set to evaluation mode

# Load existing features (if script was interrupted before)
if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
    print("🔄 Resuming from previous progress...")
    image_features = list(np.load(FEATURES_FILE))  # Convert to list for appending
    image_names = list(np.load(NAMES_FILE))  # Convert to list for appending
else:
    image_features = []
    image_names = []

# Get list of all image files
all_images = sorted(os.listdir(IMAGE_FOLDER))

# Find images that still need to be processed
processed_images = set(image_names)
remaining_images = [img for img in all_images if img not in processed_images]

print(f"🖼️ Total images: {len(all_images)}, Already processed: {len(processed_images)}, Remaining: {len(remaining_images)}")

# Process remaining images
for img_name in tqdm(remaining_images, desc="Extracting Features"):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    
    try:
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = resnet(img_tensor)
        
        # Flatten feature vector
        features = features.view(features.shape[0], -1).numpy()
        
        # Store results
        image_features.append(features)
        image_names.append(img_name)

        # Save progress every 100 images
        if len(image_features) % 100 == 0:
            np.save(FEATURES_FILE, np.vstack(image_features))
            np.save(NAMES_FILE, np.array(image_names))
            print(f"💾 Progress saved! {len(image_features)} images processed.")

    except Exception as e:
        print(f"❌ Failed to process {img_name}: {e}")

# Final save
np.save(FEATURES_FILE, np.vstack(image_features))
np.save(NAMES_FILE, np.array(image_names))
print(f"✅ Feature extraction complete! Saved {len(image_features)} feature vectors.")
