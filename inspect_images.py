import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pandas as pd

# Load the dataset
df_images = pd.read_csv("E:/Study/GSoC/HumanAI/opendata/data/published_images.csv")

# Get first non-empty thumbnail URL
image_url = df_images['iiifthumburl'].dropna().iloc[0]

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Show the image
plt.imshow(img)
plt.axis("off")  # Hide axes
plt.title("Sample Image from Dataset")
plt.show()
