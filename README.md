# Painting Similarity Model

## Overview
This project applies deep learning to identify similar paintings based on their visual features. Using **ResNet50** for feature extraction and **cosine similarity** for comparison, the model groups visually similar paintings.

## Dataset
The project utilizes the **National Gallery of Art Open Data**, which provides high-resolution paintings and metadata. The dataset is processed to extract feature embeddings for similarity analysis.

## Project Structure
images/: Downloaded painting images (not included in repo)

features.npy: Extracted image features (not included in repo)

image_names.npy: Image filenames (not included in repo)

similarity_scores.npy: Full similarity matrix (not included in repo)

top_10_similar.npy: Top 10 similar paintings per image (not included in repo)

download_images.py: Script to download dataset

extract_features.py: Extracts features using ResNet50

compute_similarity.py: Computes cosine similarity matrix

extract_top_10.py: Extracts top 10 similar images

display_similar.py: Displays similar paintings

evaluate_unsupervised.py: Evaluates model performance

optimized_similarity.py: Alternative approach computing only top 10

inspect_images.py: Inspects a few images to confirm download has taken place

published_images.rar: Contains published_images.csv from which the download_images.py downloads all the images in the images folder

## Installation  
To set up the project, clone the repository and install dependencies:  

  ```
  git clone https://github.com/muhammadahmedzaheer/HumanAI-Task2.git  
  pip install -r requirements.txt
  ```

## Execution Steps  

### 1. Download the dataset  
Downloads all painting images from the National Gallery of Art dataset using published_images.csv.  
Script: download_images.py  
Output: A folder images/ containing the downloaded paintings.  

```
python download_images.py
```

### 2. Inspect Downloaded Images (Optional)  
Verifies if the images were downloaded correctly by displaying a few samples.  
Script: inspect_images.py  

```
python inspect_images.py  
```

### 3. Extract Features Using ResNet50  
Extracts deep learning feature embeddings using a pre-trained ResNet50 model.  
Script: extract_features.py  
Outputs:  
- features.npy → Feature vectors for all images.  
- image_names.npy → Corresponding filenames for the extracted features.  

```
python extract_features.py  
```

### 4. Compute Cosine Similarity  
Computes cosine similarity between all paintings based on extracted feature vectors.  
Script: compute_similarity.py  
Output:  
- similarity_scores.npy → A large similarity matrix storing all computed similarity values (not included in GitHub due to size constraints).  

```
python compute_similarity.py  
```

### 5. Extract Top 10 Most Similar Paintings  
Extracts the 10 most similar paintings for each image using cosine similarity.  
Script: extract_top_10.py  
Output:  
- top_10_similar.npy → Stores the filenames and similarity scores of the 10 closest matches.  

```
python extract_top_10.py  
```

### 6. Display Similar Paintings  
Selects a random painting and visualizes its top 10 most similar paintings with similarity scores.  
Script: display_similar.py  
Inputs: top_10_similar.npy, image_names.npy, and the images/ folder.  

```
python display_similar.py  
```

### 7. Model Evaluation (Inter- and Intra-Class Similarity)  
Evaluates the effectiveness of the similarity model.  
Script: evaluate_unsupervised.py  
Evaluation Metrics Used:  
- Intra-Class Similarity → The average similarity between an image and its top 10 matches (expected to be high).  
- Inter-Class Similarity → The average similarity between an image and randomly selected paintings (expected to be low).  

```
python evaluate_unsupervised.py  
```

## Alternative Approach: Direct Computation of Top 10 Similar Images  
Computes and stores only the top 10 most similar paintings per image, eliminating the need for a full similarity matrix.  
Script: optimized_similarity.py  
Output: top_10_similar.npy  

```
python optimized_similarity.py  
```

#### Note: While this approach reduces storage, it affects evaluation. Without similarity_scores.npy, the model lacks references for non-similar paintings, leading to an overestimation of inter-class similarity.  

Example:  
- Using full similarity matrix → Intra = 0.88, Inter = 0.58 (Clear distinction between similar and dissimilar images)  
- Using only top 10 matches → Intra = 0.88, Inter = 0.86 (Poor distinction, as random images appear more similar than they actually are)  

## Evaluation Metrics  

Evaluation of the model is based on three key metrics: **Cosine Similarity, Intra-Class Similarity, and Inter-Class Similarity**.  

### **1. Cosine Similarity**  
- **Displayed by:** `display_similar.py`  
- **Purpose:** Measures the **pairwise similarity** between two paintings based on their extracted feature vectors.  
- **How it works:** When `display_similar.py` is run, it randomly selects a painting and shows its **top 10 most similar images**, along with their **cosine similarity scores**. Higher scores indicate greater similarity.  

### **2. Intra-Class Similarity**  
- **Displayed by:** `evaluate_unsupervised.py`  
- **Purpose:** Measures how **visually similar** the **top 10 retrieved images** are to a given painting.  
- **How it works:** The average cosine similarity is calculated between an image and its **top 10 similar matches** (as stored in `top_10_similar.npy`). A **high intra-class similarity** suggests that the model correctly groups paintings that are truly alike.  

### **3. Inter-Class Similarity**  
- **Displayed by:** `evaluate_unsupervised.py`  
- **Purpose:** Measures how **dissimilar** a painting is to randomly selected paintings.  
- **How it works:** The model selects **10 random paintings** and calculates their average similarity to the given painting. A **low inter-class similarity** indicates that the model correctly differentiates between similar and non-similar paintings.  

---
Metric                 | Purpose  
---------------------- | ---------------------------------------------------------  
Cosine Similarity      | Measures pairwise similarity between paintings.  
Intra-Class Similarity | Ensures similar paintings are correctly grouped (high).  
Inter-Class Similarity | Ensures dissimilar paintings are correctly separated (low).  


### **Why Are All Three Metrics Necessary?**  

Each of these three metrics plays a crucial role in evaluating the effectiveness of the model:  

- **Cosine Similarity (display_similar.py)** only shows individual similarity scores between an image and its closest matches. While useful, it does **not** tell us how well the model separates similar and dissimilar paintings overall.  

- **Intra-Class Similarity (evaluate_unsupervised.py)** ensures that paintings retrieved as similar are truly close in feature space. If this score is low, the model **fails to retrieve relevant paintings correctly**.  

- **Inter-Class Similarity (evaluate_unsupervised.py)** ensures that randomly selected, unrelated paintings have **low similarity** to the query painting. If this score is high, it means the model **is not effectively distinguishing between different paintings**.  

The **ideal result** is:  
✅ **High Intra-Class Similarity** → The model correctly identifies paintings that belong together.  
✅ **Low Inter-Class Similarity** → The model successfully differentiates between unrelated paintings.  

If **inter-class similarity is too high**, it means the model is not discriminative enough and is incorrectly labeling unrelated paintings as similar.  

By combining these three metrics, we get a complete picture of **both retrieval quality and discrimination ability** of the model. 🚀  

#### Notes & Limitations  
- similarity_scores.npy (56GB) is not included in the repository due to storage constraints.  
- features.npy is also omitted but can be recomputed using extract_features.py.  
- The repository provides all necessary scripts for full reproducibility of the similarity model.  

## How to Reproduce  
### Clone the repository and install dependencies  
```
git clone https://github.com/YOUR_USERNAME/HumanAI-Task2.git  
cd HumanAI-Task2  
pip install -r requirements.txt  
```

### Download the dataset using download_images.py  
```
python download_images.py  
```

### Run the scripts in the order outlined above  
```
python extract_features.py  
python compute_similarity.py  
python extract_top_10.py  
python display_similar.py  
python evaluate_unsupervised.py  
``` 

## Conclusion  
This project successfully implements an unsupervised painting similarity model, leveraging deep learning and cosine similarity to identify visually related artworks. The approach enables the retrieval of stylistically similar paintings while ensuring clear differentiation through robust evaluation metrics.  
