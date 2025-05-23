import os
import torch
import open_clip
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()

def load_dataset_embeddings(dataset_folder):
    """Embed all images in dataset and return embeddings + paths."""
    dataset_embeddings = []
    image_paths = []

    for fname in tqdm(os.listdir(dataset_folder), desc="Embedding dataset images"):
        path = os.path.join(dataset_folder, fname)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(img_tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                dataset_embeddings.append(embedding.cpu().numpy())
                image_paths.append(path)
        except (UnidentifiedImageError, Exception) as e:
            print(f"❌ Skipping invalid image: {path}")

    return np.vstack(dataset_embeddings), image_paths

def compute_similarity(query_img_path, dataset_embeddings, image_paths):
    """Process query image and compute similarity with dataset."""
    try:
        query_img = Image.open(query_img_path).convert("RGB")
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = model.encode_image(query_tensor)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()

        similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()
        top_indices = similarities.argsort()[::-1][:5]
        results = [(image_paths[idx], float(similarities[idx])) for idx in top_indices]
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to process query image: {e}")
