import os
import streamlit as st
from PIL import Image
from vision_utils import load_dataset_embeddings, compute_similarity

# Config
st.set_page_config(page_title="CLIP Image Similarity", layout="wide")
UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "static/image_dataset"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load dataset embeddings
st.sidebar.success("Loading dataset...")
dataset_embeddings, image_paths = load_dataset_embeddings(DATASET_FOLDER)

# App title
st.title("🔍 Find Visually Similar Images")
st.markdown("#### A few example categories are available: Bags, Stationery, Dogs, Groceries, Cosmetics, and Electronics. Please test only with these.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

# Upload and show image
if uploaded_file:
    with st.spinner("Processing image and finding similar images..."):
        img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(img_path, caption="Uploaded Image", width=250)


        try:
            results = compute_similarity(img_path, dataset_embeddings, image_paths)
            filtered_results = [(path, score) for path, score in results if score > 0.52]

            if filtered_results:
                st.markdown("### 🎯 Top Similar Images")
                cols = st.columns(3)

                for i, (path, score) in enumerate(filtered_results[:6]):
                    with cols[i % 3]:
                        st.image(path, use_container_width=True, caption=f"Similarity: {score*100:.2f}%")
            else:
                st.info("No similar images found with confidence above 50%.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# # Gallery section
# st.markdown("---")
# st.markdown("## 🖼️ Image Gallery from Dataset")
# gallery_images = [
#     os.path.join(DATASET_FOLDER, file)
#     for file in os.listdir(DATASET_FOLDER)
#     if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"))
# ]

# cols = st.columns(4)
# for i, img in enumerate(gallery_images[:24]):  # Limit to first 24 images
#     with cols[i % 4]:
#         st.image(img, use_container_width=True, caption=os.path.basename(img))
