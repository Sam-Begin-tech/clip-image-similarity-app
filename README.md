# 🔍 CLIP-Based Image Similarity Search App

A Streamlit app that allows users to upload an image and find visually similar images from a dataset using OpenCLIP (`ViT-B-32`, pretrained on LAION2B).

![Demo](https://github.com/your-username/clip-image-similarity-app/assets/demo.gif)

## 🚀 Features

- Embed dataset images using OpenCLIP
- Upload an image and compute visual similarity
- Show top 5 most similar images with confidence scores
- Threshold filtering (only show matches > 52%)
- Simple and fast interface with Streamlit

## 📦 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sam-Begin-tech/clip-image-similarity-app.git
cd clip-image-similarity-app
```
2. Install dependencies
Use pip and ensure you have Python 3.8+.

bash
Copy
Edit
pip install -r requirements.txt
Recommended: Create a virtual environment before installing.

3. Add your image dataset
Place your reference images inside the folder:

swift
Copy
Edit
static/image_dataset/
Supported formats: .jpg, .jpeg, .png, .webp

4. Run the app
bash
Copy
Edit
streamlit run app.py
The app will open in your default browser.

🧠 Model Used
OpenCLIP: ViT-B-32 (pretrained on laion2b_s34b_b79k)

Similarity Metric: Cosine similarity

📁 Folder Structure
static/uploads/: Temporary upload directory

static/image_dataset/: Your dataset of images to compare against

utils/vision_utils.py: Handles embedding and similarity logic

app.py: Streamlit frontend and logic

🔒 Note
Make sure the dataset folder contains valid images. Unsupported files will be skipped.



