import os
import torch
import pickle
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

batch_size = 256

def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except:
        return None

def extract_clip_embeddings(dataset_split, save_path):
    clip_embeddings = []
    captions_data = []

    batch_images = []
    batch_captions = []

    for sample in tqdm(dataset_split):

        if len(clip_embeddings) >= 25000:
            break

        img = download_image(sample["image_url"])
        if img is None:
            continue

        batch_images.append(img)
        batch_captions.append(sample["caption"])

        if len(batch_images) == batch_size:
            try:
                inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    embeddings = model.get_image_features(**inputs).cpu()

                for emb, cap in zip(embeddings, batch_captions):
                    idx = len(clip_embeddings)
                    clip_embeddings.append(emb)
                    captions_data.append({
                        "caption": cap,
                        "image_id": f"cc_{idx}",
                        "clip_embedding_id": idx
                    })

            except Exception as e:
                print(f"Batch failed: {e}")

            batch_images, batch_captions = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"clip_embedding": clip_embeddings, "captions": captions_data}, f)



ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

extract_clip_embeddings(ds["train"], "./data/conceptual_clipcap.pkl")

extract_clip_embeddings(ds["validation"], "./data/conceptual_clipcap_val.pkl")