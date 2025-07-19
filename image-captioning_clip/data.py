import os
import torch
import pickle
import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

SAVE_DIR = "./data"
TRAIN_SAMPLES = 12
VAL_SAMPLES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def download_image(url):
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        return None

def get_clip_embedding(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return model.get_image_features(**inputs).cpu().squeeze()
    except:
        return None

def process_split(ds_split, max_samples, save_path):
    clip_embeddings, captions_data = [], []

    for i, sample in tqdm(enumerate(ds_split), total=max_samples):
        if i >= max_samples:
            break

        image = download_image(sample["image_url"])
        if image is None:
            continue

        embedding = get_clip_embedding(image)
        if embedding is None:
            continue

        embedding_id = len(clip_embeddings)
        clip_embeddings.append(embedding)
        captions_data.append({
            "caption": sample["caption"],
            "image_id": f"cc_{embedding_id}",
            "clip_embedding_id": embedding_id
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"clip_embedding": clip_embeddings, "captions": captions_data}, f)
    print(f"Saved {len(clip_embeddings)} embeddings to {save_path}")

ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

# print(len(ds["train"]))
# print(len(ds["validation"]))
process_split(ds["train"], TRAIN_SAMPLES, os.path.join(SAVE_DIR, "conceptual_clipcap.pkl"))
process_split(ds["validation"], VAL_SAMPLES, os.path.join(SAVE_DIR, "conceptual_clipcap_val.pkl"))
