import os
import pickle
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import time


def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except:
        return None
    


def save_images_and_captions(dataset_split, save_path, max_samples=25000):
    data = []

    for sample in tqdm(dataset_split):
        if len(data) >= max_samples:
            break

        img = download_image(sample["image_url"])
        if img is None:
            continue

        data.append({
            "image": img,
            "caption": sample["caption"]
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)



ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

start=time.time()
save_images_and_captions(ds["train"], "./data/images_train.pkl",20)
end=time.time()
print(f"took {end-start:.2f} s")
save_images_and_captions(ds["validation"], "./data/images_val.pkl",10)