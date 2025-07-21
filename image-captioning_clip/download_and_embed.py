import os
import requests
import time
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed



device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

batch_size = 256

def download_image_and_caption(sample):
    try:
        response = requests.get(sample["image_url"], timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return {"image": image, "caption": sample["caption"]}
    except:
        return None



def process_dataset(dataset_split, save_path, max_samples=25000, batch_download_size=1000):
    clip_embeddings = []
    captions_data = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    for batch_start in range(0,max_samples,batch_download_size):

        batch_indices = list(range(batch_start, batch_start + batch_download_size))
        batch=dataset_split.select(batch_indices)

        images, captions = [], []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(download_image_and_caption, sample) for sample in batch]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading batch {batch_start // batch_download_size + 1}"):
                result = future.result()
                if result:
                    images.append(result["image"])
                    captions.append(result["caption"])


        for i in range(0, len(images), batch_size):
            img_batch = images[i:i+batch_size]
            cap_batch = captions[i:i+batch_size]


            try:
                inputs = processor(images=img_batch, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    embeddings = model.get_image_features(**inputs).cpu()

                
                for emb, cap in zip(embeddings, cap_batch):
                    idx = len(clip_embeddings)
                    clip_embeddings.append(emb)
                    captions_data.append({
                        "caption": cap,
                        "image_id": f"cc_{idx}",
                        "clip_embedding_id": idx
                    })
                
            except Exception as e:
                print(f"[CLIP batch failed] {e}")

    print(f"✅ Total embeddings saved: {len(clip_embeddings)}")

    with open(save_path, "wb") as f:
        pickle.dump({
            "clip_embedding": clip_embeddings,
            "captions": captions_data
        }, f)


ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

start = time.time()
process_dataset(ds["train"], "./data/conceptual_clipcap.pkl", max_samples=12000)
end = time.time()
print(f"Train took {end - start:.2f} s")

process_dataset(ds["validation"], "./data/conceptual_clipcap_val.pkl", max_samples=1000)