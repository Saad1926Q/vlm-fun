import os
import pickle
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

batch_size = 256


def generate_clip_embeddings(pickle_path, save_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    clip_embeddings = []
    captions_data = []

    batch_images = []
    batch_captions = []

    for sample in tqdm(data):
        img = sample["image"]
        cap = sample["caption"]

        batch_images.append(img)
        batch_captions.append(cap)

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
    
    if batch_images:
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
            print(f"Final batch failed: {e}")

    print("Total embeddings generated:", len(clip_embeddings))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "clip_embedding": clip_embeddings,
            "captions": captions_data
        }, f)


generate_clip_embeddings("./data/images_train.pkl", "./data/conceptual_clipcap.pkl")
generate_clip_embeddings("./data/images_val.pkl", "./data/conceptual_clipcap_val.pkl")