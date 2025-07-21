import os
import pickle
import requests
from tqdm import tqdm
from PIL import Image
import time
from io import BytesIO
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_and_process(sample):
    try:
        response = requests.get(sample["image_url"], timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return {
            "image": image,
            "caption": sample["caption"]
        }
    except Exception as e:
        return None


# def save_images_and_captions(dataset_split, save_path, max_samples=25000):
#     data = []
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     with ThreadPoolExecutor(max_workers=20) as executor:
#         futures = []
#         for i, sample in enumerate(dataset_split):
#             if i >= max_samples:
#                 break
#             futures.append(executor.submit(download_and_process, sample))

#         for future in tqdm(as_completed(futures), total=len(futures)):
#                 try:
#                     result = future.result(timeout=20)
#                     if result is not None:
#                         data.append(result)
#                 except Exception as e:
#                     print(f"[Thread error] {e}")

#     with open(save_path, "wb") as f:
#         pickle.dump(data, f)


def save_images_and_captions(dataset_split, save_path, max_samples=25000,batch_size=1000):
    data = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for batch_start in range(0, len(dataset_split), batch_size):
        if batch_start>=max_samples:
            break
        

        batch_indices = list(range(batch_start, batch_start + batch_size))

        batch = dataset_split.select(batch_indices)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(download_and_process, sample) for sample in batch]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_start//batch_size + 1}"):
                try:
                    result = future.result(timeout=20)
                    if result is not None:
                        data.append(result)
                except Exception as e:
                    print(f"[Thread error] {e}")


        # Saving after each batch to prevent complete data  bcoz of crashes

        with open(save_path, "wb") as f:
            pickle.dump(data, f)


ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")



start=time.time()
save_images_and_captions(ds["train"], "./data/images_train.pkl", max_samples=12000)
end=time.time()
print(f"took {end-start:.2f} s")
save_images_and_captions(ds["validation"], "./data/images_val.pkl", max_samples=1000)


with open("./data/images_train.pkl", "rb") as f:
    train_data = pickle.load(f)
print(f"Total train samples saved: {len(train_data)}")

with open("./data/images_val.pkl", "rb") as f:
    val_data = pickle.load(f)
print(f"Total validation samples saved: {len(val_data)}")