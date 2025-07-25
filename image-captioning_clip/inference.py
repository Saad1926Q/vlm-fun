import torch
import os
import cv2
import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torch.optim import AdamW
from model import ClipCapModel
from torchvision import transforms
from dataset_new import CaptionsDataset
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel,GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, prefix_length=10, clip_length=10, prefix_size=512):
    model = ClipCapModel(prefix_length=prefix_length, clip_length=clip_length, prefix_size=prefix_size)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return model

def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image 


def get_clip_embedding(pil_image):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=[pil_image], return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)


    return embedding


def generate_caption(model, clip_embedding, tokenizer:GPT2Tokenizer, max_length=20,temperature=1.0):

    # We will be using greedy strategy

    model.eval()

    with torch.no_grad():
        # Mapping clip embeddings to GPT2 embedding space
        # Go from (batch_size,self.prefix_length * self.gpt_embedding_size) to (batch_size,self.prefix_length,self.gpt_embedding_size)
        prefix=model.mapping(clip_embedding).view(-1,model.prefix_length,model.gpt_embedding_size) 

        generated_ids = None

        for i in range(max_length):

            if generated_ids is not None:
                token_embeddings=model.gpt.transformer.wte(generated_ids)
                full_embeddings = torch.cat((prefix, token_embeddings), dim=1)
            else:
                full_embeddings=prefix

            output=model.gpt(inputs_embeds=full_embeddings)

            logits=output.logits   #(1 , sequence_length, vocab_size)

            logits_next = logits[:, -1, :]  # (1, vocab_size)

            if temperature > 0:
                logits_next /= temperature

            
            next_token_id = torch.argmax(logits_next, dim=-1).unsqueeze(0)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            if generated_ids is None:
                generated_ids = next_token_id
            else:
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        output_text= tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
        return output_text.strip()
    

model=load_model("./checkpoints/coco_prefix_best.pt")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

parser = argparse.ArgumentParser()
parser.add_argument('--url',type=str,default="https://t4.ftcdn.net/jpg/03/24/42/21/360_F_324422176_Lgn7NTeFyNaUKIDu0Ppls1u8zb8wsKS4.jpg" )

args = parser.parse_args()

image_url = args.url

image = preprocess_image_from_url(image_url)

clip_embedding = get_clip_embedding(image)

caption = generate_caption(model, clip_embedding, tokenizer)
print("Generated Caption:", caption)

