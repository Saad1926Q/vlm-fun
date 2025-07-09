from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):

    # \n is used as the SEP token
    # First we have sequence of image_tokens, then BOS token, then user prompt,then SEP token

    return f"{image_token*image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(
        image:Image.Image,
        size:Tuple[int,int],
        resample:Image.Resampling=None,
        reducing_gap:Optional[int]=None
)->Image.Image:
    height, width = size
    resized=image.resize((width,height),resample=resample,reducing_gap=reducing_gap)

    return resized

def rescale(image:np.ndarray,scale:float,dtype:np.dtype=np.float32)->np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image:np.ndarray,
        mean:Union[float, Iterable[float]],
        std:Union[float, Iterable[float]]
)->np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
        images:List[Image.Image],
        size=None,
        resample: Image.Resampling = None, # how new pixel values are calculated when resizing or transforming images.
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
)->List[np.ndarray]:
    height,width=size[0],size[1]

    images=[resize(image=image,size=(height,width),resample=resample) for image in images]

    images=[np.array(image) for image in images]

    images=[rescale(image=image,scale=rescale_factor) for image in images]

    images=[normalize(image=image,mean=image_mean,std=image_std) for image in images]

    images = [image.transpose(2, 0, 1) for image in images] # (H,W,C) -> (C,H,W)

    return images




# So what the PaliGemmaProcessor does is:-
# Given a text(user prompt) and an image(will load the image,preprocess it)
# and will create these text tokens with the place holders for the image tokens
class PaliGemmaProcessor:

    IMAGE_TOKEN ="<image>" # Will be used as place holder for image tokens

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length=num_image_tokens
        self.image_size=image_size


        # So Paligemma uses the tokenizer of the gemma model
        # But the tokenizer of the gemma model was not created with the special tokens for the image

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS=[
            f"<loc{i:04d}>" for i in range(1024) # These tokens are used for object detection (bounding boxes)
        ]

        EXTRA_TOKENS+=[
            f"<seg{i:03d}>" for i in range(128) # These tokens are used for segmentation
        ]

        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # We will add the BOS and EOS tokens ourselves

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
            self,
            text:List[str],
            images:List[Image.Image],
            padding:str="longest",
            truncation:bool=True
    )->dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."


        pixel_values=process_images(
            images=images,
            size=(self.image_size,self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        pixel_values=np.stack(pixel_values,axis=0)  # Convert a list of np arrays into a single np array and add the batch dimension

        pixel_values = torch.tensor(pixel_values)

        input_strings=[
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            ) for prompt in text
        ]

        # Return input_ids and attention mask
        # Input id means a number which represents the posisiton of each token in the vocabulary
        # Later using embedding layer this will be converted into embedding
        inputs=self.tokenizer( 
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data={"pixel_values": pixel_values,**inputs}

        return return_data