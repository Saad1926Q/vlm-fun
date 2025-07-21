I recently tried implementing the [**ClipCap**](https://arxiv.org/pdf/2111.09734) model - an image captioning architecture that combines CLIP and GPT-2.Though the model is not working(as of 21/07/25).

I’m honestly not sure what went wrong - especially during inference. The model just keeps generating the same caption no matter the input image. If you’re into ML and have a minute, I’d really appreciate if you could help me out. The code looks okay to me… but something’s clearly off

## How ClipCap Works

![A high level overview of ClipCap Model](clipcap.png)

Here's a basic rundown of how the model operates:

1. **Input Image → CLIP Embeddings:**  
   The image is first encoded using CLIP, which converts it into a fixed-length embedding vector.

2. **Mismatch in Embedding Spaces:**  
   Since CLIP and GPT-2 operate in different embedding spaces, you can't directly feed the CLIP embeddings into GPT-2.

3. **Mapping Network (Prefix Mapping):**  
   A mapping network is used to transform the CLIP embeddings (called the *prefix*) into the GPT-2 embedding space.

4. **Caption Generation:**  
   The mapped embeddings are then fed into GPT-2, which generates a natural language caption based on the image.

## Training Details

- The CLIP model is **not** trained or fine-tuned - its image embeddings are good enough out of the box.
- There are **two variants** of the model based on whether or not GPT-2 is fine-tuned:

### 1. GPT-2 is Fine-Tuned
- **Mapping Network:** A Multi-Layer Perceptron (MLP)
- **Training:** Both the MLP and GPT-2 are trained

### 2. GPT-2 is Frozen
- **Mapping Network:** A Transformer
- **Training:** Only the transformer mapping network is trained


