import argparse

from config import MODEL_NAME, MAX_SEQ_LEN
from dataset import prepare_dataset
from tqdm import tqdm
from unsloth import FastVisionModel

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lora_path", type=str, default=None)
args = parser.parse_args()

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=args.lora_path if args.lora_path else MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

dataset = prepare_dataset(split=f"test[:{args.num_samples}]")

correct = 0
total = 0

for batch in tqdm(dataset.iter(batch_size=args.batch_size), total=len(dataset) // args.batch_size):
    images = batch["image"]
    prompts = batch["prompt"]
    answers = batch["answer"]

    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

    inputs = tokenizer(
        images,
        texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=True,
        temperature=1.0,
        min_p=0.1,
    )

    responses = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    for response, answer in zip(responses, answers):
        if answer.strip().lower() in response.strip().lower():
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
