from config import DATASET, DATASET_SIZE, REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START
from datasets import load_dataset


def _process(example):
    image = example["image"]
    image = image.resize((512, 512))
    if image.mode != "RGB":
        image = image.convert("RGB")

    text = (
        f"{example['query'].strip()} "
        f"Provide reasoning between {REASONING_START} and {REASONING_END}, "
        f"then your answer between {SOLUTION_START} and {SOLUTION_END}."
    )

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]

    return {
        "prompt": prompt,
        "image": image,
        "answer": example["label"][0],
    }


def prepare_dataset(split=None):
    if split is None:
        split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
    dataset = load_dataset(DATASET, split=split)
    dataset = dataset.map(_process)
    return dataset.select_columns(["prompt", "image", "answer"])
