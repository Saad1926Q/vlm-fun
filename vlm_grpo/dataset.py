from config import DATASET, DATASET_SIZE, REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START
from datasets import load_dataset


def _process(example):
    image = example["images"][0]
    image = image.resize((512, 512))
    if image.mode != "RGB":
        image = image.convert("RGB")

    text = (
        f"{example['problem'].replace('<image>', '').strip()} "
        f"Provide reasoning between {REASONING_START} and {REASONING_END}, "
        f"then your integer answer between {SOLUTION_START} and {SOLUTION_END}."
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
        "answer": example["answer"],
    }


def prepare_dataset():
    split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
    dataset = load_dataset(DATASET, split=split)
    return dataset.map(_process)
