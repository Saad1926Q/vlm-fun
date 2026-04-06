from config import DATASET, DATASET_SIZE, REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START
from datasets import load_dataset


def _resize_images(example):
    """
    Resize to (512, 512).
    """
    image = example["images"][0]
    image = image.resize((512, 512))
    example["images"] = image
    return example


def _convert_to_rgb(example):
    """
    Convert to RGB.
    """
    image = example["images"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["images"] = image
    return example


def _make_conversation(example):
    text = (
        f"{example['problem'].replace('<image>', '').strip()} "
        f"Provide reasoning between {REASONING_START} and {REASONING_END}, "
        f"then your integer answer between {SOLUTION_START} and {SOLUTION_END}."
    )

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text},  # The text part of the prompt
            ],
        },
    ]

    return {
        "prompt": prompt,
        "image": example["images"],
        "answer": example["answer"],
    }


def prepare_dataset():
    split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
    dataset = load_dataset(DATASET, split=split)
    dataset = dataset.map(_resize_images)
    dataset = dataset.map(_convert_to_rgb)
    train_dataset = dataset.map(_make_conversation)
    return train_dataset
