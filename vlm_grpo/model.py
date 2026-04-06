from config import LORA_RANK, MAX_SEQ_LEN, MODEL_NAME
from unsloth import FastVisionModel


def load_model():
    """
    Load model and tokenizer from unsloth.
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
        gpu_memory_utilization=0.8,  # Reduce if out of memory
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
    )

    return model, tokenizer
