import argparse

from dataset import prepare_dataset
from model import load_model
from rewards import correctness_reward_func, formatting_reward_func
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="grpo_lora")
parser.add_argument(
    "--save_method",
    type=str,
    default="lora",
    choices=["lora", "merged_16bit", "merged_4bit", "gguf_q4_k_m", "gguf_f16", "gguf_q8"],
)
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
parser.add_argument("--hub_token", type=str, default=None)
args = parser.parse_args()

if args.push_to_hub:
    if not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is set")
    if not args.hub_token:
        parser.error("--hub_token is required when --push_to_hub is set")

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=0.5,  # Set to 1 for a full training run
    # max_steps = 60,
    save_steps=60,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
    bf16=False,
    # Below enables GSPO:
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",
)

model, tokenizer = load_model()

train_dataset = prepare_dataset()

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    # Pass the processor to handle multimodal inputs
    processing_class=tokenizer,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
    ],
    train_dataset=train_dataset,
)

trainer.train()

# Save
if args.save_method == "lora":
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
elif args.save_method == "merged_16bit":
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
elif args.save_method == "merged_4bit":
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_4bit")
elif args.save_method == "gguf_q4_k_m":
    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method="q4_k_m")
elif args.save_method == "gguf_f16":
    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method="f16")
elif args.save_method == "gguf_q8":
    model.save_pretrained_gguf(args.output_dir, tokenizer)

# Push to hub
if args.push_to_hub:
    if args.save_method == "lora":
        model.push_to_hub(args.hub_model_id, token=args.hub_token)
        tokenizer.push_to_hub(args.hub_model_id, token=args.hub_token)
    elif args.save_method in ("merged_16bit", "merged_4bit"):
        model.push_to_hub_merged(args.hub_model_id, tokenizer, save_method=args.save_method, token=args.hub_token)
    elif args.save_method in ("gguf_q4_k_m", "gguf_f16", "gguf_q8"):
        quant_map = {"gguf_q4_k_m": "q4_k_m", "gguf_f16": "f16", "gguf_q8": None}
        quant = quant_map[args.save_method]
        kwargs = {"quantization_method": quant} if quant else {}
        model.push_to_hub_gguf(args.hub_model_id, tokenizer, token=args.hub_token, **kwargs)
