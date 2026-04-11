# VLM GRPO

this is my first attempt at trying to train a model using grpo

i'll be trying to train a VLM (unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit to be precise) on ChartQA using GRPO with unsloth

i'll be following the unsloth `Qwen3_VL_(8B)-Vision-GRPO` notebook to get an idea about how to do things

---

## results

| model | dataset | accuracy |
|---|---|---|
| Qwen3-VL-8B (baseline, no training) | ChartQA test (500 samples) | 33.6% |
| Qwen3-VL-8B (GRPO, 1 epoch) | ChartQA test (500 samples) | 35.2% |

---

## setup

login to huggingface and wandb once before running:

```bash
huggingface-cli login
wandb login
```

## how to run

```bash
python train.py [args]
```

### args

| arg | what it does | default |
|---|---|---|
| `--output_dir` | where to save the lora weights locally | `grpo_lora` |
| `--push_to_hub` | push weights to huggingface after training | off |
| `--hub_model_id` | hf repo to push to, required if `--push_to_hub` is set | None |
| `--use_wandb` | log training to wandb | off |
| `--wandb_project` | wandb project name | `vlm-grpo` |
| `--wandb_run_name` | wandb run name | None |

### example

```bash
python train.py \
  --push_to_hub \
  --hub_model_id youruser/qwen3-vl-grpo \
  --use_wandb \
  --wandb_project vlm-grpo \
  --wandb_run_name run1
```

## how to eval

```bash
python eval.py [args]
```

### args

| arg | what it does | default |
|---|---|---|
| `--num_samples` | number of test samples to evaluate on | `500` |
| `--batch_size` | batch size for inference | `8` |
| `--lora_path` | path or hf repo of the trained LoRA (if not set, evaluates the base model) | None |

### example

```bash
python eval.py --lora_path youruser/qwen3-vl-grpo
```
