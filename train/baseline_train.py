"""Baseline LoRA fine-tuning script for the structured output task.

This trains a LoRA adapter on top of Qwen 3.5-0.8B using the generated
training data. The baseline is intentionally simple — there is plenty of
room for improvement (more LoRA targets, more epochs, better prompts, etc.).

Usage:
    python -m train.baseline_train [--output_dir OUTPUT_DIR]
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
TRAIN_GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 256  # samples are ~50-150 tokens; 1024 wastes O(n²) attention compute

# Inference
MAX_NEW_TOKENS = 512

# System prompt used in chat template
SYSTEM_PROMPT = (
    "You convert natural language descriptions into structured data formats. "
    "Output only the formatted data, nothing else."
)


def load_training_data(data_path: str) -> list[dict]:
    """Load training samples from JSONL file."""
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_chat_messages(sample: dict) -> list[dict]:
    """Format a sample as chat messages for the Qwen chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]


def tokenize_sample(messages: list[dict], tokenizer) -> dict:
    """Tokenize a chat message list, masking input tokens in labels."""
    # Full conversation (with assistant response)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)

    # Conversation up to assistant response (for masking)
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)

    input_ids = full_ids["input_ids"][0]
    labels = input_ids.clone()
    # Mask prompt tokens so we only compute loss on the assistant response
    prompt_len = prompt_ids["input_ids"].shape[1]
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": full_ids["attention_mask"][0],
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline LoRA training")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "train.jsonl"),
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "output" / "baseline_lora"),
        help="Directory to save the LoRA adapter",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set, override num_train_epochs and train for this many steps",
    )
    args = parser.parse_args()

    # Detect available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading model: {MODEL_NAME} (device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization only works on CUDA (bitsandbytes has no MPS/CPU support)
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        # On MPS/CPU: load in fp16 (fits in 16GB unified memory for 0.8B model)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )

    # Apply LoRA — baseline uses only q_proj and v_proj
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Compile the model for faster execution (PyTorch 2.x, works on MPS and CUDA)
    if device in ("cuda", "mps"):
        model = torch.compile(model)

    # Load and tokenize data
    print(f"Loading data from: {args.data_path}")
    raw_data = load_training_data(args.data_path)
    print(f"  {len(raw_data)} samples loaded")

    tokenized = []
    for sample in raw_data:
        messages = format_chat_messages(sample)
        tok = tokenize_sample(messages, tokenizer)
        tokenized.append({
            "input_ids": tok["input_ids"].tolist(),
            "attention_mask": tok["attention_mask"].tolist(),
            "labels": tok["labels"].tolist(),
        })

    dataset = Dataset.from_list(tokenized)

    # Training arguments — baseline settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=TRAIN_EPOCHS if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=torch.cuda.is_bf16_supported() if device == "cuda" else False,
        fp16=(not torch.cuda.is_bf16_supported() if device == "cuda" else False),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=(device == "cuda"),  # pin_memory causes issues on MPS
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    print(f"Saving adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
