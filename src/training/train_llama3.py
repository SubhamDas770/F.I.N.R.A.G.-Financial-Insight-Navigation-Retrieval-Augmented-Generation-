"""
CPU-compatible fine-tuning script using TinyLlama-1.1B + LoRA (PEFT).
Replaces unsloth (GPU-only) with standard HuggingFace transformers + peft.

Requirements:
    pip install transformers peft datasets accelerate trl
"""

import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# Force UTF-8 output so emoji print statements work on Windows (cp1252 terminals)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # ~2.2GB download, runs on CPU
DATASET_PATH    = PROJECT_ROOT / "data" / "processed" / "llama3_training_data.jsonl"
OUTPUT_DIR      = str(PROJECT_ROOT / "data" / "fine_tuned_tinyllama")
FINAL_ADAPTER   = os.path.join(OUTPUT_DIR, "final_adapters")
MAX_SEQ_LENGTH  = 512    # Keep low for CPU memory
DEVICE          = "cpu"

# ---------------------------------------------------------------------------
# 2. Load & Format Dataset
# ---------------------------------------------------------------------------
def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_record(record: dict, tokenizer) -> str:
    """
    Converts a messages-format record into a single training string using
    the model's chat template.
    """
    messages = record.get("messages", [])
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize(batch, tokenizer):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ---------------------------------------------------------------------------
# 3. Main Training Function
# ---------------------------------------------------------------------------
def train():
    # -- Load tokenizer & model -----------------------------------------------
    print(f"📦 Loading model: {MODEL_NAME} (this may take a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # TinyLlama has no separate pad token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,   # float32 required for CPU training
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    print(f"✅ Model loaded on {DEVICE.upper()}.")

    # -- Apply LoRA adapters ---------------------------------------------------
    print("🎯 Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                         # Low rank keeps memory usage manageable on CPU
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],   # Minimal modules for CPU speed
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -- Load & format dataset -------------------------------------------------
    print(f"📂 Loading dataset from {DATASET_PATH}...")
    raw_records = load_jsonl(DATASET_PATH)
    texts = [format_record(r, tokenizer) for r in raw_records]
    hf_dataset = Dataset.from_dict({"text": texts})

    tokenized_dataset = hf_dataset.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    print(f"✅ Dataset ready: {len(tokenized_dataset)} samples.")

    # -- Training Arguments ----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,      # CPU: keep batch size at 1
        gradient_accumulation_steps=4,      # Effective batch size = 4
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=False,                         # No mixed precision on CPU
        bf16=False,
        report_to="none",                   # Set to "wandb" for tracking
        use_cpu=True,                       # Explicitly use CPU
        dataloader_num_workers=0,           # Windows: must be 0
    )

    # -- Trainer ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    # -- Train -----------------------------------------------------------------
    print("🔥 Starting LoRA fine-tuning on CPU (this will be slow — grab a coffee ☕)...")
    trainer_stats = trainer.train()
    print(f"✅ Training done in {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime']/60:.1f} min).")

    # -- Save adapters ---------------------------------------------------------
    print(f"💾 Saving LoRA adapters to {FINAL_ADAPTER}...")
    model.save_pretrained(FINAL_ADAPTER)
    tokenizer.save_pretrained(FINAL_ADAPTER)
    print("🎉 Done! Adapters saved successfully.")


if __name__ == "__main__":
    train()
