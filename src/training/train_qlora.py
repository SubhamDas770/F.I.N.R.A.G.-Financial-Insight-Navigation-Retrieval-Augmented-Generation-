import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train_llama3_qlora(
    base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_path: str = "./data/processed/llama3_training_data.jsonl",
    output_dir: str = "./data/fine_tuned_llama3",
    num_train_epochs: int = 3,
):
    print(f"🚀 Starting QLoRA Fine-Tuning for {base_model_id}...")

    # 1. Load Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training data not found at {dataset_path}. Run dataset.py first.")
    
    # Hugging Face datasets library loads the JSONL effortlessly
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Split 90% Train / 10% Eval
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    print(f"📊 Dataset Loaded: {len(train_dataset)} Train, {len(eval_dataset)} Eval samples.")

    # 2. Configure 4-bit Quantization (BitsAndBytes)
    # This compresses the model footprint so an 8B model fits on a single 16GB / 24GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for stability if Ampere GPU (A10G, A100), else float16
    )

    # 3. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # Llama 3 does not have a default pad token, so we use the eos token
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # Important for CausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Freeze the base model and prepare for QLoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA Adapters
    # We target all linear layers to maximize learning capability without catastrophic forgetting
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments Setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,      # Exactly 3 epochs as requested
        per_device_train_batch_size=4,          # Adjust down to 2 if you hit Out of Memory
        gradient_accumulation_steps=4,          # Simulates a larger batch size of 16
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        evaluation_strategy="epoch",            # Evaluate at the end of each epoch
        learning_rate=2e-4,                     # Standard QLoRA learning rate
        weight_decay=0.001,
        fp16=False,
        bf16=True,                              # Use BF16 for newer GPUs
        max_grad_norm=0.3,
        warmup_ratio=0.03,                      # Linear warmup to prevent early training collapse
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",                       # Change to "wandb" if you want visual tracking
    )

    # 6. Initialize SFTTrainer
    # The SFTTrainer automatically handles the "messages" format we created in dataset.py
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=1024,                    # Matches our parent chunk size logic
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    print("🔥 Commencing Training Loop...")
    trainer.train()

    # 8. Save the resulting LoRA adapters
    final_output_path = os.path.join(output_dir, "final_adapters")
    trainer.model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print(f"✅ Training Complete. Adapters saved to: {final_output_path}")


if __name__ == "__main__":
    # Ensure you are logged into Hugging Face CLI (huggingface-cli login) to access Llama 3
    train_llama3_qlora()