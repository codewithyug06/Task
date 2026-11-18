"""
Task 1: LoRA Fine-Tuning Script
Base Model: Qwen2.5-0.5B-Instruct (loaded from local snapshot)
Target Task: Abstract -> Structured JSON Extraction
Hyperparameters:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Learning rate: 2e-4
  - LR Scheduler: cosine with 5% warmup
  - Batch size: 4 per device, gradient accumulation 4 (effective batch size 16)
  - Epochs: 4
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TRAIN_FILE = "d:/Intern/data/train.json"
VAL_FILE = "d:/Intern/data/val.json"
OUTPUT_DIR = "d:/Intern/task1_finetune/lora_adapter"

SYSTEM_PROMPT = (
    "You are an expert scientific assistant. Extract key benchmark details from the "
    "machine learning paper abstract into a valid JSON object with the exact keys:\n"
    '- "task": The primary ML task being addressed\n'
    '- "dataset": The evaluation benchmark or dataset name\n'
    '- "metric": The primary quantitative metric reported\n'
    '- "result": The numerical score or value achieved\n'
    '- "baseline": The baseline model or previous SOTA compared against\n\n'
    "Respond ONLY with the raw JSON object. Do not include introductory text or explanations."
)

def format_example(sample, tokenizer):
    abstract = sample["abstract"]
    gold_json_str = json.dumps(sample["ground_truth"], indent=2)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Abstract:\n{abstract}"},
        {"role": "assistant", "content": gold_json_str}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt

def prepare_tokenized_dataset(tokenizer, data_path, max_length=512):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    texts = [format_example(item, tokenizer) for item in raw_data]
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in tokens["input_ids"]
        ]
        tokens["labels"] = labels
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized_dataset

def main():
    print(f"Loading tokenizer from: {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Preparing datasets...", flush=True)
    train_dataset = prepare_tokenized_dataset(tokenizer, TRAIN_FILE)
    val_dataset = prepare_tokenized_dataset(tokenizer, VAL_FILE)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    print(f"Loading base model in FP16 on GPU from {MODEL_PATH}...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Enable gradient checkpointing to conserve VRAM
    model.gradient_checkpointing_enable()

    # Configure LoRA
    print("Configuring LoRA (r=16, alpha=32, all linear modules)...", flush=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=6,
        num_train_epochs=4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
        logging_first_step=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("\nStarting LoRA Fine-Tuning...", flush=True)
    train_result = trainer.train()
    
    print("\nSaving fine-tuned adapter...", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics = train_result.metrics
    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print("Fine-tuning completed successfully!", flush=True)
    print(f"Adapter saved to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()