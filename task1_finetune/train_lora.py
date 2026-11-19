"""
Task 1: LoRA Fine-Tuning Script
Base Model: Qwen2.5-0.5B-Instruct
Target Task: Abstract -> Structured JSON Extraction
Hyperparameters:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Learning rate: 2e-4
  - LR Scheduler: cosine with 5% warmup
  - Batch size: 4 per device, gradient accumulation 2 (effective batch size 8)
  - Epochs: 4
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

# Add repo root to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import get_model_path, DATA_DIR, LORA_ADAPTER_DIR
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = get_model_path()
TRAIN_FILE = str(DATA_DIR / "train.json")
VAL_FILE = str(DATA_DIR / "val.json")
OUTPUT_DIR = str(LORA_ADAPTER_DIR)

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
    target = sample.get("target", sample.get("gold", sample.get("ground_truth")))
    gold_json_str = json.dumps(target, indent=2)
    
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
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        return model_inputs
        
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized

def main():
    print(f"Loading tokenizer and model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    ).to(device)
    
    print("\nPreparing datasets...")
    train_dataset = prepare_tokenized_dataset(tokenizer, TRAIN_FILE)
    val_dataset = prepare_tokenized_dataset(tokenizer, VAL_FILE)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_total_limit=2,
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print("\nStarting LoRA Fine-Tuning on RTX 4050 GPU...")
    start_time = time.time()
    train_result = trainer.train()
    total_time = time.time() - start_time
    
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)!")
    
    print(f"\nSaving adapter model and checkpoints to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    eval_metrics = trainer.evaluate()
    print("Final Validation Loss:", eval_metrics.get("eval_loss"))
    
    metrics_summary = {
        "train_runtime_seconds": total_time,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "epochs": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Saved training metrics summary.")

if __name__ == "__main__":
    main()
