"""
Evaluate Fine-Tuned LoRA Model on the 50 Held-Out Test Set (Fixed Domain-Holdout).
Batched generation on CUDA with PEFT adapter merging for fast and reliable execution.
Supports both local snapshot and Hugging Face Hub for Colab/Kaggle portability.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add repo root to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import get_model_path, DATA_DIR, TASK1_DIR, LORA_ADAPTER_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from metrics import evaluate_predictions

MODEL_PATH = get_model_path()
ADAPTER_PATH = str(LORA_ADAPTER_DIR)
TEST_DATA_PATH = str(DATA_DIR / "test.json")
BASE_RESULTS_PATH = str(TASK1_DIR / "base_model_eval_results.json")
OUTPUT_PATH = str(TASK1_DIR / "lora_model_eval_results.json")
BATCH_SIZE = 5

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

def format_prompt(tokenizer, abstract):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Abstract:\n{abstract}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    print(f"Loading test data from {TEST_DATA_PATH}...", flush=True)
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"Loaded {len(test_samples)} test samples.", flush=True)

    print(f"Loading base model from {MODEL_PATH} and LoRA adapter from {ADAPTER_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    predictions = []
    print("\nRunning batched inference on LoRA fine-tuned model...", flush=True)
    
    for i in range(0, len(test_samples), BATCH_SIZE):
        batch = test_samples[i:i + BATCH_SIZE]
        prompts = [format_prompt(tokenizer, s["abstract"]) for s in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            gen_tokens = out[input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            predictions.append(gen_text)
            
        print(f"Processed samples {i+1} to {min(i+BATCH_SIZE, len(test_samples))} / {len(test_samples)}", flush=True)

    print("\nEvaluating LoRA model predictions on held-out test set...", flush=True)
    metrics = evaluate_predictions(test_samples, predictions)
    
    base_macro_f1 = None
    if os.path.exists(BASE_RESULTS_PATH):
        with open(BASE_RESULTS_PATH, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            base_macro_f1 = base_data.get("summary", {}).get("overall_macro_f1")
            
    results = {
        "model": "Qwen2.5-0.5B-Instruct + LoRA",
        "type": "lora_finetuned",
        "dataset_split": "data_fixed/test.json (4 unseen domains)",
        "summary": {
            "total_samples": len(test_samples),
            "valid_json_count": metrics["valid_json_count"],
            "json_valid_rate": metrics["valid_json_count"] / len(test_samples),
            "overall_macro_f1": metrics["overall_macro_f1"],
            "field_f1": metrics["field_f1"],
            "field_precision": metrics["field_precision"],
            "field_recall": metrics["field_recall"]
        },
        "base_comparison": {
            "base_macro_f1": base_macro_f1,
            "lora_macro_f1": metrics["overall_macro_f1"],
            "delta": metrics["overall_macro_f1"] - base_macro_f1 if base_macro_f1 is not None else None
        },
        "details": []
    }
    
    for idx, (sample, pred, res) in enumerate(zip(test_samples, predictions, metrics["results"])):
        results["details"].append({
            "domain": sample.get("domain", "Unknown"),
            "abstract": sample["abstract"],
            "gold": sample.get("target", sample.get("gold")),
            "raw_prediction": pred,
            "is_valid_json": res["is_valid_json"],
            "parsed": res["parsed_json"],
            "field_metrics": res["field_metrics"],
            "sample_mean_f1": res["sample_mean_f1"]
        })
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n================ LORA MODEL RESULTS ================")
    print(f"Overall Macro F1: {metrics['overall_macro_f1']:.4f}")
    if base_macro_f1 is not None:
        delta = metrics['overall_macro_f1'] - base_macro_f1
        print(f"Base Model F1:   {base_macro_f1:.4f} -> LoRA F1: {metrics['overall_macro_f1']:.4f} (Delta: {delta:+.4f})")
    print(f"JSON Validity:    {metrics['valid_json_count']}/{len(test_samples)} ({metrics['valid_json_count']/len(test_samples)*100:.1f}%)")
    for field, f1 in metrics["field_f1"].items():
        print(f"  - {field:10s} F1: {f1:.4f} (P: {metrics['field_precision'][field]:.4f}, R: {metrics['field_recall'][field]:.4f})")
    print(f"=====================================================")
    print(f"Saved evaluation results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
