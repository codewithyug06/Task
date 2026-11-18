"""
Evaluate Base Model (Qwen2.5-0.5B-Instruct) Zero-Shot on the 50 Held-Out Test Set (Fixed Domain-Holdout).
Batched generation on pure CUDA for fast and reliable execution.
Supports both local cache and Hugging Face Hub for Colab/Kaggle portability.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import evaluate_predictions

LOCAL_MODEL = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else "Qwen/Qwen2.5-0.5B-Instruct"

TEST_DATA_PATH = "d:/Intern/data_fixed/test.json" if os.path.exists("d:/Intern/data_fixed/test.json") else "data_fixed/test.json"
OUTPUT_PATH = "d:/Intern/task1_finetune/base_model_eval_results.json"
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

    print(f"Loading base model and tokenizer from: {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    predictions = []
    print("\nRunning batched inference on base model...", flush=True)
    
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

    print("\nEvaluating base model predictions on held-out test set...", flush=True)
    metrics = evaluate_predictions(test_samples, predictions)
    
    results = {
        "model": "Qwen2.5-0.5B-Instruct",
        "type": "base_zero_shot",
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
        
    print(f"\n================ BASE MODEL RESULTS ================")
    print(f"Overall Macro F1: {metrics['overall_macro_f1']:.4f}")
    print(f"JSON Validity:    {metrics['valid_json_count']}/{len(test_samples)} ({metrics['valid_json_count']/len(test_samples)*100:.1f}%)")
    for field, f1 in metrics["field_f1"].items():
        print(f"  - {field:10s} F1: {f1:.4f} (P: {metrics['field_precision'][field]:.4f}, R: {metrics['field_recall'][field]:.4f})")
    print(f"=====================================================")
    print(f"Saved evaluation results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
