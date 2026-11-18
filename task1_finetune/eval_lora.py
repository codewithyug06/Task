"""
Evaluate Fine-Tuned LoRA Model on the 50 Held-Out Test Set.
Computes field-level Precision, Recall, F1, and JSON validity.
Compares directly against the base model baseline.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from metrics import evaluate_predictions

BASE_MODEL_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
LORA_ADAPTER_PATH = "d:/Intern/task1_finetune/lora_adapter"
TEST_DATA_PATH = "d:/Intern/data/test.json"
BASE_RESULTS_PATH = "d:/Intern/task1_finetune/base_model_eval_results.json"
OUTPUT_PATH = "d:/Intern/task1_finetune/lora_model_eval_results.json"
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

    print(f"Loading base model and tokenizer from local path:\n{BASE_MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print(f"Loading fine-tuned LoRA adapter from {LORA_ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()

    predictions = []
    print("\nRunning batched inference on fine-tuned LoRA model...", flush=True)
    
    for i in range(0, len(test_samples), BATCH_SIZE):
        batch = test_samples[i:i + BATCH_SIZE]
        prompts = [format_prompt(tokenizer, s["abstract"]) for s in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode only the generated response
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            gen_tokens = out[input_len:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            predictions.append(response)
            
        print(f"  Processed {min(i + BATCH_SIZE, len(test_samples))}/{len(test_samples)} samples...", flush=True)

    print("\nEvaluating fine-tuned model predictions...", flush=True)
    summary, details = evaluate_predictions(test_samples, predictions)
    
    # Load base model results for comparison if available
    base_summary = None
    if os.path.exists(BASE_RESULTS_PATH):
        with open(BASE_RESULTS_PATH, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            base_summary = base_data.get("summary")

    print("\n" + "="*60, flush=True)
    print("        BASE MODEL vs. FINE-TUNED MODEL COMPARISON", flush=True)
    print("="*60, flush=True)
    print(f"{'Metric':<20} | {'Base Model':<15} | {'Fine-Tuned (LoRA)':<18} | {'Delta':<10}", flush=True)
    print("-" * 65, flush=True)
    
    base_macro = base_summary["overall_macro_f1"] if base_summary else 0.0
    ft_macro = summary["overall_macro_f1"]
    delta_macro = ft_macro - base_macro
    print(f"{'Overall Macro F1':<20} | {base_macro:12.4f}    | {ft_macro:15.4f}    | {delta_macro:+8.4f}", flush=True)
    
    base_valid = base_summary["json_valid_rate"]*100 if base_summary else 0.0
    ft_valid = summary["json_valid_rate"]*100
    print(f"{'JSON Valid Rate':<20} | {base_valid:11.2f}%    | {ft_valid:14.2f}%    | {ft_valid - base_valid:+7.2f}%", flush=True)
    
    print("-" * 65, flush=True)
    print("Field-Level F1 Deltas:", flush=True)
    for f in ["task", "dataset", "metric", "result", "baseline"]:
        b_f1 = base_summary["field_f1"].get(f, 0.0) if base_summary else 0.0
        ft_f1 = summary["field_f1"].get(f, 0.0)
        d_f1 = ft_f1 - b_f1
        print(f"  - {f:<15}: Base={b_f1:.4f}  -->  LoRA={ft_f1:.4f}  ({d_f1:+7.4f})", flush=True)
    print("="*60 + "\n", flush=True)

    results_data = {
        "model": "Qwen2.5-0.5B-Instruct + LoRA",
        "type": "lora_finetuned",
        "summary": summary,
        "base_comparison": {
            "base_macro_f1": base_macro,
            "lora_macro_f1": ft_macro,
            "delta": delta_macro
        },
        "details": details
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved fine-tuned evaluation results to {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    main()