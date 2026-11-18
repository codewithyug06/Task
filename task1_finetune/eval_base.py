"""
Evaluate Base Model (Qwen2.5-0.5B-Instruct) Zero-Shot on the 50 Held-Out Test Set.
Batched generation on pure CUDA for fast and reliable execution.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import evaluate_predictions

MODEL_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TEST_DATA_PATH = "d:/Intern/data/test.json"
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

    print(f"Loading base model and tokenizer from local path:\n{MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    predictions = []
    print("\nRunning batched inference on base model...", flush=True)
    
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

    print("\nEvaluating base model predictions...", flush=True)
    summary, details = evaluate_predictions(test_samples, predictions)
    
    print("\n================== BASE MODEL RESULTS ==================", flush=True)
    print(f"Total Test Samples:  {summary['total_samples']}", flush=True)
    print(f"Valid JSON Rate:     {summary['json_valid_rate'] * 100:.2f}% ({summary['valid_json_count']}/{summary['total_samples']})", flush=True)
    print(f"Overall Macro F1:    {summary['overall_macro_f1']:.4f}", flush=True)
    print("Field-Level F1 Breakdown:", flush=True)
    for f, score in summary["field_f1"].items():
        print(f"  - {f:10s}: F1 = {score:.4f} (P = {summary['field_precision'][f]:.4f}, R = {summary['field_recall'][f]:.4f})", flush=True)
    print("========================================================\n", flush=True)

    results_data = {
        "model": "Qwen2.5-0.5B-Instruct",
        "type": "base_zero_shot",
        "summary": summary,
        "details": details
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved base model evaluation results to {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    main()