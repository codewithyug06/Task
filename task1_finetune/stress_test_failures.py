"""
Stress test the fine-tuned LoRA model on difficult edge cases to identify and document 
3 concrete, nuanced failure modes with exact inputs, outputs, and diagnostic root cause analysis.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
LORA_ADAPTER_PATH = "d:/Intern/task1_finetune/lora_adapter"
OUTPUT_PATH = "d:/Intern/task1_finetune/failure_analysis_cases.json"

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

EDGE_CASES = [
    {
        "id": "failure_case_1",
        "title": "Multi-Metric & Multi-Dataset Competition",
        "abstract": "We introduce UnifiedVision, a cross-domain vision model evaluated across both ImageNet-1K classification, achieving an unprecedented 88.4% top-1 accuracy (exceeding Swin-L by 1.2%), and COCO object detection, attaining 58.7 box AP compared to 57.1 by Mask R-CNN. The model uses unified self-attention.",
        "expected_challenge": "The abstract reports two distinct tasks, datasets, metrics, results, and baselines simultaneously. Since the schema only has scalar fields, the model must collapse or truncate one, leading to loss of relational coupling.",
        "category": "Multi-Value Extraction Conflict / Schema Bottleneck"
    },
    {
        "id": "failure_case_2",
        "title": "Purely Qualitative / Implicit Empirical Claim (No Explicit Number)",
        "abstract": "Scaling laws suggest larger models generalize better. We present ScaledSparse, an efficient MoE architecture. Extensive empirical evaluations across diverse natural language benchmarks demonstrate that our method consistently outperforms dense transformer baselines and converges substantially faster while maintaining strict numerical stability.",
        "expected_challenge": "The abstract contains no numerical score or quantitative metric, only qualitative comparative claims ('consistently outperforms', 'converges substantially faster'). The model may hallucinate a number or fail schema type expectations.",
        "category": "Hallucination under Ungrounded / Implicit Claims"
    },
    {
        "id": "failure_case_3",
        "title": "Inverted Metric Semantics & Dual-Number Ambiguity (Error vs Accuracy / Relative vs Absolute)",
        "abstract": "We propose NoiseRobust-ASR for speech recognition. On the LibriSpeech test-clean benchmark, our model reduces the Word Error Rate from 4.8% obtained by Conformer-CTC down to 3.1%, representing a relative error reduction of 35.4% over previous state of the art.",
        "expected_challenge": "There are three numbers: baseline error (4.8%), achieved error (3.1%), and relative improvement (35.4%). The model may confuse the relative gain (35.4%) with the absolute result (3.1%), or conflate error with accuracy.",
        "category": "Relative Delta vs. Absolute Score Conflation"
    }
]

def main():
    print("Loading model and tokenizer for failure mode stress testing...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()
    
    results = []
    
    print("\nRunning inference on challenging failure edge cases...\n", flush=True)
    for case in EDGE_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Abstract:\n{case['abstract']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        try:
            parsed_json = json.loads(raw_output)
        except Exception as e:
            parsed_json = {"error_parsing": str(e), "raw": raw_output}
            
        print(f"=== {case['title']} ===")
        print(f"Abstract: {case['abstract']}")
        print(f"Model Output: {raw_output}\n")
        
        case_record = {
            "id": case["id"],
            "title": case["title"],
            "category": case["category"],
            "abstract": case["abstract"],
            "model_raw_output": raw_output,
            "parsed_json": parsed_json,
            "diagnostic_analysis": case["expected_challenge"]
        }
        results.append(case_record)
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved stress test failure cases to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
