"""
Evaluation Metrics for Structured Extraction:
- JSON Parse Rate / Validity Rate
- Field-Level Token Precision, Recall, F1 for each schema key
- Sample F1 and Overall Dataset Macro/Micro F1
"""

import json
import re

SCHEMA_FIELDS = ["task", "dataset", "metric", "result", "baseline"]

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove surrounding punctuation
    text = re.sub(r'[\'"]', '', text)
    return text

def compute_token_f1(pred_str, gold_str):
    pred_tokens = normalize_text(pred_str).split()
    gold_tokens = normalize_text(gold_str).split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
        
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def extract_json_from_response(response_text):
    """Robustly parse JSON object from model generation."""
    response_text = response_text.strip()
    
    # Check for markdown code blocks
    if "```json" in response_text:
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
    elif "```" in response_text:
        match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
            
    # Try direct parse
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            return data, True
    except Exception:
        pass
        
    # Try finding first { and matching }
    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = response_text[start:end+1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data, True
        except Exception:
            pass
            
    return None, False

def evaluate_predictions(test_samples, predictions):
    """
    Computes JSON Validity, Field-Level F1s, and Overall Macro F1.
    """
    total = len(test_samples)
    valid_json_count = 0
    field_scores = {f: {"precision": [], "recall": [], "f1": []} for f in SCHEMA_FIELDS}
    overall_sample_f1s = []
    
    details = []
    
    for sample, raw_pred in zip(test_samples, predictions):
        gold = sample["ground_truth"]
        parsed_dict, is_valid = extract_json_from_response(raw_pred)
        
        if is_valid:
            valid_json_count += 1
            
        sample_field_f1s = []
        sample_detail = {
            "abstract": sample["abstract"],
            "gold": gold,
            "raw_prediction": raw_pred,
            "is_valid_json": is_valid,
            "parsed": parsed_dict,
            "field_metrics": {}
        }
        
        for f in SCHEMA_FIELDS:
            gold_val = gold.get(f, "")
            pred_val = parsed_dict.get(f, "") if (is_valid and parsed_dict) else ""
            
            p, r, f1 = compute_token_f1(pred_val, gold_val)
            field_scores[f]["precision"].append(p)
            field_scores[f]["recall"].append(r)
            field_scores[f]["f1"].append(f1)
            sample_field_f1s.append(f1)
            
            sample_detail["field_metrics"][f] = {
                "pred": pred_val,
                "gold": gold_val,
                "f1": f1
            }
            
        sample_f1 = sum(sample_field_f1s) / len(sample_field_f1s)
        overall_sample_f1s.append(sample_f1)
        sample_detail["sample_mean_f1"] = sample_f1
        details.append(sample_detail)
        
    json_valid_rate = valid_json_count / total
    macro_f1 = sum(overall_sample_f1s) / total
    
    summary = {
        "total_samples": total,
        "valid_json_count": valid_json_count,
        "json_valid_rate": json_valid_rate,
        "overall_macro_f1": macro_f1,
        "field_f1": {
            f: sum(field_scores[f]["f1"]) / total for f in SCHEMA_FIELDS
        },
        "field_precision": {
            f: sum(field_scores[f]["precision"]) / total for f in SCHEMA_FIELDS
        },
        "field_recall": {
            f: sum(field_scores[f]["recall"]) / total for f in SCHEMA_FIELDS
        }
    }
    return summary, details

