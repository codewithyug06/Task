"""
Token-level Evaluation Metrics for Structured JSON Extraction:
- Precision, Recall, and F1 per schema field
- JSON Syntax Validity Rate
- Overall Macro F1 across all fields and samples
"""

import json
import re

SCHEMA_FIELDS = ["task", "dataset", "metric", "result", "baseline"]

def tokenize(text):
    if text is None:
        return []
    text_str = str(text).lower()
    tokens = re.findall(r"\w+|[^\w\s]", text_str)
    return tokens

def compute_token_f1(pred_str, gold_str):
    pred_tokens = tokenize(pred_str)
    gold_tokens = tokenize(gold_str)
    
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0
        
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
        
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
        
    overlap = 0
    for t, count in pred_counts.items():
        if t in gold_counts:
            overlap += min(count, gold_counts[t])
            
    if overlap == 0:
        return 0.0, 0.0, 0.0
        
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def extract_json_from_response(raw_text):
    clean = raw_text.strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        clean = match.group(1).strip()
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
            
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed, True
    except Exception:
        pass
        
    return None, False

def evaluate_predictions(test_samples, predictions):
    """
    Computes JSON Validity, Field-Level F1s, and Overall Macro F1.
    Robust to whether test_samples are full sample dicts or list of gold dicts.
    """
    total = len(test_samples)
    valid_json_count = 0
    field_scores = {f: {"precision": [], "recall": [], "f1": []} for f in SCHEMA_FIELDS}
    overall_sample_f1s = []
    
    details = []
    
    for sample, raw_pred in zip(test_samples, predictions):
        if isinstance(sample, dict) and any(k in sample for k in ["ground_truth", "target", "gold"]):
            gold = sample.get("ground_truth", sample.get("target", sample.get("gold")))
            abstract = sample.get("abstract", "")
            domain = sample.get("domain", "")
        else:
            gold = sample if isinstance(sample, dict) else {}
            abstract = ""
            domain = ""
            
        parsed_dict, is_valid = extract_json_from_response(raw_pred)
        
        if is_valid:
            valid_json_count += 1
            
        sample_field_f1s = []
        sample_detail = {
            "domain": domain,
            "abstract": abstract,
            "gold": gold,
            "raw_prediction": raw_pred,
            "is_valid_json": is_valid,
            "parsed_json": parsed_dict,
            "field_metrics": {}
        }
        
        for f in SCHEMA_FIELDS:
            gold_val = gold.get(f, "") if isinstance(gold, dict) else ""
            pred_val = parsed_dict.get(f, "") if (is_valid and parsed_dict and isinstance(parsed_dict, dict)) else ""
            
            p, r, f1 = compute_token_f1(pred_val, gold_val)
            field_scores[f]["precision"].append(p)
            field_scores[f]["recall"].append(r)
            field_scores[f]["f1"].append(f1)
            sample_field_f1s.append(f1)
            
            sample_detail["field_metrics"][f] = {
                "precision": p,
                "recall": r,
                "f1": f1
            }
            
        mean_sample_f1 = sum(sample_field_f1s) / len(sample_field_f1s) if sample_field_f1s else 0.0
        sample_detail["sample_mean_f1"] = mean_sample_f1
        overall_sample_f1s.append(mean_sample_f1)
        details.append(sample_detail)
        
    summary = {
        "valid_json_count": valid_json_count,
        "overall_macro_f1": sum(overall_sample_f1s) / len(overall_sample_f1s) if overall_sample_f1s else 0.0,
        "field_f1": {f: (sum(field_scores[f]["f1"]) / total) if total else 0.0 for f in SCHEMA_FIELDS},
        "field_precision": {f: (sum(field_scores[f]["precision"]) / total) if total else 0.0 for f in SCHEMA_FIELDS},
        "field_recall": {f: (sum(field_scores[f]["recall"]) / total) if total else 0.0 for f in SCHEMA_FIELDS},
        "results": details
    }
    return summary
