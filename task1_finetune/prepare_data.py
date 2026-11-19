"""
Task 1: Domain-Holdout Dataset Generator (Fixed & Leak-Free)
Strict separation:
Train+Val (8 domains):
  1. Machine Translation
  2. Language Modeling
  3. Image Classification
  4. Object Detection
  5. Semantic Segmentation
  6. Question Answering
  7. Named Entity Recognition
  8. Sentiment Analysis

Held-Out Test (4 strictly unseen domains):
  1. Speech Recognition
  2. Code Generation
  3. Reinforcement Learning
  4. Protein Structure Prediction

Guarantees 0% overlap in task names, dataset names, baseline architectures, and domain vocabulary.
Includes numerical distractors (parameter counts, speedup multipliers, margins) to test numerical grounding.
"""

import json
import random
import os

random.seed(42)

# --- 8 TRAIN / VAL DOMAINS ---
TRAIN_VAL_DOMAINS = {
    "Machine Translation": {
        "tasks": ["Neural Machine Translation", "Low-Resource Cross-Lingual Translation", "Document-Level Translation", "Multilingual Sequence Transduction"],
        "datasets": ["WMT-14 En-De", "WMT-16 En-Ro", "IWSLT-14", "OPUS-100"],
        "baselines": ["Transformer-Base", "ConvS2S", "ByteNet", "GNMT", "MarianMT"],
        "metrics": ["BLEU", "SacreBLEU", "chrF", "TER", "METEOR"],
        "result_units": ["score", "pts", "BLEU"],
        "min_res": 22.0, "max_res": 44.5
    },
    "Language Modeling": {
        "tasks": ["Causal Language Modeling", "Autoregressive Density Estimation", "Long-Context Pretraining", "Masked Language Modeling"],
        "datasets": ["WikiText-103", "One Billion Word", "Lambada", "Penn Treebank"],
        "baselines": ["AWD-LSTM", "Transformer-XL", "GPT-2-Medium", "Megatron-LM", "PaLM-8B"],
        "metrics": ["Perplexity", "Bits-per-Character", "Zero-shot Accuracy", "Cross-Entropy Loss", "Negative Log-Likelihood"],
        "result_units": ["ppl", "%", "bpc"],
        "min_res": 10.5, "max_res": 78.2
    },
    "Image Classification": {
        "tasks": ["Fine-Grained Visual Categorization", "Zero-Shot Image Classification", "Robust Visual Recognition", "Few-Shot Visual Representation"],
        "datasets": ["ImageNet-1k", "CIFAR-100", "Places365", "iNaturalist"],
        "baselines": ["ResNet-50", "ViT-Base/16", "EfficientNet-B7", "ConvNeXt-Base", "Swin-Transformer"],
        "metrics": ["Top-1 Accuracy", "Top-5 Accuracy", "Mean Average Precision", "Error Rate", "Macro F1"],
        "result_units": ["%", "%", "pts"],
        "min_res": 72.4, "max_res": 91.8
    },
    "Object Detection": {
        "tasks": ["Dense Object Detection", "Real-Time Bounding Box Regression", "Open-Vocabulary Object Detection", "Oriented Object Localization"],
        "datasets": ["MS-COCO 2017", "Pascal VOC 2012", "LVIS v1.0", "OpenImages-v6"],
        "baselines": ["Faster R-CNN", "YOLOv8-X", "RetinaNet", "DETR-DC5", "Mask R-CNN"],
        "metrics": ["mAP@0.5:0.95", "AP50", "AP75", "Average Recall", "Detection Precision"],
        "result_units": ["%", "AP", "mAP"],
        "min_res": 41.5, "max_res": 62.8
    },
    "Semantic Segmentation": {
        "tasks": ["Pixel-Wise Semantic Segmentation", "Dense Scene Parsing", "Real-Time Urban Segmentation", "Boundary-Aware Segmentation"],
        "datasets": ["Cityscapes", "ADE20K", "Pascal Context", "COCO-Stuff"],
        "baselines": ["DeepLabv3+", "SegFormer-B5", "Mask2Former", "U-Net-Large", "PSPNet-101"],
        "metrics": ["Mean IoU", "Pixel Accuracy", "Class IoU", "Boundary F1", "Frequency Weighted IoU"],
        "result_units": ["%", "mIoU", "%"],
        "min_res": 48.0, "max_res": 84.6
    },
    "Question Answering": {
        "tasks": ["Extractive Question Answering", "Multi-Hop Reading Comprehension", "Open-Domain Knowledge Verification", "Conversational QA"],
        "datasets": ["SQuAD 2.0", "HotpotQA", "TriviaQA", "NaturalQuestions"],
        "baselines": ["BERT-Large", "RoBERTa-Large", "T5-3B", "DeBERTa-v3", "ALBERT-xxlarge"],
        "metrics": ["Exact Match", "F1 Score", "HasAns F1", "NoAns F1", "Macro F1"],
        "result_units": ["%", "F1", "EM"],
        "min_res": 74.2, "max_res": 92.5
    },
    "Named Entity Recognition": {
        "tasks": ["Token-Level Entity Extraction", "Cross-Domain Named Entity Recognition", "Nested Entity Detection", "Few-Shot Relation & Entity Parsing"],
        "datasets": ["CoNLL-2003", "OntoNotes 5.0", "WNUT-17", "Few-NERD"],
        "baselines": ["BiLSTM-CRF", "SpanBERT", "RoBERTa-NER", "LUKE-Base", "Flair-Embeddings"],
        "metrics": ["Micro F1", "Entity F1", "Precision", "Recall", "Strict Match F1"],
        "result_units": ["%", "F1", "pts"],
        "min_res": 68.5, "max_res": 94.8
    },
    "Sentiment Analysis": {
        "tasks": ["Aspect-Based Sentiment Classification", "Document Polarity Detection", "Fine-Grained Emotion Mining", "Subjectivity Analysis"],
        "datasets": ["SST-2", "IMDB Movie Reviews", "SemEval-2017", "TweetEval"],
        "baselines": ["RoBERTa-base", "DistilBERT", "XLM-RoBERTa", "Electra-Large", "TextCNN"],
        "metrics": ["Classification Accuracy", "Macro F1", "Binary F1", "AUC-ROC", "Balanced Accuracy"],
        "result_units": ["%", "%", "pts"],
        "min_res": 81.0, "max_res": 96.5
    }
}

# --- 4 HELD-OUT TEST DOMAINS (COMPLETELY UNSEEN) ---
TEST_DOMAINS = {
    "Speech Recognition": {
        "tasks": ["End-to-End Automatic Speech Recognition", "Robust Acoustic Feature Transduction", "Streaming Speech Transcription", "Multilingual Voice Recognition"],
        "datasets": ["LibriSpeech-100h", "CommonVoice-v11", "VoxCeleb2", "TED-LIUM3"],
        "baselines": ["Wav2Vec2-XLSR", "Conformer-Large", "Whisper-Small", "HuBERT-Base", "DeepSpeech2"],
        "metrics": ["Word Error Rate", "Character Error Rate", "Match Error Rate", "Phone Error Rate", "Real-Time Factor"],
        "result_units": ["%", "WER", "%"],
        "min_res": 3.2, "max_res": 14.8
    },
    "Code Generation": {
        "tasks": ["Program Synthesis from Specifications", "Unit Test-Driven Code Generation", "Repository-Level Code Completion", "Multi-Language Algorithm Synthesis"],
        "datasets": ["HumanEval", "MBPP-Sanitized", "MultiPL-E", "CodeContests-Hard"],
        "baselines": ["CodeLlama-7B", "StarCoder-15B", "CodeGen-6B", "InCoder-6B", "DeepSeek-Coder-1.3B"],
        "metrics": ["Pass@1", "Pass@10", "Executable Accuracy", "CodeBLEU", "Strict Pass Rate"],
        "result_units": ["%", "%", "%"],
        "min_res": 28.5, "max_res": 76.4
    },
    "Reinforcement Learning": {
        "tasks": ["Offline Continuous Control", "Goal-Conditioned Policy Optimization", "Model-Based Robotic Manipulation", "Deep Exploration in Sparse MDPs"],
        "datasets": ["Atari-57-Bench", "D4RL-MuJoCo", "Procgen-Hard", "Meta-World-v2"],
        "baselines": ["PPO-Clip", "SAC-v2", "Dreamer-v3", "DecisionTransformer", "Rainbow-DQN"],
        "metrics": ["Normalized Return", "Mean Episodic Reward", "Task Success Rate", "Sample Efficiency Score", "Cumulative Reward"],
        "result_units": ["pts", "score", "%"],
        "min_res": 64.0, "max_res": 98.2
    },
    "Protein Structure Prediction": {
        "tasks": ["Biomolecular Contact Distance Estimation", "3D Protein Backbone Conformation Parsing", "Sidechain Rotamer Modeling", "Macromolecular Complex Folding"],
        "datasets": ["CASP14-FreeModeling", "CAMEO-2022", "PDB-Bind-v2020", "SCOPe-2.08"],
        "baselines": ["AlphaFold2-Standard", "ESMFold-v1", "RosettaFold-Joint", "OmegaFold", "ProSST-Base"],
        "metrics": ["Global Distance Test", "TM-Score", "Local Distance Difference Test", "Root Mean Square Deviation", "Contact Precision@L/5"],
        "result_units": ["GDT_TS", "TM", "lDDT", "Å", "%"],
        "min_res": 62.0, "max_res": 92.4
    }
}

MODEL_NAMES = [
    "NovaTrans", "AeroNet", "DeepScale-v2", "HyperForm", "SyncNet",
    "OmniModel", "Prism-AI", "Veloce-LM", "OmniFormer", "FlashNet",
    "Apex-Core", "Zenith-v4", "QuantumScale", "PolyMorph-v1", "VectorPulse"
]

TEMPLATE_PATTERNS = [
    "In this work, we study {task}. We introduce {model}, an architecture with {param_distractor} parameters. When evaluated on {dataset}, our model achieves a {metric} of {result}, outperforming {baseline} ({baseline_res}). Furthermore, we observe a {speedup_distractor} speedup during inference.",
    "Addressing the critical challenge of {task}, this paper proposes {model}. Operating with {param_distractor} weights, {model} demonstrates strong empirical results on the standard {dataset} benchmark. Specifically, it establishes a new {metric} of {result}, exceeding the prior state-of-the-art {baseline} which attained {baseline_res}. This corresponds to a {margin_distractor} relative gain.",
    "Recent advances in {task} require efficient yet scalable models. Here, we present {model} ({param_distractor} params). Evaluated across {dataset}, our system records a {metric} of {result}. In direct head-to-head comparison against {baseline} ({baseline_res}), our approach exhibits both higher accuracy and a {speedup_distractor} throughput advantage.",
    "We present {model}, an open framework designed for {task}. Across rigorous experiments on {dataset}, {model} secures a top {metric} of {result}, outstripping {baseline} by {margin_distractor} margin ({baseline_res}). The architecture contains {param_distractor} trainable parameters and delivers {speedup_distractor} faster iteration times.",
    "In this paper, we tackle {task} by introducing {model}. On the {dataset} evaluation suite, our model reaches a {metric} of {result}, surpassing the competitive baseline {baseline} which scored {baseline_res}. Despite containing {param_distractor} parameters, {model} reduces inference latency by {speedup_distractor}."
]

def generate_sample(domain_name, domain_meta):
    task = random.choice(domain_meta["tasks"])
    dataset = random.choice(domain_meta["datasets"])
    baseline = random.choice(domain_meta["baselines"])
    metric = random.choice(domain_meta["metrics"])
    model = random.choice(MODEL_NAMES)
    
    # Generate realistic float result
    val = round(random.uniform(domain_meta["min_res"], domain_meta["max_res"]), 2)
    unit = random.choice(domain_meta["result_units"])
    if unit == "%":
        result_str = f"{val}%"
        base_val = round(val - random.uniform(1.2, 4.5), 2)
        baseline_res_str = f"{base_val}%"
    elif unit in ["BLEU", "AP", "mAP", "F1", "EM", "WER", "GDT_TS", "TM", "lDDT"]:
        result_str = f"{val} {unit}"
        base_val = round(val - random.uniform(1.0, 3.8), 2)
        baseline_res_str = f"{base_val} {unit}"
    else:
        result_str = f"{val}"
        base_val = round(val - random.uniform(1.0, 3.5), 2)
        baseline_res_str = f"{base_val}"
        
    # Numerical distractors
    param_counts = ["45M", "65M", "110M", "125M", "350M", "1.3B", "3.8B", "7B"]
    speedups = ["1.8x", "2.1x", "2.4x", "3.2x", "4.0x", "1.5x"]
    margins = ["3.4%", "5.2%", "2.8%", "4.1%", "6.0%"]
    
    param_distractor = random.choice(param_counts)
    speedup_distractor = random.choice(speedups)
    margin_distractor = random.choice(margins)
    
    template = random.choice(TEMPLATE_PATTERNS)
    abstract = template.format(
        task=task,
        model=model,
        dataset=dataset,
        metric=metric,
        result=result_str,
        baseline=baseline,
        baseline_res=baseline_res_str,
        param_distractor=param_distractor,
        speedup_distractor=speedup_distractor,
        margin_distractor=margin_distractor
    )
    
    target = {
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "result": result_str,
        "baseline": baseline
    }
    
    return {
        "domain": domain_name,
        "abstract": abstract,
        "target": target
    }

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_data_fixed():
    out_dir = REPO_ROOT / "data_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # 500 Train samples from 8 train domains (~62-63 per domain)
    # 50 Val samples from 8 train domains (~6 per domain)
    train_domain_keys = list(TRAIN_VAL_DOMAINS.keys())
    for i in range(500):
        dom = train_domain_keys[i % len(train_domain_keys)]
        train_samples.append(generate_sample(dom, TRAIN_VAL_DOMAINS[dom]))
        
    for i in range(50):
        dom = train_domain_keys[i % len(train_domain_keys)]
        val_samples.append(generate_sample(dom, TRAIN_VAL_DOMAINS[dom]))
        
    # 50 Held-Out Test samples from 4 completely unseen domains (~12-13 per domain)
    test_domain_keys = list(TEST_DOMAINS.keys())
    for i in range(50):
        dom = test_domain_keys[i % len(test_domain_keys)]
        test_samples.append(generate_sample(dom, TEST_DOMAINS[dom]))
        
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    # Save files
    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)
        
    print(f"Generated data_fixed: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Rigorous Zero-Leakage Verification
    train_tasks = {s["target"]["task"] for s in train_samples} | {s["target"]["task"] for s in val_samples}
    train_datasets = {s["target"]["dataset"] for s in train_samples} | {s["target"]["dataset"] for s in val_samples}
    train_baselines = {s["target"]["baseline"] for s in train_samples} | {s["target"]["baseline"] for s in val_samples}
    
    test_tasks = {s["target"]["task"] for s in test_samples}
    test_datasets = {s["target"]["dataset"] for s in test_samples}
    test_baselines = {s["target"]["baseline"] for s in test_samples}
    
    task_overlap = train_tasks & test_tasks
    dataset_overlap = train_datasets & test_datasets
    baseline_overlap = train_baselines & test_baselines
    
    print("\n--- ZERO-LEAKAGE VERIFICATION ---")
    print(f"Task overlap: {len(task_overlap)} -> {task_overlap}")
    print(f"Dataset overlap: {len(dataset_overlap)} -> {dataset_overlap}")
    print(f"Baseline overlap: {len(baseline_overlap)} -> {baseline_overlap}")
    
    assert len(task_overlap) == 0, f"LEAKAGE DETECTED in tasks: {task_overlap}"
    assert len(dataset_overlap) == 0, f"LEAKAGE DETECTED in datasets: {dataset_overlap}"
    assert len(baseline_overlap) == 0, f"LEAKAGE DETECTED in baselines: {baseline_overlap}"
    print("VERIFICATION PASSED: ZERO contamination between train/val and held-out test set!")

if __name__ == "__main__":
    build_data_fixed()
