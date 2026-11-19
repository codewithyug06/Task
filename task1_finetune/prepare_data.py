"""
Task 1: Domain-Holdout Dataset Generator for Structured Benchmark Extraction.

Architecture:
- Training Domains (10 domains): Vision, NLP, Speech, GNN, TimeSeries, Diffusion, 
  Compression, Document AI, RL, Multimodal. (500 train, 50 val)
- Held-Out Test Domains (4 strictly unseen domains): 
  Biomedical AI, Robotics Manipulation, Code Synthesis, Recommender Systems. (50 test)
- Zero dataset, task, or baseline leakage between Train and Test sets.
- Includes complex real-world syntax: multiple numbers (parameters, margins, speedups),
  inverted error metrics, diverse clause arrangements.
"""

import os
import json
import random

random.seed(42)

# =============================================================================
# 10 TRAINING / VALIDATION DOMAINS
# =============================================================================
TRAIN_DOMAINS = {
    "Computer Vision": {
        "tasks": ["Image Classification", "Semantic Segmentation", "Object Detection", "Instance Segmentation"],
        "datasets": ["ImageNet-1K", "COCO-2017", "ADE20K", "Cityscapes", "Pascal-VOC"],
        "metrics": ["Top-1 Accuracy", "mIoU", "box AP", "mask AP"],
        "baselines": ["ResNet-50", "ConvNeXt-B", "Swin-Transformer", "Mask R-CNN", "DeiT-Base"],
        "ranges": {"Top-1 Accuracy": (78.0, 89.5, "%"), "mIoU": (45.0, 58.2, "%"), "box AP": (42.0, 56.4, ""), "mask AP": (38.0, 49.8, "")}
    },
    "Natural Language Processing": {
        "tasks": ["Extractive Question Answering", "Natural Language Inference", "Coreference Resolution", "Named Entity Recognition"],
        "datasets": ["SQuAD-v2.0", "MultiNLI", "CoNLL-2003", "SuperGLUE", "TriviaQA"],
        "metrics": ["Exact Match", "F1 Score", "Accuracy", "Micro-F1"],
        "baselines": ["BERT-Large", "RoBERTa-Large", "DeBERTa-v3", "ELECTRA-Large", "T5-Base"],
        "ranges": {"Exact Match": (75.0, 88.0, "%"), "F1 Score": (82.0, 93.5, "%"), "Accuracy": (80.0, 92.0, "%"), "Micro-F1": (88.0, 95.2, "%")}
    },
    "Speech Recognition": {
        "tasks": ["Automatic Speech Recognition", "Speaker Identification", "Phoneme Recognition", "Keyword Spotting"],
        "datasets": ["LibriSpeech test-clean", "CommonVoice-EN", "VoxCeleb2", "TIMIT", "TED-LIUM-v3"],
        "metrics": ["Word Error Rate", "Character Error Rate", "Top-1 Accuracy", "Equal Error Rate"],
        "baselines": ["Wav2Vec-2.0", "Conformer-Large", "Whisper-Base", "DeepSpeech2", "HuBERT-Base"],
        "ranges": {"Word Error Rate": (2.1, 7.8, "%"), "Character Error Rate": (1.2, 4.5, "%"), "Top-1 Accuracy": (92.0, 98.4, "%"), "Equal Error Rate": (1.5, 5.2, "%")}
    },
    "Graph Neural Networks": {
        "tasks": ["Node Classification", "Link Prediction", "Graph Classification", "Molecular Property Prediction"],
        "datasets": ["Cora-Citation", "OGBG-MolHIV", "PubMed-Graph", "OGBN-ArXiv", "Reddit-Graph"],
        "metrics": ["Classification Accuracy", "ROC-AUC", "Micro-F1", "Average Precision"],
        "baselines": ["GCN", "GAT-v2", "GraphSAGE", "GIN", "Graphormer"],
        "ranges": {"Classification Accuracy": (79.0, 88.5, "%"), "ROC-AUC": (74.0, 83.5, "%"), "Micro-F1": (65.0, 77.0, "%"), "Average Precision": (70.0, 82.0, "%")}
    },
    "Time Series Forecasting": {
        "tasks": ["Long-Horizon Forecasting", "Multivariate Anomaly Detection", "Financial Trend Forecasting"],
        "datasets": ["Electricity-Load", "Weather-Bench", "Traffic-PEMS", "Exchange-Rate", "ETT-m2"],
        "metrics": ["Mean Squared Error", "Mean Absolute Error", "F1 Score", "Normalized Deviation"],
        "baselines": ["Informer", "Autoformer", "PatchTST", "DLinear", "N-BEATS"],
        "ranges": {"Mean Squared Error": (0.125, 0.485, ""), "Mean Absolute Error": (0.210, 0.520, ""), "F1 Score": (80.0, 91.0, "%"), "Normalized Deviation": (0.15, 0.35, "")}
    },
    "Generative Modeling & Diffusion": {
        "tasks": ["Unconditional Image Synthesis", "Text-to-Image Generation", "Latent Trajectory Modeling"],
        "datasets": ["FFHQ-256", "CelebA-HQ", "LSUN-Bedrooms", "ImageNet-64", "CIFAR-10"],
        "metrics": ["Fréchet Inception Distance", "Inception Score", "Precision", "Recall"],
        "baselines": ["DDPM", "Score-SDE", "StyleGAN2", "Latent Diffusion", "EDM"],
        "ranges": {"Fréchet Inception Distance": (2.15, 6.80, ""), "Inception Score": (9.2, 12.8, ""), "Precision": (68.0, 82.0, "%"), "Recall": (55.0, 72.0, "%")}
    },
    "Efficient Compression": {
        "tasks": ["Post-Training Quantization", "Structured Weight Pruning", "Knowledge Distillation"],
        "datasets": ["Llama-7B-W4A16", "Mistral-7B-Instruct", "ResNet-50-Pruned", "BERT-Base-Quant"],
        "metrics": ["WikiText-2 Perplexity", "Zero-Shot Accuracy", "Top-1 Accuracy", "Latency Speedup"],
        "baselines": ["GPTQ", "AWQ", "SmoothQuant", "Magnitude Pruning", "TinyBERT"],
        "ranges": {"WikiText-2 Perplexity": (5.12, 7.85, ""), "Zero-Shot Accuracy": (68.5, 77.2, "%"), "Top-1 Accuracy": (74.2, 79.8, "%"), "Latency Speedup": (1.8, 3.2, "x")}
    },
    "Document AI": {
        "tasks": ["Key Information Extraction", "Visual Document QA", "Table Structure Recognition"],
        "datasets": ["DocVQA", "CORD-Receipts", "FUNSD", "Kleister-Charity", "PubTables-1M"],
        "metrics": ["Average Normalized Levenshtein Similarity", "F1 Score", "Tree-Edit Distance"],
        "baselines": ["LayoutLMv3", "Donut", "DocFormer", "BROS", "LayoutXLM"],
        "ranges": {"Average Normalized Levenshtein Similarity": (72.0, 86.5, "%"), "F1 Score": (81.0, 93.0, "%"), "Tree-Edit Distance": (85.0, 94.0, "%")}
    },
    "Reinforcement Learning": {
        "tasks": ["Continuous Locomotion Control", "Deep Q-Learning", "Offline Policy Optimization"],
        "datasets": ["MuJoCo HalfCheetah-v4", "Atari-2600 Breakout", "D4RL-AntMaze", "Humanoid-v4"],
        "metrics": ["Average Normalized Return", "Episodic Reward", "Success Rate"],
        "baselines": ["SAC", "PPO", "Decision Transformer", "CQL", "TD3"],
        "ranges": {"Average Normalized Return": (78.0, 115.0, ""), "Episodic Reward": (4200.0, 9800.0, ""), "Success Rate": (70.0, 94.0, "%")}
    },
    "Multimodal Reasoning": {
        "tasks": ["Visual Question Answering", "Referring Expression Grounding", "Image-Text Retrieval"],
        "datasets": ["VQA-v2.0", "RefCOCO+", "TextVQA", "Flickr30k", "GQA-Bench"],
        "metrics": ["VQA Accuracy", "Accuracy@0.5-IoU", "Recall@1", "BLEU-4"],
        "baselines": ["BLIP-2", "Flamingo-9B", "LLaVA-1.5", "CLIP-ViT-L", "OFA-Large"],
        "ranges": {"VQA Accuracy": (74.0, 83.5, "%"), "Accuracy@0.5-IoU": (71.0, 84.0, "%"), "Recall@1": (76.0, 88.0, "%"), "BLEU-4": (34.0, 42.5, "%")}
    }
}

# =============================================================================
# 4 STRICTLY HELD-OUT TEST DOMAINS (COMPLETELY UNSEEN DATASETS & BASELINES)
# =============================================================================
TEST_HOLDOUT_DOMAINS = {
    "Biomedical & Clinical AI": {
        "tasks": ["Clinical Diagnostic Prediction", "Genomic Variant Calling", "Biomedical Literature QA", "EHR Mortality Risk Scoring"],
        "datasets": ["MedQA-USMLE-4Opt", "PubMedQA-Biomedical", "ClinVar-Pathogenicity", "MIMIC-III-ICD9", "PhysioNet-Sepsis2019"],
        "metrics": ["Macro-F1", "Diagnostic Accuracy", "AUROC", "Cohen Kappa", "Balanced Accuracy"],
        "baselines": ["Clinical-T5-Large", "Med-PaLM-2", "BioBERT-v1.1", "GatorTron-OG", "EHR-Densenet"],
        "ranges": {"Macro-F1": (76.0, 87.5, "%"), "Diagnostic Accuracy": (74.0, 85.0, "%"), "AUROC": (0.812, 0.935, ""), "Cohen Kappa": (0.72, 0.86, ""), "Balanced Accuracy": (75.0, 86.0, "%")}
    },
    "Robotics & Embodied AI": {
        "tasks": ["Language-Conditioned Manipulation", "6-DoF Robotic Grasping", "Dexterous Hand Policy Control", "Visual Servoing Navigation"],
        "datasets": ["RoboNet-v2", "BridgeData-v2", "RLBench-18Tasks", "Franka-Kitchen-MultiTask", "DClaw-ScrewTurn"],
        "metrics": ["Task Success Rate", "Grasp Success Rate", "Average Completion Time", "Cumulative Reward"],
        "baselines": ["RT-1-RobotTransformer", "SayCan-PaLM", "PaLM-E-12B", "RoboCat-Agent", "BC-Z-ZeroShot"],
        "ranges": {"Task Success Rate": (68.0, 86.0, "%"), "Grasp Success Rate": (75.0, 91.5, "%"), "Average Completion Time": (12.4, 24.5, "s"), "Cumulative Reward": (650.0, 920.0, "")}
    },
    "Code Synthesis & Formal Logic": {
        "tasks": ["Automated Program Synthesis", "Unit Test Generation", "Formal Theorem Proving", "Code Vulnerability Detection"],
        "datasets": ["HumanEval-Plus", "MBPP-Sanitized", "SWE-bench-Lite", "MiniF2F-Lean4", "BigCode-SecurityBench"],
        "metrics": ["Pass@1", "Pass@10", "Formal Proof Rate", "Detection F1-Score"],
        "baselines": ["CodeLlama-34B-Python", "StarCoder-15B", "WizardCoder-13B", "DeepSeek-Coder-6.7B", "CodeGen2-16B"],
        "ranges": {"Pass@1": (52.0, 78.5, "%"), "Pass@10": (70.0, 89.0, "%"), "Formal Proof Rate": (38.0, 56.5, "%"), "Detection F1-Score": (74.0, 86.5, "%")}
    },
    "Recommender Systems": {
        "tasks": ["Next-Item Sequential Recommendation", "Click-Through Rate Prediction", "Multi-Task Session Ranking", "Cold-Start User Matching"],
        "datasets": ["MovieLens-25M", "KuaiRand-Pure1k", "Amazon-Electronics-2023", "Taobao-ClickStream", "Criteo-AdCTR-100M"],
        "metrics": ["NDCG@10", "HitRatio@10", "AUC", "Mean Reciprocal Rank"],
        "baselines": ["DLRM-v2", "SASRec-Attention", "GRU4Rec-Session", "BST-TransformerRec", "DeepFM-Large"],
        "ranges": {"NDCG@10": (0.524, 0.688, ""), "HitRatio@10": (0.642, 0.795, ""), "AUC": (0.785, 0.892, ""), "Mean Reciprocal Rank": (0.410, 0.590, "")}
    }
}

# =============================================================================
# SYNTACTIC PATTERNS & COMPLEX REALISTIC ABSTRACT COMPOSITION
# =============================================================================

def generate_sample(domain_name, domain_meta, is_test=False):
    task = random.choice(domain_meta["tasks"])
    dataset = random.choice(domain_meta["datasets"])
    metric = random.choice(domain_meta["metrics"])
    baseline = random.choice(domain_meta["baselines"])
    
    val_min, val_max, unit = domain_meta["ranges"][metric]
    if unit == "%":
        result_num = round(random.uniform(val_min, val_max), 1)
        result_str = f"{result_num}%"
        delta = round(random.uniform(1.2, 4.5), 1)
        base_num = round(result_num - delta, 1)
        base_str = f"{base_num}%"
    elif unit == "s":
        result_num = round(random.uniform(val_min, val_max), 1)
        result_str = f"{result_num}s"
        delta = round(random.uniform(2.5, 6.0), 1)
        base_num = round(result_num + delta, 1)
        base_str = f"{base_num}s"
    elif unit == "x":
        result_num = round(random.uniform(val_min, val_max), 1)
        result_str = f"{result_num}x"
        base_str = "1.0x"
    else:
        # Float
        result_num = round(random.uniform(val_min, val_max), 3)
        result_str = str(result_num)
        delta = round(random.uniform(0.025, 0.085), 3)
        base_num = round(result_num - delta, 3)
        base_str = str(base_num)
        
    model_name = random.choice([
        "AdaptiveNet", "OmniScale", "HyperTrans", "FlashArch", "GraphPrior", 
        "SynergyAI", "VectorFlow", "DualStream", "MetaCore", "LatentOpt"
    ]) + f"-{random.choice(['V2', 'Pro', 'X', 'Lite', 'Plus'])}"
    
    # Introduce secondary distractor numbers (e.g. parameter counts, latency speedup, margins)
    param_count = random.choice(["65M", "110M", "350M", "7B", "13B"])
    speedup = f"{round(random.uniform(1.4, 2.8), 1)}x"

    # Diverse sentence styles
    styles = [
        # Style 1: Result first, baseline comparison second
        (f"In this paper, we address the challenge of {task}. We propose {model_name}, a novel architecture containing {param_count} parameters. "
         f"Evaluated on the standard {dataset} benchmark, our approach achieves a {metric} of {result_str}, substantially outperforming the previous state-of-the-art "
         f"{baseline} which attained {base_str}. Furthermore, our model exhibits a {speedup} throughput improvement during inference."),
         
        # Style 2: Baseline first, contrastive improvement
        (f"Prior approaches to {task}, such as {baseline}, struggle with high computational latency and achieve only {base_str} {metric} on {dataset}. "
         f"To overcome these limitations, we introduce {model_name}. Comprehensive experiments demonstrate that {model_name} establishes a new state of the art, "
         f"attaining {result_str} {metric} on {dataset} while reducing parameter overhead by 30%."),
         
        # Style 3: Empirical evaluation narrative
        (f"We present an empirical study on {task} using {model_name} ({param_count} parameters). When benchmarked against competitive baselines including {baseline}, "
         f"our method demonstrates superior representational capacity on the challenging {dataset} dataset, reaching an unprecedented {metric} of {result_str} "
         f"(surpassing {baseline}'s score of {base_str}). Qualitative analysis confirms the stability of our optimization objective."),
         
        # Style 4: Subordinate clause and passive construction
        (f"Although {task} has witnessed rapid progress, current models remain bounded by sample efficiency. On {dataset}, {baseline} yields a {metric} of {base_str}. "
         f"In this work, {model_name} is formulated to address structural bottlenecks. On the rigorous {dataset} evaluation, a {metric} of {result_str} is attained by {model_name}, "
         f"yielding a consistent margin over {baseline} across multiple random seeds.")
    ]
    
    abstract = random.choice(styles)
    
    target_json = {
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "result": result_str,
        "baseline": baseline
    }
    
    return {
        "domain": domain_name,
        "is_held_out": is_test,
        "abstract": abstract,
        "ground_truth": target_json
    }

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    out_dir = REPO_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating 500 Training and 50 Validation samples from 10 Training Domains...", flush=True)
    all_train_val = []
    domain_keys = list(TRAIN_DOMAINS.keys())
    
    # 550 samples across 10 domains (~55 per domain)
    for _ in range(550):
        dom = random.choice(domain_keys)
        all_train_val.append(generate_sample(dom, TRAIN_DOMAINS[dom], is_test=False))
        
    random.shuffle(all_train_val)
    train_data = all_train_val[:500]
    val_data = all_train_val[500:]
    
    print("Generating 50 Held-Out Test samples EXCLUSIVELY from 4 Unseen Domains...", flush=True)
    test_data = []
    test_keys = list(TEST_HOLDOUT_DOMAINS.keys())
    
    # 50 samples across 4 strictly held-out unseen domains
    for _ in range(50):
        dom = random.choice(test_keys)
        test_data.append(generate_sample(dom, TEST_HOLDOUT_DOMAINS[dom], is_test=True))
        
    # Save files
    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
        
    print(f"Successfully created:")
    print(f"  - Train: {len(train_data)} samples (10 domains: {', '.join(domain_keys[:3])}...)")
    print(f"  - Val:   {len(val_data)} samples (10 domains)")
    print(f"  - Test:  {len(test_data)} samples (4 UNSEEN HELD-OUT domains: {', '.join(test_keys)})")

if __name__ == "__main__":
    main()
