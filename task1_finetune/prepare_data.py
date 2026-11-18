"""
Dataset preparation for Task 1: Abstract -> Structured JSON Extraction.
Target Schema:
{
  "task": str,
  "dataset": str,
  "metric": str,
  "result": str,
  "baseline": str
}
Generates 500 train, 50 validation, and 50 held-out test examples.
"""

import json
import random
import os

SEED = 42
random.seed(SEED)

DOMAINS = [
    {
        "task": "Machine Translation",
        "datasets": ["WMT'14 English-German", "WMT'14 English-French", "IWSLT'14 De-En", "Flores-101"],
        "metric": "BLEU",
        "baselines": ["Transformer-Base", "ConvS2S", "mBART-50", "NLLB-200", "Deep-Transformer"],
        "min_res": 24.0, "max_res": 44.5, "fmt": "{:.1f}"
    },
    {
        "task": "Language Modeling",
        "datasets": ["Wikitext-103", "Penn Treebank", "OpenWebText", "The Pile"],
        "metric": "Perplexity",
        "baselines": ["Transformer-XL", "AWD-LSTM", "GPT-2-Medium", "Megatron-LM", "LLaMA-7B"],
        "min_res": 12.4, "max_res": 58.2, "fmt": "{:.1f}"
    },
    {
        "task": "Image Classification",
        "datasets": ["ImageNet-1K", "CIFAR-100", "Food-101", "iNaturalist 2021"],
        "metric": "Top-1 Accuracy",
        "baselines": ["ResNet-50", "DenseNet-121", "ViT-Base/16", "Swin-Transformer", "ConvNeXt-Base"],
        "min_res": 76.5, "max_res": 89.4, "fmt": "{:.1f}%"
    },
    {
        "task": "Object Detection",
        "datasets": ["MS COCO val2017", "Pascal VOC 2007", "LVIS v1.0", "Objects365"],
        "metric": "mAP@0.5:0.95",
        "baselines": ["Faster R-CNN", "YOLOv3", "DETR", "DINO-DETR", "CenterNet2"],
        "min_res": 42.1, "max_res": 56.8, "fmt": "{:.1f}"
    },
    {
        "task": "Semantic Segmentation",
        "datasets": ["Cityscapes", "ADE20K", "Pascal Context", "COCO-Stuff"],
        "metric": "mIoU",
        "baselines": ["DeepLabV3+", "SegFormer-B4", "MaskFormer", "PSPNet", "HRNetV2"],
        "min_res": 74.2, "max_res": 84.9, "fmt": "{:.1f}%"
    },
    {
        "task": "Question Answering",
        "datasets": ["SQuAD v2.0", "TriviaQA", "Natural Questions", "HotpotQA"],
        "metric": "F1",
        "baselines": ["BERT-Large", "RoBERTa-Large", "DeBERTa-v3-Large", "SpanBERT", "T5-Large"],
        "min_res": 83.5, "max_res": 92.8, "fmt": "{:.1f}"
    },
    {
        "task": "Named Entity Recognition",
        "datasets": ["CoNLL-2003", "OntoNotes 5.0", "WNUT-17", "Few-NERD"],
        "metric": "F1",
        "baselines": ["BiLSTM-CRF", "BERT-Base", "Flair-Embeddings", "LUKE-Base", "RoBERTa-CRF"],
        "min_res": 88.0, "max_res": 95.5, "fmt": "{:.1f}"
    },
    {
        "task": "Sentiment Analysis",
        "datasets": ["SST-2", "IMDb Reviews", "Yelp Polarity", "Amazon-5"],
        "metric": "Accuracy",
        "baselines": ["RoBERTa-Base", "ELECTRA-Large", "DistilBERT", "XLNet-Large", "BiLSTM-Attention"],
        "min_res": 91.2, "max_res": 97.4, "fmt": "{:.1f}%"
    },
    {
        "task": "Speech Recognition",
        "datasets": ["LibriSpeech test-other", "Common Voice En", "TIMIT", "TedLium v3"],
        "metric": "Word Error Rate",
        "baselines": ["Conformer-CTC", "Wav2Vec 2.0", "Whisper-Small", "DeepSpeech 2", "Hubert-Large"],
        "min_res": 3.8, "max_res": 7.5, "fmt": "{:.1f}%"
    },
    {
        "task": "Graph Representation Learning",
        "datasets": ["OGB-Arxiv", "Cora", "CiteSeer", "PubMed-Graph"],
        "metric": "Accuracy",
        "baselines": ["GCN", "GAT", "GraphSAGE", "GIN", "Node2Vec"],
        "min_res": 68.5, "max_res": 82.3, "fmt": "{:.1f}%"
    },
    {
        "task": "Code Generation",
        "datasets": ["HumanEval", "MBPP", "APPS", "DS-1000"],
        "metric": "Pass@1",
        "baselines": ["CodeLlama-7B", "StarCoder-Base", "DeepSeek-Coder-6.7B", "Codex", "CodeGen-6B"],
        "min_res": 42.0, "max_res": 66.8, "fmt": "{:.1f}%"
    },
    {
        "task": "Mathematical Reasoning",
        "datasets": ["GSM8K", "MATH", "SVAMP", "Minerva-Math"],
        "metric": "Accuracy",
        "baselines": ["LLaMA-2-7B", "Mistral-7B", "PaLM-540B", "Qwen-7B", "Chinchilla-70B"],
        "min_res": 45.2, "max_res": 74.0, "fmt": "{:.1f}%"
    },
    {
        "task": "Reinforcement Learning",
        "datasets": ["Atari 2600 Breakout", "MuJoCo HalfCheetah-v3", "Humanoid-v3", "Procgen"],
        "metric": "Average Return",
        "baselines": ["DQN", "PPO", "SAC", "Rainbow", "DreamerV3"],
        "min_res": 340.0, "max_res": 620.0, "fmt": "{:.1f}"
    },
    {
        "task": "Protein Structure Prediction",
        "datasets": ["CASP14", "CASP15", "CAMEO", "PDB-Blind"],
        "metric": "GDT-TS",
        "baselines": ["AlphaFold v1", "RoseTTAFold", "ESMFold", "TrRosetta", "RaptorX"],
        "min_res": 82.5, "max_res": 94.2, "fmt": "{:.1f}"
    }
]

INTRO_PATTERNS = [
    "In this paper, we tackle the critical challenge of {task}.",
    "We investigate novel neural architectures for {task}.",
    "Addressing current efficiency and scaling bottlenecks in {task}, we introduce an adaptive formulation.",
    "Despite significant progress in {task}, prevailing methods struggle with distributional shift.",
    "In this work, we propose a lightweight parameter-efficient architecture for {task}.",
    "Existing paradigms for {task} require exorbitant computational resources during both training and evaluation.",
    "We present a generalized attention-based framework targeted at {task}."
]

BODY_PATTERNS = [
    "Prior standards such as {baseline} exhibit severe degradation under low-resource or noisy settings.",
    "While established baselines including {baseline} offer competitive scores, their parameter efficiency remains suboptimal.",
    "Traditional models like {baseline} struggle to capture fine-grained relational semantics across distant tokens.",
    "In contrast to {baseline}, our design decouples feature extraction from cross-modal alignment.",
    "Compared to the prevailing baseline {baseline}, our method incorporates dynamic token sparsification."
]

EVAL_PATTERNS = [
    "We conduct rigorous empirical validations on the standard {dataset} benchmark.",
    "Evaluated extensively across {dataset}, our framework demonstrates consistent empirical superiority.",
    "On the widely referenced {dataset} dataset, our proposed model achieves state-of-the-art results.",
    "When deployed and evaluated on {dataset}, the architecture reveals substantial gains.",
    "Benchmarking our approach on {dataset} confirms that our design principles scale effectively."
]

RESULT_PATTERNS = [
    "Specifically, our model delivers a {metric} of {result}, clearly surpassing {baseline}.",
    "Our approach records an impressive {metric} of {result}, outperforming {baseline} by a notable margin.",
    "Experimental results demonstrate a {metric} of {result}, establishing a strong new baseline over {baseline}.",
    "The proposed method achieves an outstanding {metric} of {result}, whereas {baseline} fails to match this benchmark.",
    "Our system reaches a top {metric} of {result}, confirming the efficacy of our architectural modifications."
]

def build_dataset():
    seen_abstracts = set()
    data = []
    
    # We want 600 total (500 train, 50 val, 50 test)
    while len(data) < 600:
        domain = random.choice(DOMAINS)
        task = domain["task"]
        dataset = random.choice(domain["datasets"])
        metric = domain["metric"]
        baseline = random.choice(domain["baselines"])
        
        # generate numeric result
        val = random.uniform(domain["min_res"], domain["max_res"])
        res_str = domain["fmt"].format(val)
        
        intro = random.choice(INTRO_PATTERNS).format(task=task)
        body = random.choice(BODY_PATTERNS).format(baseline=baseline)
        evl = random.choice(EVAL_PATTERNS).format(dataset=dataset)
        res = random.choice(RESULT_PATTERNS).format(metric=metric, result=res_str, baseline=baseline)
        
        abstract = f"{intro} {body} {evl} {res} Extensive ablation studies verify that every proposed component contributes positively to the overall performance."
        
        if abstract not in seen_abstracts:
            seen_abstracts.add(abstract)
            ground_truth = {
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "result": res_str,
                "baseline": baseline
            }
            data.append({
                "abstract": abstract,
                "ground_truth": ground_truth
            })
            
    random.shuffle(data)
    
    train_set = data[:500]
    val_set = data[500:550]
    test_set = data[550:600]
    
    os.makedirs("d:/Intern/data", exist_ok=True)
    with open("d:/Intern/data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2)
    with open("d:/Intern/data/val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2)
    with open("d:/Intern/data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)
        
    print(f"Dataset successfully created:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

if __name__ == "__main__":
    build_dataset()