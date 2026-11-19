# Applied Generative AI Engineering: Production Systems & Optimization
### Complete Implementation & Technical Walkthrough for 3-Task Engineering Assessment
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20PEFT-yellow.svg)](https://huggingface.co/)
[![CUDA Available](https://img.shields.io/badge/CUDA-RTX%204050%20(6GB)-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![PDF Compliance](https://img.shields.io/badge/Report-Strict%204--Page%20Ceiling-brightgreen.svg)](report/final_report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Overview

This repository houses a production-grade, mathematically verified submission for the **Applied Generative AI Engineering Assessment**. Rather than treating the assignment as three disconnected toy problems, the entire codebase is unified into a cohesive **Machine Learning Literature Intelligence & Local Serving Architecture** built over 10 landmark AI research papers (*Attention Is All You Need, LoRA, QLoRA, FlashAttention, ResNet, BERT, Chinchilla, GPT-3, Llama 2, DDPM*):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END SYSTEM ARCHITECTURE                                  │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│   TASK 1: REPRESENTATION   │    TASK 2: RETRIEVAL        │     TASK 3: INFERENCE       │
│  Domain-Holdout LoRA (0.5B)│ Scratch Hybrid RAG (335 pp) │ Local Serving & Optimization│
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ • Abstract → 5-Key JSON    │ • Zero LangChain/LlamaIndex │ • FP16 vs NF4 4-bit profiled│
│ • Strict 8/4 Domain Split  │ • Pure Python Okapi BM25    │ • 2.16x VRAM compression    │
│ • 0 Overlap Train vs Test  │ • FAISS IndexFlatIP (Dense) │ • PagedAttention vLLM Plan  │
│ • Macro F1: 0.9011 → 0.9832│ • Vector Dilution Analysis  │ • 20 Users on 24GB (7.77x)  │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

Every numerical figure reported is grounded in **our own empirical runs** executed on local NVIDIA CUDA hardware, backed by raw JSON logs, and formatted into both interactive Jupyter/Colab notebooks and a strict **4-page executive report**.

---

## Key Empirical Findings & Metrics

### 1. Task 1 — LoRA Fine-Tuning on Strict Domain Holdout
- **Dataset Partition**: 500 train + 50 validation samples spanning 8 standard domains (*Vision, Translation, QA, NLP*). **Held-out test set** comprises 50 samples across 4 **strictly unseen domains** (*Speech Recognition, Code Generation, Reinforcement Learning, Protein Structure Prediction*). **Zero overlap** in tasks, datasets, or baselines.
- **Evaluation Metric**: Token-level Precision, Recall, Field $F_1$, and Overall Macro $F_1$.

| Model Setting | Overall Macro $F_1$ | `task` $F_1$ | `dataset` $F_1$ | `metric` $F_1$ | `result` $F_1$ | `baseline` $F_1$ | Valid JSON |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Zero-Shot (`0.5B-Instruct`)** | **0.9011** | 0.7854 | 0.9300 | 0.9508 | **0.8394** | 1.0000 | 100.0% |
| **LoRA Fine-Tuned (`r=16, \alpha=32`)** | **0.9832** | **0.9219** | **1.0000** | **0.9971** | **0.9971** | **1.0000** | 100.0% |
| **Empirical Delta ($\Delta$)** | **+0.0821** | **+0.1365** | **+0.0700** | **+0.0463** | **+0.1577** | 0.0000 | 0.0% |

> **Key Takeaway**: LoRA fine-tuning overcomes the base model's vulnerability to numerical distractors (`result` $F_1: 0.8394 \to 0.9971$) and generalizes strongly to unseen technical vocabulary (`task` $F_1: 0.7854 \to 0.9219$).

---

### 2. Task 2 — Scratch RAG Engine & Chunk Size Benchmark (335 Real arXiv Pages)
- **Document Corpus**: 10 seminal arXiv PDFs, 335 authentic pages, 169,731 words parsed via `pypdf`/`PyMuPDF` ([corpus_real_sections.json](data/corpus_real_sections.json)). Zero high-level frameworks used.
- **Benchmark Suite**: 15 hand-authored technical test questions paired with exact ground-truth passage IDs and excerpts ([benchmark_15_questions.json](data/benchmark_15_questions.json)).

| Chunking Configuration | Total Chunks | Retrieval Mode | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Setting A: 256 tokens** | **1,732 chunks** | Sparse BM25 | 33.3% | 60.0% | 80.0% | 0.4656 |
| *(overlap: 50 tokens)* | | Dense FAISS | **46.7%** | 53.3% | **60.0%** | **0.5133** |
| | | **Hybrid (Scratch)** | **33.3%** | **53.3%** | **66.7%** | **0.4444** |
| **Setting B: 512 tokens** | **913 chunks** | Sparse BM25 | 33.3% | 73.3% | 80.0% | 0.5000 |
| *(overlap: 100 tokens)* | | Dense FAISS | **40.0%** | 53.3% | **53.3%** | **0.4500** |
| | | **Hybrid (Scratch)** | **26.7%** | **66.7%** | **73.3%** | **0.4556** |

> **Empirical Discovery ("Vector Dilution Effect")**: Setting B (512 tokens) caused a direct decline in Dense Recall@1 from **46.7% down to 40.0%** and MRR from **0.5133 to 0.4500**. Squashing 512 tokens of complex math into a single 384-dimensional dense vector dilutes fine-grained technical assertions. Setting A (256 tokens) provides superior semantic resolution.

---

### 3. Task 3 — Local GPU Inference & Serving Optimization
- **Hardware Profile**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), PyTorch 2.1, BitsAndBytes. Evaluated across 10 identical multi-domain technical prompts.

| Hardware Metric | Setting A: FP16 (16-bit Baseline) | Setting B: 4-bit NF4 (BitsAndBytes) | Quantitative Impact |
| :--- | :---: | :---: | :--- |
| **Model Weights VRAM** | **942.32 MB** | **436.35 MB** | **$2.16\times$ VRAM Reduction ($-53.7\%$)** |
| **Peak Generation VRAM** | **957.62 MB** | **457.66 MB** | **$2.09\times$ Peak Reduction ($-52.2\%$)** |
| **Average Decoding Speed** | **7.97 tokens/sec** | **5.19 tokens/sec** | $-34.9\%$ decoding speed (dequant penalty) |
| **Total Generation Time (10 prompts)** | **168.93 s** | **255.47 s** | $+51.2\%$ total latency |

- **20-Concurrent-User Serving Plan (24 GB GPU)**: Serving `Qwen2.5-7B` in 4-bit AWQ (3.85 GB weights) leaves a dedicated **17.00 GB dynamic KV cache pool**. Under GQA ($H_{KV}=4$), 20 users with 2,048 tokens consume only **2.24 GB**, providing a **$7.77\times$ safety headroom factor** with PagedAttention and continuous batching.

---

## Interactive Notebooks (Pre-Populated with Execution Outputs)

All tasks are packaged into self-contained, reproducible Jupyter Notebooks in [`notebooks/`](notebooks/). Every code cell is **pre-populated with verified execution outputs** matching our technical report:

| Notebook | Focus Area | Contents & Validated Outputs |
| :--- | :--- | :--- |
| **[`Applied_GenAI_Master_Submission.ipynb`](notebooks/Applied_GenAI_Master_Submission.ipynb)** | **Master End-to-End Pipeline** | Full linear reproduction of Task 1, Task 2, and Task 3 in a single executable workflow. |
| **[`Task1_LoRA_Finetuning.ipynb`](notebooks/Task1_LoRA_Finetuning.ipynb)** | **Fine-Tuning & 7B Memory** | Zero-leakage data split, base evaluation, LoRA training, 3 failure modes, and 7B memory derivation. |
| **[`Task2_RAG_From_Scratch.ipynb`](notebooks/Task2_RAG_From_Scratch.ipynb)** | **Scratch RAG Engine** | 10 arXiv PDF download/extraction (335 pp), BM25 + FAISS index, 256 vs 512 evaluation, grounded QA. |
| **[`Task3_Local_Optimization_and_Serving.ipynb`](notebooks/Task3_Local_Optimization_and_Serving.ipynb)** | **Serving & Quantization** | FP16 vs NF4 profiling, 10-prompt quality table, KV cache / quantization theory, 24GB serving math. |

---

## Repository Directory Structure

```
├── config.py                              # Centralized dynamic path & HF Hub configuration
├── build_notebooks.py                     # Programmatic notebook builder with output embedding
├── README.md                              # Engineering documentation & reproduction guide
├── notebooks/                             # Executable notebooks with rich saved outputs
│   ├── Applied_GenAI_Master_Submission.ipynb
│   ├── Task1_LoRA_Finetuning.ipynb
│   ├── Task2_RAG_From_Scratch.ipynb
│   └── Task3_Local_Optimization_and_Serving.ipynb
├── data_fixed/                            # Leak-free domain-holdout dataset (8 train / 4 test)
│   ├── train.json                         # 500 training samples
│   ├── val.json                           # 50 validation samples
│   └── test.json                          # 50 strictly unseen held-out domain test samples
├── data/                                  # Real 10-paper academic literature corpus
│   ├── corpus_real_sections.json          # 335 authentic arXiv pages structured by section
│   ├── corpus_real.txt                    # Continuous text of 335-page corpus (169k words)
│   ├── benchmark_15_questions.json        # 15 technical questions with ground-truth excerpts
│   └── arxiv_pdfs/                        # 10 downloaded original arXiv PDFs
├── task1_finetune/                        # Task 1: LoRA Fine-Tuning Pipeline
│   ├── prepare_data.py                    # Domain-holdout dataset generator & distractor injector
│   ├── train_lora.py                      # LoRA training script (r=16, alpha=32, all linear modules)
│   ├── eval_base.py                       # Base model zero-shot evaluation on held-out set
│   ├── eval_lora.py                       # Fine-tuned model held-out evaluation
│   ├── stress_test_failures.py            # Failure mode stress testing
│   ├── metrics.py                         # Token-level Precision, Recall, and Macro F1
│   ├── base_model_eval_results.json       # Base model empirical metrics (Macro F1: 0.9011)
│   ├── lora_model_eval_results.json       # Fine-tuned model empirical metrics (Macro F1: 0.9832)
│   ├── failure_analysis_cases.json        # 3 documented empirical failure edge cases
│   └── lora_adapter/                      # Saved adapter weights (adapter_model.safetensors)
├── task2_rag/                             # Task 2: Scratch RAG Engine (Zero Frameworks)
│   ├── build_corpus.py                    # 10-PDF downloader & page extractor
│   ├── build_15_questions.py              # Verification script for 15 technical questions
│   ├── rag_engine.py                      # Pure Python Chunker, Scratch BM25, and Hybrid RAG
│   ├── evaluate_rag.py                    # Chunk size evaluation (256 vs 512 tokens)
│   ├── generate_qa_answers.py             # Grounded question answering with citations
│   ├── rag_evaluation_results.json        # Empirical retrieval metrics (Recall@K, MRR@5)
│   └── qa_generation_answers.json         # 15 synthesized answers with page citations
├── task3_optimization/                    # Task 3: Local Serving & Quantization
│   ├── benchmark_inference.py             # FP16 vs 4-bit NF4 hardware profiler
│   ├── inference_benchmark_results.json   # Latency, memory, and 10-prompt comparison
│   ├── Modelfile                          # Standalone Ollama Modelfile configuration
│   └── run_ollama_benchmark.py            # Companion Ollama API benchmark runner
└── report/                                # Submission Reports
    ├── final_report.md                    # Primary 4-Page Submission Report (PDF-Ready)
    ├── final_report.html                  # HTML source with custom KaTeX & page breaks
    ├── final_report.pdf                   # Rendered 4-Page PDF (verified via pypdf)
    └── technical_appendix.md              # Extended technical appendix & raw derivations
```

---

## Step-by-Step Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/yxgx06/dem1.git
cd dem1
pip install torch transformers peft datasets accelerate bitsandbytes sentence-transformers faiss-cpu pypdf pymupdf nbformat
```

### 1. Task 1: Domain-Holdout LoRA Fine-Tuning
```bash
# Generate leak-free domain-holdout dataset (8 train / 4 test domains)
python task1_finetune/prepare_data.py

# Evaluate Base Model Zero-Shot Baseline on 50 held-out samples
python task1_finetune/eval_base.py

# Train LoRA Adapter (r=16, alpha=32, 4 epochs) on GPU
python task1_finetune/train_lora.py

# Evaluate Fine-Tuned Model on Held-Out Test Set
python task1_finetune/eval_lora.py

# Run failure mode stress testing
python task1_finetune/stress_test_failures.py
```

### 2. Task 2: Scratch RAG Pipeline over 335-Page Corpus
```bash
# Download 10 landmark arXiv PDFs and extract 335 pages
python task2_rag/build_corpus.py

# Verify the 15 technical benchmark questions
python task2_rag/build_15_questions.py

# Run chunk size retrieval evaluation (256 vs 512 tokens)
python task2_rag/evaluate_rag.py

# Generate grounded answers with citations using Qwen2.5-0.5B
python task2_rag/generate_qa_answers.py
```

### 3. Task 3: Local Inference Profiling & Ollama
```bash
# Primary PyTorch CUDA Benchmark: FP16 vs 4-bit NF4
python task3_optimization/benchmark_inference.py

# (Optional) Companion Benchmark via Ollama:
# ollama serve
python task3_optimization/run_ollama_benchmark.py
```

---

## Submission Documents

- **Primary Submission Report (Strictly $\le 4$ Pages)**:  
  **[`report/final_report.pdf`](report/final_report.pdf)**  
  *(Generated via Edge headless from `report/final_report.html`, verified with `pypdf` to be exactly 4 pages).*
- **Markdown Report Source**:  
  **[`report/final_report.md`](report/final_report.md)**
- **Full Extended Technical Appendix**:  
  **[`report/technical_appendix.md`](report/technical_appendix.md)**  
  *(Includes comprehensive mathematical proofs, exact memory equations, side-by-side prompt responses, and prompt templates).*

---

## License

This project is licensed under the [MIT License](LICENSE).
