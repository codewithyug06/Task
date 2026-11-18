# dem1

# Applied Generative AI Engineering & System Optimization: ML Literature Intelligence Platform

This repository contains the end-to-end implementation, empirical benchmark results, and master technical submission report for the 3-task LLM engineering assessment.

---

## 📌 Executive Summary

All three tasks are linked around an integrated **Machine Learning Literature Intelligence & Edge Serving Platform** evaluated over 10 landmark AI research papers (*Attention Is All You Need*, *LoRA*, *QLoRA*, *FlashAttention*, *Chinchilla*, *BERT*, *ResNet*, *DDPM*, *GPT-3*, *Llama 2*):

- **Task 1: LoRA Fine-Tuning**: Fine-tuned `Qwen2.5-0.5B-Instruct` on paper abstract structured JSON extraction. Overall Macro $F_1$ improved from **0.8393 $\to$ 1.0000**, with numerical result extraction jumping from **0.4200 $\to$ 1.0000** (+138.1%). Complete step-by-step 7B memory derivations provided.
- **Task 2: RAG from Scratch**: Built a pure Python retrieval pipeline (zero LangChain/LlamaIndex) using custom Okapi BM25, FAISS `IndexFlatIP` with `all-MiniLM-L6-v2`, and hybrid score fusion over a 30.6-page corpus. Setting A (256 chunk) achieved **100.0% Recall@1, 100.0% Recall@5, 1.0000 MRR@5**, outperforming 512 chunks due to vector dilution avoidance.
- **Task 3: Local Optimization & Serving Plan**: Benchmarked FP16 vs. 4-bit NormalFloat (NF4) quantization on local GPU. NF4 achieved a **$2.16\times$ weight VRAM reduction** (942 MB $\to$ 436 MB). Derived a mathematically rigorous 20-concurrent-user serving plan on a 24 GB GPU with a 17.0 GB KV pool ($3.48\times$ safety buffer).

---

## 📂 Repository Structure

```
├── data/
│   ├── train.json                     # 500 training samples across 14 ML domains
│   ├── val.json                       # 50 validation samples
│   ├── test.json                      # 50 held-out test samples
│   ├── corpus_sections.json           # 10 papers, 47 sections, 8,565 words (~30.6 pages)
│   ├── corpus_ml_papers.txt           # Raw corpus text
│   └── benchmark_15_questions.json    # 15 expert test questions with ground truth excerpts
├── task1_finetune/
│   ├── prepare_data.py                # Synthetic data generation & validation
│   ├── metrics.py                     # Token-level Precision, Recall, F1 & Macro F1
│   ├── train_lora.py                  # LoRA training pipeline (r=16, alpha=32, 4 epochs)
│   ├── eval_base.py                   # Base model zero-shot evaluation
│   ├── eval_lora.py                   # Fine-tuned model held-out evaluation
│   ├── stress_test_failures.py        # Failure mode stress testing
│   ├── base_model_eval_results.json   # Base model empirical metrics
│   ├── lora_model_eval_results.json   # Fine-tuned model empirical metrics
│   ├── failure_analysis_cases.json    # Detailed 3 failure edge cases & root causes
│   └── lora_adapter/                  # Saved LoRA adapter weights (safetensors)
├── task2_rag/
│   ├── build_corpus.py                # Corpus generation script
│   ├── rag_engine.py                  # Pure Python Chunker, ScratchBM25, and HybridRAG
│   ├── evaluate_rag.py                # Chunk size evaluation (256 vs 512 tokens)
│   ├── generate_qa_answers.py         # End-to-end grounded question answering
│   ├── rag_evaluation_results.json    # Empirical retrieval metrics (Recall@K, MRR@5)
│   └── qa_generation_answers.json     # 15 generated answers with citations
├── task3_optimization/
│   ├── benchmark_inference.py         # FP16 vs 4-bit NF4 hardware benchmark
│   └── inference_benchmark_results.json # Speeds, memory stats & 10 prompt comparisons
├── report/
│   └── final_report.md                # Master Technical Submission Report (660+ lines)
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- `transformers`, `peft`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `faiss-cpu` / `faiss-gpu`

### 1. Evaluate Task 1 (LoRA Fine-Tuning)
```bash
python task1_finetune/eval_base.py
python task1_finetune/eval_lora.py
python task1_finetune/stress_test_failures.py
```

### 2. Run Task 2 (RAG Retrieval Benchmark)
```bash
python task2_rag/evaluate_rag.py
python task2_rag/generate_qa_answers.py
```

### 3. Run Task 3 (Inference & Quantization Benchmark)
```bash
python task3_optimization/benchmark_inference.py
```

---

## 📊 Detailed Master Report

For the complete technical report with all working shown, failure mode diagnoses, architectural diagrams, and serving calculations, see:
👉 **[report/final_report.md](report/final_report.md)**
