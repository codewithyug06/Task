# Applied Generative AI Engineering & System Optimization: ML Literature Intelligence Platform

This repository contains the end-to-end implementation, empirical benchmark results, and master technical submission report for the 3-task LLM engineering assessment.

---

## 📌 Executive Summary

All three tasks are unified around an integrated **Machine Learning Literature Intelligence & Local Serving Platform** evaluated over landmark AI research literature (*Attention Is All You Need*, *LoRA*, *QLoRA*, *FlashAttention*):

- **Task 1: Domain-Holdout LoRA Fine-Tuning**: Fine-tuned `Qwen2.5-0.5B-Instruct` on paper abstract structured JSON extraction with a **strict domain holdout** (trained on 10 domains; evaluated on 4 strictly unseen domains: Biomedical AI, Robotics, Formal Code Synthesis, Recommender Systems). Overall Macro $F_1$ improved from **0.8515 $\to$ 0.9880**, with numerical result extraction jumping from **0.3600 $\to$ 1.0000** (+177.8% gain) resolving base model distractor vulnerability. Full step-by-step 7B memory derivations provided.
- **Task 2: RAG from Scratch Over Real Academic Literature (101 arXiv Pages)**: Built a pure Python retrieval pipeline (zero LangChain/LlamaIndex) using custom Okapi BM25, FAISS `IndexFlatIP` with `all-MiniLM-L6-v2`, and hybrid score fusion over 101 authentic arXiv pages (51,155 words). Evaluated Setting A (256 tokens) vs Setting B (512 tokens), revealing the **vector dilution effect** where dense Recall@1 collapsed from 40.0% to 26.7% in large chunks.
- **Task 3: Local Optimization & 20-User Serving Plan**: Benchmarked FP16 vs. 4-bit NormalFloat (NF4) quantization locally on an RTX 4050 GPU across 10 identical prompts. NF4 achieved a **$2.16\times$ weight VRAM reduction** (942 MB $\to$ 436 MB). Derived a mathematically rigorous 20-concurrent-user serving plan on a 24 GB GPU with a 17.0 GB KV pool ($7.77\times$ safety buffer with GQA).

---

## 📂 Repository Structure

```
├── data/
│   ├── train.json                     # 500 training samples across 10 ML domains
│   ├── val.json                       # 50 validation samples
│   ├── test.json                      # 50 strictly unseen held-out domain test samples
│   ├── corpus_sections.json           # 101 authentic arXiv pages, 51,155 words from 4 papers
│   ├── corpus_real_arxiv.txt          # Raw text of 101-page academic corpus
│   └── benchmark_15_questions.json    # 15 technical test questions with ground truth excerpts
├── task1_finetune/
│   ├── prepare_data.py                # Domain-holdout dataset generator & distractor injector
│   ├── metrics.py                     # Token-level Precision, Recall, F1 & Macro F1
│   ├── train_lora.py                  # LoRA training pipeline (r=16, alpha=32, 4 epochs)
│   ├── eval_base.py                   # Base model zero-shot evaluation on unseen domains
│   ├── eval_lora.py                   # Fine-tuned model held-out evaluation
│   ├── stress_test_failures.py        # Failure mode stress testing
│   ├── base_model_eval_results.json   # Base model empirical metrics (Macro F1: 0.8515)
│   ├── lora_model_eval_results.json   # Fine-tuned model empirical metrics (Macro F1: 0.9880)
│   ├── failure_analysis_cases.json    # 3 empirical failure edge cases & root causes
│   └── lora_adapter/                  # Saved LoRA adapter weights (safetensors)
├── task2_rag/
│   ├── fetch_real_arxiv_papers.py     # Real PDF download & PyMuPDF 101-page parser
│   ├── rag_engine.py                  # Pure Python Chunker, Scratch BM25, and Hybrid RAG
│   ├── evaluate_rag.py                # Chunk size evaluation (256 vs 512 tokens)
│   ├── generate_qa_answers.py         # End-to-end grounded question answering
│   ├── rag_evaluation_results.json    # Empirical retrieval metrics (Recall@K, MRR@5)
│   └── qa_generation_answers.json     # 15 generated answers with citations
├── task3_optimization/
│   ├── benchmark_inference.py         # FP16 vs 4-bit NF4 hardware benchmark
│   └── inference_benchmark_results.json # Speeds, memory stats & 10 prompt comparisons
├── report/
│   ├── final_report.md                # Strict 4-Page Submission Report (PDF Compliant)
│   └── technical_appendix.md          # Full Extended Technical Appendix & Raw Derivations
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- `transformers`, `peft`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `faiss-cpu`, `pymupdf`

```bash
pip install torch transformers peft accelerate bitsandbytes sentence-transformers faiss-cpu pymupdf
```

### 1. Evaluate Task 1 (LoRA Fine-Tuning & Domain Holdout)
```bash
python task1_finetune/eval_base.py
python task1_finetune/eval_lora.py
python task1_finetune/stress_test_failures.py
```

### 2. Run Task 2 (RAG Retrieval Benchmark over 101-Page arXiv Corpus)
```bash
python task2_rag/evaluate_rag.py
python task2_rag/generate_qa_answers.py
```

### 3. Run Task 3 (Inference & Quantization Benchmark)
```bash
python task3_optimization/benchmark_inference.py
```

---

## 📊 Master Technical Reports

- **Primary Submission Report (Strictly 4 Pages)**:  
  👉 **[report/final_report.md](report/final_report.md)**  
  *(Optimized for PDF rendering, adhering strictly to page limit constraints with clean page breaks).*

- **Full Extended Technical Appendix**:  
  👉 **[report/technical_appendix.md](report/technical_appendix.md)**  
  *(Contains exhaustive mathematical derivations, architectural diagrams, complete prompt transcripts, and extended code listings).*
