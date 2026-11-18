# Applied Generative AI Engineering & System Optimization: ML Literature Platform

This repository contains the complete implementation, interactive Jupyter/Colab notebooks, empirical benchmark results, and master technical submission report for the 3-task LLM engineering assessment.

---

## 📌 Executive Summary

All three tasks are unified around an integrated **Machine Learning Literature Intelligence & Local Serving Platform** evaluated over 10 landmark AI research papers (*Attention Is All You Need*, *LoRA*, *QLoRA*, *FlashAttention*, *ResNet*, *BERT*, *Chinchilla*, *GPT-3*, *Llama 2*, *DDPM*):

- **Task 1: Domain-Holdout LoRA Fine-Tuning**: Fine-tuned `Qwen2.5-0.5B-Instruct` on paper abstract structured JSON extraction with a **strict domain holdout** (trained on 8 domains; evaluated on 4 strictly unseen domains: *Speech Recognition, Code Generation, Reinforcement Learning, Protein Structure Prediction*). Overall Macro $F_1$ improved from **0.9011 $\to$ 0.9832** (+0.0821 gain), with `task` $F_1$ jumping from **0.7854 $\to$ 0.9219** and numerical `result` extraction jumping from **0.8394 $\to$ 0.9971** overcoming numeric distractor tokens. Full 7B GPU fine-tuning memory calculations provided.
- **Task 2: RAG from Scratch Over Real Academic Literature (335 arXiv Pages)**: Built a pure Python retrieval pipeline (zero LangChain/LlamaIndex) using custom Okapi BM25, FAISS `IndexFlatIP` with `all-MiniLM-L6-v2`, and hybrid score fusion over 335 authentic arXiv pages (169,731 words). Evaluated Setting A (256 tokens, 1,732 chunks) vs Setting B (512 tokens, 913 chunks), demonstrating the **vector dilution effect** where dense Recall@1 degraded from 46.7% to 40.0% on larger chunks.
- **Task 3: Local Optimization & 20-User Serving Plan**: Benchmarked FP16 vs. 4-bit NormalFloat (NF4) quantization locally on an RTX 4050 GPU across 10 identical prompts. NF4 achieved a **$2.16\times$ weight VRAM reduction** (942 MB $\to$ 436 MB). Derived a mathematically rigorous 20-concurrent-user serving plan on a 24 GB GPU with a 17.0 GB KV pool ($7.77\times$ safety buffer with GQA).

---

## 📓 Interactive Jupyter / Google Colab / Kaggle Notebooks

All tasks are packaged into self-contained, reproducible Jupyter Notebooks in the [`notebooks/`](notebooks/) directory:

| Notebook | Focus Area | Key Features & Output |
| :--- | :--- | :--- |
| **[`Applied_GenAI_Master_Submission.ipynb`](notebooks/Applied_GenAI_Master_Submission.ipynb)** | **All-in-One Master Pipeline** | End-to-end reproduction of Task 1, Task 2, and Task 3 in a single executable Colab/Kaggle notebook. |
| **[`Task1_LoRA_Finetuning.ipynb`](notebooks/Task1_LoRA_Finetuning.ipynb)** | **LoRA Fine-Tuning & Memory** | Zero-leakage domain-holdout split, base zero-shot baseline, LoRA training, checkpoint saving, 3 failure modes, 7B memory math. |
| **[`Task2_RAG_From_Scratch.ipynb`](notebooks/Task2_RAG_From_Scratch.ipynb)** | **Scratch RAG Engine** | 10 arXiv PDF download/parsing (335 pages), pure Python BM25 + FAISS IndexFlatIP, 256 vs 512 chunk comparison, QA citations. |
| **[`Task3_Local_Optimization_and_Serving.ipynb`](notebooks/Task3_Local_Optimization_and_Serving.ipynb)** | **Serving & Quantization** | FP16 vs NF4 profiling on GPU, 10-prompt side-by-side analysis, KV cache / quantization theory, 24 GB serving derivation. |

---

## 📂 Repository Structure

```
├── notebooks/
│   ├── Applied_GenAI_Master_Submission.ipynb  # All-in-one reproduction notebook
│   ├── Task1_LoRA_Finetuning.ipynb            # Task 1 standalone notebook
│   ├── Task2_RAG_From_Scratch.ipynb           # Task 2 standalone notebook
│   └── Task3_Local_Optimization_and_Serving.ipynb # Task 3 standalone notebook
├── data_fixed/
│   ├── train.json                     # 500 training samples across 8 ML domains
│   ├── val.json                       # 50 validation samples
│   └── test.json                      # 50 strictly unseen held-out domain test samples
├── data/
│   ├── corpus_real_sections.json      # 335 authentic arXiv pages from 10 papers
│   ├── corpus_real.txt                # Raw text of 335-page academic corpus (169k words)
│   ├── benchmark_15_questions.json    # 15 technical test questions with ground truth excerpts
│   └── arxiv_pdfs/                    # 10 downloaded original arXiv PDFs
├── task1_finetune/
│   ├── prepare_data_fixed.py          # Domain-holdout dataset generator & distractor injector
│   ├── metrics.py                     # Token-level Precision, Recall, F1 & Macro F1
│   ├── train_lora.py                  # LoRA training pipeline (r=16, alpha=32, 4 epochs)
│   ├── eval_base.py                   # Base model zero-shot evaluation on unseen domains
│   ├── eval_lora.py                   # Fine-tuned model held-out evaluation
│   ├── stress_test_failures.py        # Failure mode stress testing
│   ├── base_model_eval_results.json   # Base model empirical metrics (Macro F1: 0.9011)
│   ├── lora_model_eval_results.json   # Fine-tuned model empirical metrics (Macro F1: 0.9832)
│   ├── failure_analysis_cases.json    # 3 empirical failure edge cases & root causes
│   └── lora_adapter/                  # Saved LoRA adapter weights & checkpoints
├── task2_rag/
│   ├── build_corpus_real.py           # Real 10-PDF download & PyMuPDF/pypdf parser
│   ├── build_15_questions.py          # Verification script for 15 technical questions
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
```bash
pip install torch transformers peft datasets accelerate bitsandbytes sentence-transformers faiss-cpu pypdf pymupdf nbformat
```

### 1. Evaluate Task 1 (LoRA Fine-Tuning & Domain Holdout)
```bash
python task1_finetune/prepare_data_fixed.py
python task1_finetune/eval_base.py
python task1_finetune/train_lora.py
python task1_finetune/eval_lora.py
python task1_finetune/stress_test_failures.py
```

### 2. Run Task 2 (RAG Retrieval Benchmark over 335-Page arXiv Corpus)
```bash
python task2_rag/build_corpus_real.py
python task2_rag/build_15_questions.py
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
