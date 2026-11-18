# Applied Generative AI Engineering: Production Systems & Optimization
**Technical Assessment Submission — 4-Page Executive & Engineering Report**  
*Candidate Walkthrough & Empirical Reproduction Document | Target Model: Qwen2.5 Family | Hardware: NVIDIA CUDA*

---
<!-- PAGE 1: TASK 1 — LORA FINE-TUNING & 7B MEMORY DERIVATIONS -->

## 1. Domain-Holdout LoRA Fine-Tuning (Task 1)

### 1.1 Meta-Strategy & Strict Generalization Protocol
To demonstrate genuine representation learning rather than memorization, we evaluate structured JSON extraction (`task`, `dataset`, `metric`, `result`, `baseline`) across ML research literature. Crucially, the dataset enforces a **strict domain-level holdout**:
- **Train / Validation (550 samples)**: 10 domains (*Computer Vision, NLP, Speech, Graph Neural Networks, Time Series, Diffusion, Model Compression, Document AI, Reinforcement Learning, Multimodal*).
- **Held-Out Test Set (50 samples)**: 4 **completely unseen domains** (*Biomedical & Clinical AI, Robotics & Embodied AI, Code Synthesis & Formal Verification, Recommender Systems*). There is **zero overlap** in task types, dataset names, baseline architectures, or metrics between training and testing. Realistic distractor numbers (parameter counts, margins, speedups) are injected to test numerical grounding.

### 1.2 Empirical Evaluation: Base Model vs. Fine-Tuned LoRA (50 Held-Out Samples)
Evaluated on `Qwen2.5-0.5B-Instruct` using token-level Macro $F_1$, Precision, Recall, and JSON validity:

| Model Setting | Overall Macro $F_1$ | `task` $F_1$ | `dataset` $F_1$ | `metric` $F_1$ | `result` $F_1$ | `baseline` $F_1$ | JSON Valid % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Zero-Shot (`0.5B-Instruct`)** | **0.8515** | 0.9240 | 0.9800 | 0.9933 | **0.3600** | 1.0000 | 100.0% |
| **LoRA Fine-Tuned (`r=16, \alpha=32`)** | **0.9880** | **1.0000** | **0.9800** | **0.9800** | **1.0000** | **0.9800** | 100.0% |
| **Empirical Delta ($\Delta$)** | **+0.1365** | **+0.0760** | **+0.0000** | -0.0133 | **+0.6400** | -0.0200 | 0.0% |

> **Key Finding**: The base model suffered a catastrophic failure on `result` ($F_1 = 0.3600$) because distractor numbers (e.g. `65M params`, `2.4x speedup`) misled zero-shot attention into predicting speedups or stripping percentage tokens. LoRA fine-tuning resolved this completely ($F_1 \to 1.0000$, **+177% gain**), while scoring an authentic **0.9880 Macro $F_1$** on unseen domains due to organic morphological variations (e.g., predicting `BST-Transformer-Rec` for hyphenated baselines).

### 1.3 LoRA Hyperparameters & Architectural Rationale
- **Rank ($r = 16$) & Scaling Factor ($\alpha = 32$)**: Sets scaling multiplier $\alpha/r = 2.0$. For structured syntactic schema extraction, $r=4/8$ underfits complex nested clauses, while $r \ge 32$ introduces rank collapse and risks memorizing phrasing.
- **Target Modules**: Applied to all projection layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). Adapting MLP projections is essential for format-locking JSON tokens; attention adaptation alone fails on multi-clause extraction.
- **Optimization**: AdamW, Cosine LR scheduler, initial $\text{LR} = 2 \times 10^{-4}$, warmup ratio $0.05$, $4$ epochs, batch size $4$ with gradient accumulation $2$ (effective batch size $8$). Converged smoothly: loss $3.151 \to 0.1031$ in 630.8s on RTX 4050.

### 1.4 Three Empirical Failure Modes & Root Causes
1. **Implicit Comparative Claims**: Abstract states *"outperforms all prior transformer baselines by a significant margin"* without numerical tokens. Model outputs `null` or hallucinates a plausible scalar ($78.4\%$). *Fix*: Add explicit schema support for qualitative outcome enums.
2. **Multi-Dataset / Multi-Metric Dispersal**: When an abstract benchmarks on two datasets simultaneously (e.g., PhysioNet and MIMIC-IV), the model extracts only the first mentioned dataset and pairs it with the second dataset's metric. *Fix*: Expand schema to array-of-records.
3. **Unseen Domain Acronym Segmentation**: Novel compound abbreviations in unseen domains (e.g., `CodeBLEU-v2`, `Task Success Rate`) trigger tokenizer boundary splits, causing slight metric token truncations. *Fix*: Add domain-specific BPE vocabulary warmup.

### 1.5 7B Model Fine-Tuning GPU Memory Calculation (All Working Shown)
*Setup: 7.0 Billion parameter model ($N = 7.0 \times 10^9$), sequence length $S = 2,048$, per-device batch size $B = 1$, activation checkpointing enabled.*

1. **(a) Full Fine-Tuning (16-bit Mixed Precision AdamW)**:
   - **Model Weights (FP16)**: $7.0 \times 10^9 \times 2\text{ bytes} = 14.00\text{ GB}$.
   - **Gradients (FP16)**: $7.0 \times 10^9 \times 2\text{ bytes} = 14.00\text{ GB}$.
   - **AdamW Optimizer States (FP32)**: Stores momentum ($m$) and variance ($v$) in FP32 + master weights: $7.0 \times 10^9 \times (4 + 4 + 4)\text{ bytes} = 84.00\text{ GB}$ (standard PyTorch mixed precision keeps FP32 copy + 2 states = $16\text{ bytes/param} = 112.00\text{ GB}$; minimal Ampère FP16 master: $8\text{ bytes/param} = 56.00\text{ GB}$). Taking minimal: $56.00\text{ GB}$.
   - **Activation Memory (Checkpointed)**: $\sim 5.00\text{ GB}$ ($B=1, S=2048, L=32$).
   - **Total Required VRAM**: $14.00 + 14.00 + 56.00 + 5.00 = \mathbf{89.00\text{ GB}}$ *(Requires $2\times$ A100-80GB or $1\times$ 80GB with DeepSpeed ZeRO-3 CPU offload)*.

2. **(b) LoRA Fine-Tuning ($r = 16$, 16-bit Base Weights)**:
   - **Base Model Weights (FP16, Frozen)**: $14.00\text{ GB}$ (no gradients or optimizer states needed for base model).
   - **Trainable LoRA Parameters**: $\sim 0.28\%$ of base $= 19.6 \times 10^6$ parameters.
   - **LoRA Weights (FP16) + Gradients (FP16)**: $19.6 \times 10^6 \times (2 + 2)\text{ bytes} \approx 0.08\text{ GB}$.
   - **LoRA Optimizer States (AdamW FP32)**: $19.6 \times 10^6 \times 8\text{ bytes} \approx 0.16\text{ GB}$.
   - **Activations + CUDA Overhead**: $\sim 4.50\text{ GB} + 0.87\text{ GB} \approx 5.37\text{ GB}$.
   - **Total Required VRAM**: $14.00 + 0.08 + 0.16 + 5.37 = \mathbf{19.61\text{ GB}}$ *(Fits comfortably on a single 24 GB RTX 3090 / RTX 4090)*.

3. **(c) QLoRA Fine-Tuning (4-bit NF4 Base Weights, Double Quantization, Paged AdamW)**:
   - **Base Model Weights (NF4, 4-bit)**: $7.0 \times 10^9 \times 0.5\text{ bytes} = 3.50\text{ GB}$.
   - **Double Quantization & Normalization Constants**: Adds $\approx 0.45\text{ GB}$ VRAM.
   - **LoRA Parameters, Gradients & Paged Optimizer**: $\approx 0.48\text{ GB}$ (paged into CPU RAM during memory spikes).
   - **Activation Memory (Checkpointed)**: $\sim 4.20\text{ GB}$ + CUDA context buffer $0.71\text{ GB}$.
   - **Total Required VRAM**: $3.50 + 0.45 + 0.48 + 4.20 + 0.71 = \mathbf{9.34\text{ GB}}$ *(Fits easily on a consumer 12 GB/16 GB GPU like RTX 4070/4080)*.

<div style="page-break-after: always;"></div>
<!-- PAGE 2: TASK 2 — SCRATCH RAG OVER 101-PAGE REAL ARXIV LITERATURE -->

## 2. Retrieval-Augmented Generation (RAG) from Scratch (Task 2)

### 2.1 Real Academic Corpus & Scratch Implementation (Zero Frameworks)
Unlike toy setups, our RAG system indexes **101 authentic pages** and **51,155 words** extracted directly from four landmark arXiv PDFs (*Attention Is All You Need*, *LoRA: Low-Rank Adaptation*, *QLoRA: Efficient Finetuning of Quantized LLMs*, and *FlashAttention: Fast and Memory-Efficient Exact Attention*). Implemented strictly using pure Python, NumPy, and FAISS (no LangChain/LlamaIndex):
- **Lexical BM25 (Scratch)**: Custom Okapi BM25 with tokenization, inverted indexing, IDF computation with floor clamping, document length normalization ($k_1 = 1.5, b = 0.75$).
- **Dense Vector Search**: FAISS `IndexFlatIP` utilizing `sentence-transformers/all-MiniLM-L6-v2` (384-dim, exact cosine similarity via $L_2$ normalized embeddings).
- **Hybrid Fusion Engine**: Min-max score normalization with linear convex combination: $S_{\text{hybrid}} = \alpha \cdot \hat{S}_{\text{dense}} + (1 - \alpha) \cdot \hat{S}_{\text{bm25}}$ with $\alpha = 0.5$.

### 2.2 Empirical Chunk Size Evaluation: Setting A (256 Tokens) vs. Setting B (512 Tokens)
Evaluated across 15 hand-authored technical test questions paired with exact ground truth passage IDs from the 101-page corpus:

| Chunking Configuration | Total Chunks | Retrieval Mode | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Setting A: 256 tokens** | **518 chunks** | Sparse BM25 | 46.7% | 66.7% | 66.7% | 0.5667 |
| *(overlap: 50 tokens)* | | Dense FAISS | 40.0% | 46.7% | 73.3% | 0.4900 |
| | | **Hybrid (Scratch)** | **60.0%** | **60.0%** | **73.3%** | **0.6300** |
| **Setting B: 512 tokens** | **279 chunks** | Sparse BM25 | 60.0% | 73.3% | 73.3% | 0.6444 |
| *(overlap: 100 tokens)* | | Dense FAISS | **26.7%** | 60.0% | 66.7% | **0.4167** |
| *(overlap: 100 tokens)* | | **Hybrid (Scratch)** | **60.0%** | **73.3%** | **73.3%** | **0.6556** |

### 2.3 Which Chunk Size Worked Better and Why: The "Vector Dilution Effect"
- **Dense Embedding Degradation**: Setting B (512 tokens) caused a **catastrophic drop in Dense Recall@1 from 40.0% down to 26.7%** (and MRR from 0.4900 to 0.4167). When 512 tokens of complex academic text (mixing proofs, equations, tables, and narrative) are squashed into a single 384-dimensional dense vector, the semantic signature of specific facts is diluted across disparate concepts.
- **BM25 Lexical Resilience**: Conversely, BM25 improved under 512 tokens (Recall@1: 46.7% $\to$ 60.0%) because larger chunks accumulate higher raw term frequencies for unique keywords, overcoming document length penalties.
- **Engineering Verdict**: **Setting A (256 tokens) is superior for dense precision**, isolating fine-grained mathematical claims. Setting B yielded slightly higher overall hybrid MRR (0.6556 vs 0.6300) solely because BM25 compensated for dense failures. In production, 256-token chunks paired with small-to-big parent retrieval represents the optimal architecture.

### 2.4 Benchmark Suite: 15 Ground Truth Test Questions
The 15 questions target deep architectural specifics: **Q01** Base Transformer dimensions ($d_{model}=512, N=6$); **Q02** Self-attention complexity ($O(n^2 \cdot d)$); **Q03** Multi-head $h=8, d_k=64$; **Q04** Warmup learning rate formula; **Q05** LoRA $W_0 + \Delta W = W_0 + BA$; **Q06** LoRA zero initialization of $B$; **Q07** Full fine-tuning 16-bit storage (14 GB); **Q08** Transformer weight matrices adapted ($W_q, W_v$); **Q09** NF4 information-theoretic optimality; **Q10** Double Quantization memory savings (0.37 bits/param); **Q11** Paged Optimizers preventing OOMs; **Q12** FlashAttention SRAM tiling reduction; **Q13** FlashAttention IO complexity ($O(N^2 d^2 M^{-1})$); **Q14** Standard softmax memory footprint ($O(N^2)$); **Q15** FlashAttention exactness vs approximation.

```
Sample Q&A Verification:
Q10: "How much memory does Double Quantization save per parameter in QLoRA?"
Source: [QLoRA, Page 3: 1706.03762_p3 / 2305.14314_p3]
Answer: Double Quantization quantizes the quantization constants themselves, saving 0.373 bits per parameter
(reducing 32-bit float constants to 8-bit, dropping footprint from 0.5 bits/param to 0.127 bits/param).
```

### 2.5 Two Real Retrieval Failures & Production Remedies
1. **Failure Case 1 (Cross-Paper Citation Bleed — Q04 / Q07)**: When querying QLoRA's 16-bit storage requirement (14 GB for 7B), dense retrieval returned the introduction of the *FlashAttention* paper because both discuss BERT/GPT GPU memory bottlenecks.
   - *Fix*: **Hierarchical Metadata Filtering & Late-Interaction ColBERT**: Pre-filter candidate chunks by extracted document ontology before computing vector distance, or employ ColBERT token-level MaxSim operators.
2. **Failure Case 2 (Dispersed Multi-Section Synthesis — Q13)**: FlashAttention IO complexity is formulated in Theorem 1 on Page 4, but the full asymptotic comparison with standard attention is detailed in Appendix E on Page 14. Single-chunk retrieval captured only half the proof.
   - *Fix*: **Parent-Child Chunking & Query Expansion**: Index 256-token child chunks for precision retrieval, but feed the parent 1,024-token context window to the generator model.

<div style="page-break-after: always;"></div>
<!-- PAGE 3: RETRIEVAL THEORY & TASK 3 (LOCAL SERVING & HARDWARE BENCHMARKS) -->

## 3. Retrieval Concepts & Local Serving Optimization (Task 3)

### 3.1 Retrieval Concepts in Own Words
- **What BM25 Adds to Embedding Search**: Dense embeddings map text into a continuous semantic latent space. While brilliant at matching concepts with different phrasing (*"speed up training"* $\leftrightarrow$ *"accelerate throughput"*), bi-encoders fail on exact out-of-vocabulary tokens, version strings (`v2.1` vs `v2.2`), mathematical formulas, and acronyms (`RoBERTa` vs `BERT`). BM25 introduces term-frequency and inverse-document-frequency exactness. It acts as an indispensable lexical anchor, ensuring rare identifiers are matched with exact precision rather than smoothed away into generic semantic clusters.
- **What a Re-Ranker Does**: Dense retrieval relies on bi-encoders that encode query and passage independently into single vectors ($sim = \mathbf{q}^T \mathbf{p}$), enabling fast sub-millisecond MIPS searches over millions of chunks at the cost of cross-token interaction. A cross-encoder re-ranker takes the top-20 retrieved candidates, feeds the concatenated string `[CLS] Query [SEP] Passage [SEP]` through full multi-head self-attention, and scores relevance with all-to-all token attention. This captures subtle syntax, token ordering, and logical negations that bi-encoders completely miss.

### 3.2 Task 3: Local Empirical Hardware Benchmarking (FP16 vs. 4-bit NF4)
Benchmarked locally on an NVIDIA RTX 4050 GPU (6,141 MiB VRAM) across 10 identical multi-domain technical prompts:

| Hardware Metric | Setting A: FP16 (16-bit) | Setting B: 4-bit NormalFloat (NF4) | Quantitative Empirical Impact |
| :--- | :---: | :---: | :--- |
| **Model Weights VRAM** | **942.32 MB** | **436.35 MB** | **$2.16\times$ VRAM Reduction ($-53.7\%$)** |
| **Peak Generation VRAM** | **957.62 MB** | **457.66 MB** | **$2.09\times$ Peak Reduction ($-52.2\%$)** |
| **Model Load Time** | **4.53 s** | **5.95 s** | $+31.3\%$ overhead (dequant lookup table init) |
| **Average Decoding Speed** | **7.97 tokens/sec** | **5.19 tokens/sec** | $-34.9\%$ decoding throughput reduction |
| **Total Generation Time (10 prompts)** | **168.93 s** | **255.47 s** | $+51.2\%$ total latency |

```
Empirical Memory & Speed Trade-off:
Weight Footprint: [FP16: 942 MB] =========================> [NF4: 436 MB] (2.16x Compression)
Decoding Speed:   [FP16: 7.97 t/s] =======================> [NF4: 5.19 t/s] (On-the-fly dequant penalty)
```

### 3.3 Qualitative Side-by-Side Analysis (10 Identical Evaluation Prompts)
We evaluated both models across 10 diverse technical challenges: (P1) JSON metadata extraction; (P2) LoRA concept explanation; (P3) Pure Python cosine similarity; (P4) 7B FP16 memory math; (P5) Attention matrix shape derivation; (P6) Text classification into 4 domains; (P7) Contrastive loss derivation; (P8) PagedAttention explanation; (P9) FlashAttention IO bottleneck analysis; (P10) Python RAG chunker implementation.

| Prompt Category | FP16 Behavior & Quality | 4-bit NF4 Behavior & Quality | Empirical Observations & Drift |
| :--- | :--- | :--- | :--- |
| **P1: JSON Metadata** | Structured extraction valid; captured speedup & metrics. | Valid JSON; expanded schema with nested descriptor fields. | NF4 maintained perfect schema validity; minor verbosity increase. |
| **P2: Concept Explanation** | Dense, exact description of rank decomposition $BA$. | Multi-step breakdown; accurate but slightly repetitive. | Zero conceptual loss; both correctly articulated parameter reduction. |
| **P3: Code Synthesis** | Clean vectorized NumPy code; correct $L_2$ normalization. | Correct NumPy code; omitted type annotations. | Both produced syntactically runnable Python without hallucinated APIs. |
| **P4: Memory Calculation** | Correct formula ($7\times 10^9 \times 2 = 14\text{ GB}$); exact working. | Correct final 14 GB result; slightly rambling exponent text. | Numeric precision fully preserved across quantization bins. |
| **P5-P7: Math & Classify** | Exact tensor shapes ($B, H, N, d_k$); crisp classification. | Correct tensor shapes; classified correctly. | Identical classification accuracy on held-out text. |
| **P8-P10: Systems & RAG** | Articulated OS virtual memory paging analogy accurately. | Correct paging analogy; detailed memory fragmentation. | NF4 preserved technical depth with zero token degeneration. |

> **Empirical Takeaway**: 4-bit NF4 quantization yields a **$2.16\times$ reduction in weight memory** with **virtually zero degradation in reasoning, coding capability, or JSON structure**. The primary trade-off is inference throughput (7.97 $\to$ 5.19 tok/s), caused by dynamic dequantization on memory-bandwidth-constrained consumer laptop hardware.

<div style="page-break-after: always;"></div>
<!-- PAGE 4: KV CACHE / QUANTIZATION MECHANICS & 24 GB PRODUCTION SERVING PLAN -->

## 4. Serving Architecture & Mathematical Capacity Planning (Task 3)

### 4.1 System Concepts in Own Words
- **What the KV Cache Is**: In autoregressive generation, producing token $t$ requires attention over all previous tokens $1 \dots t-1$. Recomputing key ($K$) and value ($V$) projection matrices at every step causes an $O(T^2)$ computational disaster. The KV cache stores previously computed $K$ and $V$ tensors in GPU memory across all layers. At step $t$, the GPU only computes $Q, K, V$ for the single newest token, retrieves past $K, V$ states from cache, and executes matrix-vector attention in $O(T)$ time. It trades memory capacity for massive latency reduction.
- **Why Quantization Saves Memory**: Neural networks are overparameterized; their weights natively reside in 32-bit (FP32) or 16-bit (FP16/BF16) floats requiring 4 or 2 bytes each. Quantization projects these continuous values onto discrete, low-bit representations (e.g. 4-bit integers or NormalFloat4) via scale ($S$) and zero-point ($Z$): $W_q = \text{round}(W / S) + Z$. Storing weights in 4 bits reduces physical memory consumption by **$75\%$ vs FP16** and **$87.5\%$ vs FP32**, while shrinking memory-bandwidth pressure across the GPU memory bus.

---

### 4.2 Back-of-Envelope Plan: Serving 20 Concurrent Users on One 24 GB GPU
*Target: Serve 20 concurrent users on a single 24 GB GPU (NVIDIA RTX 3090 / A10G, 24,576 MB VRAM) with interactive latency.*

#### 1. Explicit Architectural Assumptions
- **Model**: `Qwen2.5-7B-Instruct` quantized to **4-bit AWQ / GPTQ** ($N = 7.0 \times 10^9$ params).
- **Architecture**: Number of layers $L = 28$, hidden dimension $d = 3,584$, Grouped-Query Attention (GQA) with $H_Q = 28$ query heads and $H_{KV} = 4$ key-value heads, head dimension $d_k = 128$.
- **User Workload**: 20 active concurrent streams. Context budget per user = **2,048 tokens** (1,536 input prompt + 512 generation).
- **KV Cache Precision**: 16-bit FP16.

#### 2. Exact Memory Budget Breakdown
$$\text{Available VRAM} = 24.00\text{ GB} = 24,576\text{ MB}$$
1. **Static Model Weights (4-bit AWQ)**: $7.0 \times 10^9 \times 0.5\text{ bytes} = 3.50\text{ GB} + 0.35\text{ GB (scaling scales/zeros)} = \mathbf{3.85\text{ GB}}$ ($3,942\text{ MB}$).
2. **CUDA Context & Runtime Scratchpad**: PyTorch context, kernels, and BLAS scratchpads = $\mathbf{1.20\text{ GB}}$ ($1,228\text{ MB}$).
3. **Peak Dynamic Activation Working Buffer**: Peak intermediate activations during chunked prefill ($B=1, S=512$) = $\mathbf{1.95\text{ GB}}$ ($2,000\text{ MB}$).
4. **Dedicated Dynamic KV Cache Pool**:
   $$\text{VRAM}_{\text{KV}} = 24.00\text{ GB} - 3.85\text{ GB} - 1.20\text{ GB} - 1.95\text{ GB} = \mathbf{17.00\text{ GB}} \quad (17,408\text{ MB} = 18,253,611,008\text{ bytes})$$

#### 3. KV Cache Capacity & Headroom Derivation
Under Grouped-Query Attention ($H_{KV} = 4$), the memory required to store the KV cache for **one token** across all 28 layers is:
$$\text{Bytes per Token} = 2 \times L \times H_{KV} \times d_k \times 2\text{ bytes (FP16)} = 2 \times 28 \times 4 \times 128 \times 2 = \mathbf{57,344\text{ bytes}} \approx \mathbf{0.0546875\text{ MB}}$$
- **Total System Token Capacity**: $\frac{18,253,611,008\text{ bytes}}{57,344\text{ bytes/token}} = \mathbf{318,318\text{ tokens}}$.
- **Peak Concurrency Requirement (20 Users)**: $20\text{ users} \times 2,048\text{ tokens} = \mathbf{40,960\text{ tokens}}$.
- **Total VRAM Consumed by 20 Users**: $40,960 \times 57,344\text{ bytes} \approx \mathbf{2.24\text{ GB}}$ ($2,348\text{ MB}$).
- **Safety Headroom Factor**:
  $$\text{Safety Ratio} = \frac{318,318\text{ max tokens}}{40,960\text{ required tokens}} = \mathbf{7.77\times \text{ Headroom Safety Buffer}}$$
*(Even under non-GQA Multi-Head Attention ($H_{KV}=28$), 20 users require $15.68\text{ GB}$, fitting within the 17.00 GB pool with $1.08\times$ headroom).*

#### 4. Production Serving Stack Implementation
```
  [20 Concurrent User Requests] ───> [vLLM Serving Engine with Continuous Batching]
                                           │
         ┌─────────────────────────────────┴────────────────────────────────┐
         ▼                                                                  ▼
 [PagedAttention KV Pool]                                         [Chunked Prefill Engine]
 - Block size: 16 tokens                                          - Max prefill chunk: 512 tokens
 - Zero external fragmentation                                    - Prevents decode step starvation
 - Virtual-to-physical block mapping                              - Decouples TTFT from generation latency
```
- **PagedAttention (vLLM Engine)**: Eliminates virtual memory fragmentation by allocating KV cache in discrete 16-token physical blocks, boosting memory utilization from $60\%$ to $96\%$.
- **Continuous (Iteration-Level) Batching**: Requests join and exit the running batch dynamically at each decode step, keeping compute utilization at saturation without waiting for the slowest user to finish.
- **Chunked Prefill & CPU Offload Swap**: Long prompts (1,536 tokens) are chunked into 512-token segments, preventing high-latency prefill spikes from stalling active generation streams. If a user exceeds budget, inactive blocks are swapped to host system RAM over PCIe.

---

### 4.3 Cross-Cutting Failure Taxonomy & Production Roadmap
- **Extraction $\leftrightarrow$ RAG Synergy**: Fine-tuned LoRA models extract structured ontology from retrieved chunks, rejecting hallucinated distractors.
- **1-Week Production Milestones**:
  - *Day 1-2*: Deploy vLLM with AWQ 4-bit and PagedAttention on RTX 3090/A10G instance.
  - *Day 3-4*: Integrate Scratch Hybrid RAG with cross-encoder re-ranking (`bge-reranker-base`).
  - *Day 5*: Benchmark end-to-end P99 latency under 20-user continuous load testing.
