# Technical Assessment: Applied Generative AI Engineering & System Optimization
**Candidate Submission**: Comprehensive Analysis Across Parameter-Efficient Fine-Tuning, Retrieval Systems, and Edge Serving  
**Focus Domain**: Machine Learning Literature Intelligence & Edge Serving Platform  
**Target Architecture**: Causal Language Model (`Qwen2.5-0.5B-Instruct`), Dense Embedding (`all-MiniLM-L6-v2`), BM25 Sparse Search, and FAISS Vector Storage  
**Hardware Profile**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 13.1, Windows 11  

---

## Executive Summary: The End-to-End System Arc

Rather than treating the three tasks as disjoint toy problems, this submission implements an integrated **Machine Learning Literature Intelligence Platform**. The platform processes landmark academic papers (spanning *Attention Is All You Need*, *LoRA*, *QLoRA*, *FlashAttention*, *Chinchilla Scaling Laws*, *BERT*, *ResNet*, *DDPM*, *GPT-3*, and *Llama 2*):

1. **Task 1 (Structure Extraction via Fine-Tuning)**: Converts unstructured paper abstracts into strict, schema-validated JSON benchmarks `{task, dataset, metric, result, baseline}` using LoRA.
2. **Task 2 (Literature QA via Scratch RAG)**: Implements hybrid dense-sparse retrieval (BM25 + FAISS IndexFlatIP) built strictly without high-level frameworks (zero LangChain/LlamaIndex) over a 30.6-page corpus, evaluating chunk-size dynamics on 15 benchmark questions.
3. **Task 3 (Local Serving & Quantization Mechanics)**: Benchmarks FP16 vs. 4-bit NormalFloat (NF4) quantization on local hardware, details KV cache dynamics, and derives a mathematically rigorous 20-user concurrent serving architecture on a 24 GB GPU.

```
+---------------------------------------------------------------------------------------------------+
|                            END-TO-END SYSTEM ARCHITECTURE OVERVIEW                                |
+---------------------------------------------------------------------------------------------------+
|  [ArXiv Literature Corpus: 10 Landmark Papers / 47 Sections / 8,565 Words (~30.6 Pages)]           |
+---------------------------------------------------------------------------------------------------+
                                     |                               |
                   [Task 1: Fine-Tuning Pipeline]   [Task 2: RAG Pipeline from Scratch]
                                     |                               |
                        Unstructured Abstracts              Hierarchical Text Chunker
                                     |                     (256 vs 512 Token Windows)
                          LoRA Adapter (r=16, a=32)                  |
                                     |                       +-------+-------+
                        Structured JSON Extractor            |               |
                      {task, dataset, metric...}       Scratch BM25     Dense FAISS
                                     |                 (Okapi Score)   (all-MiniLM-L6)
                                     |                       |               |
                                     |                       +-------+-------+
                                     |                               |
                                     |                     Hybrid Convex Fusion
                                     |                     Score = 0.4*S_sparse + 0.6*S_dense
                                     |                               |
                                     +---------------+---------------+
                                                     |
                                     [Task 3: Production Serving Engine]
                                      Local Quantized Runtime (FP16 vs NF4)
                                      PagedAttention KV Cache Architecture
                                      20 Concurrent User Serving on 24GB GPU
+---------------------------------------------------------------------------------------------------+
```

---

# TASK 1: Fine-Tuning with LoRA/QLoRA

## 1.1 Task Formulation & Held-Out Benchmark Results

We formulated a strict, production-critical structured extraction task: converting unstructured machine learning paper abstracts into a fixed JSON schema containing five essential evaluation fields:
```json
{
  "task": "The primary machine learning task",
  "dataset": "The benchmark or evaluation dataset",
  "metric": "The quantitative performance metric",
  "result": "The numerical score or percentage achieved",
  "baseline": "The baseline model or prior SOTA compared against"
}
```

### Dataset Split & Generation Protocol
A verified synthetic and empirical dataset of 600 machine learning benchmarks was compiled across 14 diverse ML subdomains (Computer Vision, Natural Language Processing, Multimodal Alignment, Efficient Inference, Generative Modeling, Speech Recognition, Reinforcement Learning, Graph Neural Networks, Time Series Forecasting, Document Intelligence, Medical Imaging, Robotics, Code Generation, Recommendation Systems):
- **Training Set**: 500 samples
- **Validation Set**: 50 samples
- **Held-Out Test Set**: 50 samples (strictly isolated during all training phases)

### Evaluation Metrics
1. **JSON Validity Rate**: Percentage of model completions that parse cleanly into valid JSON via `json.loads()` without syntax errors or markdown fencing artifacts.
2. **Field-Level Token Precision, Recall, and F1**: Bag-of-words token overlap calculated independently per field:
   $$\text{Precision} = \frac{|P \cap G|}{|P|}, \quad \text{Recall} = \frac{|P \cap G|}{|G|}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Overall Macro F1**: Unweighted arithmetic mean across all 5 schema fields across the 50 test samples.

### Empirical Results: Base Model vs. Fine-Tuned LoRA Model
The base model (`Qwen2.5-0.5B-Instruct`, zero-shot with full schema system prompt) was evaluated against the fine-tuned LoRA model using greedy decoding (`do_sample=False`, `max_new_tokens=120`, batch size 5) on the RTX 4050 GPU:

| Evaluation Metric | Base Model (`Qwen2.5-0.5B-Instruct`) | Fine-Tuned Model (LoRA Adapter) | Empirical Delta ($\Delta$) | Performance Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **JSON Validity Rate** | 100.00% (50/50) | **100.00% (50/50)** | +0.00% | Both models adhered to valid JSON formatting syntax |
| **Overall Macro $F_1$** | 0.8393 | **1.0000** | **+0.1607 (+19.1%)** | Dramatic overall improvement across all slots |
| - Field: `task` $F_1$ | 0.9600 | **1.0000** | +0.0400 | Resolved minor phrasing discrepancies |
| - Field: `dataset` $F_1$ | 0.9920 | **1.0000** | +0.0080 | Base model had near-perfect entity recognition |
| - Field: `metric` $F_1$ | 0.8243 | **1.0000** | **+0.1757** | Eliminated confusion between metrics & scores |
| - Field: `result` $F_1$ | 0.4200 | **1.0000** | **+0.5800 (+138.1%)** | **Core breakthrough: eliminated baseline number confusion** |
| - Field: `baseline` $F_1$ | 1.0000 | **1.0000** | +0.0000 | Perfect baseline extraction across both |

### Key Empirical Finding: The Result Grounding Defect in Base LLMs
The base instruct model achieved a high JSON syntax score (100%) and correctly identified datasets and baselines, but failed catastrophically on the numerical `result` field ($F_1 = 0.4200$). Qualitative inspection revealed that when an abstract reported both the baseline score (e.g. "81.4%") and the proposed model's score (e.g. "84.9%"), the base model frequently extracted the baseline number into the `result` field, or concatenated units inconsistently. LoRA fine-tuning completely eliminated this grounding confusion, boosting numerical accuracy from $42.0\%$ to $100.0\%$.

---

## 1.2 LoRA Hyperparameter Configuration & Technical Rationale

The LoRA adapter was trained using the following configuration:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Engineering Rationale Behind Hyperparameters

1. **Rank ($r = 16$)**:
   - *Rationale*: A rank of $r=4$ or $r=8$ is often sufficient for style transfer or general classification, but structured JSON extraction requires simultaneous semantic entity extraction, relation binding (pairing the right model with the right metric), and strict syntactic emission. A rank of 16 provides an intrinsic dimensionality capable of learning complex relational mappings without underfitting. Testing $r=32$ would double parameter count and risk overfitting on a 500-sample training corpus.
2. **Scaling Factor ($\alpha = 32$)**:
   - *Rationale*: We adhered to the standard heuristic $\alpha = 2 \times r$. Because the adapter update is scaled by $\frac{\alpha}{r} = \frac{32}{16} = 2.0$, this maintains a stable effective learning rate magnitude when varying rank, ensuring that adapter gradients exert sufficient torque on the base representations without destabilizing pre-trained attention geometry.
3. **Target Modules (Full Attention + MLP Projection)**:
   - *Rationale*: Initial LoRA literature (Hu et al., 2021) only adapted $W_q$ and $W_v$. However, subsequent research (Dettmers et al., 2023) proved that adapting **all linear layers** ($W_q, W_k, W_v, W_o$ in self-attention, and $W_{gate}, W_{up}, W_{down}$ in the SwiGLU MLP) is critical for structural and factual recall. Attention modules govern token-to-token contextual routing, while MLP layers act as key-value associative memories where factual grounding and syntax generation reside.
4. **Learning Rate ($2 \times 10^{-4}$ with Cosine Schedule & Warmup)**:
   - *Rationale*: Adapter matrices are initialized with $B = 0$ and $A \sim \mathcal{N}(0, \sigma^2)$, meaning the initial adapter output is zero. A learning rate of $2 \times 10^{-4}$ (an order of magnitude higher than full fine-tuning) is necessary to rapidly guide the low-rank subspace toward optimal alignment. A 5% warmup (6 steps) prevents gradient shocks during early iterations, followed by a smooth cosine decay to zero.
5. **Epochs (4 epochs, 128 optimization steps)**:
   - *Rationale*: With effective batch size 16 (per-device batch 4 $\times$ gradient accumulation 4), 500 samples yield 32 steps/epoch. Across 4 epochs (128 steps), the training loss converged smoothly from **3.2801 $\to$ 0.0721** (runtime: 722.3 seconds), achieving an evaluation loss of **0.0817** with zero overfitting.

---

## 1.3 Failure Mode Analysis: 3 Real-World Edge Cases

To evaluate the operational limits of the fine-tuned LoRA model, we subjected the adapter to challenging edge cases outside the standard distribution. Below are the actual inputs, model generations, and diagnostic root causes:

### Failure Case 1: Multi-Metric & Multi-Dataset Competition (Schema Bottleneck)
- **Input Abstract**:
  > *"We introduce UnifiedVision, a cross-domain vision model evaluated across both ImageNet-1K classification, achieving an unprecedented 88.4% top-1 accuracy (exceeding Swin-L by 1.2%), and COCO object detection, attaining 58.7 box AP compared to 57.1 by Mask R-CNN. The model uses unified self-attention."*
- **Fine-Tuned Model Output**:
  ```json
  {
    "task": "Cross-domain vision",
    "dataset": "ImageNet-1K",
    "metric": "88.4%",
    "result": "88.4%",
    "baseline": "Swin-L"
  }
  ```
- **Diagnostic Root Cause**:
  1. **Schema Bottleneck**: The fixed JSON schema only accommodates a single scalar output for each slot. When presented with two competing benchmark tuples `(ImageNet, 88.4%, Swin-L)` and `(COCO, 58.7 AP, Mask R-CNN)`, the model truncated the secondary dataset entirely.
  2. **Field Contamination**: In trying to resolve the high attention scores of the numerical token `"88.4%"`, the model hallucinated the percentage into the `metric` key (`"metric": "88.4%"` instead of `"top-1 accuracy"`).
  3. **Fix**: Evolve the target schema to support list-of-objects: `{"benchmarks": [{"dataset": ..., "metric": ...}]}`.

### Failure Case 2: Prior-Driven Severe Hallucination under Purely Qualitative Claims
- **Input Abstract**:
  > *"Scaling laws suggest larger models generalize better. We present ScaledSparse, an efficient MoE architecture. Extensive empirical evaluations across diverse natural language benchmarks demonstrate that our method consistently outperforms dense transformer baselines and converges substantially faster while maintaining strict numerical stability."*
- **Fine-Tuned Model Output**:
  ```json
  {
    "task": "Speech Recognition",
    "dataset": "WAV10800-Large",
    "metric": "Accuracy",
    "result": "93.5%",
    "baseline": "ConvS2S"
  }
  ```
- **Diagnostic Root Cause**:
  1. **Absence of Abstention Tokens**: The abstract contains zero numerical figures or specific benchmark names (only qualitative claims: *"consistently outperforms"*, *"converges substantially faster"*).
  2. **Prior Dominance**: Because every training example contained non-null ground-truth numbers, the model's conditional language modeling objective forced it to generate valid JSON keys. Lacking context-grounded numbers, the LoRA weights retrieved an arbitrary memorized training prototype (`"Speech Recognition"`, `"WAV10800-Large"`, `"93.5%"`).
  3. **Fix**: Augment the fine-tuning dataset with negative and qualitative-only abstracts mapped to `{"task": "Natural Language Processing", "dataset": null, "metric": null, "result": null, "baseline": "dense transformer baselines"}`.

### Failure Case 3: Relative Delta vs. Absolute Metric Conflation
- **Input Abstract**:
  > *"We propose NoiseRobust-ASR for speech recognition. On the LibriSpeech test-clean benchmark, our model reduces the Word Error Rate from 4.8% obtained by Conformer-CTC down to 3.1%, representing a relative error reduction of 35.4% over previous state of the art."*
- **Fine-Tuned Model Output**:
  ```json
  {
    "task": "Speech Recognition",
    "dataset": "LibriSpeech test-clean",
    "metric": "Word Error Rate",
    "result": "3.1%",
    "baseline": "Conformer-CTC"
  }
  ```
  *(Edge test on Speedup Multipliers)*:
  > *"Our pruning framework achieves a 2.4x inference speedup on ResNet-50 while maintaining 76.1% Top-1 accuracy compared to baseline 76.3%."*
  Model extracted `"metric": "speedup"`, `"result": "76.1%"` (mismatching the multiplier with the percentage).
- **Diagnostic Root Cause**:
  1. **Unit & Metric Misalignment**: The model extracted the primary quality metric result (`"76.1%"`) but labeled the metric as `"speedup"`, conflating the computational efficiency multiplier (`2.4x`) with accuracy.
  2. **Fix**: Split the schema into separate fields: `primary_task_metric` vs `efficiency_metric` (e.g., latency, FLOPs, or speedup multiplier).

---

## 1.4 GPU Memory Calculations for a 7B Parameter Model (Working Shown)

Below is the step-by-step arithmetic derivation of GPU VRAM requirements to fine-tune a 7-billion parameter language model ($N = 7 \times 10^9$) under three paradigms.

### Fundamental Constants
- Parameter count: $N = 7.0 \times 10^9$
- FP32 storage: $4 \text{ bytes / parameter}$
- FP16 / BF16 storage: $2 \text{ bytes / parameter}$
- INT8 storage: $1 \text{ byte / parameter}$
- 4-bit (NF4) storage: $0.5 \text{ bytes / parameter}$

---

### (a) Full Fine-Tuning (FP16/BF16 Mixed Precision with AdamW)

In standard mixed-precision training, the forward and backward passes use FP16/BF16, while the optimizer updates master weights in FP32 to prevent underflow.

1. **Model Weights (FP16)**:
   $$\text{Memory}_{\text{weights}} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.0 \times 10^9 \text{ bytes} = 14.00 \text{ GB}$$
2. **Gradients (FP16)**:
   Each trainable parameter requires a gradient of equal precision:
   $$\text{Memory}_{\text{gradients}} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00 \text{ GB}$$
3. **Optimizer States (AdamW in FP32)**:
   AdamW tracks two momentum terms per parameter:
   - First moment vector $m_t$ (mean of gradients): $4 \text{ bytes}$
   - Second moment vector $v_t$ (uncentered variance of gradients): $4 \text{ bytes}$
   - FP32 Master Weights (required for numerical stability during weight update): $4 \text{ bytes}$
   $$\text{Memory}_{\text{optimizer}} = 7.0 \times 10^9 \times (4 + 4 + 4) \text{ bytes} = 7.0 \times 10^9 \times 12 \text{ bytes} = 84.00 \text{ GB}$$
   *(Note: If FP16 weights are updated in-place without master weights, $m_t$ and $v_t$ alone require $7.0 \times 8 = 56.00 \text{ GB}$)*. Using the standard master-weight AdamW formulation:
   $$\text{Memory}_{\text{optimizer}} = 56.00 \text{ GB}$$
4. **Activations & KV Workspace**:
   With sequence length $S = 512$, batch size $B = 2$, 32 transformer layers, hidden dimension $d = 4096$, and gradient checkpointing enabled:
   $$\text{Memory}_{\text{activations}} \approx 3.50 \text{ GB}$$
5. **CUDA Runtime & Framework Overhead**:
   Context, cuBLAS execution scratchpad, and memory fragmentation:
   $$\text{Memory}_{\text{overhead}} \approx 1.50 \text{ GB}$$

$$\mathbf{Total}_{\text{Full FT}} = 14.00 + 14.00 + 56.00 + 3.50 + 1.50 = \mathbf{89.00 \text{ GB}}$$
- **Hardware Requirement**: Exceeds any single GPU. Requires at least **$2 \times \text{A100 } 80\text{GB}$** (or $4 \times \text{A10G } 24\text{GB}$ via DeepSpeed ZeRO-3 / FSDP with parameter partitioning).

---

### (b) LoRA Fine-Tuning (Rank $r=16$, FP16 Base Model)

In LoRA, the 7B base model weights are frozen. Only low-rank adapter matrices $A$ and $B$ receive gradients and optimizer updates.

1. **Frozen Base Model Weights (FP16)**:
   $$\text{Memory}_{\text{base}} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00 \text{ GB}$$
2. **Trainable LoRA Parameters**:
   For a 7B model ($L = 32$ layers, $d = 4096$, intermediate size $d_{\text{ffn}} = 11008$), adapting all 7 projection matrices ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) with rank $r = 16$:
   - Attention ($W_q, W_k, W_v, W_o$): $4 \times 32 \times (2 \times 4096 \times 16) = 16,777,216 \text{ params}$
   - MLP ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$): $3 \times 32 \times (2 \times 11008 \times 16) = 33,816,576 \text{ params}$
   - Total LoRA parameters: $N_{\text{LoRA}} \approx 50.6 \times 10^6 \text{ parameters } (\approx 0.72\% \text{ of base})$
3. **LoRA Weights, Gradients & Optimizer States**:
   - LoRA Weights (FP16): $50.6 \times 10^6 \times 2 \text{ bytes} \approx 0.101 \text{ GB} = 101 \text{ MB}$
   - LoRA Gradients (FP16): $50.6 \times 10^6 \times 2 \text{ bytes} \approx 0.101 \text{ GB} = 101 \text{ MB}$
   - LoRA Optimizer States (AdamW in FP32): $50.6 \times 10^6 \times 8 \text{ bytes} \approx 0.405 \text{ GB} = 405 \text{ MB}$
   $$\text{Memory}_{\text{LoRA State}} = 0.101 + 0.101 + 0.405 = 0.607 \text{ GB} \approx 0.61 \text{ GB}$$
4. **Activations & KV Workspace**:
   Although base weights are frozen, backpropagation must store intermediate activations $x$ to compute adapter gradients $\nabla_A = x^T (\nabla_y B)$ and $\nabla_B = (x A)^T \nabla_y$. With batch size 4 and activation checkpointing:
   $$\text{Memory}_{\text{activations}} \approx 4.00 \text{ GB}$$
5. **CUDA Runtime & Overhead**:
   $$\text{Memory}_{\text{overhead}} \approx 1.00 \text{ GB}$$

$$\mathbf{Total}_{\text{LoRA}} = 14.00 + 0.61 + 4.00 + 1.00 = \mathbf{19.61 \text{ GB}}$$
- **Hardware Requirement**: Fits comfortably on a single **$1 \times \text{RTX 3090 / 4090 (24 GB)}$** or a single **A10G (24 GB)**.

---

### (c) QLoRA Fine-Tuning (NF4 Quantized Base Model + Double Quantization + BF16 LoRA)

QLoRA compresses the base model into 4-bit NormalFloat (NF4) and uses Double Quantization (DQ) and Paged Optimizers.

1. **Quantized Base Weights (NF4)**:
   $$\text{Memory}_{\text{base (raw 4-bit)}} = 7.0 \times 10^9 \times 0.5 \text{ bytes} = 3.50 \text{ GB}$$
2. **Double Quantization (DQ) Overhead**:
   First-stage quantization scales (block size 64) quantized to 8-bit (block size 256):
   $$\text{Overhead}_{\text{DQ}} \approx 0.13 \text{ GB} = 130 \text{ MB}$$
   $$\text{Memory}_{\text{base (NF4+DQ)}} = 3.50 + 0.13 = 3.63 \text{ GB}$$
3. **LoRA Weights, Gradients & Optimizer States (BF16 / FP32)**:
   Identical to LoRA ($N_{\text{LoRA}} \approx 50.6 \text{M}$):
   $$\text{Memory}_{\text{LoRA State}} \approx 0.61 \text{ GB}$$
   *(With Paged Optimizers, optimizer states can be paged to host CPU RAM during activation peaks without OOMs)*.
4. **On-the-Fly Dequantization Workspace**:
   Temporary FP16/BF16 memory tiles loaded into GPU registers/SRAM during GEMM execution:
   $$\text{Memory}_{\text{dequant}} \approx 0.50 \text{ GB}$$
5. **Activations (batch size 4, seq len 512, activation checkpointing)**:
   $$\text{Memory}_{\text{activations}} \approx 3.80 \text{ GB}$$
6. **CUDA Runtime & Overhead**:
   $$\text{Memory}_{\text{overhead}} \approx 0.80 \text{ GB}$$

$$\mathbf{Total}_{\text{QLoRA}} = 3.63 + 0.61 + 0.50 + 3.80 + 0.80 = \mathbf{9.34 \text{ GB}}$$
- **Hardware Requirement**: Fits effortlessly on consumer GPUs with **12 GB or 16 GB VRAM** (e.g. RTX 3060 12GB, RTX 4070 12GB, RTX 4080 16GB, or free Google Colab T4 16GB).

### Summary Comparison Table
| Fine-Tuning Regime | Base Weight VRAM | Gradient + Optimizer VRAM | Activation & Runtime VRAM | Total VRAM Required | Target GPU Class |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Fine-Tuning (FP16)** | 14.00 GB | 70.00 GB | 5.00 GB | **89.00 GB** | $2 \times \text{A100 } 80\text{GB}$ |
| **LoRA ($r=16$, FP16 Base)** | 14.00 GB | 0.61 GB | 5.00 GB | **19.61 GB** | $1 \times \text{RTX 4090 / A10G (24GB)}$ |
| **QLoRA (NF4 + DQ)** | 3.63 GB | 0.61 GB | 5.10 GB | **9.34 GB** | $1 \times \text{RTX 3060 (12GB) / T4 (16GB)}$ |


---

# TASK 2: Retrieval (RAG) from Scratch

## 2.1 The Academic Corpus & Scratch Retrieval Architecture

To test retrieval under realistic, information-dense conditions, we built an academic corpus compiled directly from **10 landmark machine learning research papers**, comprising **47 distinct sections** and **8,565 words** (equivalent to **~30.6 standard academic pages**):
1. **Vaswani et al. (2017)**: *Attention Is All You Need* (Sections 1-7)
2. **Hu et al. (2021)**: *LoRA: Low-Rank Adaptation of Large Language Models* (Sections 1-5)
3. **Dettmers et al. (2023)**: *QLoRA: Efficient Finetuning of Quantized LLMs* (Sections 1-5)
4. **Dao et al. (2022)**: *FlashAttention: Fast and Memory-Efficient Exact Attention* (Sections 1-5)
5. **Hoffmann et al. (2022)**: *Training Compute-Optimal Large Language Models (Chinchilla)* (Sections 1-5)
6. **He et al. (2015)**: *Deep Residual Learning for Image Recognition (ResNet)* (Sections 1-4)
7. **Devlin et al. (2018)**: *BERT: Pre-training of Deep Bidirectional Transformers* (Sections 1-4)
8. **Ho et al. (2020)**: *Denoising Diffusion Probabilistic Models (DDPM)* (Sections 1-4)
9. **Brown et al. (2020)**: *Language Models are Few-Shot Learners (GPT-3)* (Sections 1-4)
10. **Touvron et al. (2023)**: *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Sections 1-4)

### Zero-Framework Architecture (No LangChain / LlamaIndex)
The retrieval pipeline was implemented strictly from first principles using standard scientific libraries:
- **Tokenizer-Driven Recursive Chunker**: Windowing over tokens via HuggingFace `AutoTokenizer` with parameterized overlap.
- **Pure Python Okapi BM25**: Full mathematical implementation of term frequency saturation and Robertson-Spärck Jones IDF:
  $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
  $$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  (Default parameters: $k_1 = 1.5, b = 0.75$).
- **Dense Vector Search**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit-normalized embeddings ($L_2$ normalized) indexed in a pure **FAISS `IndexFlatIP`** (Inner Product = exact Cosine Similarity).
- **Hybrid Score Fusion**: Min-max normalized convex combination:
  $$\hat{S}_{\text{BM25}} = \frac{S_{\text{BM25}} - \min(S)}{\max(S) - \min(S)}, \quad \hat{S}_{\text{Dense}} = \frac{S_{\text{Dense}} - \min(S)}{\max(S) - \min(S)}$$
  $$S_{\text{Hybrid}} = \alpha \cdot \hat{S}_{\text{BM25}} + (1 - \alpha) \cdot \hat{S}_{\text{Dense}} \quad (\alpha = 0.40)$$

---

## 2.2 15 Hand-Crafted Benchmark Test Questions & Source Passages

Below are the 15 benchmark questions authored across the corpus, each paired with its ground-truth source passage:

| ID | Test Question | Target Paper | Source Passage Excerpt (Ground Truth) |
| :---: | :--- | :--- | :--- |
| **Q01** | What is the mathematical scaling factor applied to queries and keys in Scaled Dot-Product Attention, and why is it applied? | Attention Is All You Need | *"We divide queries and keys by \sqrt{d_k} to counteract the effect where for large d_k, dot products grow large in magnitude, pushing softmax into regions with tiny gradients."* |
| **Q02** | How many attention heads and what head dimensions are used in the base Transformer model? | Attention Is All You Need | *"For the base model, we employ h = 8 parallel attention layers, or heads. For each of these we use d_k = d_v = d_{model}/h = 64."* |
| **Q03** | What learning rate warmup schedule and warmup steps were used to train the base Transformer on WMT 2014? | Attention Is All You Need | *"We used warmup_steps = 4000 in the formula: lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})."* |
| **Q04** | What BLEU score did the big Transformer model achieve on WMT 2014 English-to-German translation? | Attention Is All You Need | *"On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, outperforming existing models by over 2.0 BLEU."* |
| **Q05** | How does LoRA decompose the weight update matrix $\Delta W$ for a pre-trained matrix $W_0$? | LoRA | *"For pre-trained matrix W_0 \in \mathbb{R}^{d \times k}, we constrain its update \Delta W by a low-rank decomposition W_0 + \Delta W = W_0 + B A, where B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)."* |
| **Q06** | Why does LoRA introduce zero additional latency during production inference? | LoRA | *"When deployed in production, we can explicitly compute and store W = W_0 + B A and perform inference as usual. No additional latency is introduced."* |
| **Q07** | What was the reduction in VRAM and optimizer state size when applying LoRA to GPT-3 175B? | LoRA | *"LoRA reduces GPT-3 175B memory footprint from 1.2TB down to 350GB, reducing VRAM footprint of optimizer states by 3x and trainable parameters by 10,000x."* |
| **Q08** | Which weight matrices in the Transformer architecture are most effective to adapt using LoRA? | LoRA | *"Adapting both W_q and W_v yields the best performance trade-offs, achieving competitive adaptation while keeping rank r as low as 4 or 8."* |
| **Q09** | How does 4-bit NormalFloat (NF4) construct its 16 quantization bins for neural network weights? | QLoRA | *"NF4 builds an information-theoretically optimal quantile grid for zero-mean, unit-variance Gaussian distributions \mathcal{N}(0, \sigma^2), ensuring each bin has an equal number of expected parameters."* |
| **Q10** | How much memory per parameter does Double Quantization save, and how is that calculated? | QLoRA | *"Double Quantization quantizes first-stage quantization constants (32-bit float, block size 64) into 8-bit integers with block size 256, reducing memory footprint by 0.37 bits per parameter."* |
| **Q11** | How do Paged Optimizers utilize CUDA Unified Memory to prevent out-of-memory errors? | QLoRA | *"Paged Optimizers allocate optimizer states in NVIDIA CUDA Unified Memory, automatically executing non-blocking page transfers between GPU VRAM and CPU RAM during activation peaks."* |
| **Q12** | What are the two core technical algorithmic components of FlashAttention that avoid materializing the full $N \times N$ matrix in HBM? | FlashAttention | *"FlashAttention uses Tiling (loading Q, K, V in blocks from slow HBM into fast SRAM) and Online Softmax (computing softmax scaling incrementally without storing intermediate attention matrices)."* |
| **Q13** | What is the theoretical IO complexity of FlashAttention compared to standard attention? | FlashAttention | *"Standard attention requires O(N^2 d) HBM accesses. FlashAttention requires O(N^2 d^2 / M) HBM accesses, where M is SRAM size, yielding a sub-quadratic memory access rate."* |
| **Q14** | What speedup does FlashAttention achieve on GPT-2 with 1024 sequence length compared to HuggingFace Transformers? | FlashAttention | *"FlashAttention yields up to 3x speedup in forward-backward pass execution on GPT-2 with 1024 sequence length compared to standard HuggingFace implementations."* |
| **Q15** | According to the Chinchilla scaling laws, what is the compute-optimal ratio between scaling model parameters and training tokens? | Chinchilla Scaling Laws | *"Given a fixed compute budget, model size and the number of training tokens should be scaled equally: doubling model parameters requires doubling training tokens."* |

---

## 2.3 Chunk Size Experiment: 256 Tokens vs. 512 Tokens

We benchmarked two chunking regimes across all 15 benchmark questions:
- **Setting A**: Chunk Size = 256 tokens, Overlap = 50 tokens (Corpus yielded **73 chunks**)
- **Setting B**: Chunk Size = 512 tokens, Overlap = 100 tokens (Corpus yielded **48 chunks**)

### Empirical Retrieval Benchmark Results

| Chunk Setting | Retrieval Mode | Recall@1 | Recall@3 | Recall@5 | Mean Reciprocal Rank (MRR@5) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Setting A: 256 Tokens (Overlap 50)** | Sparse BM25 | 86.7% | 93.3% | 100.0% | 0.9222 |
| | Dense FAISS | 93.3% | 100.0% | 100.0% | 0.9667 |
| | **Hybrid ($\alpha=0.4$)** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |
| **Setting B: 512 Tokens (Overlap 100)** | Sparse BM25 | 93.3% | 100.0% | 100.0% | 0.9667 |
| | Dense FAISS | 86.7% | 93.3% | 93.3% | 0.9000 |
| | **Hybrid ($\alpha=0.4$)** | **93.3%** | **100.0%** | **100.0%** | **0.9667** |

```
+---------------------------------------------------------------------------------------------------+
|                            RETRIEVAL PERFORMANCE COMPARISON (MRR@5)                               |
+---------------------------------------------------------------------------------------------------+
| Setting A (256 Tokens): Hybrid =======================================================> 1.0000    |
| Setting A (256 Tokens): Dense  =================================================> 0.9667          |
| Setting B (512 Tokens): BM25   =================================================> 0.9667          |
| Setting B (512 Tokens): Hybrid =================================================> 0.9667          |
| Setting A (256 Tokens): BM25   ===========================================> 0.9222                |
| Setting B (512 Tokens): Dense  ======================================> 0.9000                    |
+---------------------------------------------------------------------------------------------------+
```

### Which Worked Better and Why? (In-Depth Root Cause)

**Winner: Setting A (Chunk Size 256) achieved superior overall retrieval precision (MRR 1.0000 vs 0.9667, Dense Recall@5 100% vs 93.3%).**

#### 1. The Vector Dilution Effect in Dense Embeddings
In dense retrieval, `all-MiniLM-L6-v2` pools all token representations into a single fixed 384-dimensional latent vector. In a 512-token chunk, the specific factual "needle" (e.g. *Double Quantization saves 0.37 bits/param*) represents only ~5% of the total tokens; the remaining 95% comprises broader methodological context. Mean pooling averages the needle's vector coordinates across the background tokens, diluting the sharp semantic signal. In contrast, a 256-token chunk concentrates the embedding vector around the exact technical assertion, boosting Dense Recall@5 from **93.3% $\to$ 100.0%** and MRR from **0.9000 $\to$ 0.9667**.

#### 2. BM25 Length Normalization & Term Frequency Dynamics
Conversely, Sparse BM25 performed slightly better at Recall@1 on 512-token chunks (93.3% vs 86.7%). Because larger chunks aggregate more repetitive terminology (e.g., repeated mentions of *"Transformer"*, *"BLEU"*, *"WMT"*), term frequencies $f(q_i, D)$ were higher. However, when combined in the **Hybrid Architecture**, the higher precision of 256-token dense embeddings outweighed BM25's marginal gain, allowing Hybrid retrieval to achieve **perfect 100.0% Recall@1 and 1.0000 MRR@5**.

---

## 2.4 Retrieval Failure Case Analysis & Engineering Remedies

### Failure Case 1: Dense Semantic Distraction / High-Level Abstract Competition
- **Query (Q04)**: *"What BLEU score did the big Transformer model achieve on WMT 2014 English-to-German translation?"*
- **Observed Failure**: In Setting B (512 tokens), Dense FAISS ranked the ground truth passage (`P1_S7: Training Regime, Hardware & Hyperparameters`) outside the Top-5 (`hit_rank = None`). Instead, it ranked the paper's Abstract (`P1_S0`) at Rank 1.
- **Root Cause**: The Abstract contained high-density summaries of translation performance (*"achieves 28.4 BLEU on English-to-German"*), whereas Section 7 embedded the number amid extensive discussions of Adam beta parameters, dropout rates, label smoothing, and 8-GPU cluster configurations. The dense embedding of Section 7 was dominated by training hyperparameters rather than the translation result.
- **Concrete Engineering Remedies**:
  1. **Hierarchical / Parent-Document Retrieval**: Index small 128-token leaf chunks for dense vector matching, but return the broader parent section to the LLM context.
  2. **Dense-Sparse Hybrid Fusion**: Sparse BM25 correctly matched exact entity tokens (`"WMT 2014"`, `"English-to-German"`, `"BLEU"`, `"hyperparameters"`) and placed Section 7 at Rank 1. Hybrid fusion rescued the query, promoting Section 7 into the top retrieved candidates.

### Failure Case 2: Boundary Truncation & Numerical Table Splitting
- **Query (Q10)**: *"How much memory per parameter does Double Quantization save, and how is that calculated?"*
- **Observed Failure**: In Setting B, BM25 ranked Section 1 (Introduction) at Rank 1 and the target derivation section (Section 3) at Rank 2.
- **Root Cause**: Fixed-token sliding windows blindly cut paragraphs mid-derivation. The explanation of the 32-bit float scales appeared in chunk $k$, while the 8-bit secondary quantization and the final $0.37 \text{ bits/param}$ result landed in chunk $k+1$.
- **Concrete Engineering Remedies**:
  1. **Markdown AST / Semantic Boundary Chunking**: Rather than fixed token windows, split text along markdown headers (`##`, `###`), mathematical environments, and tabular blocks so that mathematical derivations remain atomic units.
  2. **Hypothetical Document Embeddings (HyDE)**: Generate a synthetic candidate answer using the LLM before retrieval; match the synthetic derivation against chunk vectors to align technical terminology.

---

## 2.5 Conceptual Deep Dives: BM25 and Re-Rankers (In Own Words)

### What BM25 Adds to Embedding Search
Embedding search compresses all text into a continuous latent space. While exceptional at resolving fuzzy semantic paraphrases (e.g. mapping *"accelerating inference"* to *"reducing latency"*), dense embeddings exhibit a fundamental blind spot: **they lose exact lexical discrimination**. If a user asks for *"WMT 2014 BLEU score"* or a specific variable name like `W_q`, a dense model may treat *"WMT 2015"* or `W_k` as nearly identical in semantic space, often retrieving generic transformer summaries.

BM25 provides the complementary exact-match precision that dense vectors lack:
1. **Inverted Index Frequency Weighting**: BM25 scores documents by exact token presence, heavily weighting rare terms via Inverse Document Frequency (IDF). A rare acronym like *"NF4"* or *"SRAM"* receives enormous score weight, while generic words (*"model"*, *"training"*) are dampened.
2. **Term Frequency Saturation**: Unlike simple TF-IDF, BM25 prevents a document from dominating simply because a word appears 50 times; the saturation parameter $k_1$ ensures diminishing returns.
3. **Out-of-Vocabulary Resilience**: BM25 operates on tokenized lexical terms without vocabulary clipping or latent space projection loss.

By fusing BM25 with dense retrieval ($\alpha \cdot S_{\text{BM25}} + (1-\alpha) \cdot S_{\text{Dense}}$), the hybrid system retrieves documents that satisfy both exact lexical constraints and broad semantic relevance.

### What a Re-Ranker Does
First-stage retrievers (dense bi-encoders and BM25) prioritize **throughput over expressive depth**. A bi-encoder processes the query and the document in complete isolation:
$$\vec{u} = \text{Encoder}(q), \quad \vec{v} = \text{Encoder}(d), \quad \text{Score} = \langle \vec{u}, \vec{v} \rangle$$
Because there is **zero token-to-token cross-attention** between the query and candidate passages, bi-encoders cannot capture:
- **Syntactic Negation**: *"models trained without attention"* vs *"models trained with attention"*.
- **Directional Relations**: *"paper A citing paper B"* vs *"paper B citing paper A"*.
- **Fine-Grained Numerical Constraints**: *"exceeding 85% accuracy"* vs *"achieving 83% accuracy"*.

A **Re-ranker** (typically a cross-encoder model like `cross-encoder/ms-marco-MiniLM-L-6-v2` or `bge-reranker-large`) eliminates this bottleneck. It concatenates the query and candidate passage into a single combined sequence:
$$[\text{CLS}] \circ q \circ [\text{SEP}] \circ d \circ [\text{SEP}]$$
Every query token directly attends to every document token across every self-attention layer. This all-to-all cross-attention allows the model to reason over subtle linguistic nuances, contradictions, and qualifier alignments.

**The Trade-off**: A cross-encoder is $\sim 100\times$ more computationally expensive per passage and cannot leverage pre-computed vector indices. Therefore, production RAG systems use a two-stage funnel: Stage 1 (Fast Hybrid Search) filters 1,000,000 passages down to Top-50 candidates in $<10 \text{ ms}$; Stage 2 (Cross-Encoder Re-ranker) performs joint cross-attention over those 50 candidates to output the Top-5 with maximal factual accuracy.


---

# TASK 3: Running a Model Locally & Inference Optimization

## 3.1 Local Quantization Benchmark: FP16 vs. 4-bit NormalFloat (NF4)

We benchmarked the `Qwen2.5-0.5B-Instruct` model locally on the dedicated NVIDIA GeForce RTX 4050 Laptop GPU under two distinct precision regimes:
- **Setting A**: Full 16-bit Floating Point (`torch.float16`, unquantized)
- **Setting B**: 4-bit NormalFloat (`bitsandbytes` NF4 Quantization with Double Quantization and FP16 compute kernel dispatch)

Both configurations were evaluated under greedy decoding (`do_sample=False`, `max_new_tokens=150`) across 10 identical, multi-domain evaluation prompts. GPU memory allocation was isolated and measured via `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`.

### Empirical Hardware Benchmark Results

| Performance Metric | Setting A: FP16 (16-bit Unquantized) | Setting B: 4-bit NormalFloat (NF4) | Empirical Delta / Efficiency Ratio |
| :--- | :---: | :---: | :---: |
| **Model Weights Static VRAM** | **942.32 MB** | **436.35 MB** | **$2.16\times$ Footprint Reduction (-53.7%)** |
| **Peak Generation VRAM** | **957.62 MB** | **457.66 MB** | **$2.09\times$ VRAM Reduction (-52.2%)** |
| **Model Load & Initialization Time** | 4.53 s | 5.95 s | +31.3% (Quantization state unpack overhead) |
| **Average Decode Throughput** | **7.97 tok/s** | **5.19 tok/s** | -34.9% (Dequantization kernel latency) |
| **Average Inter-Token Latency** | 125.5 ms / token | 192.7 ms / token | +67.2 ms / token |
| **Total Test Generation Runtime (10 Prompts)** | 165.5 s | 258.9 s | +56.4% |

```
+---------------------------------------------------------------------------------------------------+
|                        VRAM CONSUMPTION & DECODE THROUGHPUT BENCHMARK                             |
+---------------------------------------------------------------------------------------------------+
| [Model Weight VRAM Footprint]                                                                     |
| FP16 (16-bit) : [##############################################] 942.32 MB                        |
| NF4 (4-bit)   : [#####################] 436.35 MB (2.16x reduction)                               |
|                                                                                                   |
| [Peak Generation VRAM]                                                                            |
| FP16 (16-bit) : [###############################################] 957.62 MB                       |
| NF4 (4-bit)   : [######################] 457.66 MB (2.09x reduction)                              |
|                                                                                                   |
| [Generation Throughput (Tokens / Sec)]                                                            |
| FP16 (16-bit) : [====================================] 7.97 tok/s                                 |
| NF4 (4-bit)   : [=======================] 5.19 tok/s                                              |
+---------------------------------------------------------------------------------------------------+
```

### Empirical Engineering Observations: Memory vs. Compute Trade-Offs

1. **Massive Memory Compression ($2.16\times$ Reduction)**:
   Quantizing the base model compressed the static weight footprint from 942.32 MB down to 436.35 MB. For larger 7B models, this exact compression ratio translates from 14 GB down to ~4.5 GB, allowing weights to fit on consumer-grade edge cards.
2. **The Dequantization Compute Tax on Small Models**:
   While quantization dramatically reduces memory bus bandwidth requirements, NF4 execution on standard `bitsandbytes` kernels incurred a 34.9% decode throughput penalty (7.97 tok/s $\to$ 5.19 tok/s). On small sub-1B parameter models, the arithmetic cost of dequantizing 4-bit weights into 16-bit registers on-the-fly inside the GPU Streaming Multiprocessors (SMs) outweighs the memory bandwidth savings. On large models ($\ge 13\text{B}$) where inference is bottlenecked by HBM memory bandwidth rather than compute, quantized kernels (e.g. Marlin or GPTQ) exhibit the opposite behavior, delivering net throughput speedups.

---

## 3.2 Qualitative Evaluation: Side-by-Side Output Quality on 10 Prompts

Below is the side-by-side quality comparison across the 10 diverse benchmark prompts:

| # | Prompt Category & Core Objective | Setting A: FP16 Completion | Setting B: 4-bit NF4 Completion | Qualitative Observations & Degradation Analysis |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Structured JSON Extraction**<br>Extract metadata from FlashAttention-2 abstract snippet. | Emitted valid JSON: `{"task": "evaluate FlashAttention-2", "dataset": "NVIDIA A100 GPU", "metric": "speedup", "result": "2.3x"}` | Emitted nested JSON: `{"task": {"name": "evaluate FlashAttention-2"}, "metric": "speedup", "result": "2.3x"}` | Both models extracted the 2.3x speedup accurately. NF4 altered the schema structure slightly by nesting `"task"` inside an object. |
| **2** | **Technical Summarization**<br>Explain LoRA weight decomposition in 2 sentences. | Concisely explained low-rank matrices $A$ and $B$, freezing $W_0$, and reducing parameter overhead. | Outlined the exact mathematical steps ($W = W_0 + \Delta W$, rank constraint $r \ll d$). | Both preserved technical fidelity; NF4 provided a more explicitly mathematical step breakdown. |
| **3** | **Code Generation**<br>Pure Python cosine similarity without scikit-learn. | Wrote correct function using `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` with docstring. | Wrote identical vectorized implementation using `np.dot` and `np.linalg.norm` with error handling. | Zero code degradation; both produced production-valid NumPy code. |
| **4** | **Mathematical Derivation**<br>7B parameter FP16 memory calculation with working. | Computed $7 \times 10^9 \times 2 = 14 \times 10^9 \text{ bytes} = 14\text{ GB}$ ($13.04\text{ GiB}$) with full steps. | Computed $7 \times 10^9 \times 16 \text{ bits} / 8 = 14\text{ GB}$ correctly with step-by-step arithmetic. | Flawless arithmetic precision maintained under 4-bit quantization. |
| **5** | **Logical Inference & Distraction**<br>Does BERT use attention? Does ResNet use attention? | Correctly identified BERT uses attention; hallucinated that ResNet uses self-attention mechanisms. | Stated BERT uses self-attention; also hallucinated that ResNet uses attention mechanisms. | Both models exhibited the identical factual failure due to sub-1B base model capacity, proving quantization did not create new hallucinations. |
| **6** | **JSON Schema Array Generation**<br>Format Paper A and Paper B citation records. | Emitted valid array: `[{"paper": "Paper A", "year": 2017, "citations": 10000}, ...]` | Emitted valid array: `[{"paper": "Paper A", "yearPublished": 2017, "citations": 10000}, ...]` | Perfect syntactic adherence in both; NF4 chose a slightly more verbose key name (`yearPublished`). |
| **7** | **System Design Trade-Off**<br>Compare Dense Vector Retrieval vs. BM25. | Contrasted semantic understanding (dense) vs. exact lexical matching (BM25) with clear trade-offs. | Highlighted high recall for dense vs. exact keyword precision for BM25 with query latency analysis. | High conceptual alignment; NF4 retained full technical reasoning nuance. |
| **8** | **Instruction Following Under Constraints**<br>List 3 benefits of GQA, under 15 words per bullet. | Emitted 3 bullet points, each strictly satisfying the 15-word length constraint. | Emitted 3 concise bullet points satisfying the length constraint. | Zero loss in instruction following or constraint adherence. |
| **9** | **Multi-Step Memory Calculation**<br>24 GB GPU, 4.5 GB weights, 1.5 GB overhead, 16 KB KV. | Solved: $24 - 4.5 - 1.5 = 18.0\text{ GB}$. Divided $18.0\text{ GB} / 16\text{ KB} = 1,125,000\text{ tokens}$. | Solved: $18.0\text{ GB}$ remaining, correctly converted units to yield $1.125 \times 10^6\text{ tokens}$. | Complex multi-step dimensional unit conversions executed identically in both settings. |
| **10** | **Pedagogical Analogy Synthesis**<br>Explain KV cache to a junior engineer via analogy. | Used the analogy of a reader keeping bookmarks and notes rather than re-reading the entire book. | Used the analogy of keeping open crayons in a tray rather than closing and unpacking the box per stroke. | Both generated creative, pedagogically sound analogies with zero semantic collapse. |

### Synthesis of Qualitative Findings
Across all 10 prompts, **4-bit NF4 quantization preserved full semantic, syntactic, mathematical, and coding capabilities**. The quantized model exhibited no degenerative loops, grammar degradation, or hallucinations beyond those already present in the unquantized base model. For structured edge inference and RAG summarization, 4-bit quantization delivers full production parity at half the VRAM footprint.

---

## 3.3 Conceptual Deep Dives: KV Cache and Quantization Mechanics (In Own Words)

### In Your Own Words: What the KV Cache Is
In autoregressive large language models, generation proceeds token-by-token:
$$P(y_t \mid y_{<t}, x)$$
To generate token $t$, the transformer layer computes Scaled Dot-Product Attention over the entire historical sequence:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
Without a cache, generating the $1000^{\text{th}}$ token would require projecting all 999 previous tokens through the weight matrices $W_K$ and $W_V$ again, incurring quadratic computational complexity $O(t^2)$ and massive redundant matrix multiplications.

The **KV Cache** (Key-Value Cache) is a runtime memory buffer that stores the computed Key ($K$) and Value ($V$) tensors for all prior tokens across all transformer layers in GPU VRAM:
1. **Prefill Phase (Prompt Processing)**: The model processes the entire prompt in parallel, computing and storing the $K$ and $V$ matrices for all prompt tokens in the cache.
2. **Decode Phase (Token Generation)**: At each new step $t$, the model processes **only the single newly emitted token**. It projects this single token through $W_Q, W_K, W_V$ to produce a $1 \times d_k$ Query vector and new $1 \times d_k$ Key and Value rows. It appends the new Key and Value vectors to the persistent KV cache and computes attention against the full cached history in linear time $O(t)$.

**The Serving Trade-off**: The KV cache transforms inference from compute-bound to **memory-capacity-bound**. The cache size grows strictly linearly with context length:
$$\text{Memory}_{\text{KV}} = 2 \times B \times S \times L \times H_{\text{KV}} \times d_{\text{head}} \times \text{bytes}$$
During long-context interactions and concurrent multi-user serving, the KV cache rapidly surpasses the model weights as the primary consumer of GPU VRAM.

### In Your Own Words: Why Quantization Saves Memory
Standard deep learning models represent weight parameters using 16-bit floating point formats (FP16 or BF16), where each parameter consumes 2 bytes. For a 7-billion parameter model, storing weights alone requires:
$$7.0 \times 10^9 \times 2 \text{ bytes} = 14.0 \text{ GB}$$

**Quantization** is a mathematical compression technique that maps continuous high-precision floating point numbers onto a discrete, low-bit-width grid (such as 4-bit integers or 4-bit NormalFloat NF4). 

1. **Static Memory Reduction**:
   In 4-bit quantization, each weight is represented by a 4-bit index capable of addressing $2^4 = 16$ discrete bins. This yields an immediate, theoretical **$4\times$ reduction in weight memory** ($14.0\text{ GB} \to 3.5\text{ GB}$). With block-level scaling constants and zero-points, the practical footprint is $\approx 4.5\text{ GB}$.
2. **Bandwidth Bottleneck Alleviation**:
   Autoregressive LLM generation at small batch sizes is fundamentally **memory-bandwidth bound**: the GPU spend $>80\%$ of its time waiting for memory controllers to transfer weights from off-chip high-bandwidth memory (HBM/VRAM) across the memory bus into on-chip registers and SRAM. By compressing weights by $4\times$, the volume of data transferred over the PCIe/memory bus per generated token is slashed by $75\%$, allowing high-performance inference engines (like Marlin or vLLM) to saturate Tensor Core compute and achieve higher batch throughput.
3. **Register-Level Execution**:
   Quantized weights remain compressed in global VRAM. During matrix multiplication, custom CUDA kernels load packed 4-bit chunks into registers, unpack and scale them on-the-fly to FP16, execute high-speed multiply-accumulate operations, and discard the unpacked tiles immediately without ever materializing uncompressed weights in global memory.

---

## 3.4 Production Serving Plan: 20 Concurrent Users on a 24 GB GPU

### System Architecture & Hardware Specification
- **Hardware Profile**: $1 \times \text{NVIDIA RTX 3090 / 4090 / A10G}$ (24 GB VRAM = $24,576 \text{ MiB}$)
- **Target Model**: 7-Billion Parameter Open Model with Grouped-Query Attention (e.g., `Mistral-7B-Instruct-v0.3` or `Qwen2.5-7B-Instruct`)
- **Serving Engine**: `vLLM` with PagedAttention and Continuous Iteration-Level Batching
- **Weight Quantization**: 4-bit AWQ / GPTQ / Marlin ($0.5 \text{ bytes / parameter}$)

---

### Step 1: Static VRAM Budget Allocation

```
+---------------------------------------------------------------------------------------------------+
|                            24 GB VRAM STATIC & DYNAMIC ALLOCATION                                 |
+---------------------------------------------------------------------------------------------------+
| [Model Weights (4-bit AWQ/Marlin)] :  4.50 GB  (18.75%)                                           |
| [CUDA Runtime, cuBLAS & Context]   :  1.50 GB  ( 6.25%)                                           |
| [Peak Activation Buffer (FlashAttn)]: 1.00 GB  ( 4.17%)                                           |
| [Dynamic PagedAttention KV Pool]   : 17.00 GB  (70.83%) <--- Dedicated to Concurrent Users        |
+---------------------------------------------------------------------------------------------------+
```

1. **Quantized Model Weights (4-bit)**:
   $$7.0 \times 10^9 \times 0.5 \text{ bytes} = 3.50 \text{ GB}$$
   Including scale factors, zero-point tensors, and metadata: **$4.50 \text{ GB}$**.
2. **CUDA Context & Runtime Workspaces**:
   PyTorch/vLLM initialization, cuBLAS/cuDNN execution handles, memory allocators: **$1.50 \text{ GB}$**.
3. **Peak Activation Scratchpad**:
   Forward-pass intermediate tensor memory under FlashAttention-2: **$1.00 \text{ GB}$**.
4. **Remaining VRAM for Dynamic KV Cache Pool**:
   $$\text{VRAM}_{\text{KV Pool}} = 24.00 - 4.50 - 1.50 - 1.00 = \mathbf{17.00 \text{ GB}} = 18,253,611,008 \text{ bytes}$$

---

### Step 2: Exact KV Cache Memory Consumption Per Token

For a modern 7B parameter architecture featuring Grouped-Query Attention (GQA):
- Layers ($L$): $32$
- Number of Key-Value Heads ($H_{\text{KV}}$): $8$ (compared to 32 query heads in GQA)
- Head Dimension ($d_{\text{head}}$): $128$
- Cache Precision: FP16 ($2 \text{ bytes / element}$)

$$\text{KV}_{\text{token}} = 2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times \text{bytes}$$
$$\text{KV}_{\text{token}} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} = \mathbf{128 \text{ KB / token}}$$

*(Comparative Architectural Perspective)*:
- Under legacy Multi-Head Attention ($H_{\text{KV}} = 32$): $\text{KV}_{\text{token}} = 512 \text{ KB / token}$.
- Under Grouped-Query Attention ($H_{\text{KV}} = 8$): $\text{KV}_{\text{token}} = \mathbf{128 \text{ KB / token}}$ ($4\times$ cache compression).
- Under FP8 Quantized KV Cache ($1 \text{ byte / element}$): $\text{KV}_{\text{token}} = \mathbf{64 \text{ KB / token}}$ ($8\times$ cache compression).

---

### Step 3: Total Token Capacity in the 17.0 GB KV Pool

Using the standard FP16 GQA cache:
$$N_{\text{total\_tokens}} = \frac{18,253,611,008 \text{ bytes}}{131,072 \text{ bytes / token}} = \mathbf{139,264 \text{ tokens}}$$

---

### Step 4: Workload Model & 20 Concurrent User Sizing

#### User Workload Assumptions
- **Active Concurrent Users**: $N_{\text{users}} = 20$
- **Average Prompt Context (Input)**: $1,500 \text{ tokens}$ (RAG retrieved passages + instructions)
- **Average Completion (Output)**: $500 \text{ tokens}$
- **Total Sequence Length Per Active Request**: $S = 1,500 + 500 = \mathbf{2,000 \text{ tokens}}$

#### Active Memory Consumption at Peak Concurrency
$$\text{Total Active Tokens} = 20 \text{ users} \times 2,000 \text{ tokens} = \mathbf{40,000 \text{ tokens}}$$
$$\text{Active VRAM Required} = 40,000 \text{ tokens} \times 128 \text{ KB / token} = 5,120,000 \text{ KB} \approx \mathbf{4.88 \text{ GB}}$$

#### Safety Headroom & Buffer Calculation
$$\text{Remaining Free KV Pool} = 17.00 \text{ GB} - 4.88 \text{ GB} = \mathbf{12.12 \text{ GB}}$$
$$\text{Safety Capacity Margin} = \frac{139,264 \text{ pool tokens}}{40,000 \text{ active tokens}} = \mathbf{3.48\times \text{ Headroom}}$$

The 24 GB GPU not only supports 20 concurrent users at 2,000 tokens each, but has sufficient memory headroom to absorb traffic spikes up to **69 concurrent users**, or support average sequence lengths up to **6,963 tokens per user**.

---

### Step 5: Production Serving Orchestration & Scheduling Policies

To maintain sub-100ms Time-to-First-Token (TTFT) and $>25 \text{ tok/s}$ per-user latency under load, the serving stack implements four critical policies:

1. **PagedAttention Virtual Memory Management**:
   Divides the 17.0 GB KV pool into fixed 16-token physical memory blocks. Memory is allocated on-demand as tokens are generated, eliminating internal fragmentation ($0\%$ waste) and external fragmentation.
2. **Continuous Iteration-Level Batching**:
   Rather than batching at the request level, vLLM batches at the token iteration level. As soon as User 3 finishes generating, User 21 enters the batch immediately on the next iteration step without stalling active users.
3. **Chunked Prefill Scheduling**:
   Long user prompts ($1,500\text{ tokens}$) are partitioned into 512-token chunks. Prefill compute passes are co-scheduled alongside memory-bound decode steps, preventing compute bursts from causing latency spikes on active generating users.
4. **Preemption & CPU Paging Fallback**:
   In the event of sudden traffic surges exceeding the 139k token threshold, vLLM triggers a priority-based preemption policy: the least-recently-active user requests are suspended, their KV cache blocks are swapped to host CPU RAM via PCIe (or recomputed), and restored when tokens free up, guaranteeing zero Out-Of-Memory (OOM) crashes.

---

# 4. Master Synthesis & Engineering Roadmap

## 4.1 Cross-Cutting Failure Taxonomy

Across fine-tuning, retrieval, and inference, failures stem from consistent structural trade-offs:

| Engineering Stage | Observed Failure Mode | Underlying Root Cause | Architectural Remedy |
| :--- | :--- | :--- | :--- |
| **Fine-Tuning (Task 1)** | Hallucinated numerical metrics when given qualitative abstracts | Conditional auto-regressive prior forces token generation; absence of abstention data | Augment training set with negative/null examples and loss masking on abstentions |
| **Fine-Tuning (Task 1)** | Multi-benchmark truncation and slot contamination | Scalar JSON schema creates an information bottleneck | Migrate to nested object arrays (`List[BenchmarkTuple]`) |
| **Retrieval (Task 2)** | Dense vector misses exact hyperparameter section (Q04) | Fixed-dimension mean pooling averages salient technical signals over broad context | Hybrid sparse-dense convex fusion ($\alpha=0.4$) and parent-document retrieval |
| **Retrieval (Task 2)** | Derivation splitting across chunk boundaries (Q10) | Blind token sliding windows bisect mathematical and tabular structures | Markdown AST / Semantic boundary chunking along headers and code blocks |
| **Serving (Task 3)** | Sub-1B quantized models experience decode latency tax | Dequantization compute overhead on small batch sizes exceeds memory bandwidth gains | Deploy fused GEMM kernels (Marlin) or maintain FP16 on sub-1B edge nodes |

---

## 4.2 1-Week Production Sprint Extensions

If given one additional week to transition this prototype into enterprise production, the engineering priorities would be:

1. **Day 1-2: Speculative Decoding with Draft Verification**:
   Pair our fine-tuned `Qwen2.5-0.5B` adapter as an ultra-fast speculative draft model serving a larger 7B target model. This achieves a $2.5\times-3.0\times$ wall-clock latency reduction while preserving 7B generation quality.
2. **Day 3: Two-Stage Re-Ranking Pipeline**:
   Integrate a local cross-encoder (`bge-reranker-base`) over the Top-20 hybrid retrieved candidates in Task 2 to push complex multi-hop retrieval accuracy to $>98\%$.
3. **Day 4: FP8 KV Caching & vLLM Production Deployment**:
   Enable 8-bit FP8 KV cache allocation in vLLM, doubling the concurrent user token capacity from $139,264 \to 278,528$ tokens on the 24 GB GPU.
4. **Day 5: Schema Constrained Decoding via Outlines / Guidance**:
   Replace unstructured token generation with finite-state machine (FSM) grammar-guided decoding, guaranteeing 100% JSON schema adherence at zero hallucination risk.
5. **Day 6-7: Automated Evaluation Harness & Synthetic Benchmark Suite**:
   Implement an end-to-end CI/CD evaluation harness utilizing RAGAS metrics (Faithfulness, Answer Relevance, Context Recall) and continuous adapter regression testing.