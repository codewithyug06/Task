"""
Optional Companion Script: Run Benchmark via Local Ollama Runtime.
Connects to local Ollama server (http://localhost:11434) to benchmark
16-bit vs 4-bit quantization levels (e.g. qwen2.5:0.5b-fp16 vs qwen2.5:0.5b).
"""

import json
import time
import urllib.request
import urllib.error

OLLAMA_API_URL = "http://localhost:11434/api/generate"

PROMPTS = [
    "Extract key metadata as JSON {task, dataset, metric, result} from: 'In this paper, we evaluate FlashAttention-2 on an NVIDIA A100 GPU across sequence lengths from 512 to 16k tokens. FlashAttention-2 yields a 2.3x speedup over standard attention, achieving 225 TFLOPs/s on GPT-style models.'",
    "Explain how LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices in 2 concise sentences.",
    "Write a pure Python function to compute the exact cosine similarity between two 1D NumPy arrays without using sklearn.",
    "If a model has 7 billion parameters and each parameter is stored in 16-bit float, what is the exact memory required for the weights alone in gigabytes? Show calculation.",
    "All Transformer models use attention mechanisms. BERT is a Transformer model. ResNet is a convolutional model. Does BERT use attention? Does ResNet use attention?",
    "Format the following information into a strict JSON array of objects: Paper A was published in 2017 with 10,000 citations. Paper B was published in 2020 with 5,000 citations.",
    "Compare Dense Vector Retrieval against BM25 keyword retrieval. State one key advantage and one key limitation of each.",
    "List exactly 3 key benefits of Grouped Query Attention (GQA) over Multi-Head Attention (MHA). Use bullet points and keep each under 15 words.",
    "A GPU has 24 GB of VRAM. A 4-bit quantized 7B model takes 4.5 GB of VRAM. System overhead is 1.5 GB. How much VRAM remains for the KV cache? If each token takes 16 KB of KV cache, how many total tokens can be stored?",
    "Explain what the KV cache is in autoregressive language models using a real-world analogy suitable for a junior engineer."
]

def check_ollama_running():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def query_ollama(model_name, prompt):
    data = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 150
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    
    # Ollama returns eval_count (tokens) and eval_duration (nanoseconds)
    eval_count = res.get("eval_count", 0)
    eval_duration_sec = res.get("eval_duration", 1) / 1e9
    tok_sec = (eval_count / eval_duration_sec) if eval_duration_sec > 0 else 0
    return {
        "response": res.get("response", ""),
        "eval_count": eval_count,
        "eval_duration_sec": eval_duration_sec,
        "tok_sec": tok_sec,
        "wall_time": elapsed
    }

def main():
    print("=" * 65)
    print("OLLAMA BENCHMARK RUNNER (Task 3 Companion)")
    print("=" * 65)
    
    if not check_ollama_running():
        print("Notice: Ollama daemon is not currently active at http://localhost:11434.")
        print("To run with Ollama on your machine:")
        print("  1. Start Ollama: 'ollama serve'")
        print("  2. Pull models:  'ollama pull qwen2.5:0.5b' (4-bit default Q4_K_M)")
        print("                   'ollama pull qwen2.5:0.5b-fp16' (16-bit unquantized)")
        print("  3. Re-run this script: 'python task3_optimization/run_ollama_benchmark.py'")
        print("-" * 65)
        print("The primary reproducible benchmark is in: 'benchmark_inference.py'")
        print("which executes direct PyTorch CUDA profiling (FP16 vs NF4 4-bit).")
        print("=" * 65)
        return

    models = ["qwen2.5:0.5b-fp16", "qwen2.5:0.5b"]
    for m in models:
        print(f"\nBenchmarking Ollama Model: {m}")
        speeds = []
        for i, p in enumerate(PROMPTS):
            try:
                res = query_ollama(m, p)
                speeds.append(res["tok_sec"])
                print(f"  Prompt {i+1}/10: {res['eval_count']} tokens in {res['eval_duration_sec']:.2f}s ({res['tok_sec']:.1f} tok/s)")
            except Exception as e:
                print(f"  Prompt {i+1} failed: {e}")
        if speeds:
            print(f"Average Decoding Speed for {m}: {sum(speeds)/len(speeds):.2f} tokens/sec")

if __name__ == "__main__":
    main()
