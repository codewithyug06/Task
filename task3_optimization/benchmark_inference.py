"""
Task 3: Local Model Inference & Quantization Optimization Benchmark
Compares Qwen2.5-0.5B-Instruct in:
- Setting A: FP16 (16-bit unquantized)
- Setting B: 4-bit (NF4 quantized via bitsandbytes)

Measures:
- Peak VRAM footprint (MB)
- Prefill throughput & Decode throughput (tokens/sec)
- Latency per token (ms)
- Output quality comparison across 10 diverse prompts
"""

import os
import json
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
OUTPUT_PATH = "d:/Intern/task3_optimization/inference_benchmark_results.json"

TEST_PROMPTS = [
    # 1. Domain extraction (task consistency)
    "Extract key metadata as JSON {task, dataset, metric, result} from: 'In this paper, we evaluate FlashAttention-2 on an NVIDIA A100 GPU across sequence lengths from 512 to 16k tokens. FlashAttention-2 yields a 2.3x speedup over standard attention, achieving 225 TFLOPs/s on GPT-style models.'",
    # 2. Technical summarization
    "Explain how LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices in 2 concise sentences.",
    # 3. Code generation (Python)
    "Write a pure Python function to compute the exact cosine similarity between two 1D NumPy arrays without using sklearn.",
    # 4. Mathematical reasoning
    "If a model has 7 billion parameters and each parameter is stored in 16-bit float, what is the exact memory required for the weights alone in gigabytes? Show calculation.",
    # 5. Logical inference
    "All Transformer models use attention mechanisms. BERT is a Transformer model. ResNet is a convolutional model. Does BERT use attention? Does ResNet use attention?",
    # 6. Structured JSON output
    "Format the following information into a strict JSON array of objects: Paper A was published in 2017 with 10,000 citations. Paper B was published in 2020 with 5,000 citations.",
    # 7. System design trade-off
    "Compare Dense Vector Retrieval against BM25 keyword retrieval. State one key advantage and one key limitation of each.",
    # 8. Instruction following with constraints
    "List exactly 3 key benefits of Grouped Query Attention (GQA) over Multi-Head Attention (MHA). Use bullet points and keep each under 15 words.",
    # 9. Complex multi-step reasoning
    "A GPU has 24 GB of VRAM. A 4-bit quantized 7B model takes 4.5 GB of VRAM. System overhead is 1.5 GB. How much VRAM remains for the KV cache? If each token takes 16 KB of KV cache, how many total tokens can be stored?",
    # 10. Creative/Synthesizing explanation
    "Explain what the KV cache is in autoregressive language models using a real-world analogy suitable for a junior engineer."
]

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def benchmark_setting(setting_name, load_kwargs, prompts, tokenizer):
    print(f"\n=======================================================")
    print(f"BENCHMARKING: {setting_name}")
    print(f"=======================================================")
    clear_gpu()
    start_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    
    print("Loading model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        **load_kwargs
    )
    load_time = time.time() - t0
    
    post_load_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    model_weight_vram = post_load_vram - start_vram
    print(f"Model loaded in {load_time:.2f}s | Weight VRAM: {model_weight_vram:.2f} MB", flush=True)
    
    # Warmup
    print("Running warmup...", flush=True)
    warmup_ids = tokenizer("Warmup test", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**warmup_ids, max_new_tokens=10)
    
    prompt_results = []
    total_generated_tokens = 0
    total_generation_time = 0.0
    
    for idx, prompt_text in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_token_count = inputs["input_ids"].shape[1]
        
        # Measure peak VRAM prior to generation
        torch.cuda.reset_peak_memory_stats()
        
        # Synchronize for precise timing
        torch.cuda.synchronize()
        t_gen_start = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        t_gen_end = time.time()
        
        gen_time = t_gen_end - t_gen_start
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        generated_tokens = output_ids[0][input_token_count:]
        num_gen = len(generated_tokens)
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        tok_per_sec = num_gen / gen_time if gen_time > 0 else 0
        total_generated_tokens += num_gen
        total_generation_time += gen_time
        
        print(f"  Prompt {idx+1:2d} | In: {input_token_count:3d} tok | Out: {num_gen:3d} tok | Time: {gen_time:5.2f}s | Speed: {tok_per_sec:6.2f} tok/s | Peak VRAM: {peak_vram:6.1f} MB", flush=True)
        
        prompt_results.append({
            "prompt_id": idx + 1,
            "prompt": prompt_text,
            "input_tokens": input_token_count,
            "output_tokens": num_gen,
            "generation_time_s": gen_time,
            "tokens_per_second": tok_per_sec,
            "peak_vram_mb": peak_vram,
            "response": response_text
        })
    
    avg_speed = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0
    overall_peak_vram = max(p["peak_vram_mb"] for p in prompt_results)
    
    results = {
        "setting": setting_name,
        "load_time_s": load_time,
        "weight_vram_mb": model_weight_vram,
        "overall_peak_vram_mb": overall_peak_vram,
        "total_generated_tokens": total_generated_tokens,
        "total_generation_time_s": total_generation_time,
        "avg_tokens_per_second": avg_speed,
        "prompts": prompt_results
    }
    
    del model
    clear_gpu()
    return results

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    all_benchmarks = {}
    
    # 1. Setting A: FP16 Unquantized
    load_kwargs_fp16 = {
        "torch_dtype": torch.float16,
        "device_map": "cuda"
    }
    all_benchmarks["FP16"] = benchmark_setting("Setting A: FP16 (16-bit)", load_kwargs_fp16, TEST_PROMPTS, tokenizer)
    
    # 2. Setting B: 4-bit NF4 Quantization (bitsandbytes)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    load_kwargs_q4 = {
        "quantization_config": bnb_config,
        "device_map": "cuda"
    }
    all_benchmarks["4bit_NF4"] = benchmark_setting("Setting B: 4-bit (NF4 Quantized)", load_kwargs_q4, TEST_PROMPTS, tokenizer)
    
    # Print Comparison Table
    print("\n" + "="*70)
    print("           TASK 3 INFERENCE & QUANTIZATION BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Metric':<30} | {'FP16 (16-bit)':<18} | {'NF4 (4-bit)':<18} | {'Delta / Ratio'}")
    print("-" * 75)
    
    fp16 = all_benchmarks["FP16"]
    q4 = all_benchmarks["4bit_NF4"]
    
    vram_fp16 = fp16["weight_vram_mb"]
    vram_q4 = q4["weight_vram_mb"]
    vram_ratio = vram_fp16 / vram_q4 if vram_q4 > 0 else 0
    print(f"{'Model Weight VRAM (MB)':<30} | {vram_fp16:15.2f} MB | {vram_q4:15.2f} MB | {vram_ratio:.2f}x reduction")
    
    peak_fp16 = fp16["overall_peak_vram_mb"]
    peak_q4 = q4["overall_peak_vram_mb"]
    peak_ratio = peak_fp16 / peak_q4 if peak_q4 > 0 else 0
    print(f"{'Peak Generation VRAM (MB)':<30} | {peak_fp16:15.2f} MB | {peak_q4:15.2f} MB | {peak_ratio:.2f}x reduction")
    
    spd_fp16 = fp16["avg_tokens_per_second"]
    spd_q4 = q4["avg_tokens_per_second"]
    spd_diff = ((spd_q4 - spd_fp16) / spd_fp16) * 100 if spd_fp16 > 0 else 0
    print(f"{'Throughput (Tokens/Sec)':<30} | {spd_fp16:15.2f} tok/s | {spd_q4:15.2f} tok/s | {spd_diff:+.1f}%")
    
    print("="*70)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)
    print(f"\nSaved benchmark results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

