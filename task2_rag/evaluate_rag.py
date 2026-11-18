"""
Task 2 Evaluation:
Compare Chunk Size 256 (overlap 50) vs Chunk Size 512 (overlap 100)
Metrics: Recall@1, Recall@3, Recall@5, MRR@5 across 15 hand-crafted benchmark questions.
Generates full retrieval breakdown and analysis of failures.
"""

import json
import time
from rag_engine import TextChunker, HybridRAG, EMBEDDING_MODEL_PATH

SECTIONS_FILE = "d:/Intern/data/corpus_sections.json"
QUESTIONS_FILE = "d:/Intern/data/benchmark_15_questions.json"
OUTPUT_REPORT = "d:/Intern/task2_rag/rag_evaluation_results.json"
TOKENIZER_PATH = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

def evaluate_retrieval(rag_system, questions, top_k=5, alpha=0.4):
    modes = ["sparse_bm25", "dense_faiss", "hybrid"]
    metrics = {m: {"r@1": 0, "r@3": 0, "r@5": 0, "mrr@5": 0.0} for m in modes}
    query_details = []

    for q in questions:
        target_sec = q["target_section_id"]
        q_record = {
            "id": q["id"],
            "question": q["question"],
            "target_section_id": target_sec,
            "target_paper": q["target_paper"],
            "results_by_mode": {}
        }

        for m in modes:
            submode = "bm25" if m == "sparse_bm25" else ("dense" if m == "dense_faiss" else "hybrid")
            retrieved = rag_system.retrieve(q["question"], top_k=top_k, alpha=alpha, mode=submode)
            
            # Check rank of target_section_id
            hit_rank = None
            for idx, r in enumerate(retrieved):
                if r["section_id"] == target_sec:
                    hit_rank = idx + 1
                    break
                    
            if hit_rank == 1:
                metrics[m]["r@1"] += 1
            if hit_rank is not None and hit_rank <= 3:
                metrics[m]["r@3"] += 1
            if hit_rank is not None and hit_rank <= 5:
                metrics[m]["r@5"] += 1
            if hit_rank is not None:
                metrics[m]["mrr@5"] += 1.0 / hit_rank

            q_record["results_by_mode"][m] = {
                "hit_rank": hit_rank,
                "top_1_header": retrieved[0]["header"] if retrieved else "",
                "top_1_score": retrieved[0]["final_score"] if retrieved else 0.0
            }

        query_details.append(q_record)

    total_q = len(questions)
    summary = {}
    for m in modes:
        summary[m] = {
            "recall@1": metrics[m]["r@1"] / total_q,
            "recall@3": metrics[m]["r@3"] / total_q,
            "recall@5": metrics[m]["r@5"] / total_q,
            "mrr@5": metrics[m]["mrr@5"] / total_q
        }
    return summary, query_details

def main():
    print(f"Loading sections from {SECTIONS_FILE}...", flush=True)
    with open(SECTIONS_FILE, "r", encoding="utf-8") as f:
        sections = json.load(f)
        
    print(f"Loading 15 benchmark questions from {QUESTIONS_FILE}...", flush=True)
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # =========================================================================
    # SETTING A: Chunk size 256 tokens, overlap 50 tokens
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print("RUNNING SETTING A: Chunk Size 256, Overlap 50", flush=True)
    print("="*70, flush=True)
    chunker_256 = TextChunker(TOKENIZER_PATH, chunk_size=256, chunk_overlap=50)
    chunks_256 = chunker_256.chunk_sections(sections)
    print(f"Total chunks created for Setting A: {len(chunks_256)}", flush=True)
    
    rag_256 = HybridRAG(chunks_256, EMBEDDING_MODEL_PATH)
    summary_256, details_256 = evaluate_retrieval(rag_256, questions, top_k=5, alpha=0.4)

    # =========================================================================
    # SETTING B: Chunk size 512 tokens, overlap 100 tokens
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print("RUNNING SETTING B: Chunk Size 512, Overlap 100", flush=True)
    print("="*70, flush=True)
    chunker_512 = TextChunker(TOKENIZER_PATH, chunk_size=512, chunk_overlap=100)
    chunks_512 = chunker_512.chunk_sections(sections)
    print(f"Total chunks created for Setting B: {len(chunks_512)}", flush=True)
    
    rag_512 = HybridRAG(chunks_512, EMBEDDING_MODEL_PATH)
    summary_512, details_512 = evaluate_retrieval(rag_512, questions, top_k=5, alpha=0.4)

    # =========================================================================
    # COMPARATIVE SUMMARY TABLE
    # =========================================================================
    print("\n" + "#"*70, flush=True)
    print("               TASK 2 RETRIEVAL BENCHMARK RESULTS", flush=True)
    print("#"*70, flush=True)
    print(f"{'Setting':<25} | {'Retrieval Mode':<15} | {'Recall@1':<10} | {'Recall@5':<10} | {'MRR@5':<10}", flush=True)
    print("-" * 75, flush=True)
    
    for mode in ["sparse_bm25", "dense_faiss", "hybrid"]:
        s = summary_256[mode]
        print(f"{'Chunk 256 (overlap 50)':<25} | {mode:<15} | {s['recall@1']*100:6.1f}%   | {s['recall@5']*100:6.1f}%   | {s['mrr@5']:.4f}", flush=True)
    print("-" * 75, flush=True)
    for mode in ["sparse_bm25", "dense_faiss", "hybrid"]:
        s = summary_512[mode]
        print(f"{'Chunk 512 (overlap 100)':<25} | {mode:<15} | {s['recall@1']*100:6.1f}%   | {s['recall@5']*100:6.1f}%   | {s['mrr@5']:.4f}", flush=True)
    print("#"*70 + "\n", flush=True)

    results = {
        "setting_A_256": {
            "chunk_size": 256,
            "chunk_overlap": 50,
            "total_chunks": len(chunks_256),
            "summary": summary_256,
            "details": details_256
        },
        "setting_B_512": {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "total_chunks": len(chunks_512),
            "summary": summary_512,
            "details": details_512
        }
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed RAG evaluation metrics saved to {OUTPUT_REPORT}", flush=True)

if __name__ == "__main__":
    main()

