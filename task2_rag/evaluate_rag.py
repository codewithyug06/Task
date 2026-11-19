"""
Task 2 Evaluation:
Compare Chunk Size 256 (overlap 50) vs Chunk Size 512 (overlap 100)
Corpus: 335 pages across 10 landmark arXiv papers (corpus_real_sections.json)
Questions: 15 hand-crafted benchmark questions (benchmark_15_questions.json)
Metrics: Recall@1, Recall@3, Recall@5, MRR@5 across Sparse BM25, Dense FAISS, and Hybrid.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add repo root to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, TASK2_DIR
from rag_engine import TextChunker, HybridRAG, EMBEDDING_MODEL_PATH, DEFAULT_TOKENIZER_PATH

SECTIONS_FILE = str(REPO_ROOT / "data" / "corpus_real_sections.json") if (REPO_ROOT / "data" / "corpus_real_sections.json").exists() else str(REPO_ROOT / "data" / "corpus_sections.json")
QUESTIONS_FILE = str(REPO_ROOT / "data" / "benchmark_15_questions.json")
OUTPUT_REPORT = str(TASK2_DIR / "rag_evaluation_results.json")

def evaluate_retrieval(rag_system, questions, top_k=5, alpha=0.5):
    modes = ["sparse_bm25", "dense_faiss", "hybrid"]
    metrics = {m: {"r@1": 0, "r@3": 0, "r@5": 0, "mrr@5": 0.0} for m in modes}
    query_details = []

    for q in questions:
        target_sec = q["target_section_id"]
        q_record = {
            "id": q["id"],
            "difficulty": q.get("difficulty", "Medium"),
            "question": q["question"],
            "target_section_id": target_sec,
            "target_paper": q.get("target_paper", q.get("paper", "")),
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
    print("=" * 70)
    print("TASK 2: RETRIEVAL ACCURACY BENCHMARK (SETTING A VS SETTING B)")
    print("=" * 70)

    print(f"Loading corpus sections from: {SECTIONS_FILE}")
    with open(SECTIONS_FILE, "r", encoding="utf-8") as f:
        sections = json.load(f)
    print(f"Loaded {len(sections)} pages from 10 arXiv papers.")

    print(f"Loading benchmark test questions from: {QUESTIONS_FILE}")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} technical test questions with ground truth pages.")

    # Setting A: 256 tokens, overlap 50
    print("\n" + "=" * 50)
    print("EVALUATING SETTING A: Chunk Size = 256, Overlap = 50")
    print("=" * 50)
    chunker_A = TextChunker(tokenizer_path=DEFAULT_TOKENIZER_PATH, chunk_size=256, chunk_overlap=50)
    chunks_A = chunker_A.chunk_sections(sections)
    print(f"Total chunks generated for Setting A: {len(chunks_A)}")
    rag_A = HybridRAG(chunks_A, embedding_model_path=EMBEDDING_MODEL_PATH)
    summary_A, details_A = evaluate_retrieval(rag_A, questions)

    # Setting B: 512 tokens, overlap 100
    print("\n" + "=" * 50)
    print("EVALUATING SETTING B: Chunk Size = 512, Overlap = 100")
    print("=" * 50)
    chunker_B = TextChunker(tokenizer_path=DEFAULT_TOKENIZER_PATH, chunk_size=512, chunk_overlap=100)
    chunks_B = chunker_B.chunk_sections(sections)
    print(f"Total chunks generated for Setting B: {len(chunks_B)}")
    rag_B = HybridRAG(chunks_B, embedding_model_path=EMBEDDING_MODEL_PATH)
    summary_B, details_B = evaluate_retrieval(rag_B, questions)

    results = {
        "corpus_stats": {
            "total_papers": 10,
            "total_pages": len(sections),
            "setting_A_chunks": len(chunks_A),
            "setting_B_chunks": len(chunks_B)
        },
        "setting_A_256": {
            "chunk_size": 256,
            "chunk_overlap": 50,
            "total_chunks": len(chunks_A),
            "summary": summary_A,
            "details": details_A
        },
        "setting_B_512": {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "total_chunks": len(chunks_B),
            "summary": summary_B,
            "details": details_B
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARATIVE RETRIEVAL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Setting A (256)':<24} | {'Setting B (512)':<24}")
    print("-" * 70)
    for m in ["sparse_bm25", "dense_faiss", "hybrid"]:
        r1_A = summary_A[m]["recall@1"] * 100
        r5_A = summary_A[m]["recall@5"] * 100
        mrr_A = summary_A[m]["mrr@5"]
        r1_B = summary_B[m]["recall@1"] * 100
        r5_B = summary_B[m]["recall@5"] * 100
        mrr_B = summary_B[m]["mrr@5"]
        print(f"[{m.upper()}]")
        print(f"  Recall@1         | {r1_A:5.1f}%                   | {r1_B:5.1f}%")
        print(f"  Recall@5         | {r5_A:5.1f}%                   | {r5_B:5.1f}%")
        print(f"  MRR@5            | {mrr_A:5.4f}                   | {mrr_B:5.4f}")
    print("=" * 70)
    print(f"Saved complete evaluation report to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
