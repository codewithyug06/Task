"""
Task 2 End-to-End QA Demonstration:
Uses Hybrid RAG (Chunk Size 256) over 335 pages of 10 arXiv papers to retrieve top passages,
and uses Qwen2.5-0.5B-Instruct to generate grounded factual answers with citations.
Saves QA pairs with citations to d:/Intern/task2_rag/qa_generation_answers.json.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_engine import TextChunker, HybridRAG, EMBEDDING_MODEL_PATH, DEFAULT_TOKENIZER_PATH

SECTIONS_FILE = "d:/Intern/data/corpus_real_sections.json" if os.path.exists("d:/Intern/data/corpus_real_sections.json") else "data/corpus_real_sections.json"
QUESTIONS_FILE = "d:/Intern/data/benchmark_15_questions.json" if os.path.exists("d:/Intern/data/benchmark_15_questions.json") else "data/benchmark_15_questions.json"
OUTPUT_FILE = "d:/Intern/task2_rag/qa_generation_answers.json"

LOCAL_MODEL = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print("Initializing RAG system for end-to-end question answering...", flush=True)
    with open(SECTIONS_FILE, "r", encoding="utf-8") as f:
        sections = json.load(f)
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    chunker = TextChunker(DEFAULT_TOKENIZER_PATH, chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk_sections(sections)
    rag = HybridRAG(chunks, EMBEDDING_MODEL_PATH)

    print(f"Loading generator LLM from: {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    results = []
    print(f"\nGenerating answers for {len(questions)} benchmark questions...\n", flush=True)

    for q in questions:
        q_text = q["question"]
        retrieved = rag.retrieve(q_text, top_k=2, alpha=0.5, mode="hybrid")
        context_str = "\n\n".join([f"[{r['header']}]\n{r['text']}" for r in retrieved])

        prompt = (
            f"You are a factual research assistant. Answer the question accurately using ONLY the provided context excerpts.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {q_text}\n"
            f"Answer:"
        )

        messages = [
            {"role": "system", "content": "You are a concise, factual scientific assistant. Rely only on the provided context."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"[{q['id']}] {q_text[:70]}...")
        print(f"Answer: {answer[:120]}...\n")

        results.append({
            "id": q["id"],
            "difficulty": q.get("difficulty", "Medium"),
            "question": q_text,
            "target_section_id": q["target_section_id"],
            "target_paper": q.get("target_paper", q.get("paper", "")),
            "retrieved_contexts": [
                {
                    "section_id": r["section_id"],
                    "header": r["header"],
                    "score": r["final_score"],
                    "snippet": r["text"][:200]
                }
                for r in retrieved
            ],
            "ground_truth_excerpt": q.get("ground_truth_excerpt", ""),
            "generated_answer": answer
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved all {len(results)} grounded QA generations to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
