"""
Task 2: Pure RAG Engine (Zero LangChain / LlamaIndex).
Components:
- Tokenizer-based Recursive / Sliding Window Chunker (256 vs 512 tokens)
- Okapi BM25 Sparse Search Engine built strictly from scratch
- Dense FAISS IndexFlatIP Retriever using sentence-transformers/all-MiniLM-L6-v2
- Hybrid Score Fusion (BM25 + Dense Cosine Similarity)
- Ground-truth passage matching for Recall@K and MRR@K evaluation
"""

import math
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

EMBEDDING_MODEL_PATH = "D:/ai_cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

class TextChunker:
    def __init__(self, tokenizer_path, chunk_size=256, chunk_overlap=50):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_sections(self, sections):
        chunks = []
        chunk_id = 0
        
        for sec in sections:
            sec_id = sec["section_id"]
            title = sec["header"]
            text = sec["text"]
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            step = self.chunk_size - self.chunk_overlap
            
            if len(tokens) <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "section_id": sec_id,
                    "header": title,
                    "text": text,
                    "token_count": len(tokens)
                })
                chunk_id += 1
            else:
                for i in range(0, len(tokens), step):
                    window = tokens[i:i + self.chunk_size]
                    if len(window) < 30 and i > 0:
                        continue  # Avoid trailing tiny fragments
                    chunk_text = self.tokenizer.decode(window, skip_special_tokens=True).strip()
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "section_id": sec_id,
                        "header": title,
                        "text": chunk_text,
                        "token_count": len(window)
                    })
                    chunk_id += 1
                    if i + self.chunk_size >= len(tokens):
                        break
                        
        return chunks

class ScratchBM25:
    """
    Okapi BM25 implementation from scratch.
    Uses Robertson-Spärck Jones IDF formula:
      IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_freqs = {}
        self.term_freqs = []

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def fit(self, documents):
        self.corpus_size = len(documents)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_tokens += doc_len

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_tokens / max(1, self.corpus_size)

    def _idf(self, term):
        n_q = self.doc_freqs.get(term, 0)
        return math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def score(self, query):
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size)

        for q in q_tokens:
            if q not in self.doc_freqs:
                continue
            idf = self._idf(q)
            for idx in range(self.corpus_size):
                f_qd = self.term_freqs[idx].get(q, 0)
                if f_qd == 0:
                    continue
                d_len = self.doc_lens[idx]
                numerator = f_qd * (self.k1 + 1.0)
                denominator = f_qd + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                scores[idx] += idf * (numerator / denominator)

        return scores

class HybridRAG:
    def __init__(self, chunks, embedding_model_path=EMBEDDING_MODEL_PATH):
        self.chunks = chunks
        self.chunk_texts = [f"{c['header']}\n{c['text']}" for c in chunks]
        
        # 1. Initialize BM25
        print(f"Building BM25 index over {len(self.chunks)} chunks...", flush=True)
        self.bm25 = ScratchBM25(k1=1.5, b=0.75)
        self.bm25.fit(self.chunk_texts)

        # 2. Initialize Dense FAISS
        print(f"Loading dense embedding model: {embedding_model_path}...", flush=True)
        self.embed_model = SentenceTransformer(embedding_model_path)
        
        print("Encoding chunk embeddings...", flush=True)
        embeddings = self.embed_model.encode(
            self.chunk_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        self.dim = embeddings.shape[1]
        
        # FAISS IndexFlatIP computes exact inner product (equivalent to cosine similarity on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        print("FAISS index populated successfully.", flush=True)

    def retrieve(self, query, top_k=5, alpha=0.4, mode="hybrid"):
        """
        Retrieves top_k chunks.
        mode in ['bm25', 'dense', 'hybrid']
        alpha: weight for BM25 (1 - alpha for dense)
        """
        N = len(self.chunks)
        
        # Dense retrieval
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        dense_scores_all = np.dot(self.faiss_index.reconstruct_n(0, N), q_emb[0])
        
        # Sparse BM25 retrieval
        bm25_scores = self.bm25.score(query)
        
        # Normalize scores to [0, 1] for balanced combination
        def min_max_norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-6:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        norm_bm25 = min_max_norm(bm25_scores)
        norm_dense = min_max_norm(dense_scores_all)

        if mode == "bm25":
            final_scores = norm_bm25
        elif mode == "dense":
            final_scores = norm_dense
        else: # hybrid
            final_scores = alpha * norm_bm25 + (1.0 - alpha) * norm_dense

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx].copy()
            chunk["rank"] = rank + 1
            chunk["final_score"] = float(final_scores[idx])
            chunk["bm25_score"] = float(bm25_scores[idx])
            chunk["dense_score"] = float(dense_scores_all[idx])
            results.append(chunk)

        return results