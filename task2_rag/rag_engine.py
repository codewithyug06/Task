"""
Task 2: Pure RAG Engine (Zero LangChain / LlamaIndex).
Components:
- Tokenizer-based Recursive / Sliding Window Chunker (256 vs 512 tokens)
- Okapi BM25 Sparse Search Engine built strictly from scratch
- Dense FAISS IndexFlatIP Retriever using sentence-transformers/all-MiniLM-L6-v2
- Hybrid Score Fusion (BM25 + Dense Cosine Similarity)
- Ground-truth passage matching for Recall@K and MRR@K evaluation
- Fully portable for both local environment and Colab/Kaggle
"""

import os
import math
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

LOCAL_EMB = "D:/ai_cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_MODEL_PATH = LOCAL_EMB if os.path.exists(LOCAL_EMB) else "sentence-transformers/all-MiniLM-L6-v2"

LOCAL_TOK = "D:/ai_cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_TOKENIZER_PATH = LOCAL_TOK if os.path.exists(LOCAL_TOK) else "Qwen/Qwen2.5-0.5B-Instruct"

class TextChunker:
    def __init__(self, tokenizer_path=DEFAULT_TOKENIZER_PATH, chunk_size=256, chunk_overlap=50):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_sections(self, sections):
        chunks = []
        chunk_id = 0
        
        for sec in sections:
            sec_id = sec["section_id"]
            title = sec.get("header", f"[{sec.get('paper', '')}] Page {sec.get('page', '')}")
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
                        # Skip tiny trailing snippets
                        continue
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
    Pure Python Okapi BM25 implementation from first principles.
    k1 = 1.5, b = 0.75
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.avg_dl = 0.0
        self.doc_freqs = {}
        self.inverted_index = {}
        
        self._build_index(corpus)

    def _tokenize(self, text):
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 1]
        return tokens

    def _build_index(self, corpus):
        total_len = 0
        for doc_id, doc in enumerate(corpus):
            tokens = self._tokenize(doc["text"])
            dl = len(tokens)
            self.doc_lengths.append(dl)
            total_len += dl
            
            term_counts = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
                
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    self.doc_freqs[term] = 0
                self.inverted_index[term].append((doc_id, count))
                self.doc_freqs[term] += 1
                
        self.avg_dl = total_len / max(1, self.corpus_size)

    def search(self, query, top_k=5):
        tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        
        for term in tokens:
            if term not in self.inverted_index:
                continue
            df = self.doc_freqs[term]
            # IDF with Robertson-Spärck Jones formulation
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            idf = max(0.01, idf)
            
            for doc_id, freq in self.inverted_index[term]:
                dl = self.doc_lengths[doc_id]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                scores[doc_id] += idf * (numerator / denominator)
                
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0.0]

class HybridRAG:
    """
    Hybrid Retrieval Engine combining Scratch BM25 and FAISS IndexFlatIP (Dense).
    Applies Min-Max score normalization and convex combination.
    """
    def __init__(self, chunks, embedding_model_path=EMBEDDING_MODEL_PATH):
        self.chunks = chunks
        print(f"Initializing Scratch BM25 over {len(chunks)} chunks...")
        self.bm25 = ScratchBM25(chunks)
        
        print(f"Loading dense embedding model: {embedding_model_path}...")
        self.encoder = SentenceTransformer(embedding_model_path)
        
        print(f"Encoding {len(chunks)} chunks into dense vectors...")
        texts = [f"{c['header']}\n{c['text']}" for c in chunks]
        embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"FAISS IndexFlatIP built successfully with dimension {dim} and {self.index.ntotal} vectors.")

    def retrieve(self, query, top_k=5, alpha=0.5, mode="hybrid"):
        bm25_hits = dict(self.bm25.search(query, top_k=top_k * 4))
        
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        dense_scores, dense_indices = self.index.search(q_emb, top_k * 4)
        dense_hits = {int(idx): float(score) for idx, score in zip(dense_indices[0], dense_scores[0]) if idx != -1}
        
        if mode == "bm25":
            sorted_hits = sorted(bm25_hits.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return self._format_results(sorted_hits, bm25_hits, dense_hits)
            
        if mode == "dense":
            sorted_hits = sorted(dense_hits.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return self._format_results(sorted_hits, bm25_hits, dense_hits)
            
        all_candidate_ids = set(bm25_hits.keys()) | set(dense_hits.keys())
        if not all_candidate_ids:
            return []
            
        bm25_vals = [bm25_hits.get(cid, 0.0) for cid in all_candidate_ids]
        b_min, b_max = min(bm25_vals), max(bm25_vals)
        b_range = b_max - b_min if b_max > b_min else 1.0
        
        dense_vals = [dense_hits.get(cid, 0.0) for cid in all_candidate_ids]
        d_min, d_max = min(dense_vals), max(dense_vals)
        d_range = d_max - d_min if d_max > d_min else 1.0
        
        hybrid_scores = {}
        for cid in all_candidate_ids:
            norm_b = (bm25_hits.get(cid, 0.0) - b_min) / b_range
            norm_d = (dense_hits.get(cid, 0.0) - d_min) / d_range
            hybrid_scores[cid] = alpha * norm_d + (1.0 - alpha) * norm_b
            
        sorted_hits = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._format_results(sorted_hits, bm25_hits, dense_hits)

    def _format_results(self, ranked_tuples, bm25_scores, dense_scores):
        results = []
        for idx, final_score in ranked_tuples:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "section_id": chunk["section_id"],
                "header": chunk["header"],
                "text": chunk["text"],
                "final_score": final_score,
                "bm25_score": bm25_scores.get(idx, 0.0),
                "dense_score": dense_scores.get(idx, 0.0)
            })
        return results
