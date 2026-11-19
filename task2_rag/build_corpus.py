"""
Task 2: Build Real Academic Corpus from 10 Landmark arXiv Papers
Downloads PDFs directly from arXiv and extracts genuine page text using pypdf / PyMuPDF.

10 Papers:
1. Attention Is All You Need (1706.03762)
2. LoRA: Low-Rank Adaptation of Large Language Models (2106.09685)
3. QLoRA: Efficient Finetuning of Quantized LLMs (2305.14314)
4. FlashAttention: Fast and Memory-Efficient Exact Attention (2205.14135)
5. Deep Residual Learning for Image Recognition / ResNet (1512.03385)
6. BERT: Pre-training of Deep Bidirectional Transformers (1810.04805)
7. Training Compute-Optimal Large Language Models / Chinchilla (2203.15556)
8. Language Models are Few-Shot Learners / GPT-3 (2005.14165)
9. Llama 2: Open Foundation and Fine-Tuned Chat Models (2307.09288)
10. Denoising Diffusion Probabilistic Models / DDPM (2006.11239)
"""

import os
import sys
import json
import time
import urllib.request

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import fitz # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

PAPERS = [
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "year": 2017},
    {"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "year": 2021},
    {"arxiv_id": "2305.14314", "title": "QLoRA: Efficient Finetuning of Quantized LLMs", "year": 2023},
    {"arxiv_id": "2205.14135", "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "year": 2022},
    {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition", "year": 2015},
    {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "year": 2018},
    {"arxiv_id": "2203.15556", "title": "Training Compute-Optimal Large Language Models", "year": 2022},
    {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners", "year": 2020},
    {"arxiv_id": "2307.09288", "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "year": 2023},
    {"arxiv_id": "2006.11239", "title": "Denoising Diffusion Probabilistic Models", "year": 2020}
]

def download_pdf(arxiv_id, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"  [Cached] {arxiv_id} already exists ({os.path.getsize(out_path)} bytes)")
        return True
    
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  [Downloading] {url} -> {out_path} ...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            print(f"  [Success] Downloaded {arxiv_id} ({os.path.getsize(out_path)} bytes)")
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"  [Attempt {attempt+1} failed]: {e}")
            time.sleep(2.0)
    return False

def extract_pages(pdf_path, title, arxiv_id):
    pages = []
    # Try pypdf first as requested
    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) > 100:
                    pages.append({
                        "section_id": f"{arxiv_id}_p{idx+1}",
                        "paper": title,
                        "arxiv_id": arxiv_id,
                        "page": idx + 1,
                        "text": text
                    })
            if len(pages) > 0:
                return pages
        except Exception as e:
            print(f"  pypdf error on {pdf_path}: {e}")
            
    # Fallback to PyMuPDF (fitz)
    if HAS_FITZ:
        doc = fitz.open(pdf_path)
        for idx in range(len(doc)):
            page = doc[idx]
            text = page.get_text() or ""
            text = text.strip()
            if len(text) > 100:
                pages.append({
                    "section_id": f"{arxiv_id}_p{idx+1}",
                    "paper": title,
                    "arxiv_id": arxiv_id,
                    "page": idx + 1,
                    "text": text
                })
        doc.close()
        
    return pages

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    pdf_dir = str(REPO_ROOT / "data" / "arxiv_pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(str(REPO_ROOT / "data"), exist_ok=True)
    
    print("=" * 70)
    print("BUILDING REAL ACADEMIC CORPUS FROM 10 LANDMARK ARXIV PAPERS")
    print("=" * 70)
    
    all_sections = []
    total_words = 0
    
    for item in PAPERS:
        arxiv_id = item["arxiv_id"]
        title = item["title"]
        pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
        
        ok = download_pdf(arxiv_id, pdf_path)
        if not ok:
            print(f"WARNING: Failed to download {arxiv_id}")
            continue
            
        pages = extract_pages(pdf_path, title, arxiv_id)
        paper_words = sum(len(p["text"].split()) for p in pages)
        total_words += paper_words
        all_sections.extend(pages)
        print(f"  -> Extracted {len(pages)} pages ({paper_words:,} words) for '{title}'")
        
    print("-" * 70)
    print(f"TOTAL EXTRACTED CORPUS: {len(all_sections)} pages across {len(PAPERS)} papers")
    print(f"TOTAL WORD COUNT: {total_words:,} words")
    print("-" * 70)
    
    # Save structured sections JSON
    sections_out = str(REPO_ROOT / "data" / "corpus_real_sections.json")
    with open(sections_out, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2)
    print(f"Saved structured sections to: {sections_out}")
    
    # Save continuous text file corpus_real.txt
    text_out = str(REPO_ROOT / "data" / "corpus_real.txt")
    with open(text_out, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(f"\n\n========================================\n")
            f.write(f"PAPER: {sec['paper']} (arXiv: {sec['arxiv_id']})\n")
            f.write(f"PAGE: {sec['page']} [SECTION ID: {sec['section_id']}]\n")
            f.write(f"========================================\n\n")
            f.write(sec['text'])
            f.write("\n")
    print(f"Saved full corpus text to: {text_out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
