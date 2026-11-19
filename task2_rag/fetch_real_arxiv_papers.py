"""
Task 2: Download and parse real landmark arXiv papers (75 pages total).
Papers:
1. Attention Is All You Need (1706.03762) - 15 pages
2. LoRA: Low-Rank Adaptation of Large Language Models (2106.09685) - 14 pages
3. QLoRA: Efficient Finetuning of Quantized LLMs (2305.14314) - 26 pages
4. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2205.14135) - 20 pages

Extracts full text into genuine page sections, outputting:
- data/corpus_sections.json
- data/corpus_real_arxiv.txt
"""

import os
import time
import requests
import fitz  # PyMuPDF
import json

PAPERS = [
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al. (2017)",
        "url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    {
        "id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al. (2021)",
        "url": "https://arxiv.org/pdf/2106.09685.pdf"
    },
    {
        "id": "2305.14314",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Dettmers et al. (2023)",
        "url": "https://arxiv.org/pdf/2305.14314.pdf"
    },
    {
        "id": "2205.14135",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": "Dao et al. (2022)",
        "url": "https://arxiv.org/pdf/2205.14135.pdf"
    }
]

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = str(REPO_ROOT / "data" / "arxiv_pdfs")
SECTIONS_OUTPUT = str(REPO_ROOT / "data" / "corpus_sections.json")
TEXT_OUTPUT = str(REPO_ROOT / "data" / "corpus_real_arxiv.txt")

HEADERS = {
    "User-Agent": "ResearchEvaluationBot/1.0 (academic research assessment; mailto:student@university.edu)"
}

def download_paper(paper):
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"{paper['id']}.pdf")
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
        print(f"  [Cached] {paper['title']} ({pdf_path})", flush=True)
        return pdf_path
        
    print(f"  [Downloading] {paper['title']} from {paper['url']}...", flush=True)
    resp = requests.get(paper['url'], headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(resp.content)
    print(f"  [Saved] {pdf_path} ({len(resp.content)/1024:.1f} KB)", flush=True)
    time.sleep(2)  # Etiquette delay between arXiv downloads
    return pdf_path

def parse_pdf_sections(pdf_path, paper):
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    print(f"  Processing '{paper['title']}': {page_count} pages...", flush=True)
    
    sections = []
    # Segment by page chunks or natural sections
    # To maintain grounded provenance for retrieval evaluation, chunk into pages/sub-pages
    for page_idx in range(page_count):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        if len(text) < 100:
            continue
            
        # Clean basic header/footer noise
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = " ".join(lines)
        
        section_record = {
            "section_id": f"{paper['id']}_p{page_idx+1}",
            "paper_id": paper["id"],
            "paper_title": paper["title"],
            "page": page_idx + 1,
            "header": f"[{paper['title']}] Page {page_idx+1}",
            "text": clean_text,
            "word_count": len(clean_text.split())
        }
        sections.append(section_record)
        
    doc.close()
    return sections, page_count

def main():
    print("="*60, flush=True)
    print("TASK 2: DOWNLOADING & INGESTING REAL ARXIV PAPERS", flush=True)
    print("="*60, flush=True)
    
    all_sections = []
    total_pages = 0
    full_text_dump = []
    
    for paper in PAPERS:
        try:
            pdf_path = download_paper(paper)
            sections, pages = parse_pdf_sections(pdf_path, paper)
            all_sections.extend(sections)
            total_pages += pages
            for s in sections:
                full_text_dump.append(f"--- {s['header']} ---\n{s['text']}\n")
        except Exception as e:
            print(f"  [ERROR] Failed processing {paper['title']}: {e}", flush=True)
            
    total_words = sum(s["word_count"] for s in all_sections)
    print("\n" + "="*60, flush=True)
    print(f"CORPUS INGESTION SUMMARY:", flush=True)
    print(f"  - Total Papers:   {len(PAPERS)}", flush=True)
    print(f"  - Total Pages:    {total_pages} (Requirement: >= 30 pages)", flush=True)
    print(f"  - Total Sections: {len(all_sections)}", flush=True)
    print(f"  - Total Words:    {total_words:,} words", flush=True)
    print("="*60, flush=True)
    
    with open(SECTIONS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2)
    with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text_dump))
        
    print(f"Saved real corpus sections to: {SECTIONS_OUTPUT}")
    print(f"Saved full corpus text to:     {TEXT_OUTPUT}")

if __name__ == "__main__":
    main()
