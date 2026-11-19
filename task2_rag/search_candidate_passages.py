import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / 'data' / 'corpus_real_sections.json', encoding='utf-8') as f:
    sections = json.load(f)

def search_text(term, max_matches=3):
    matches = []
    for s in sections:
        if term.lower() in s['text'].lower():
            idx = s['text'].lower().find(term.lower())
            start = max(0, idx - 100)
            end = min(len(s['text']), idx + 250)
            snippet = s['text'][start:end].replace('\n', ' ')
            matches.append((s['section_id'], s['paper'], s['page'], snippet))
            if len(matches) >= max_matches:
                break
    return matches

search_terms = [
    "label smoothing",
    "Grassmann",
    "Double Quantization",
    "Paged Optimizers",
    "Theorem 1",
    "degradation problem",
    "80% of the time",
    "Kaplan et al.",
    "alternating dense",
    "Ghost Attention",
    "simplified training objective",
    "bottleneck building block",
    "next sentence prediction",
    "Chinchilla",
    "cosine schedule"
]

for term in search_terms:
    res = search_text(term, max_matches=2)
    print(f"\n=== SEARCH: '{term}' ===")
    for sec_id, paper, page, snip in res:
        print(f"[{sec_id}] {paper} (p.{page}): {snip[:150]}...")
