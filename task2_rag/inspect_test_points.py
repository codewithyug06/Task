import json

with open('d:/Intern/data/corpus_real_sections.json', encoding='utf-8') as f:
    sections = json.load(f)

def find_sec(arxiv_id, page):
    sec_id = f'{arxiv_id}_p{page}'
    for s in sections:
        if s['section_id'] == sec_id:
            return s
    return None

test_points = [
    ('1706.03762', 3, 'encoder is composed of a stack'),
    ('1706.03762', 8, 'label smoothing'),
    ('2106.09685', 4, 'alpha'),
    ('2106.09685', 8, 'Grassmann'),
    ('2305.14314', 3, 'Double Quantization'),
    ('2305.14314', 4, 'Paged Optimizers'),
    ('2205.14135', 4, 'Theorem 1'),
    ('2205.14135', 5, 'SRAM'),
    ('1512.03385', 2, 'degradation'),
    ('1512.03385', 5, 'shortcut'),
    ('1810.04805', 3, '80%'),
    ('2203.15556', 4, 'Kaplan'),
    ('2005.14165', 8, 'alternating dense'),
    ('2307.09288', 13, 'Ghost Attention'),
    ('2006.11239', 3, 'alpha')
]

for aid, p, query in test_points:
    s = find_sec(aid, p)
    if s:
        print(f"=== {s['paper']} (p.{p}) [ID: {s['section_id']}] ===")
        found = False
        for l in s['text'].split('\n'):
            if query.lower() in l.lower():
                print("  MATCH:", l.strip()[:140])
                found = True
                break
        if not found:
            print("  (query term not in single line, text length:", len(s['text']), ")")
    else:
        print(f"MISSING: {aid} p.{p}")
