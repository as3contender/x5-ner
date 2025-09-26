import ast, re, pandas as pd
from collections import Counter, defaultdict

def parse_spans(s):
    try:
        return [(int(a), int(b), str(t)) for a,b,t in ast.literal_eval(s)]
    except Exception:
        return []

def to_words(text, spans):
    out=[]
    for s,e,lab in spans:
        out.append((text[s:e], lab))
    return out

df = pd.read_csv("diff_report.csv")
df["gold_spans"] = df["gold"].apply(parse_spans)
df["pred_spans"] = df["pred"].apply(parse_spans)

buckets = Counter()
by_label = Counter()
boundary_off = 0
type_swap = Counter()
miss_by_tag = Counter()
extra_by_tag = Counter()
short_brand_fp = []
percent_misses = []
volume_misses = []

for _, r in df.iterrows():
    text = r["sample"]
    g = {(s,e,l) for s,e,l in r["gold_spans"]}
    p = {(s,e,l) for s,e,l in r["pred_spans"]}
    g_spans_no_lab = {(s,e) for s,e,_ in g}
    p_spans_no_lab = {(s,e) for s,e,_ in p}

    # полные совпадения — убираем
    inter = g & p
    g_only = g - inter
    p_only = p - inter

    # boundary-only: те же окна, другой лейбл — type swap
    for s,e,l in list(g_only):
        for sp,ep,lp in list(p_only):
            if (s,e)==(sp,ep) and (l!=lp):
                type_swap[(l,lp)] += 1
                try:
                    g_only.remove((s,e,l)); p_only.remove((sp,ep,lp))
                except KeyError:
                    pass

    # boundary shift: есть предсказание рядом (перекрытие >0), но окна разные
    def overlaps(a,b):
        (s1,e1),(s2,e2)=a,b
        return max(0, min(e1,e2)-max(s1,s2))>0

    for s,e,l in g_only:
        if any(overlaps((s,e),(sp,ep)) for sp,ep,_ in p_only):
            boundary_off += 1

    # FN/FP по тегам
    for _,_,l in g_only:
        if l.startswith("B-") or l.startswith("I-") or l=="O":
            miss_by_tag[l.split("-")[-1]] += 1
    for _,_,l in p_only:
        if l.startswith("B-") or l.startswith("I-") or l=="O":
            extra_by_tag[l.split("-")[-1]] += 1

    # частные кейсы: короткие бренды FP, пропуски процентов/объёмов
    for s,e,l in g_only:
        token = text[s:e]
        if l.endswith("PERCENT") or re.search(r"(?:^|\\s)(\\d+[.,]?\\d*)\\s*%|проц", text[s-2:e+2].lower()):
            percent_misses.append((text, token))
        if l.endswith("VOLUME") or re.search(r"(\\d+[.,]?\\d*)\\s*(л|мл|шт|уп|г|кг|рулон|пак|бут)", text.lower()):
            volume_misses.append((text, text[s:e]))

    for s,e,l in p_only:
        token = text[s:e]
        if l.endswith("BRAND") and len(token) <= 4 and re.search(r"[A-Za-z]", token):
            short_brand_fp.append(token)

print("=== TYPE SWAP (gold -> pred) top 10 ===")
for (g,pred),c in type_swap.most_common(10):
    print(f"{g:>10} -> {pred:<10} : {c}")

print("\\n=== Boundary shift count ===", boundary_off)
print("\\n=== Misses by tag (FN) ===", dict(miss_by_tag))
print("=== Extras by tag (FP) ===", dict(extra_by_tag))

print("\\n=== Short BRAND FP (samples) ===", Counter(short_brand_fp).most_common(15))
print("\\nMissed PERCENT examples:", percent_misses[:10])
print("\\nMissed VOLUME examples:", volume_misses[:10])