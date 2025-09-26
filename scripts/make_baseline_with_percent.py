import argparse, ast, pandas as pd, regex as re

RE_PERCENT = re.compile(r"(?<!\d)\d{1,2}(?:[.,]\d)?\s*%")
RE_PERCENTWORD = re.compile(r"\b\d{1,2}(?:[.,]\d)?\s*(?:проц|процент(?:а|ов)?)\b", re.I)
DAIRY = {"молоко", "кефир", "сливки", "сметана", "творог", "сыр", "йогурт", "ряженка", "простокваша", "масло"}


def parse(s):
    try:
        return [tuple(x) for x in ast.literal_eval(s)]
    except:
        return []


ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)  # data/submission_baseline.csv
ap.add_argument("--out_csv", required=True)  # data/submission_baseline_v2.csv
args = ap.parse_args()

df = pd.read_csv(args.in_csv, sep=";")
rows = []
for _, r in df.iterrows():
    text = str(r["sample"])
    ents = [(int(s), int(e), str(t)) for (s, e, t) in parse(r["annotation"])]

    taken = [(s, e) for s, e, _ in ents]

    def overlaps(s, e):
        return any(ss < e and ee > s for ss, ee in taken)

    # 1) Явные проценты
    for m in RE_PERCENT.finditer(text):
        s, e = m.span()
        if not overlaps(s, e):
            ents.append((s, e, "B-PERCENT"))
            taken.append((s, e))
    for m in RE_PERCENTWORD.finditer(text):
        s, e = m.span()
        if not overlaps(s, e):
            ents.append((s, e, "B-PERCENT"))
            taken.append((s, e))

    # 2) Голые числа (молочные → PERCENT 1..40)
    for m in re.finditer(r"\b(\d{1,2})(?:[.,]\d)?\b", text):
        s, e = m.span()
        if overlaps(s, e):
            continue
        num = int(m.group(1))
        low = text.lower()
        if num == 0 and ("балтик" in low or "ноль" in low):
            ents.append((s, e, "B-PERCENT"))
            taken.append((s, e))
            continue
        if 1 <= num <= 40 and any(w in low for w in DAIRY):
            ents.append((s, e, "B-PERCENT"))
            taken.append((s, e))
            continue
        # иначе не трогаем

    ents = sorted(set(ents), key=lambda x: (x[0], x[1], x[2]))
    rows.append({"sample": text, "annotation": str(ents)})

pd.DataFrame(rows, columns=["sample", "annotation"]).to_csv(args.out_csv, sep=";", index=False)
print("Saved:", args.out_csv)
