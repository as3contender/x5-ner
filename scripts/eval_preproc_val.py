#!/usr/bin/env python3
"""
Compare gold val.csv vs submission.csv at character-level for selected entity types.

Reports all samples where labels differ for any of the selected types.
Types are drawn from: TYPE, BRAND, VOLUME, PERCENT, O
- For O: we treat it as "not in any selected entity type".

Usage:
  PYTHONPATH=$(pwd) python scripts/diff_val_vs_submission.py \
    --gold data/val.csv \
    --pred submission_val.csv \
    --types TYPE,BRAND,VOLUME,PERCENT,O \
    --out diff_report.csv \
    --show 20
"""
import argparse
import pandas as pd
import ast
from typing import List, Tuple, Set, Dict

def parse_ann(s: str) -> List[Tuple[int,int,str]]:
    try:
        arr = eval(s)
        out = []
        for triple in arr:
            if isinstance(triple, (list,tuple)) and len(triple)==3:
                s0,e0,lab = int(triple[0]), int(triple[1]), str(triple[2])
                out.append((s0,e0,lab))
        return out
    except Exception:
        return []

def normalize_type(tag: str) -> str:
    # BIO → plain type
    if not tag:
        return ""
    if "-" in tag:
        return tag.split("-", 1)[-1]
    return tag

def build_char_labels(text: str, entities: List[Tuple[int,int,str]], selected_types: Set[str]) -> List[str]:
    """
    Returns a list of labels for each character position in text:
    - 'TYPE'/'BRAND'/'VOLUME'/'PERCENT' if the char is covered by at least one entity of that type (BIO ignored)
    - 'O' otherwise
    Only entities whose normalized type is in selected_types (excluding 'O' itself) are applied.
    """
    n = len(text)
    labels = ["O"] * n
    for s,e,lab in entities:
        t = normalize_type(lab).upper()
        if t in ("", "O"):
            continue
        if t not in {"TYPE","BRAND","VOLUME","PERCENT"}:
            continue
        if t not in selected_types:
            continue
        ss = max(0, min(s, n))
        ee = max(0, min(e, n))
        for i in range(ss, ee):
            labels[i] = t
    # If 'O' is not in selected_types, we can map non-selected back to 'O' or keep as 'O' (already)
    if "O" not in selected_types:
        # map all non-selected types back to 'O' already handled above
        pass
    return labels

def labels_diff(a: List[str], b: List[str], selected_types: Set[str]) -> bool:
    """Whether there is any difference for selected types (including 'O' if requested)."""
    # When O included, we compare exact labels among allowed set; otherwise compare only on entity positions
    if "O" in selected_types:
        # strict char-by-char equality check but we normalize non-selected labels to 'O'
        return any(x != y for x,y in zip(a,b))
    else:
        # Difference exists if any char is labeled with an entity type (from selected_types) in one and different in the other
        for x,y in zip(a,b):
            x_ent = x if x in selected_types else "O"
            y_ent = y if y in selected_types else "O"
            if x_ent != y_ent:
                return True
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold val.csv (sample;annotation)")
    ap.add_argument("--pred", required=True, help="Path to predicted submission.csv (sample;annotation)")
    ap.add_argument("--types", default="TYPE,BRAND,VOLUME,PERCENT,O",
                    help="Comma-separated list of types to compare. Allowed: TYPE,BRAND,VOLUME,PERCENT,O")
    ap.add_argument("--out", default="diff_report.csv", help="CSV with differing rows")
    ap.add_argument("--show", type=int, default=20, help="Print first N diffs")
    args = ap.parse_args()

    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    allowed = {"TYPE","BRAND","VOLUME","PERCENT","O"}
    bad = [t for t in types if t not in allowed]
    if bad:
        raise ValueError(f"Unknown types: {bad}. Allowed: {sorted(list(allowed))}")
    selected_types = set(types)

    gold_df = pd.read_csv(args.gold, sep=";")
    pred_df = pd.read_csv(args.pred, sep=";")
    if not {"sample","annotation"}.issubset(gold_df.columns) or not {"sample","annotation"}.issubset(pred_df.columns):
        raise ValueError("Both CSVs must have columns: sample;annotation (sep=';').")

    # Join on sample
    df = gold_df.merge(pred_df, on="sample", how="inner", suffixes=("_gold","_pred"))
    print(f"Samples compared (intersection on 'sample'): {len(df)}")

    diffs = []
    for _, row in df.iterrows():
        text = str(row["sample"])
        gold_ents = parse_ann(row["annotation_gold"])
        pred_ents = parse_ann(row["annotation_pred"])

        gold_chars = build_char_labels(text, gold_ents, selected_types)
        pred_chars = build_char_labels(text, pred_ents, selected_types)

        if labels_diff(gold_chars, pred_chars, selected_types):
            # build short human-friendly preview of mismatched segments
            # find ranges where labels differ
            spans = []
            cur = None
            for i,(g,p) in enumerate(zip(gold_chars, pred_chars)):
                if g != p:
                    if cur is None:
                        cur = [i,i+1]
                    else:
                        cur[1] = i+1
                else:
                    if cur is not None:
                        spans.append(tuple(cur)); cur=None
            if cur is not None:
                spans.append(tuple(cur))

            diffs.append({
                "sample": text,
                "gold": row["annotation_gold"],
                "pred": row["annotation_pred"],
                "mismatch_spans": spans
            })

    diff_df = pd.DataFrame(diffs, columns=["sample","gold","pred","mismatch_spans"])
    diff_df.to_csv(args.out, index=False)
    print(f"Found {len(diff_df)} diffs. Saved to {args.out}")

    # Show first N
    for i in range(min(args.show, len(diff_df))):
        r = diff_df.iloc[i]
        print("— sample:", r["sample"])
        print("  gold:", r["gold"])
        print("  pred:", r["pred"])
        print("  mismatch_spans:", r["mismatch_spans"])

if __name__ == "__main__":
    main()
