#!/usr/bin/env python3
import argparse, ast, math, random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import pandas as pd

TYPES = ("TYPE", "BRAND", "VOLUME", "PERCENT")


def _read_any_csv(path: str) -> pd.DataFrame:
    for sep in (";", "\t", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")


def parse_ann(s: str) -> List[Tuple[int, int, str]]:
    try:
        items = ast.literal_eval(s)
        out = []
        for it in items:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                out.append((int(it[0]), int(it[1]), str(it[2])))
        return out
    except Exception:
        return []


def combo_label(ann: List[Tuple[int, int, str]]) -> str:
    lab = set()
    for _, _, tag in ann:
        if tag and tag != "O":
            typ = tag.split("-", 1)[-1]
            if typ in TYPES:
                lab.add(typ)
    return "NONE" if not lab else "+".join(sorted(lab))


def alloc_counts_per_group(group_sizes: Dict[str, int], val_size, rng: random.Random):
    groups = list(group_sizes.keys())
    n_total = sum(group_sizes.values())
    target = float(val_size) * n_total if isinstance(val_size, float) else float(val_size)
    # доли + метод наибольших остатков
    frac = {g: group_sizes[g] * target / n_total for g in groups}
    base = {g: int(frac[g]) for g in groups}
    rem = {g: frac[g] - base[g] for g in groups}
    need = int(round(target)) - sum(base.values())
    for g, _ in sorted(rem.items(), key=lambda kv: kv[1], reverse=True)[: max(0, need)]:
        base[g] += 1
    for g in groups:
        base[g] = min(base[g], group_sizes[g])
    return base


def stratified_split(df: pd.DataFrame, val_size, seed=42):
    rng = random.Random(seed)
    labels = df["annotation"].map(parse_ann).map(combo_label)
    groups = defaultdict(list)
    for i, g in enumerate(labels):
        groups[g].append(i)
    counts = {g: len(ix) for g, ix in groups.items()}
    take = alloc_counts_per_group(counts, val_size, rng)
    val_idx, tr_idx = [], []
    for g, ix in groups.items():
        rng.shuffle(ix)
        k = take[g]
        val_idx.extend(ix[:k])
        tr_idx.extend(ix[k:])
    tr_idx.sort()
    val_idx.sort()
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def summarize(df: pd.DataFrame, name: str):
    from collections import Counter

    c = Counter(df["annotation"].map(parse_ann).map(combo_label))
    total = sum(c.values())
    parts = ", ".join(f"{k}:{v} ({v/total:.1%})" for k, v in sorted(c.items()))
    print(f"[{name}] n={total} | {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="data/train.csv")
    ap.add_argument("--train_out", default="data/train_strat.csv")
    ap.add_argument("--val_out", default="data/val.csv")
    ap.add_argument("--val_size", default="0.1", help="доля (0..1) или целое число")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = _read_any_csv(args.train_in)
    if "sample" not in df.columns or "annotation" not in df.columns:
        df = pd.read_csv(args.train_in, sep=";")

    try:
        val_size = float(args.val_size) if "." in str(args.val_size) else int(args.val_size)
    except Exception:
        raise ValueError("--val_size должно быть числом, напр. 0.1 или 5000")

    summarize(df, "FULL")
    tr, val = stratified_split(df, val_size, seed=args.seed)
    summarize(tr, "TRAIN_STRAT")
    summarize(val, "VAL")

    tr.to_csv(args.train_out, sep=";", index=False)
    val.to_csv(args.val_out, sep=";", index=False)
    print(f"Saved: {args.train_out} ({len(tr)})")
    print(f"Saved: {args.val_out} ({len(val)})")


if __name__ == "__main__":
    main()
