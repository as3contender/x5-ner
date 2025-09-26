import argparse
import pandas as pd
from ner.dataset import read_train, preview_alignment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", default="cointegrated/rubert-tiny")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="alignment_preview.csv")
    args = ap.parse_args()

    df = read_train(args.train)
    prev = preview_alignment(args.model, df, n=args.n)
    out_rows = []
    for _, r in prev.iterrows():
        out_rows.append({
            "text": r["text"],
            "gold_spans": r["gold_spans"],
            "triplets": r["triplets"]
        })
    out = pd.DataFrame(out_rows)
    out.to_csv(args.out, index=False)
    print(f"Saved preview to {args.out}")

if __name__ == "__main__":
    main()