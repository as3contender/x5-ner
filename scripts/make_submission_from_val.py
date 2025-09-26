#!/usr/bin/env python3
import argparse, ast
import pandas as pd
from ner.infer import NERPipeline
from ner.postprocess import postprocess_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--out", default="submission_val.csv")
    ap.add_argument("--model", default="artifacts/ner-checkpoint")
    ap.add_argument("--split_type", action="store_true")
    ap.add_argument("--boost_numeric", action="store_true")
    ap.add_argument("--brand_thresh", type=float, default=0.85)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--include_o", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv, sep=";")
    assert "sample" in df.columns and "annotation" in df.columns, "val.csv должен быть sample;annotation"

    pipe = NERPipeline(args.model)
    out_rows = []

    k = 0
    for i, row in df.iterrows():
        q = str(row["sample"])
        ents, log = pipe.predict_entities(q, include_o=args.include_o)  # [(start, end, tag)]
        # в postprocess_all ждём [(s,e,'B-...')] без score
        ents_pp = ents
        ents_pp = postprocess_all(
            q,
            ents_pp,
            do_split_type=args.split_type,
            do_boost_numeric=args.boost_numeric,
            brand_thresh=args.brand_thresh,
        )
        # сериализация, как в baseline: [(0,5,'B-TYPE'), ...]
        ser = "[" + ", ".join(f"({s}, {e}, '{t}')" for (s, e, t) in ents_pp) + "]"
        out_rows.append({"sample": q, "annotation": ser})

        if args.debug and k < 10 and ser != row["annotation"]:
            k += 1
            print("Q:", q)
            print("gold:", row["annotation"])
            print("raw:", [(t, s, e) for (s, e, t) in ents])
            print("pp :", ents_pp)
            print("log:", log)
            if ser == row["annotation"]:
                print("OK")
            else:
                print("NOT OK")
            print()

    sub = pd.DataFrame(out_rows, columns=["sample", "annotation"])
    # ТАК ЖЕ, как gold: ';' разделитель!
    sub.to_csv(args.out, sep=";", index=False)

    print(f"[make_submission_from_val] Saved {len(sub)} rows to {args.out}")


if __name__ == "__main__":
    main()
