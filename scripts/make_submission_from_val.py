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
    ap.add_argument("--logs_length", type=int, default=10)
    ap.add_argument("--logs_stop", action="store_true")
    ap.add_argument("--logs_out", default="logs_val.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv, sep=";")
    assert "sample" in df.columns and "annotation" in df.columns, "val.csv должен быть sample;annotation"

    pipe = NERPipeline(args.model)
    out_rows = []
    logs_output_rows = []

    k = 0
    error_count = 0

    for i, row in df.iterrows():
        if args.logs_stop and error_count >= args.logs_length:
            break

        q = str(row["sample"])
        ents, log, log_details = pipe.predict_entities(q)  # [(start, end, tag)]
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

        if ser != row["annotation"]:
            error_count += 1
            error_text = "1"
        else:
            error_text = "0"

        # Добавляем логи для каждой детали
        for detail in log_details:
            detail_with_error = detail + f";{error_text}"
            logs_output_rows.append(detail_with_error)

        if args.debug and k < args.logs_length and ser != row["annotation"]:
            k += 1
            print("Q:", q)
            print("gold:", row["annotation"])
            print("raw:", ents)
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

    # Разделяем строки на колонки по разделителю ;
    logs_data = []
    for i, row in enumerate(logs_output_rows):
        split_row = row.split(";")
        logs_data.append(split_row)

    logs_sub = pd.DataFrame(
        logs_data,
        columns=[
            "text",
            "token_text",
            "start",
            "end",
            "label",
            "reason",
            "p_brand",
            "p_type",
            "in_lex",
            "fuzzy_hit",
            "pure_lat",
            "short_lat",
            "has_vowel",
            "p_brand_sum",
            "p_type_sum",
            "p_O",
            "error",
        ],
    )
    logs_sub.to_csv(args.logs_out, sep=";", index=True)

    print(f"[make_submission_from_val] Saved {len(sub)} rows to {args.out} with {error_count} errors")


if __name__ == "__main__":
    main()
