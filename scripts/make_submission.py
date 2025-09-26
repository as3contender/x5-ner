#!/usr/bin/env python3
import argparse
import pandas as pd
from tqdm import tqdm

from ner.infer import NERPipeline
from ner.utils import bio_to_entities, serialize_entities
from ner.postprocess import postprocess_all


# --- Robust CSV reader -----------------------------------------------------------
def _read_any_csv(path: str) -> pd.DataFrame:
    """Try common separators first, then fall back to python engine sniffing."""
    for sep in (";", "\t", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            # Minimal sanity: at least 1 column
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
                return df
        except Exception:
            pass
    # last resort: let pandas sniff
    return pd.read_csv(path, sep=None, engine="python")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True, help="CSV с колонкой 'sample'")
    ap.add_argument("--out", type=str, default="submission.csv", help="Куда сохранить сабмит")
    ap.add_argument("--model", type=str, required=True, help="Путь к чекпойнту (директория)")
    # ВАЖНО: явные флаги постпроцесса и порог брендов
    ap.add_argument("--split_type", action="store_true", help="Разбивать TYPE по словам (baseline style)")
    ap.add_argument("--boost_numeric", action="store_true", help="Добивать VOLUME/PERCENT регулярками")
    ap.add_argument("--brand_thresh", type=float, default=0.85, help="Порог уверенности для BRAND (0..1)")
    ap.add_argument("--add_percent", action="store_true", help="Добавлять PERCENT в постпроцессе")
    ap.add_argument("--use_new_numeric", action="store_true", help="Use improved numeric preprocessor (PERCENT/VOLUME)")
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"[make_submission] model={args.model}")
    print(f"[make_submission] test={args.test}")
    print(f"[make_submission] out={args.out}")
    print(
        f"[make_submission] split_type={args.split_type}  boost_numeric={args.boost_numeric}  brand_thresh={args.brand_thresh}"
    )

    # quick look at test columns
    try:
        _tmp = _read_any_csv(args.test)
        print(f"[make_submission] test columns: {list(_tmp.columns)}")
    except Exception as e:
        print(f"[make_submission] cannot preview test columns: {e}")

    pipe = NERPipeline(model_path=args.model)

    test_df = _read_any_csv(args.test)
    # Normalize columns: allow either 'sample' or 'search_query' (+ optional 'id')
    cols = {c.strip(): c for c in test_df.columns}
    if "sample" not in cols:
        if "search_query" in cols:
            test_df = test_df.rename(columns={cols["search_query"]: "search_query"})
            test_df["sample"] = test_df["search_query"].astype(str)
        else:
            raise AssertionError("Ожидаю колонку 'sample' или 'search_query' в тестовом CSV")
    else:
        if cols["sample"] != "sample":
            test_df = test_df.rename(columns={cols["sample"]: "sample"})
    # Keep only needed column
    test_df = test_df[["sample"]].copy()

    rows = []
    for idx, q in enumerate(tqdm(test_df["sample"], desc="Infer")):
        # 1) сырые BIO-спаны с фильтрацией слабых BRAND
        spans = pipe.predict_bio_tokens(q, brand_thresh=args.brand_thresh)
        # 2) BIO->entities
        ents = bio_to_entities(spans)
        # 3) постпроцесс: расширение/дробление TYPE + VOLUME/PERCENT regex-boost
        ents_pp = postprocess_all(
            q,
            ents,
            do_split_type=args.split_type,
            do_boost_numeric=args.boost_numeric,
            add_percent=args.add_percent,
            use_new_numeric=args.use_new_numeric,
            brand_thresh=args.brand_thresh,
        )

        # DEBUG: первые 10 — показываем до/после
        if idx < 10:

            def _fmt(es):  # [('TYPE', start, end), ...]
                return [(t, s, e) for (s, e, t) in es]

            print("\nQ:", q)
            print("raw:", _fmt(ents))
            print("pp :", _fmt(ents_pp))

        rows.append({"sample": q, "annotation": serialize_entities(ents_pp)})

    out_df = pd.DataFrame(rows, columns=["sample", "annotation"])  # фиксируем порядок колонок
    out_df.to_csv(args.out, sep=";", index=False)
    print(f"[make_submission] Saved {len(out_df)} rows to {args.out}")
    try:
        print(out_df.head(3).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
