# scripts/eval_submission_vs_baseline.py
import argparse
import ast
import pandas as pd
from collections import defaultdict

ENTITY_TYPES = ("TYPE", "BRAND", "VOLUME", "PERCENT")


def parse_ann(s):
    try:
        v = ast.literal_eval(s)
        out = []
        for t in v:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                start, end, tag = t
                out.append((int(start), int(end), str(tag)))
        return out
    except Exception:
        return []


def merge_bio_spans(spans):
    """
    Склеиваем BIO-спаны в цельные сущности (etype, start, end).
    Предполагаем, что в файлах аннотации как минимум начинаются с B-XXX
    и I-XXX продолжают ту же сущность.
    """
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    ents = []

    cur_type, cur_s, cur_e = None, None, None

    def flush():
        nonlocal cur_type, cur_s, cur_e
        if cur_type is not None:
            ents.append((cur_type, cur_s, cur_e))
        cur_type, cur_s, cur_e = None, None, None

    for s, e, tag in spans:
        if not tag or tag == "O":
            flush()
            continue
        if "-" in tag:
            bi, et = tag.split("-", 1)
        else:
            bi, et = "B", tag

        if et not in ENTITY_TYPES:
            flush()
            continue

        if bi == "B" or cur_type is None:
            flush()
            cur_type, cur_s, cur_e = et, s, e
        elif bi == "I":
            if cur_type == et and s <= cur_e:  # примыкание/перекрытие
                cur_e = max(cur_e, e)
            else:
                flush()
                cur_type, cur_s, cur_e = et, s, e
        else:
            flush()
            cur_type, cur_s, cur_e = et, s, e

    flush()
    return set(ents)


def evaluate(gold_df, pred_df):
    """
    gold_df: sample;annotation (эталон — твой исходный baseline submission.csv)
    pred_df: sample;annotation (твои предсказания)
    Сравнение делаем на пересечении по 'sample'.
    """
    g = gold_df.copy()
    p = pred_df.copy()
    g["gold_spans"] = g["annotation"].apply(parse_ann)
    p["pred_spans"] = p["annotation"].apply(parse_ann)

    df = pd.merge(g[["sample", "gold_spans"]], p[["sample", "pred_spans"]], on="sample", how="inner")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    mismatches = []  # собираем примеры, где не совпало

    for _, row in df.iterrows():
        text = row["sample"]
        gold_ents = {(t, s, e) for (t, s, e) in merge_bio_spans(row["gold_spans"]) if t in ENTITY_TYPES}
        pred_ents = {(t, s, e) for (t, s, e) in merge_bio_spans(row["pred_spans"]) if t in ENTITY_TYPES}

        inter = gold_ents & pred_ents
        for t, _, _ in inter:
            tp[t] += 1
        for t, _, _ in pred_ents - inter:
            fp[t] += 1
        for t, _, _ in gold_ents - inter:
            fn[t] += 1

        if gold_ents != pred_ents:
            mismatches.append({"sample": text, "gold": sorted(list(gold_ents)), "pred": sorted(list(pred_ents))})

    per_type = {}
    for t in ENTITY_TYPES:
        P = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        R = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        per_type[t] = {"precision": P, "recall": R, "f1": F1}

    macro_f1 = sum(per_type[t]["f1"] for t in ENTITY_TYPES) / len(ENTITY_TYPES)

    # общий (микро) F1 по всем сущностям
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    fn_sum = sum(fn.values())
    P_all = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    R_all = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    F1_all = 2 * P_all * R_all / (P_all + R_all) if (P_all + R_all) > 0 else 0.0

    return per_type, macro_f1, {"precision": P_all, "recall": R_all, "f1": F1_all}, len(df), mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_csv", required=True, help="Путь к исходному baseline submission.csv (sample;annotation)")
    ap.add_argument("--pred_csv", required=True, help="Путь к твоему новому submission.csv (sample;annotation)")
    ap.add_argument("--show", type=int, default=10, help="Сколько примеров расхождений вывести")
    args = ap.parse_args()

    print(f"Gold path: {args.gold_csv}\nPred path: {args.pred_csv}")

    gold = pd.read_csv(args.gold_csv, sep=";")
    pred = pd.read_csv(args.pred_csv, sep=";")

    for need in ("sample", "annotation"):
        if need not in gold.columns:
            raise ValueError(f"{args.gold_csv} должен содержать колонку '{need}'")
        if need not in pred.columns:
            raise ValueError(f"{args.pred_csv} должен содержать колонку '{need}'")

    per_type, macro_f1, overall, used, mismatches = evaluate(gold, pred)

    print(f"\nSamples compared (intersection on 'sample'): {used}")
    print("\nPer-type metrics (vs baseline):")
    for t in ENTITY_TYPES:
        m = per_type[t]
        print(f"  {t:7s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"\nMacro-F1 over types: {macro_f1:.4f}")
    print(f"Overall (micro)      P={overall['precision']:.4f}  R={overall['recall']:.4f}  F1={overall['f1']:.4f}")

    if args.show > 0 and mismatches:
        print(f"\nExamples of mismatches (first {min(args.show, len(mismatches))}):")
        for ex in mismatches[: args.show]:
            print("— sample:", ex["sample"])
            print("  gold:", ex["gold"])
            print("  pred:", ex["pred"])
            print()


if __name__ == "__main__":
    main()
