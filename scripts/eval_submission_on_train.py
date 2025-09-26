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
    На вход: список (s, e, tag) в BIO (могут быть только B- или B- + I- на соседних кусках).
    Выход: множество цельных сущностей в виде (etype, start, end), где etype ∈ ENTITY_TYPES.
    Логика:
      - Сортируем по началу.
      - Склеиваем последовательности: B-TYPE [I-TYPE I-TYPE ...] в одну сущность (s..e).
      - Если пришёл только B-XXX (без I-), берём его как одиночную сущность.
      - Игнорируем 'O'.
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
        if tag == "O" or not tag:
            flush()
            continue
        if "-" in tag:
            bi, et = tag.split("-", 1)
        else:
            # если пришёл TYPE без префикса B-/I- — трактуем как B-
            bi, et = "B", tag

        if et not in ENTITY_TYPES:
            # неизвестные типы — пропустим
            flush()
            continue

        if bi == "B" or cur_type is None:
            # начинаем новую сущность
            flush()
            cur_type, cur_s, cur_e = et, s, e
        elif bi == "I":
            # продолжаем только если тип совпадает и спан примыкает/перекрывается
            if cur_type == et and s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                # некорректная I без B — начнём новую сущность
                flush()
                cur_type, cur_s, cur_e = et, s, e
        else:
            # на всякий случай: любые иные метки — завершить и начать заново
            flush()
            cur_type, cur_s, cur_e = et, s, e

    flush()
    return set(ents)


def evaluate(gold_df, pred_df):
    """
    gold_df: DataFrame с колонками 'sample', 'annotation'
    pred_df: DataFrame с колонками 'sample', 'annotation'
    Оценка на пересечении по sample (строгий матч).
    """
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()
    gold_df["gold_spans"] = gold_df["annotation"].apply(parse_ann)
    pred_df["pred_spans"] = pred_df["annotation"].apply(parse_ann)

    # соединим по sample (strict)
    df = pd.merge(gold_df[["sample", "gold_spans"]], pred_df[["sample", "pred_spans"]], on="sample", how="inner")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for _, row in df.iterrows():
        gold_ents = merge_bio_spans(row["gold_spans"])
        pred_ents = merge_bio_spans(row["pred_spans"])

        # оставим только известные типы
        gold_ents = {(t, s, e) for (t, s, e) in gold_ents if t in ENTITY_TYPES}
        pred_ents = {(t, s, e) for (t, s, e) in pred_ents if t in ENTITY_TYPES}

        inter = gold_ents & pred_ents
        for t, _, _ in inter:
            tp[t] += 1
        for t, _, _ in pred_ents - inter:
            fp[t] += 1
        for t, _, _ in gold_ents - inter:
            fn[t] += 1

    # посчитаем метрики по типам
    per_type = {}
    for t in ENTITY_TYPES:
        P = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        R = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        per_type[t] = {"precision": P, "recall": R, "f1": F1}

    macro_f1 = sum(per_type[t]["f1"] for t in ENTITY_TYPES) / len(ENTITY_TYPES)

    # ещё — «общий» F1 по всем сущностям (микро)
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    fn_sum = sum(fn.values())
    P_all = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    R_all = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    F1_all = 2 * P_all * R_all / (P_all + R_all) if (P_all + R_all) > 0 else 0.0

    return per_type, macro_f1, {"precision": P_all, "recall": R_all, "f1": F1_all}, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Путь к train.csv (sep=';')")
    ap.add_argument("--submission_csv", required=True, help="Путь к submission.csv (sep=';')")
    args = ap.parse_args()

    gold = pd.read_csv(args.train_csv, sep=";")
    pred = pd.read_csv(args.submission_csv, sep=";")

    # определим колонки
    if not {"sample", "annotation"} <= set(gold.columns):
        raise ValueError("train.csv должен содержать колонки: sample;annotation")
    if not {"sample", "annotation"} <= set(pred.columns):
        raise ValueError("submission.csv должен содержать колонки: sample;annotation")

    per_type, macro_f1, overall, used = evaluate(gold, pred)

    print(f"\nSamples used (intersection on 'sample'): {used}")
    print("\nPer-type metrics:")
    for t in ENTITY_TYPES:
        m = per_type[t]
        print(f"  {t:7s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"\nMacro-F1 over types: {macro_f1:.4f}")
    print(f"Overall (micro)      P={overall['precision']:.4f}  R={overall['recall']:.4f}  F1={overall['f1']:.4f}\n")


if __name__ == "__main__":
    main()
