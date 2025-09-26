#!/usr/bin/env python3
import argparse, ast, random
from typing import List, Tuple
import pandas as pd
import regex as re

DAIRY = [
    "молоко",
    "кефир",
    "сливки",
    "сметана",
    "творог",
    "сыр",
    "йогурт",
    "ряженка",
    "простокваша",
    "масло",
    "сырок",
    "творожок",
]
ADJ = ["ультрапастеризованное", "пастеризованное", "деревенское", "фермерское", "обезжиренное", "классическое"]

# проценты: целые и десятичные (рус/англ разделитель)
PERCENTS_INT = [1, 2, 3, 3, 3, 3, 5, 9, 10, 10, 15, 20, 25, 33]  # частоты похожи на рынок
PERCENTS_DEC = ["1.5", "2.5", "3.2", "3,2", "6.5"]

FMT = [
    "{prod} {p}%",
    "{prod} {p} %",
    "{prod} {p} процентов",
    "{prod} {p} проц",
    "{prod} {adj} {p}%",
    "{prod} {adj} {p} %",
]


def make_sample(rng: random.Random) -> str:
    prod = rng.choice(DAIRY)
    use_adj = rng.random() < 0.5
    adj = rng.choice(ADJ)
    if rng.random() < 0.2:
        p = rng.choice(PERCENTS_DEC)
    else:
        p = str(rng.choice(PERCENTS_INT))
    fmt = rng.choice(FMT)
    txt = fmt.format(prod=prod, p=p, adj=adj)
    # нормализуем двойные пробелы
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def annotate_percent(text: str) -> List[Tuple[int, int, str]]:
    """Размечаем TYPE по словам-товарам и PERCENT на процентной группе."""
    ents: List[Tuple[int, int, str]] = []
    # TYPE: разбиваем первые 1-2 слова (prod и, возможно, adj) до числа
    # найдём число в тексте
    m = re.search(r"\d+[.,]?\d?\s*(?:%|проц|процент)", text, flags=re.IGNORECASE)
    p_start, p_end = m.span() if m else (len(text), len(text))

    # отметим TYPE как слова до p_start
    for m2 in re.finditer(r"\p{L}[\p{L}\p{N}-]*", text[:p_start], flags=re.UNICODE):
        s, e = m2.span()
        ents.append((s, e, "B-TYPE"))

    # PERCENT спан = вся найденная группа с %/словом
    if m:
        ents.append((p_start, p_end, "B-PERCENT"))

    # сжать дубли и отсортировать
    ents = sorted(set(ents), key=lambda x: (x[0], x[1], x[2]))
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="data/train_strat.csv")
    ap.add_argument("--train_out", default="data/train_aug.csv")
    ap.add_argument("--n_new", type=int, default=2000, help="сколько синтетических строк добавить")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.train_in, sep=";")

    # сгенерим N примеров
    rows = []
    for _ in range(args.n_new):
        txt = make_sample(rng)
        ents = annotate_percent(txt)
        rows.append({"sample": txt, "annotation": str(ents)})

    df_new = pd.DataFrame(rows, columns=["sample", "annotation"])
    out = pd.concat([df, df_new], ignore_index=True)
    out.to_csv(args.train_out, sep=";", index=False)
    print(f"Base train: {len(df)}  | Added PERCENT: {len(df_new)}  | Saved: {args.train_out} = {len(out)}")


if __name__ == "__main__":
    main()
