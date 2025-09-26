#!/usr/bin/env python
import pandas as pd
import ast, re, sys
from collections import Counter
from pathlib import Path

SRC_FILES = ["data/train.csv", "data/val.csv", "data/test_with_submission.csv"]
OUT_PATH = Path("artifacts/brand_lexicon.txt")
MIN_LEN = 3
MAX_LEN = 30

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def norm_text(s: str) -> str:
    s = s.lower()
    # оставляем только буквы/цифры латиницы и кириллицы
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "", s)


def parse_ann(s):
    if pd.isna(s):
        return []
    # пробуем безопасный парсер
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    # запасной вариант: вытащить кортежи вида (start, end, 'TAG')
    try:
        tuples = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']+)'\s*\)", str(s))
        return [(int(a), int(b), c) for a, b, c in tuples]
    except Exception:
        return []


def base_tag(tag: str) -> str:
    return tag.split("-", 1)[-1]


def iter_rows(df: pd.DataFrame):
    # определяем колонку с текстом
    if "sample" in df.columns:
        col_q = "sample"
    elif "search_query" in df.columns:
        col_q = "search_query"
    else:
        raise KeyError("Не найдены колонки 'sample' или 'search_query' в датасете")
    for _, row in df.iterrows():
        q = str(row[col_q]).strip()
        spans_raw = row.get("annotation", "")
        spans = parse_ann(str(spans_raw))
        yield q, spans


def build_lexicon():
    lex = Counter()
    diag_total_rows = 0
    diag_rows_with_brand = 0
    diag_total_brand_spans = 0

    for path in SRC_FILES:
        p = Path(path)
        if not p.exists():
            continue
        # файлы ;-разделённые, а в annotation есть запятые
        try:
            df = pd.read_csv(
                p,
                sep=";",
                engine="python",
                header=0,  # первая строка — заголовки
                dtype=str,
                keep_default_na=False,  # не превращать пустые строки в NaN
            )
        except Exception:
            # фолбэк: если заголовка нет/кривой — читаем без него
            try:
                df = pd.read_csv(
                    p,
                    sep=";",
                    engine="python",
                    header=None,
                    names=["sample", "annotation"],
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception as e:
                print(f"[warn] не смог прочитать {path}: {e}", file=sys.stderr)
                continue

        for q, spans in iter_rows(df):
            diag_total_rows += 1
            had_brand = False
            for s, e, lab in spans:
                if "BRAND" in base_tag(lab):
                    diag_total_brand_spans += 1
                    had_brand = True
                    span_txt = q[s:e]
                    # добавляем целиком нормализованный спан
                    whole = norm_text(span_txt)
                    if MIN_LEN <= len(whole) <= MAX_LEN:
                        lex[whole] += 1
                    # и каждое слово из него
                    for m in WORD_RE.finditer(span_txt):
                        w = norm_text(m.group(0))
                        if MIN_LEN <= len(w) <= MAX_LEN:
                            lex[w] += 1
            if had_brand:
                diag_rows_with_brand += 1

    return lex, {
        "rows_total": diag_total_rows,
        "rows_with_brand": diag_rows_with_brand,
        "brand_spans_total": diag_total_brand_spans,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lex, info = build_lexicon()

    print("[diag]", info)
    if not lex:
        print("[diag] Лексикон пуст. Проверь:")
        print("  • что в CSV есть колонка 'annotation' и она содержит списки спанов")
        print("  • что в аннотациях встречается 'BRAND' (а не, например, 'BREND'/'Brand')")
        print("  • индексы спанов соответствуют строкам (s/e внутри границ текста)")
        sys.exit(0)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for w, _ in lex.most_common():
            f.write(w + "\n")

    print(f"[ok] saved {len(lex)} brand entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
