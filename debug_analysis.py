from __future__ import annotations
import pandas as pd
import regex as re
from collections import Counter
from typing import List, Tuple

TRAIN_PATH = "data/train.csv"
WORD_RE = re.compile(r"\p{L}[\p{L}\p{N}-]*", re.UNICODE)

# regex-шаблоны
RE_PERCENT_SIGN = re.compile(r"(?<!\d)\d{1,2}(?:[.,]\d)?\s*%")
RE_PERCENT_WORD = re.compile(r"\b\d{1,2}(?:[.,]\d)?\s*(?:проц|процент(?:а|ов)?)\b", re.IGNORECASE)
UNITS = ["мл", "ml", "l", "л", "г", "гр", "kg", "кг", "шт", "уп", "пак", "ш", "к"]
RE_VOLUME = re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{'|'.join(UNITS)})\b", re.IGNORECASE)
RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _read_train(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def parse_ann(s: str):
    try:
        return [(int(a[0]), int(a[1]), a[2]) for a in eval(s)]
    except Exception:
        return []


def build_fatty_lexicon(df: pd.DataFrame, top_k=200):
    mask = df["annotation"].map(parse_ann).map(lambda a: any(t.endswith("PERCENT") for *_, t in a))
    dfp = df[mask]
    cnt = Counter()
    for txt in dfp["sample"]:
        for m in WORD_RE.finditer(str(txt).lower()):
            w = m.group(0)
            if len(w) > 2:
                cnt[w] += 1
    stop = {"для", "и", "в", "на", "по", "без", "со", "из", "от", "до", "за", "процент", "проц"}
    return [w for w, _ in cnt.most_common() if w not in stop][:top_k]


try:
    df_train = _read_train(TRAIN_PATH)
    FATTY_WORDS = build_fatty_lexicon(df_train)
except Exception:
    FATTY_WORDS = ["молоко", "кефир", "сливки", "сметана", "творог", "сыр"]


def extract_explicit_numeric(text: str):
    ents = []
    for rx in (RE_PERCENT_SIGN, RE_PERCENT_WORD):
        for m in rx.finditer(text):
            ents.append((m.start(), m.end(), "B-PERCENT"))
    for m in RE_VOLUME.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    return sorted(ents)


PACK_WORDS = ["бутыл", "банка", "пакет", "упаков", "рулон", "лист", "пачк", "флакон"]


def infer_implicit_numeric(text: str, fatty_words=FATTY_WORDS):
    ents = []
    for m in RE_NUMBER.finditer(text):
        s, e = m.span()
        raw = text[s:e]
        if re.match(r"\d+%", raw):
            continue
        try:
            val = float(raw.replace(",", "."))
        except:
            continue
        ctx = text[max(0, s - 20) : min(len(text), e + 20)].lower()
        if val == 0:
            ents.append((s, e, "B-PERCENT"))
        elif 1 <= val <= 40 and any(w in ctx for w in fatty_words):
            ents.append((s, e, "B-PERCENT"))
        elif val >= 100 or any(w in ctx for w in PACK_WORDS):
            ents.append((s, e, "B-VOLUME"))
    return ents


# Отладочный анализ
def debug_preprocess(text: str):
    print(f"Анализируем: '{text}'")
    print(f"FATTY_WORDS: {FATTY_WORDS[:10]}...")

    # Проверяем явное извлечение
    explicit = extract_explicit_numeric(text)
    print(f"Явное извлечение: {explicit}")

    # Проверяем неявное извлечение
    print("\nНеявное извлечение:")
    for m in RE_NUMBER.finditer(text):
        s, e = m.span()
        raw = text[s:e]
        print(f"  Найдено число: '{raw}' на позиции {s}-{e}")

        if re.match(r"\d+%", raw):
            print(f"    Пропускаем - уже содержит %")
            continue

        try:
            val = float(raw.replace(",", "."))
            print(f"    Значение: {val}")
        except:
            print(f"    Не удалось преобразовать в число")
            continue

        ctx = text[max(0, s - 20) : min(len(text), e + 20)].lower()
        print(f"    Контекст: '{ctx}'")

        # Проверяем условия
        if val == 0:
            print(f"    → B-PERCENT (val == 0)")
        elif 1 <= val <= 40 and any(w in ctx for w in FATTY_WORDS):
            print(f"    → B-PERCENT (1<=val<=40 + жирные слова)")
        elif val >= 100 or any(w in ctx for w in PACK_WORDS):
            print(f"    → B-VOLUME (val>=100 или упаковка)")
        else:
            print(f"    → НЕ ОПРЕДЕЛЕНО")


# Проверим содержимое FATTY_WORDS
print("Проверяем лексикон жирных слов:")
print(f"Всего слов: {len(FATTY_WORDS)}")
print(f"Первые 20: {FATTY_WORDS[:20]}")

# Ищем слова связанные с маслом
oil_words = [w for w in FATTY_WORDS if "масл" in w]
print(f"\nСлова с 'масл': {oil_words}")

# Проверим есть ли 'масло' и 'сливочное'
print(f"'масло' в лексиконе: {'масло' in FATTY_WORDS}")
print(f"'сливочное' в лексиконе: {'сливочное' in FATTY_WORDS}")

# Проверим контекст для "масло сливочное 72"
text = "масло сливочное 72"
ctx = text.lower()
print(f"\nКонтекст: '{ctx}'")
print(f"Содержит 'масло': {'масло' in ctx}")
print(f"Содержит 'сливочное': {'сливочное' in ctx}")

# Проверим какие жирные слова есть в контексте
found_fatty = [w for w in FATTY_WORDS if w in ctx]
print(f"Найденные жирные слова в контексте: {found_fatty[:10]}")

print("\n" + "=" * 50 + "\n")

# Тестируем проблемные случаи
debug_preprocess("масло сливочное 72")
print("\n" + "=" * 50 + "\n")
debug_preprocess("масло сливочное 82")
