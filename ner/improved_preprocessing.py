from __future__ import annotations
import pandas as pd
import regex as re
from collections import Counter
from typing import List, Tuple
import difflib

from rapidfuzz import fuzz


TRAIN_PATH = "data/train.csv"
WORD_RE = re.compile(r"\p{L}[\p{L}\p{N}-]*", re.UNICODE)

# --- Normalization & fuzzy matching helpers (for noisy user words) ---
YO_MAP = str.maketrans({"ё": "е", "Ё": "е"})


def normalize_token(tok: str) -> str:
    # lower, map ё->е, strip non-letters/digits/dash, collapse long repeats
    tok = tok.lower().translate(YO_MAP)
    tok = re.sub(r"[^\p{L}\p{N}-]+", "", tok)
    # collapse 3+ repeated chars -> single (e.g., "сметааана" -> "сметана")
    tok = re.sub(r"(\p{L})\1{2,}", r"\1", tok)
    return tok


def edit_distance(a: str, b: str) -> int:
    # quick exact / early exits
    if a == b:
        return 0
    # fallback: O(len^2), but strings are short
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return max(la, lb)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))  # deletion  # insertion  # substitution
            prev = cur
    return dp[lb]


def token_close_to_lexicon(tok: str, lex_set: set, max_ed_small=1, max_ed_big=2) -> bool:
    """
    Returns True if token is close to any word in lexicon by edit distance (len-aware),
    or fuzzy ratio above threshold when rapidfuzz is available.
    """
    t = normalize_token(tok)
    if not t:
        return False
    if t in lex_set:
        return True
    # try fast fuzzy ratio first if available (faster than full ED scan on large lexicons)

    # quick filter by length to reduce comparisons
    candidates = [w for w in lex_set if abs(len(w) - len(t)) <= 2]
    # threshold slightly ниже, чтобы ловить пары типа "малако" ~ "молоко"
    RF_THRESH = 80
    if any(fuzz.ratio(t, w) >= RF_THRESH for w in candidates):
        return True
    # fallback: проверим классическим edit distance с len-aware лимитом
    limit = max_ed_small if len(t) <= 5 else max_ed_big
    for w in candidates:
        if edit_distance(t, w) <= limit:
            return True

    return False


# regex-шаблоны
# проценты: допускаем пробелы вокруг разделителя и до 2 знаков после запятой/точки
# примеры: 1%, 1 %, 1,5%, 1 ,5 %, 10.25 %
RE_PERCENT_SIGN = re.compile(r"(?<!\d)\d{1,2}(?:\s*[.,]\s*\d{1,2})?\s*%")
RE_PERCENT_WORD = re.compile(r"\b\d{1,2}(?:[.,]\d)?\s*(?:проц|процент(?:а|ов)?)\b", re.IGNORECASE)
UNITS = ["мл", "ml", "l", "л", "г", "гр", "kg", "кг", "шт", "уп", "пак", "ш", "к", "литров", "литровый", "литра"]
RE_VOLUME = re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{'|'.join(UNITS)})\b", re.IGNORECASE)
RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# MULTIPACK: 6x1л, 6 x 1 л, 2*0.5 л, 3×200 мл
RE_MULTIPACK = re.compile(rf"\b\d+\s*[x×*]\s*\d+(?:[.,]\d+)?\s*(?:{'|'.join(UNITS)})\b", re.IGNORECASE)

# --- Word-number volumes like "пять литров", "две упаковки", "полтора литра", "пол-литра" ---
UNIT_WORDS = [
    r"литр(?:а|ов)?",
    r"миллилитр(?:а|ов)?",
    r"килограмм(?:а|ов)?",
    r"грамм(?:а|ов)?",
    r"бутылк(?:а|и|ок)",
    r"банк(?:а|и|ок)",
    r"пакет(?:а|ов)?",
    r"упаковк(?:а|и|ок)",
    r"рулон(?:а|ов)?",
    r"лист(?:а|ов)?",
    r"флакон(?:а|ов)?",
    r"штук(?:а|и|)?",
    r"шт",
]
# simple numerals 1..19 + tens (20..90) with feminine forms (две/одна)
ONE_WORD = r"(?:один|одна|одно)"
TWO_WORD = r"(?:два|две)"
THREE_TO_NINE = r"(?:три|четыре|пять|шесть|семь|восемь|девять)"
TEN_TO_NINETEEN = r"(?:десять|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|семнадцать|восемнадцать|девятнадцать)"
TENS = r"(?:двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто)"
UNIT_1_19 = rf"(?:{ONE_WORD}|{TWO_WORD}|{THREE_TO_NINE}|{TEN_TO_NINETEEN})"
# composites like "двадцать пять"
NUM_WORD = rf"(?:{TENS}(?:\s+{THREE_TO_NINE})?|{UNIT_1_19})"
# пол / полтора
HALF_NUM = r"(?:пол)(?:\s*|-)?"
POLTORA = r"(?:полтор(?:а|ы))"

# --- Word-number percents like "пять процентов", "один процент", "полпроцента" ---
ZERO_WORD = r"(?:ноль)"
PERC_WORD_FORMS = r"(?:процент(?:а|ов)?|проц\.?|проц)"
RE_WORD_PERCENT = re.compile(rf"\b(?:{ZERO_WORD}|{NUM_WORD})\s+{PERC_WORD_FORMS}\b", re.IGNORECASE)
# "пол процента" / "полпроцента" (редко встречается, но поддержим)
RE_HALF_PERCENT = re.compile(rf"\b{HALF_NUM}?{PERC_WORD_FORMS}\b", re.IGNORECASE)

RE_WORD_VOLUME = re.compile(rf"\b({NUM_WORD})\s+({'|'.join(UNIT_WORDS)})\b", re.IGNORECASE)
RE_HALF_VOLUME = re.compile(
    rf"\b(?:{HALF_NUM}({'|'.join(UNIT_WORDS)})|{POLTORA}\s+({'|'.join(UNIT_WORDS)}))\b", re.IGNORECASE
)


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
            w = normalize_token(m.group(0))
            if len(w) > 2:
                cnt[w] += 1
    stop = {"для", "и", "в", "на", "по", "без", "со", "из", "от", "до", "за", "процент", "проц"}
    vocab = [w for w, _ in cnt.most_common() if w not in stop][:top_k]
    return set(vocab)


try:
    df_train = _read_train(TRAIN_PATH)
    FATTY_WORDS = build_fatty_lexicon(df_train)
    if not isinstance(FATTY_WORDS, set):
        FATTY_WORDS = set(FATTY_WORDS)
except Exception:
    FATTY_WORDS = {normalize_token(w) for w in ["молоко", "кефир", "сливки", "сметана", "творог", "сыр"]}


def extract_explicit_numeric(text: str):
    ents = []
    for rx in (RE_PERCENT_SIGN, RE_PERCENT_WORD):
        for m in rx.finditer(text):
            ents.append((m.start(), m.end(), "B-PERCENT"))
    # word-number percents: "пять процентов", "один процент", "ноль процентов"
    for m in RE_WORD_PERCENT.finditer(text):
        ents.append((m.start(), m.end(), "B-PERCENT"))
    # half-percent: "пол процента", "полпроцента"
    for m in RE_HALF_PERCENT.finditer(text):
        ents.append((m.start(), m.end(), "B-PERCENT"))
    # multipacks first (to avoid double-capturing the unit part)
    for m in RE_MULTIPACK.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    # regular volumes
    for m in RE_VOLUME.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    # word-number based volumes
    ents.extend(extract_word_number_volume(text))

    # adjective+noun phrases like "большой объем" -> treat as VOLUME (B-/I- across tokens)
    # Handles variants: "большая/большое/большие", typos like "обьем", and "объём".
    tokens = [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    norm_tokens = [normalize_token(t) for t, _, _ in tokens]

    def _is_size_adj(t: str) -> bool:
        # catch "больш*" (большой/ая/ое/ие, больший/е и пр.) and "огромн*"
        return t.startswith("больш") or t.startswith("огромн")

    def _is_volume_noun(t: str) -> bool:
        # normalize covers ё->е, so "объём" -> "объем"; accept common typo "обьем"
        return t.startswith("объем") or t.startswith("обьем")

    for i in range(len(tokens) - 1):
        t1 = norm_tokens[i]
        t2 = norm_tokens[i + 1]
        if not t1 or not t2:
            continue
        if _is_size_adj(t1) and _is_volume_noun(t2):
            s1, e1 = tokens[i][1], tokens[i][2]
            s2, e2 = tokens[i + 1][1], tokens[i + 1][2]
            ents.append((s1, e1, "B-VOLUME"))
            ents.append((s2, e2, "I-VOLUME"))

    return sorted(ents)


def extract_word_number_volume(text: str):
    ents = []
    # "пять литров", "двадцать пять упаковок", "две бутылки"
    for m in RE_WORD_VOLUME.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    # "пол-литра", "пол литра", "полтора литра"
    for m in RE_HALF_VOLUME.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    return ents


PACK_WORDS = {normalize_token(w) for w in ["бутыл", "банка", "пакет", "упаков", "рулон", "лист", "пачк", "флакон"]}
SIZE_WORDS = {normalize_token(w) for w in ["размер"]}


def infer_implicit_numeric(text: str, fatty_words=FATTY_WORDS):
    ents = []
    # pre-tokenize once and keep normalized tokens with positions
    tokens = [(normalize_token(m.group(0)), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    norm_tokens = [t for t, _, _ in tokens if t]

    # collect explicit spans (percents/units/multipacks) to avoid re-labelling them implicitly
    explicit_spans = []
    for rx in (RE_PERCENT_SIGN, RE_PERCENT_WORD, RE_MULTIPACK, RE_VOLUME):
        for mm in rx.finditer(text):
            explicit_spans.append(mm.span())

    def overlaps_explicit(s: int, e: int) -> bool:
        return any(es < e and ee > s for (es, ee) in explicit_spans)

    # helpers to check proximity around a number position
    def neighbors(pos: int, window_chars: int = 20) -> List[str]:
        left = pos - window_chars
        right = pos + window_chars
        return [t for (t, s, e) in tokens if s < right and e > left and t]

    for m in RE_NUMBER.finditer(text):
        s, e = m.span()
        if overlaps_explicit(s, e):
            continue
        raw = text[s:e]
        # skip explicit percent like "10%"
        if RE_PERCENT_SIGN.search(raw):
            continue
        try:
            val = float(raw.replace(",", "."))
        except:
            continue
        neigh = neighbors((s + e) // 2)
        # guard: 'размер 5' and similar size patterns are NOT percent/volume
        if any(w in SIZE_WORDS for w in neigh):
            continue
        # check fatty/pack context with fuzzy tolerance
        has_fatty = any(token_close_to_lexicon(w, fatty_words) for w in neigh)
        has_pack = any(token_close_to_lexicon(w, PACK_WORDS) for w in neigh)
        label = None
        if val == 0:
            label = "B-PERCENT"
        elif 1 <= val <= 99 and has_fatty:
            label = "B-PERCENT"
        elif val >= 100 or (has_pack and val >= 2):
            label = "B-VOLUME"
        if label:
            ents.append((s, e, label))
    return ents


def split_percent_with_spaces(text: str, entities: List[Tuple[int, int, str]]):
    """
    Разбивает сущности B-PERCENT, содержащие пробелы, на B-PERCENT и I-PERCENT части.
    Например: (7, 10, 'B-PERCENT') для "1 %" -> [(7, 8, 'B-PERCENT'), (8, 10, 'I-PERCENT')]
    """
    result = []

    for start, end, label in entities:
        if label == "B-PERCENT":
            # Извлекаем текст сущности
            entity_text = text[start:end]

            # Проверяем есть ли пробелы внутри
            if " " in entity_text:
                # Находим первый пробел
                first_space_idx = entity_text.find(" ")

                # Разбиваем на две части
                # Первая часть (число) - B-PERCENT
                first_end = start + first_space_idx
                result.append((start, first_end, "B-PERCENT"))

                # Вторая часть (пробел + %) - I-PERCENT
                result.append((first_end + 1, end, "I-PERCENT"))
            else:
                # Если пробелов нет, оставляем как есть
                result.append((start, end, label))
        else:
            # Для других типов сущностей оставляем как есть
            result.append((start, end, label))

    return result


def _split_entity_with_spaces(text: str, entities: List[Tuple[int, int, str]], base_label: str):
    """
    Разбивает сущности base_label, если внутри есть пробел (на B- и I- части).
    Пример: "2 л" -> [(s, s_num_end, 'B-VOLUME'), (s_num_end, e, 'I-VOLUME')]
    """
    result = []
    for start, end, label in entities:
        if label == f"B-{base_label}":
            entity_text = text[start:end]
            if " " in entity_text:
                first_space_idx = entity_text.find(" ")
                first_end = start + first_space_idx
                result.append((start, first_end, f"B-{base_label}"))
                result.append((first_end + 1, end, f"I-{base_label}"))
            else:
                result.append((start, end, label))
        else:
            result.append((start, end, label))
    return result


def split_volume_with_spaces(text: str, entities: List[Tuple[int, int, str]]):
    return _split_entity_with_spaces(text, entities, base_label="VOLUME")


def split_volume_multipack(text: str, entities: List[Tuple[int, int, str]]):
    """
    Если внутри VOLUME есть множитель (x, ×, *), разбиваем на B-/I- вокруг первого разделителя.
    Примеры:
      "6x1л"   -> [(start, start+1, B-VOLUME), (start+1, end, I-VOLUME)]  (условно на границе 'x')
      "6 x 1 л"-> [(s, s_delim, B-VOLUME), (s_delim, e, I-VOLUME)]
    """
    result = []
    for start, end, label in entities:
        if label != "B-VOLUME":
            result.append((start, end, label))
            continue
        seg = text[start:end]
        # ищем первый символ-мультипликатор
        m = re.search(r"[x×*]", seg)
        if not m:
            result.append((start, end, label))
            continue
        cut = start + m.start()  # позиция разделителя в исходном тексте
        # Если разделитель в самом начале/конце — не режем
        if cut <= start or cut >= end:
            result.append((start, end, label))
            continue
        result.append((start, cut, "B-VOLUME"))
        result.append((cut, end, "I-VOLUME"))
    return result


def merge_overlapping_entities(entities: List[Tuple[int, int, str]]):
    """
    Улучшенное схлопывание перекрывающихся сущностей.
    Особенно хорошо обрабатывает случаи типа "1 %" -> [(7,8,'B-PERCENT'), (9,10,'I-PERCENT')]
    """
    if not entities:
        return []

    # Сортируем по начальной позиции
    entities = sorted(entities)
    result = []

    for start, end, label in entities:
        # Проверяем, есть ли перекрытие с последней добавленной сущностью
        if result:
            last_start, last_end, last_label = result[-1]

            # Если сущности одного типа и перекрываются или находятся рядом
            if label == last_label and not (end <= last_start or start >= last_end):

                # Если текущая сущность полностью входит в последнюю - пропускаем
                if start >= last_start and end <= last_end:
                    continue
                # Если последняя сущность полностью входит в текущую - заменяем
                elif last_start >= start and last_end <= end:
                    result[-1] = (start, end, label)
                    continue
                # Если частично перекрываются - берем объединение
                else:
                    result[-1] = (min(start, last_start), max(end, last_end), label)
                    continue

        result.append((start, end, label))

    return result


def preprocess_query(text: str):
    explicit = extract_explicit_numeric(text)
    implicit = infer_implicit_numeric(text)
    all_entities = sorted(explicit + implicit)

    # Сначала схлопываем перекрывающиеся сущности
    merged_entities = merge_overlapping_entities(all_entities)

    # Затем разбиваем B-PERCENT с пробелами на B-PERCENT и I-PERCENT
    final_entities = split_percent_with_spaces(text, merged_entities)

    # Разбиваем VOLUME с пробелами на B-/I- части (например, "2 л")
    final_entities = split_volume_with_spaces(text, final_entities)

    # Разбиваем мультипликативные объёмы: "6x1л", "2 x 1 л"
    final_entities = split_volume_multipack(text, final_entities)

    return final_entities


if __name__ == "__main__":
    # Тестируем на примерах
    test_cases = [
        "молоко 1 %",
        "кефир 1%",
        "масло сливочное 72",
        "масло сливочное 82",
        "сливки 10%",
        "сливки 33 %",
        "сметна 20",  # опечатка: 'сметна' ~ 'сметана' -> PERCENT
        "малако 3,2",  # опечатка: 'малако' ~ 'молоко' -> PERCENT
        "бумага туалетная 12",  # без юнитов -> VOLUME (pack context)
        "молоко 2 л",  # явный VOLUME, нужно B-VOLUME + I-VOLUME
        "вода 1 л",  # явный VOLUME, нужно B-VOLUME + I-VOLUME
        "вода 6x1л",  # мультипак без пробелов
        "вода 6 x 1 л",  # мультипак с пробелами
        "сок 2*0.5 л",  # мультипак со звёздочкой
        "напиток 3×200 мл",  # мультипак со знаком ×
        "молоко ультрапастеризованное 3.2",
    ]

    test_cases_volume = [
        "вода питьевая большой объём",
        "вода пять литров",
        "вода шишкин лес 5литров",
        "колготки размер 5",
        "молоко 2 л",
        "пакеты 60 л",
    ]

    test_cases_volume.extend(
        [
            "вода пять литров",
            "вода две упаковки",
            "сахар пол килограмма",
            "масло пол-литра",
            "молоко полтора литра",
        ]
    )

    # Добавим тесты на проценты словами
    test_cases += [
        "кефир пять процентов",
        "творог один процент",
        "сыр ноль процентов",
        "сливки пол процента",
        "майонез двадцать пять процентов",
    ]

    print("Результаты улучшенной предобработки:")
    for text in test_cases_volume:
        result = preprocess_query(text)
        print(f"{text} => {result}")

    # Также проверим на реальных данных из percent_example.csv
    print("\n" + "=" * 60 + "\n")
    print("Проверка на реальных данных:")
    try:
        df_examples = pd.read_csv("data/volume_example.csv", sep=";")
        for idx, row in df_examples.iterrows():
            sample = row["sample"]
            annotation = row["annotation"]
            result = preprocess_query(sample)
            print(f"{sample} {annotation} => {result}")
            if idx >= 30:  # Ограничиваем вывод
                break
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
