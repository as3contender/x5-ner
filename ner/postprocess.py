from __future__ import annotations
from typing import List, Tuple, Set
import regex as re
import os

from ner.improved_preprocessing import preprocess_query as _numeric_preproc


# --------- Регэкспы для числовых сущностей ---------
# проценты: 3%, 3 %, 3.2%, 3,2 %
RE_PERCENT = re.compile(
    r"""
    (?<!\d)                         # не цифра слева
    (?P<val>\d{1,2}(?:[.,]\d)?)     # 1-2 цифры + необязательная десятичная
    \s*%                            # пробелы и знак процента
""",
    re.VERBOSE,
)

# объёмы/вес/шт: 200мл, 1 л, 0.5л, 200 г, 2 шт
RE_VOLUME = re.compile(
    r"""
    (?<!\d)
    (?P<val>\d+(?:[.,]\d+)?)
    \s*
    (?P<u>мл|ml|l|л|г|гр|kg|кг|шт|уп|пак)\b
""",
    re.IGNORECASE | re.VERBOSE,
)

# мультипаки: 6x1л, 2*200г, 3×90 г
RE_MULTIPACK = re.compile(
    r"""
    (?<!\d)
    (?P<count>\d+)\s*[x×*]\s*(?P<val>\d+(?:[.,]\d+)?)\s*
    (?P<u>мл|ml|l|л|г|гр|kg|кг|шт)\b
""",
    re.IGNORECASE | re.VERBOSE,
)

# --- helper: detect numeric-only text (e.g., "3", "3,2", "  3.2  ") ---
RE_NUMERIC_ONLY = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*$")

# --------- Предлоги для замены сущностей после них ---------
DEFAULT_PREPOSITIONS = {
    # Simple and common Russian prepositions (lowercase)
    "в",
    "во",
    "на",
    "к",
    "ко",
    "от",
    "до",
    "из",
    "изо",
    "с",
    "со",
    "у",
    "за",
    "для",
    "по",
    "о",
    "об",
    "обо",
    "при",
    "через",
    "над",
    "под",
    "перед",
    "между",
    "про",
    "без",
    "около",
    "вокруг",
    "после",
    "среди",
    "вне",
    "кроме",
    "ради",
    "согласно",
    "насчёт",
    "насчет",
    "вместо",
    "вроде",
    "наперекор",
    "вопреки",
    "сквозь",
    "путём",
    "путем",
    "благодаря",
    "из-за",
    "изза",
    "из-под",
    "изпод",
    "вслед",
    "навстречу",
    "мимо",
    "вдоль",
    "поперёк",
    "поперек",
    "вглубь",
    "вширь",
    "вокрест",
    "попросту",
    "доя",
    "мытья",
    "дл",
}


def _span_is_numeric(text: str, s: int, e: int) -> bool:
    s -= 1
    e += 1
    if s < 0 or e > len(text) or s >= e:
        return False
    return bool(RE_NUMERIC_ONLY.match(text[s:e]))


BRAND_LEX_PATH = os.environ.get("BRAND_LEXICON", "artifacts/brand_lexicon.txt")
_BRAND_LEX = None


def _overlaps(s: int, e: int, spans: List[Tuple[int, int]]) -> bool:
    for ss, ee in spans:
        if ss < e and ee > s:
            return True
    return False


def _only_separators(text: str, s: int, e: int) -> bool:
    """
    Returns True if the substring text[s:e] contains only separators (spaces/punct),
    i.e., no letters or digits. Used to decide whether spans are 'adjacent'.
    """
    if s >= e:
        return True
    # if there's any letter or digit between spans, they're not adjacent
    return re.search(r"[\p{L}\p{N}]", text[s:e]) is None


def normalize_token(text: str) -> str:
    """Нормализует токен для сравнения с предлогами."""
    return text.strip().strip("\t\r\n .,!?:;\"'«»()[]{}-—").lower()


def replace_after_prepositions(
    text: str, entities: List[Tuple[int, int, str]], prepositions: Set[str] = None
) -> List[Tuple[int, int, str]]:
    """
    Заменяет сущности, которые идут сразу после предлогов или слова "все", на 'O'.
    Если сущность имеет метку 'O' и является предлогом или словом "все", то следующая сущность также становится 'O'.
    """
    if not entities:
        return entities

    if prepositions is None:
        prepositions = DEFAULT_PREPOSITIONS

    # Добавляем "все" к списку слов, которые зануляют следующую сущность
    words_to_zero_next = prepositions | {"все"}

    # Work on a mutable copy
    mutable: List[List[object]] = [[a, b, c] for (a, b, c) in entities]
    i = 0
    while i < len(mutable) - 1:
        start, end, label = mutable[i]
        if label == "O":
            token_text = text[start:end]
            if normalize_token(token_text) in words_to_zero_next:
                # Set the next segment label to 'O'
                nxt = mutable[i + 1]
                nxt[2] = "O"
                # Note: we only change the immediate next segment, as requested
        i += 1
    return [(int(a), int(b), str(c)) for (a, b, c) in mutable]


def stitch_consecutive_B_to_I(text: str, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Convert sequences of consecutive B-TYPE (or B-BRAND) that are adjacent
    (separated only by spaces/punct) into B-... followed by I-... for continuity.
    Example:
      [(0,4,'B-TYPE'), (5,10,'B-TYPE')] -> [(0,4,'B-TYPE'), (5,10,'I-TYPE')]
    Works analogously for BRAND. Leaves other labels untouched.
    """
    if not entities:
        return entities
    ents = sorted(entities, key=lambda x: (x[0], x[1], x[2]))
    out: List[Tuple[int, int, str]] = []
    prev_s = prev_e = None
    prev_core = None
    prev_tag = None
    for s, e, t in ents:
        if t.startswith("B-") and prev_tag is not None:
            core = t.split("-", 1)[-1]
            if prev_core == core and _only_separators(text, prev_e, s):
                # same class (TYPE or BRAND) and adjacent -> make it I-*
                t = f"I-{core}"
        # push current
        out.append((s, e, t))
        # update prev only for TYPE/BRAND chainable labels (B-* or I-*)
        cur_core = t.split("-", 1)[-1] if "-" in t else None
        if cur_core in {"TYPE", "BRAND"}:
            prev_s, prev_e, prev_tag, prev_core = s, e, t, cur_core
        else:
            prev_s = prev_e = prev_tag = prev_core = None
    return out


def _merge_and_dedup_entities(text: str, base_ents: List[Tuple[int, int, str]], add_ents: List[Tuple[int, int, str]]):
    """
    Merge model entities with extra numeric entities from improved preprocessor.
    Conflict policy:
      - If overlap TYPE/BRAND (base) vs numeric (add) -> keep TYPE/BRAND (base), drop numeric (add)
      - If overlap numeric (base) vs numeric (add)   -> prefer numeric (add), drop numeric (base)
      - If exact duplicate -> keep one
      - Otherwise append
    """
    if not add_ents:
        return sorted(set(base_ents), key=lambda x: (x[0], x[1], x[2]))

    NUMERIC = {"B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT", "VOLUME", "PERCENT"}

    merged = list(base_ents)

    def is_numeric(tag: str) -> bool:
        t = tag.upper()
        return t in NUMERIC or t.split("-")[-1].upper() in {"VOLUME", "PERCENT"}

    out = []
    # We'll rebuild merged with conflict resolution to allow dropping base numeric when add numeric is better.
    for s0, e0, t0 in merged:
        # print(f"merged: {s0}, {e0}, {t0}")
        out.append((s0, e0, t0))

    for s, e, t in add_ents:
        # print(f"add_ents: {s}, {e}, {t}")
        keep_add = True
        # resolve conflicts against current 'out'
        new_out = []
        for ss, ee, tt in out:
            # exact duplicate -> keep one, skip adding duplicate
            if s == ss and e == ee and t == tt:
                keep_add = False
                new_out.append((ss, ee, tt))
                continue

            # overlap?
            if ss < e and ee > s:
                base_is_num = is_numeric(tt)
                add_is_num = is_numeric(t)

                if add_is_num and base_is_num:
                    # numeric vs numeric -> prefer 'add' (from improved preproc)
                    # drop the base numeric span and keep the add later
                    # i.e., do not carry over (ss,ee,tt)
                    continue
                elif (not base_is_num) and add_is_num:
                    # Default policy: TYPE/BRAND (base) vs numeric (add) -> prefer base (drop add)
                    # Exception: if base is TYPE and the overlapping text is purely numeric ("3", "3,2"),
                    # we prefer NUMERIC (from improved preproc), because numbers inside TYPE are almost always %/volume.
                    # print(f"base_is_num: {base_is_num}, add_is_num: {add_is_num}")
                    prefer_add = False
                    if tt.startswith(("B-TYPE", "I-TYPE")):
                        # print(f"tt.startswith(('B-TYPE', 'I-TYPE')): {tt.startswith(('B-TYPE', 'I-TYPE'))}")
                        os_ = max(s, ss)
                        oe_ = min(e, ee)
                        # print(f"os_: {os_}, oe_: {oe_}")
                        if _span_is_numeric(text, os_, oe_):
                            # print(f"_span_is_numeric(text, os_, oe_): {_span_is_numeric(text, os_, oe_)}")
                            prefer_add = True
                    if prefer_add:
                        # drop the base TYPE span; do not carry it over, keep add later
                        # print(f"drop the base TYPE span; do not carry it over, keep add later")
                        continue
                    else:
                        keep_add = False
                        new_out.append((ss, ee, tt))
                        continue
                else:
                    # Other overlaps (e.g., base numeric? or unknown types): be conservative, keep base and drop add
                    keep_add = False
                    new_out.append((ss, ee, tt))
                    continue
            else:
                # no overlap, keep base
                new_out.append((ss, ee, tt))

        # print(f"new_out: {new_out}")
        out = new_out
        if keep_add:
            out.append((s, e, t))

    return sorted(set(out), key=lambda x: (x[0], x[1], x[2]))


# --------- Дробление TYPE на слова ---------
# слово = непрерывная последовательность букв/цифр (по \p{L}\p{N})
RE_WORD = re.compile(r"\p{L}[\p{L}\p{N}-]*", re.UNICODE)

# --------- Расширение TYPE вправо перед дроблением ---------


def _skip_spaces_punct(i: int, text: str) -> int:
    """Сдвинуться вправо, пропуская пробелы и простую пунктуацию."""
    while i < len(text) and text[i] in " \t.,;:!?":
        i += 1
    return i


def _expand_right_to_words(text: str, s: int, e: int, max_words: int = 2) -> tuple[int, int]:
    """Расширяет спан вправо, захватывая до max_words следующих слов, если они примыкают."""
    cur_e = e
    added = 0
    while added < max_words:
        j = _skip_spaces_punct(cur_e, text)
        if j >= len(text):
            break
        m = RE_WORD.match(text, j)
        if not m:
            break
        ws, we = m.span()
        cur_e = we
        added += 1
    return s, cur_e


def expand_phrases_right(
    text: str,
    entities: List[Tuple[int, int, str]],
    tag_bases: Tuple[str, ...] = ("TYPE", "BRAND"),
    max_words: int = 4,
) -> List[Tuple[int, int, str]]:
    """
    For each (s,e,'B-{TAG}') where TAG in tag_bases, expand span to the right capturing up to `max_words`
    next words if they are adjacent (separated only by spaces/punct), not overlapping other entities.
    """
    out: List[Tuple[int, int, str]] = []
    # Build "other" spans map per tag to avoid overlaps with non-target tags
    all_other = [(s, e) for (s, e, t) in entities if (not t.startswith("B-")) or (t.split("-", 1)[-1] not in tag_bases)]
    for s, e, tag in entities:
        if not tag.startswith("B-"):
            out.append((s, e, tag))
            continue
        base = tag.split("-", 1)[-1]
        if base not in tag_bases:
            out.append((s, e, tag))
            continue
        ns, ne = _expand_right_to_words(text, s, e, max_words=max_words)
        # избегаем пересечений с другими сущностями
        if _overlaps(ns, ne, all_other):
            out.append((s, e, tag))
        else:
            out.append((ns, ne, tag))
    return sorted(set(out), key=lambda x: (x[0], x[1], x[2]))


def split_tags_into_words(
    text: str,
    entities: List[Tuple[int, int, str]],
    tag_bases: Tuple[str, ...] = ("TYPE", "BRAND"),
    min_word_len: int = 1,
) -> List[Tuple[int, int, str]]:
    """
    Replaces each (s,e,'B-{TAG}') where TAG in tag_bases with a set of (s_i,e_i,'B-{TAG}') per word inside the span.
    Needed to align with gold where TYPE/BRAND are often annotated per word.
    """
    out: List[Tuple[int, int, str]] = []
    for s, e, tag in entities:
        base = tag.split("-", 1)[-1]
        if base not in tag_bases:
            out.append((s, e, tag))
            continue

        span_text = text[s:e]
        # print(f"span_text: {span_text}")
        if " " in span_text:
            span_text_split = span_text.split(" ")
            # print(f"span_text_split: {span_text_split}")
            if tag.startswith("B-"):
                is_b = True
            for span_text_part in span_text_split:
                if is_b:
                    pr = "B"
                    is_b = False
                else:
                    pr = "I"
                out.append((s, s + len(span_text_part), f"{pr}-{base}"))
                s += len(span_text_part) + 1
        else:
            out.append((s, e, tag))

    # keep untouched non-target tags already appended
    out = sorted(set(out), key=lambda x: (x[0], x[1], x[2]))
    return out


def merge_across_joiners(
    text: str,
    entities: List[Tuple[int, int, str]],
    joiners: Tuple[str, ...] = ("-", "–", "—", ".", "+", "/"),
    allow_spaces: bool = True,
) -> List[Tuple[int, int, str]]:
    """
    Склеивает соседние спаны одной сущности (TYPE/BRAND), если между ними только
    символы-соединители (joiners) и, опционально, пробелы.
    BIO сохраняем: оставляем предыдущий B-*, просто расширяем его end.
    """
    if not entities:
        return entities

    def base(tag: str) -> str:
        return tag.split("-", 1)[-1] if "-" in tag else tag

    def gap_is_joiners(prev_end: int, cur_start: int) -> bool:
        gap = text[prev_end:cur_start]
        if not gap:
            return True
        for ch in gap:
            if allow_spaces and ch.isspace():
                continue
            if ch in joiners:
                continue
            return False
        return True

    ents = sorted(entities, key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int, str]] = []

    for s, e, t in ents:
        b = base(t).upper()
        if out and base(out[-1][2]).upper() == b and gap_is_joiners(out[-1][1], s):
            # расширяем предыдущий спан; метка остаётся как была (обычно B-*)
            out[-1] = (out[-1][0], e, out[-1][2])
        else:
            out.append((s, e, t))
    return out


def fix_first_span(text: str, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Fix first word of TYPE/BRAND spans.
    """
    out = []
    if not entities:
        return entities
    first_span = entities[0]
    # Пропускаем начальные пробелы в тексте и корректируем индексы сущности
    # Найти количество начальных пробелов в text
    num_leading_spaces = len(text) - len(text.lstrip())
    if first_span[0] != num_leading_spaces:
        out.append((num_leading_spaces, first_span[1], f"B-{first_span[2].split('-', 1)[-1]}"))
    else:
        out.append(first_span)
    out.extend(entities[1:])

    return out


def formatted_out(out: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Format output to be like [(0,5,'B-TYPE'), ...]
    """
    return [(s, e, t) for (s, e, t) in out]


def zeroize_percent_after_size(text: str, entities: List[Tuple[int, int, str]]):
    """
    Turn PERCENT spans to 'O' for patterns like "размер 5" (size, not percent).
    Heuristic: find "размер <digits>" and zero any overlapping PERCENT spans
    that do not contain '%' in the underlying text segment.
    """
    if not entities:
        return entities
    # find numeric positions after the word 'размер'
    size_num_spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"\bразмер\b\s*(\d+)\b", text, re.IGNORECASE | re.U):
        size_num_spans.append(m.span(1))  # span of the digits subgroup

    if not size_num_spans:
        return entities

    def overlaps_num(s: int, e: int) -> bool:
        for ss, ee in size_num_spans:
            if ss < e and ee > s:
                return True
        return False

    out: List[Tuple[int, int, str]] = []
    for s, e, t in entities:
        if t.endswith("PERCENT") and overlaps_num(s, e):
            seg = text[s:e]
            if "%" not in seg:
                out.append((s, e, "O"))
                continue
        out.append((s, e, t))
    return out


def postprocess_all(
    text: str,
    entities: List[Tuple[int, int, str]],
    *,
    do_split_type: bool = True,
    do_boost_numeric: bool = True,
    do_replace_after_prepositions: bool = True,
    brand_thresh: float = 0.85,
) -> List[Tuple[int, int, str]]:
    """
    Комбинированный постпроцесс:
      1) (опц.) расширяем TYPE вправо (до 2 слов),
      2) (опц.) дробим TYPE по словам,
      3) (опц.) добавляем проценты/объёмы по регуляркам,
      4) (опц.) заменяем сущности после предлогов и слова "все" на 'O',
      5) сортируем и дедупим.
    """
    out = entities
    if do_split_type:
        # expand both TYPE and BRAND to include following words (up to 4 by default)
        # out = expand_phrases_right(text, out, tag_bases=("TYPE", "BRAND"), max_words=4)
        # then split both TYPE and BRAND into word-level spans
        out = fix_first_span(text, out)
        # out = split_tags_into_words(text, out, tag_bases=("TYPE", "BRAND"))
    if do_boost_numeric:
        extra = _numeric_preproc(text)
        # DEBUG: to inspect what improved_preprocessing produced, enable env var
        if os.environ.get("DEBUG_NUMERIC") == "1":
            print("[numeric_v2]", repr(text), "->", extra)
        # print("extra", extra)
        # print("out", out)
        out = _merge_and_dedup_entities(text, out, extra)

    # out = filter_brands_by_lexicon(text, out, min_len=2)

    # Normalize BIO for consecutive TYPE/BRAND chunks: B-... B-... -> B-... I-...
    out = stitch_consecutive_B_to_I(text, out)
    # out = merge_across_joiners(text, out)

    # Zeroize false PERCENT like "размер 5" (sizes) -> O
    out = zeroize_percent_after_size(text, out)

    # Replace entities after prepositions with 'O'
    if do_replace_after_prepositions:
        out = replace_after_prepositions(text, out)

    out = formatted_out(out)

    return sorted(set(out), key=lambda x: (x[0], x[1], x[2]))


# start region: brand lexicon
def _load_brand_lexicon():
    global _BRAND_LEX
    if _BRAND_LEX is None:
        try:
            with open(BRAND_LEX_PATH, "r", encoding="utf-8") as f:
                _BRAND_LEX = {line.strip().lower() for line in f if line.strip()}
        except Exception:
            _BRAND_LEX = set()
    return _BRAND_LEX


def filter_brands_by_lexicon(text, entities, min_len=2):
    lex = _load_brand_lexicon()
    out = []
    for s, e, t in entities:
        if t != "B-BRAND":
            out.append((s, e, t))
            continue
        tok = re.sub(r"[^\w\-]+", "", text[s:e].lower(), flags=re.U)
        if len(tok) < min_len:
            continue
        if tok in lex:
            out.append((s, e, t))
    return out


if __name__ == "__main__":
    # text = "dr.bakers ванильный"
    text = "сок ананасовый без сахара"
    # ents = [(0, 9, "B-BRAND"), (10, 19, "B-TYPE")]
    ents = [(0, 3, "B-TYPE"), (4, 14, "B-TYPE")]
    # add_ents = [(0, 6, "B-TYPE"), (7, 10, "B-PERCENT")]
    # _merge_and_dedup_entities("молоко 3,2", base_ents, add_ents)
    print(postprocess_all(text, ents, do_boost_numeric=True, do_split_type=True, do_replace_after_prepositions=True))

# end region: brand lexicon
