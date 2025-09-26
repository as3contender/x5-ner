# robust regex import: prefer the 'regex' package for Unicode categories like \p{L};
# fall back to stdlib 're' with an explicit Cyrillic/Latin pattern
try:
    import regex as re  # supports \p{L}, \p{Nd}

    WORD_RE = re.compile(r"[\p{L}\p{Nd}]+")
except Exception:
    import re  # stdlib

    WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
import torch
from typing import List, Dict, Tuple
from .utils import id2label, label2id
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .train import XLMRCRFForTokenClassification

# optional fuzzy matching for brand lexicon
try:
    from rapidfuzz import fuzz, process as rf_process
except Exception:  # if not installed, we degrade gracefully
    fuzz = None
    rf_process = None

# normalize token for lexicon/fuzzy checks (use stdlib re to avoid \p escapes)
import re as _stdre

TYPE_HINTS = set(
    """
молоко сыр творог сметана кефир йогурт сливки масло хлеб батон булка напиток вода сок чай кофе конфеты шоколад печенье паста макароны рис гречка крупа
""".split()
)

STOPWORDS = set(
    """
и в во на но да или либо для без со от до по о об при над через из у к с а как чем же же-то то ли
""".split()
)


def norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ё", "е")
    return s


def _looks_like_common_type(word: str) -> bool:
    w = norm(word)  # твоя нормализация (lower, ё→е и т.п.)
    return w in TYPE_HINTS


def _normalize_token(s: str) -> str:
    s = s.lower()
    return _stdre.sub(r"[^0-9a-zA-Zа-яё]+", "", s)


def _lex_norm(s: str) -> str:
    # normalization for lexicon keys: lower, ё->е, remove non-alnum
    s = norm(s)
    return _stdre.sub(r"[^0-9a-zA-Zа-я]+", "", s)


def load_brand_lexicon(path: str = "artifacts/brand_lexicon.txt"):
    try:
        with open(path, "r") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return set(_lex_norm(x) for x in items)
    except Exception:
        return set()


BRAND_LEXICON = load_brand_lexicon()
BRAND_LEXICON_LIST = list(BRAND_LEXICON) if BRAND_LEXICON else []


def _is_short_latin(word: str) -> bool:
    w = word.strip()
    return 1 <= len(w) <= 4 and re.fullmatch(r"[A-Za-z]+", w) is not None


class NERPipeline:
    def __init__(self, model_dir: str):
        # Try to load as CRF model first, fallback to standard model
        try:
            self.model = XLMRCRFForTokenClassification.from_pretrained(model_dir)
        except Exception:
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # start region predict_bio_tokens
    def predict_bio_tokens(
        self,
        text: str,
        brand_thresh: float = 0.85,
        entity_thresh: float = 0.55,
        cont_I_thresh: float = 0.50,
        include_o: bool = True,
        base_label: str = "B-TYPE",
    ) -> Tuple[List[Dict], str]:
        log_lines = []
        log_lines.append(
            f"[start] text='{text}' | brand_thresh={brand_thresh} entity_thresh={entity_thresh} cont_I_thresh={cont_I_thresh}"
        )

        # ---------- токенизация с оффсетами ----------
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=False,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        offsets = enc["offset_mapping"][0].tolist()
        word_ids = enc.word_ids(0)

        # ---------- форвард ----------
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attn)
            logits = out.logits if hasattr(out, "logits") else (out["logits"] if isinstance(out, dict) else out[0])

            # МЯГКИЕ вероятности для уверенности
            probs_tok_soft = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # [T, C]

            # CRF используем ТОЛЬКО для траектории меток по сабтокенам
            token_label_ids = None
            if hasattr(self.model, "crf"):
                mask = attn.bool()
                token_label_ids = self.model.crf.viterbi_decode(logits, mask=mask)[0]  # List[int] длиной T

        # ---------- соответствие word_id -> sabtokens ----------
        word_to_tok: Dict[int, List[int]] = {}
        order: List[int] = []
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in word_to_tok:
                word_to_tok[wid] = []
                order.append(wid)
            word_to_tok[wid].append(i)

        # ---------- агрегируем вероятности по словам (СРЕДНЕЕ softmax) ----------
        from numpy import mean as _np_mean

        word_probs: Dict[int, List[float]] = {}
        word_span: Dict[int, Tuple[int, int]] = {}

        def get_fallback_label(p_type: float) -> str:
            if p_type >= 0.15:  # TODO: fix this
                return base_label
            return "O"

        for wid in order:
            idxs = word_to_tok[wid]
            ws = min(offsets[i][0] for i in idxs)
            we = max(offsets[i][1] for i in idxs)
            word_span[wid] = (ws, we)

            sub = probs_tok_soft[idxs]  # [k, C] мягкие вероятности
            wp = sub.mean(axis=0)  # усреднение по сабтокенам
            word_probs[wid] = wp

        # ---------- помощники ----------
        def trim_span(s: int, e: int) -> Tuple[int, int]:
            while s < e and text[s].isspace():
                s += 1
            while s < e and text[e - 1].isspace():
                e -= 1
            return s, e

        def normalize_token(s: str) -> str:
            return _stdre.sub(r"[^0-9a-zA-Zа-яё]+", "", s.lower())

        labels_word: Dict[int, str] = {}
        prev_lab = "O"

        for wid in order:
            s, e = word_span[wid]
            if s >= e:
                labels_word[wid] = "O"
                log_lines.append(f"[decide] wid={wid} span=({s},{e}) -> O (s>=e)")
                continue

            token_text = text[s:e]
            tok_norm = normalize_token(token_text)
            pure_lat = bool(_stdre.fullmatch(r"[A-Za-z]+", tok_norm or ""))

            p = word_probs[wid]
            pred_id = int(p.argmax())
            lab = id2label[pred_id]

            # ---- вспомогательное ----
            def _pure_latin(s: str) -> bool:
                return _stdre.fullmatch(r"[A-Za-z]+", s) is not None

            def _has_vowel_latin(s: str) -> bool:
                return _stdre.search(r"[AEIOUYaeiouy]", s) is not None

            def _looks_like_type_word(s: str) -> bool:
                # быстрый фильтр: словарь типовых товаров + не числа + не стоп-слово
                t = _normalize_token(s)
                return (t in TYPE_HINTS) and (t not in STOPWORDS) and (not t.isdigit())

            # ---- извлечь вероятности для класса ----
            p_O = float(p[label2id["O"]]) if "O" in label2id else 0.0
            p_B_BRAND = float(p[label2id["B-BRAND"]]) if "B-BRAND" in label2id else 0.0
            p_I_BRAND = float(p[label2id["I-BRAND"]]) if "I-BRAND" in label2id else 0.0
            p_brand = max(p_B_BRAND, p_I_BRAND)
            p_brand_sum = p_B_BRAND + p_I_BRAND

            p_B_TYPE = float(p[label2id["B-TYPE"]]) if "B-TYPE" in label2id else 0.0
            p_I_TYPE = float(p[label2id["I-TYPE"]]) if "I-TYPE" in label2id else 0.0
            p_type = max(p_B_TYPE, p_I_TYPE)
            p_type_sum = p_B_TYPE + p_I_TYPE

            tok_norm = _normalize_token(token_text)
            in_lex = _lex_norm(token_text) in BRAND_LEXICON
            fuzzy_hit = False
            if (not in_lex) and rf_process and fuzz and BRAND_LEXICON_LIST:
                # жёстче предыдущего: короткие слова не фаззим, требуем высокий скор
                if len(tok_norm) >= 4:
                    match = rf_process.extractOne(tok_norm, BRAND_LEXICON_LIST, scorer=fuzz.WRatio, score_cutoff=95)
                    fuzzy_hit = match is not None

            pure_lat = _pure_latin(tok_norm)
            short_lat = pure_lat and (len(tok_norm) <= 2)
            has_vowel = _has_vowel_latin(tok_norm)

            log_lines.append(f"[wid={wid}] p_brand: {p_brand:.3f}, p_type: {p_type:.3f}, p_O: {p_O:.3f}")
            log_lines.append(
                f"[wid={wid}] p_brand_sum: {p_brand_sum:.3f}, p_type_sum: {p_type_sum:.3f}, p_O: {p_O:.3f}"
            )

            # 0) Если текст короче 2 символов - отдаем O
            if len(tok_norm) <= 2 and p_type < 0.4 and p_brand < 0.4:
                lab = "O"
                reason = "short_text"
                break

            # 0) Сильная уверенноть - отдаем приоритер ответу модели
            reason = ""
            for threshold in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]:
                if p_brand_sum >= threshold * 1.3 or p_brand > threshold:
                    lab = "I-BRAND" if prev_lab.endswith("BRAND") else "B-BRAND"
                    reason = f"strong_brand_{threshold}"
                    break
                elif p_type_sum >= threshold * 1.3 or p_type > threshold:
                    lab = "I-TYPE" if prev_lab.endswith("TYPE") else "B-TYPE"
                    reason = f"strong_type_{threshold}"
                    break
                elif p_O >= threshold:
                    lab = "O"
                    reason = f"strong_o_{threshold}"
                    break

            # ---- приоритет отрицаний ----
            # 0) короткая чистая латиница не из лексикона — не брендируем никогда
            if reason != "":
                pass
            elif short_lat and (not in_lex) and (not fuzzy_hit):
                lab = "O"
                reason = "short_lat_no_lex"

            # 1) если слово «похоже на TYPE», то брендировать можно только при ОЧЕНЬ сильной модели и отсутствии конфликта
            elif _looks_like_type_word(token_text):
                strong_brand = p_brand >= max(brand_thresh, p_type + 0.20, entity_thresh + 0.10)
                if strong_brand and (in_lex or fuzzy_hit):
                    lab = "I-BRAND" if prev_lab.endswith("BRAND") else "B-BRAND"
                    reason = "brand_over_type_very_strong"
                else:
                    # даём шанс TYPE, но без «надувания» из воздуха
                    if p_type >= max(entity_thresh, p_brand + 0.07):
                        lab = "I-TYPE" if prev_lab.endswith("TYPE") else "B-TYPE"
                        reason = "clear_type"
                    else:
                        lab = get_fallback_label(p_type)
                        reason = "looks_type_guard"

            # 2) лексикон / фаззи (как «сильная подсказка»), НО не слепо
            elif in_lex or fuzzy_hit:
                # если TYPE уверенно выигрывает — отдать приоритет TYPE, несмотря на лексикон
                if p_type >= max(entity_thresh, p_brand + 0.07):
                    lab = "I-TYPE" if prev_lab.endswith("TYPE") else "B-TYPE"
                    reason = "type_over_lex"
                # иначе требуем немного уверенности от модели, чтобы отсечь явно шумные случаи
                elif p_brand >= 0.20 and (p_brand >= p_type - 0.05):
                    lab = "I-BRAND" if prev_lab.endswith("BRAND") else "B-BRAND"
                    reason = "lex_or_fuzzy_brand"
                else:
                    lab = get_fallback_label(p_type)
                    reason = "lex_conf_too_low"

            # 3) чистая латиница вне лексикона — только очень уверенный бренд
            elif pure_lat and (not in_lex) and (not fuzzy_hit):
                # нужна гласная, длина>=4 и запас над TYPE
                if has_vowel and len(tok_norm) >= 4 and p_brand >= max(brand_thresh, p_type + 0.15, 0.80):
                    lab = "I-BRAND" if prev_lab.endswith("BRAND") else "B-BRAND"
                    reason = "strong_pure_lat_brand"
                else:
                    lab = get_fallback_label(p_type)
                    reason = "weak_pure_lat"

            # 4) TYPE — только если уверенно выигрывает у BRAND
            elif p_type >= max(entity_thresh, p_brand + 0.07):
                lab = "I-TYPE" if prev_lab.endswith("TYPE") else "B-TYPE"
                reason = "clear_type"

            # 5) BRAND — только если уверенно и выше порога
            elif p_brand >= max(brand_thresh, p_type + 0.10):
                lab = "I-BRAND" if prev_lab.endswith("BRAND") else "B-BRAND"
                reason = "clear_brand"

            # 6) иначе — O
            else:
                lab = get_fallback_label(p_type)
                reason = "fallback"

            # BIO-нормализация (на всякий случай)
            if lab != "O":
                base = lab.split("-", 1)[-1]
                lab = f"I-{base}" if prev_lab.endswith(base) else f"B-{base}"

            labels_word[wid] = lab
            prev_lab = lab if lab != "O" else "O"

            log_lines.append(
                f"[decide] wid={wid} span=({s},{e}) '{token_text}' | p_brand={p_brand:.3f} p_type={p_type:.3f} in_lex={in_lex} fuzzy={fuzzy_hit} pureLat={pure_lat} shortLat={short_lat} -> {lab} ({reason})\n"
            )

        # ---------- собираем спаны ----------
        spans: List[Dict] = []
        for wid in order:
            lab = labels_word.get(wid, "O")
            if lab == "O" and not include_o:
                continue
            s, e = trim_span(*word_span[wid])
            if s < e:
                spans.append({"start": s, "end": e, "label": lab})

        return spans, "\n".join(log_lines)

    # end region predict_bio_tokens

    def predict_entities(
        self,
        text: str,
        brand_thresh: float = 0.8,
        entity_thresh: float = 0.55,
        cont_I_thresh: float = 0.35,
        include_o: bool = False,
    ) -> Tuple[List[Tuple[int, int, str]], str]:
        spans, log = self.predict_bio_tokens(
            text,
            brand_thresh=brand_thresh,
            entity_thresh=entity_thresh,
            cont_I_thresh=cont_I_thresh,
            include_o=include_o,
        )
        result = []
        for span in spans:
            if span["label"] == "O" and not include_o:
                continue
            result.append((span["start"], span["end"], span["label"]))
        return result, log
