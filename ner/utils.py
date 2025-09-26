import random
import numpy as np
import torch
from typing import List, Dict, Tuple

BIO_TAGS = [
    "O",
    "B-TYPE",
    "I-TYPE",
    "B-BRAND",
    "I-BRAND",
    "B-VOLUME",
    "I-VOLUME",
    "B-PERCENT",
    "I-PERCENT",
]

id2label = {i: l for i, l in enumerate(BIO_TAGS)}
label2id = {l: i for i, l in enumerate(BIO_TAGS)}


def set_seed(seed: int = 42):
    """Фиксируем все random-числа для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bio_to_entities(spans: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    Конвертация токен-уровня BIO (список dict'ов от predict_bio_tokens)
    в агрегированные сущности (start, end, 'B-XXX').

    Вход: [{"start_index": s, "end_index": e, "entity": "B-XXX"/"I-XXX"}, ...]
    Выход: [(s, e, "B-XXX"), ...]  — один B-спан на объединённую сущность.
    """
    if not spans:
        return []

    entities: List[Tuple[int, int, str]] = []
    cur_type, cur_s, cur_e = None, None, None  # активная сущность

    def flush():
        nonlocal cur_type, cur_s, cur_e
        if cur_type is not None:
            entities.append((cur_s, cur_e, f"B-{cur_type}"))
        cur_type, cur_s, cur_e = None, None, None

    for seg in spans:
        s = int(seg["start_index"])
        e = int(seg["end_index"])
        lab = str(seg["entity"]) if seg["entity"] is not None else "O"
        if lab == "O":
            flush()
            continue
        if "-" in lab:
            bi, typ = lab.split("-", 1)
        else:
            bi, typ = "B", lab

        if bi == "B" or cur_type is None:
            flush()
            cur_type, cur_s, cur_e = typ, s, e
        elif bi == "I":
            if cur_type == typ and s <= cur_e:  # примыкание/перекрытие
                cur_e = max(cur_e, e)
            else:
                # некорректная I без согласования — начнём новую
                flush()
                cur_type, cur_s, cur_e = typ, s, e
        else:
            flush()
            cur_type, cur_s, cur_e = typ, s, e

    flush()
    # стабильный порядок
    entities.sort(key=lambda x: (x[0], x[1], x[2]))
    return entities


def serialize_entities(ents: List[Tuple[int, int, str]]) -> str:
    """
    Стабильная сериализация [(s,e,'B-XXX'), ...] в строку как в baseline submission.
    """
    ents = sorted(ents, key=lambda x: (x[0], x[1], x[2]))
    # str(list_of_tuples) даёт нужный формат c одинарными кавычками
    return str([(int(s), int(e), str(tag)) for (s, e, tag) in ents])
