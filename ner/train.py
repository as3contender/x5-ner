import argparse
import os
import yaml
import random
import numpy as np
import re
import torch.nn as nn

try:
    from TorchCRF import CRF
except Exception as _e:
    CRF = None  # will assert later if use_crf is requested
from typing import List, Dict

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, classification_report

from ner.utils import BIO_TAGS, id2label, label2id, set_seed
from ner.dataset import read_train, encode_with_alignment
from seqeval.metrics.sequence_labeling import get_entities


class XLMRCRFForTokenClassification(nn.Module):
    """XLM-R (or any AutoModel) with CRF head for BIO tagging.
    Returns dict with 'loss' and 'logits' (emissions). Use .decode() for Viterbi paths.
    """

    def __init__(self, model_name: str, num_labels: int, id2label_map: Dict[int, str], label2id_map: Dict[str, int]):
        super().__init__()
        from transformers import AutoModel

        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        if CRF is None:
            raise ImportError("torchcrf is not installed. Run: pip install torchcrf")
        self.crf = CRF(num_labels)
        self.id2label = id2label_map
        self.label2id = label2id_map

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)  # (B, T, C)
        loss = None
        if labels is not None:
            # build CRF mask from labels: valid where label != -100
            mask = labels.ne(-100)
            # replace masked labels with zero to satisfy type constraints
            safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
            loss = -self.crf(emissions, safe_labels, mask=mask).mean()
        return {"loss": loss, "logits": emissions}

    @torch.no_grad()
    def decode(self, input_ids=None, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs.last_hidden_state)
        # approximate mask: treat all non-pad tokens as valid, but drop specials later in postproc if needed
        mask = attention_mask.bool()
        # Use viterbi_decode method from CRF
        return self.crf.viterbi_decode(emissions, mask=mask)

    @classmethod
    def from_pretrained(cls, model_dir: str):
        """Load a pretrained CRF model from directory"""
        import json
        import os

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract model info
        model_name = config.get("model_name", "xlm-roberta-base")
        num_labels = len(config.get("id2label", {}))
        id2label_map = config.get("id2label", {})
        label2id_map = config.get("label2id", {})

        # Convert string keys to int for id2label
        id2label_map = {int(k): v for k, v in id2label_map.items()}

        # Create model instance
        model = cls(model_name, num_labels, id2label_map, label2id_map)

        # Load state dict
        state_dict_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(state_dict_path):
            from safetensors.torch import load_file

            state_dict = load_file(state_dict_path)
        else:
            # Fallback to pytorch format
            state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
            state_dict = torch.load(state_dict_path, map_location="cpu")

        model.load_state_dict(state_dict)
        return model


def train_val_split(df, val_split=0.1, seed: int = 42):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(df) * (1 - val_split))
    tr_idx, val_idx = idx[:cut], idx[cut:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# ===== Length-preserving text noise (keeps all span offsets valid) =====
KB_NEIGHBORS = {
    "о": "о0",
    "0": "0о",
    "a": "aqs",
    "e": "e3",
    "3": "3e",
    "c": "cs",
    "k": "kij",
    "м": "мпн",
    "н": "нм",
    "т": "тгь",
    "p": "pr",
    "o": "o0",
    "l": "l1",
    "1": "1l",
}
HOMO = {
    "a": "а",
    "e": "е",
    "o": "о",
    "p": "р",
    "c": "с",
    "x": "х",
    "y": "у",
    "k": "к",
    "А": "A",
    "Е": "E",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Х": "X",
    "У": "Y",
    "К": "K",
}
RE_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def _rnd_replace_char(ch: str) -> str:
    opts = KB_NEIGHBORS.get(ch)
    if not opts:
        return ch
    return random.choice(opts)


def _homoglyph(ch: str) -> str:
    return HOMO.get(ch, ch)


def augment_text_length_preserving(
    text: str, prob=0.3, typo_prob=0.25, homoglyph_prob=0.25, space_punct_prob=0.15
) -> str:
    if random.random() > prob:
        return text
    chars = list(text)
    # within-word changes (length preserved)
    for m in RE_WORD.finditer(text):
        s, e = m.span()
        for i in range(s, e):
            r = random.random()
            if r < typo_prob:
                chars[i] = _rnd_replace_char(chars[i])
            elif r < typo_prob + homoglyph_prob:
                chars[i] = _homoglyph(chars[i])
    # swap single-char separators without changing length
    for i, ch in enumerate(chars):
        if ch == " " and random.random() < space_punct_prob:
            chars[i] = "-" if random.random() < 0.5 else "."
        elif ch in "-." and random.random() < (space_punct_prob * 0.3):
            chars[i] = " "
    return "".join(chars)


def encode_dataframe(df, tokenizer, max_len: int, augment_fn=None):
    """
    Преобразует DataFrame со столбцами:
      - sample: str
      - annotation_list: List[(start, end, tag)]
    в dict для Dataset.from_dict с полями:
      - input_ids, attention_mask, labels (списки списков int)
    """
    input_ids, attention_mask, labels = [], [], []
    for _, row in df.iterrows():
        text = row["sample"]
        if augment_fn is not None:
            text = augment_fn(text)
        spans = row["annotation_list"]
        enc = encode_with_alignment(
            tokenizer=tokenizer,
            text=text,
            spans=spans,
            max_len=max_len,
            return_tensors=False,  # получаем списки, не тензоры (для Dataset.from_dict удобно)
        )
        input_ids.append(enc["input_ids"])
        attention_mask.append(enc["attention_mask"])
        labels.append(enc["labels"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_datasets(train_df, val_df, tokenizer, max_len: int, train_augment_fn=None) -> DatasetDict:
    train_enc = encode_dataframe(train_df, tokenizer, max_len, augment_fn=train_augment_fn)
    val_enc = encode_dataframe(val_df, tokenizer, max_len, augment_fn=None)
    ds = DatasetDict(
        train=Dataset.from_dict(train_enc),
        validation=Dataset.from_dict(val_enc),
    )
    return ds


def seq_to_tags(seq_ids: List[int]) -> List[str]:
    """Конвертирует список id-меток в BIO-строки, игнорируя -100 (маска спец-токенов)."""
    out = []
    for i in seq_ids:
        if i == -100:
            continue
        out.append(id2label.get(i, "O"))
    return out


def _per_type_f1(y_true, y_pred, types=("TYPE", "BRAND", "VOLUME", "PERCENT")):
    """
    Считает TP/FP/FN и F1 отдельно по каждому типу сущностей
    на уровне entity spans (совпадение типа и границ токенов).
    """
    # соберём списки сущностей по каждому примеру
    # get_entities ждёт BIO-последовательности по токенам
    tp = {t: 0 for t in types}
    fp = {t: 0 for t in types}
    fn = {t: 0 for t in types}

    for true_tags, pred_tags in zip(y_true, y_pred):
        true_ents = get_entities(true_tags)  # [('TYPE', start, end), ...]  end включительно
        pred_ents = get_entities(pred_tags)

        # свернём в множества для точного сравнения спанов
        true_set = {(t, s, e) for (t, s, e) in true_ents if t in types}
        pred_set = {(t, s, e) for (t, s, e) in pred_ents if t in types}

        # TP — пересечение множеств
        inter = true_set & pred_set
        for t, _, _ in inter:
            tp[t] += 1

        # FP — предсказано, но нет в правде
        for t, _, _ in pred_set - inter:
            fp[t] += 1

        # FN — есть в правде, но не предсказано
        for t, _, _ in true_set - inter:
            fn[t] += 1

    # посчитаем precision/recall/f1 по каждому типу
    per_type = {}
    for t in types:
        P = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        R = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        per_type[t] = {"precision": P, "recall": R, "f1": F1}

    # macro по типам (как в ТЗ)
    macro_f1 = sum(per_type[t]["f1"] for t in types) / len(types)

    return per_type, macro_f1


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)

    # Собираем BIO-строки без спец-токенов
    y_true, y_pred = [], []
    for p_row, l_row in zip(pred_ids, labels):
        true_tags = []
        pred_tags = []
        for p, l in zip(p_row, l_row):
            if l == -100:  # спец-токен
                continue
            true_tags.append(id2label.get(int(l), "O"))
            pred_tags.append(id2label.get(int(p), "O"))
        y_true.append(true_tags)
        y_pred.append(pred_tags)

    # Общий entity-level F1 по seqeval (по всем классам)
    overall_f1 = f1_score(y_true, y_pred)

    # Детализация по типам
    per_type, macro_f1_types = _per_type_f1(y_true, y_pred, types=("TYPE", "BRAND", "VOLUME", "PERCENT"))

    # Короткий лог в консоль
    msg = (
        f"\n[Eval] overall_f1={overall_f1:.4f}  "
        f"macro_f1_types={macro_f1_types:.4f}  |  "
        f"TYPE={per_type['TYPE']['f1']:.4f}  "
        f"BRAND={per_type['BRAND']['f1']:.4f}  "
        f"VOLUME={per_type['VOLUME']['f1']:.4f}  "
        f"PERCENT={per_type['PERCENT']['f1']:.4f}\n"
    )
    print(msg)

    # Вернём метрики, чтобы Trainer их логировал/сохранял
    metrics = {
        "f1": overall_f1,
        "macro_f1_types": macro_f1_types,
        "f1_TYPE": per_type["TYPE"]["f1"],
        "f1_BRAND": per_type["BRAND"]["f1"],
        "f1_VOLUME": per_type["VOLUME"]["f1"],
        "f1_PERCENT": per_type["PERCENT"]["f1"],
        "precision_TYPE": per_type["TYPE"]["precision"],
        "recall_TYPE": per_type["TYPE"]["recall"],
        "precision_BRAND": per_type["BRAND"]["precision"],
        "recall_BRAND": per_type["BRAND"]["recall"],
        "precision_VOLUME": per_type["VOLUME"]["precision"],
        "recall_VOLUME": per_type["VOLUME"]["recall"],
        "precision_PERCENT": per_type["PERCENT"]["precision"],
        "recall_PERCENT": per_type["PERCENT"]["recall"],
    }
    return metrics


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_name = cfg.get("model_name", "cointegrated/rubert-tiny")
    train_path = cfg.get("train_path", "data/train.csv")
    val_path = cfg.get("val_path", "data/val.csv")
    max_len = int(cfg.get("max_seq_len", 128))
    out_dir = cfg.get("output_dir", "artifacts/ner-checkpoint")
    epochs = int(cfg.get("epochs", 5))
    bs = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 3e-5))
    wd = float(cfg.get("weight_decay", 0.01))

    use_crf = bool(cfg.get("use_crf", False))
    aug_cfg = cfg.get("augment", {}) or {}
    do_aug = bool(aug_cfg.get("enable", False))
    aug_prob = float(aug_cfg.get("prob", 0.0))
    aug_typo = float(aug_cfg.get("typo_prob", 0.25))
    aug_homo = float(aug_cfg.get("homoglyph_prob", 0.25))
    aug_sp = float(aug_cfg.get("space_punct_prob", 0.15))

    train_augment_fn = (
        (
            lambda t: augment_text_length_preserving(
                t, prob=aug_prob, typo_prob=aug_typo, homoglyph_prob=aug_homo, space_punct_prob=aug_sp
            )
        )
        if do_aug
        else None
    )

    os.makedirs(out_dir, exist_ok=True)

    # 1) Данные: читаем train.csv и делим на train/val
    train_df = read_train(train_path)  # -> sample, annotation_list
    val_df = read_train(val_path)  # -> sample, annotation_list

    # 2) Токенайзер
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Датасеты с уже выровненными labels
    ds = build_datasets(train_df, val_df, tokenizer, max_len, train_augment_fn=train_augment_fn)

    # 4) Модель
    if use_crf:
        model = XLMRCRFForTokenClassification(
            model_name=model_name,
            num_labels=len(BIO_TAGS),
            id2label_map={i: BIO_TAGS[i] for i in range(len(BIO_TAGS))},
            label2id_map={BIO_TAGS[i]: i for i in range(len(BIO_TAGS))},
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(BIO_TAGS),
            id2label={i: BIO_TAGS[i] for i in range(len(BIO_TAGS))},
            label2id={BIO_TAGS[i]: i for i in range(len(BIO_TAGS))},
        )

    # 5) Коллатор
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 6) Тренинг
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        weight_decay=wd,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    # 7) Сохранение лучшего чекпойнта
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved checkpoint to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
