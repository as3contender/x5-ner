import ast
from typing import List, Tuple, Dict, Any
import pandas as pd
import torch
from transformers import AutoTokenizer
from .utils import label2id


# -----------------------------
# Reading and parsing train.csv
# -----------------------------
def read_train(path: str) -> pd.DataFrame:
    """
    Reads train.csv with ';' separator.
    Expects columns: sample, annotation (stringified list of (start,end,tag)).
    Returns DataFrame with columns: sample (str), annotation_list (list of tuples).
    """
    df = pd.read_csv(path, sep=";")

    def parse_ann(x):
        try:
            v = ast.literal_eval(x)
            out = []
            for t in v:
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    s, e, tag = t
                    out.append((int(s), int(e), str(tag)))
            return out
        except Exception:
            return []

    df["annotation_list"] = df["annotation"].apply(parse_ann)
    df["sample"] = df["sample"].astype(str)
    return df[["sample", "annotation_list"]]


# ------------------------------------------
# Alignment: char-level spans -> token-level
# ------------------------------------------
def spans_to_bio_labels(
    text: str,
    spans: List[Tuple[int, int, str]],
    offsets: List[Tuple[int, int]],
) -> List[str]:
    """
    Convert character-level spans to BIO labels for each token offset.
    - offsets: list of (start_char, end_char) per token; special tokens have (0,0)
    - spans: list of (start_char, end_char, tag), where tag is already in BIO (B-*, I-* or just TYPE)
    """
    L = ["O"] * len(offsets)
    special = [(s == 0 and e == 0) for (s, e) in offsets]
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))

    for s, e, tag in spans_sorted:
        if tag == "O":
            continue
        ent_type = tag.split("-", 1)[1] if "-" in tag else tag
        began = False
        for i, (ts, te) in enumerate(offsets):
            if special[i]:
                continue
            if ts < e and te > s:  # overlap
                if not began:
                    L[i] = f"B-{ent_type}"
                    began = True
                else:
                    L[i] = f"I-{ent_type}"
    return L


def encode_with_alignment(
    tokenizer: "AutoTokenizer",
    text: str,
    spans: List[Tuple[int, int, str]],
    max_len: int = 128,
    return_tensors: bool = True,
    first_subtoken_only: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize raw text and align gold character-level spans to token-level BIO ids.
    Returns dict with input_ids, attention_mask, labels, (optionally) offsets.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt" if return_tensors else None,
    )
    offsets_tensor = enc["offset_mapping"]
    if return_tensors:
        offsets = [(int(s), int(e)) for (s, e) in offsets_tensor[0].tolist()]
    else:
        offsets = [tuple(x) for x in offsets_tensor]
    bio = spans_to_bio_labels(text, spans, offsets)

    # Determine first-subtoken-of-word using simple space-delimited word boundaries
    # A token is considered first-subtoken if its start is 0 or the previous char is a space
    first_sub_mask = []
    for s, e in offsets:
        if s == 0 and e == 0:
            first_sub_mask.append(False)  # special tokens
        else:
            first_sub_mask.append(bool(s == 0 or (s > 0 and text[s - 1].isspace())))

    label_ids = []
    for lab, (s, e), is_first in zip(bio, offsets, first_sub_mask):
        if s == 0 and e == 0:
            label_ids.append(-100)  # mask for special tokens
        else:
            if first_subtoken_only and not is_first:
                label_ids.append(-100)
            else:
                label_ids.append(label2id.get(lab, label2id["O"]))

    if return_tensors:
        enc.pop("offset_mapping")
        enc["labels"] = torch.tensor([label_ids], dtype=torch.long)
        return enc
    else:
        enc["labels"] = label_ids
        return enc


# ------------------------------------------
# Utility: small preview for sanity checking
# ------------------------------------------
def preview_alignment(
    tokenizer_name: str,
    df: pd.DataFrame,
    n: int = 5,
    max_len: int = 128,
) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    rows = []
    for i in range(min(n, len(df))):
        text = df.iloc[i]["sample"]
        spans = df.iloc[i]["annotation_list"]
        enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
        offsets = enc["offset_mapping"]
        labels = spans_to_bio_labels(text, spans, offsets)
        tokens = tok.convert_ids_to_tokens(enc["input_ids"])
        triplets = [(t, tuple(off), lab) for t, off, lab in zip(tokens, offsets, labels)]
        rows.append(
            {
                "text": text,
                "gold_spans": spans,
                "tokens": tokens,
                "offsets": [tuple(o) for o in offsets],
                "labels": labels,
                "triplets": triplets,
            }
        )
    return pd.DataFrame(rows)
