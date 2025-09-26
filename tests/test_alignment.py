from transformers import AutoTokenizer
from ner.dataset import spans_to_bio_labels

tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")


def test_simple_percent():
    text = "молоко 3,2% 1л"
    spans = [
        (0, 6, "B-TYPE"),
        (7, 11, "B-PERCENT"),
        (12, 14, "B-VOLUME"),
    ]
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=64)
    labels = spans_to_bio_labels(text, spans, enc["offset_mapping"])
    assert any(l.startswith("B-TYPE") for l in labels), labels
    assert any(l.startswith("B-PERCENT") for l in labels), labels
    assert any(l.startswith("B-VOLUME") for l in labels), labels


def test_multitoken_brand():
    text = "абрикосы 500г global village"
    spans = [
        (0, 8, "B-TYPE"),
        (9, 13, "B-VOLUME"),
        (14, 20, "B-BRAND"),
        (21, 28, "I-BRAND"),
    ]
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=64)
    labels = spans_to_bio_labels(text, spans, enc["offset_mapping"])
    saw_b, saw_i = False, False
    for lab in labels:
        if lab == "B-BRAND":
            saw_b = True
        if lab == "I-BRAND" and saw_b:
            saw_i = True
    assert saw_b and saw_i, labels
