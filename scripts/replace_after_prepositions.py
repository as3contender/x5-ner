#!/usr/bin/env python3
import argparse
import ast
from pathlib import Path
from typing import List, Tuple, Optional, Set


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


def load_prepositions(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set(DEFAULT_PREPOSITIONS)
    items: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if not w or w.startswith("#"):
                continue
            items.add(w)
    return items or set(DEFAULT_PREPOSITIONS)


def normalize_token(text: str) -> str:
    return text.strip().strip("\t\r\n .,!?:;\"'«»()[]{}-—").lower()


def replace_after_prepositions(
    sample: str, annotations: List[Tuple[int, int, str]], preps: Set[str]
) -> List[Tuple[int, int, str]]:
    if not annotations:
        return annotations
    # Work on a mutable copy
    mutable: List[List[object]] = [[a, b, c] for (a, b, c) in annotations]
    i = 0
    while i < len(mutable) - 1:
        start, end, label = mutable[i]
        if label == "O":
            token_text = sample[start:end]
            if normalize_token(token_text) in preps:
                # Set the next segment label to 'O'
                nxt = mutable[i + 1]
                nxt[2] = "O"
                # Note: we only change the immediate next segment, as requested
        i += 1
    return [(int(a), int(b), str(c)) for (a, b, c) in mutable]


def process_file(src: Path, dst: Path, preps_path: Optional[Path]) -> None:
    preps = load_prepositions(preps_path)
    with src.open("r", encoding="utf-8") as fin:
        lines = fin.readlines()
    if not lines:
        dst.write_text("", encoding="utf-8")
        return

    header = lines[0].rstrip("\n")
    out_lines: List[str] = [header + "\n"]

    changed = 0
    total = 0
    for raw in lines[1:]:
        total += 1
        line = raw.rstrip("\n")
        if not line:
            out_lines.append("\n")
            continue
        try:
            sample, ann_str = line.split(";", 1)
        except ValueError:
            out_lines.append(raw)
            continue
        try:
            anns = ast.literal_eval(ann_str)
            if isinstance(anns, tuple):
                anns = [anns]
            anns_typed: List[Tuple[int, int, str]] = [(int(a), int(b), str(c)) for a, b, c in anns]
        except Exception:
            out_lines.append(raw)
            continue

        before = repr(anns_typed)
        after_anns = replace_after_prepositions(sample, anns_typed, preps)
        after = repr(after_anns)
        if after != before:
            changed += 1
        out_lines.append(f"{sample};{after}\n")

    with dst.open("w", encoding="utf-8") as fout:
        fout.writelines(out_lines)

    print(f"Processed: {total} lines, modified: {changed} lines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace entity immediately after 'O' prepositions with 'O'.")
    parser.add_argument(
        "--source", "--исходный_файл", dest="source", required=False, help="Path to source CSV (sample;annotation)"
    )
    parser.add_argument("--dest", "--файл_назначения", dest="dest", required=False, help="Path to destination CSV")
    parser.add_argument(
        "--prepositions", required=False, help="Optional path to newline-separated prepositions list (utf-8)"
    )
    args = parser.parse_args()

    if not args.source or not args.dest:
        raise SystemExit("Please provide --source/--исходный_файл and --dest/--файл_назначения")

    src = Path(args.source)
    dst = Path(args.dest)
    preps_path = Path(args.prepositions) if args.prepositions else None

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    process_file(src, dst, preps_path)


if __name__ == "__main__":
    main()
