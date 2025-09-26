#!/usr/bin/env python3
import argparse
import ast
from pathlib import Path
from typing import List, Tuple


def filter_out_o_spans(annotations: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    return [(int(a), int(b), str(c)) for a, b, c in annotations if str(c) != "O"]


def process_file(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as fin:
        lines = fin.readlines()

    if not lines:
        dst.write_text("", encoding="utf-8")
        return

    header = lines[0].rstrip("\n")
    out_lines: List[str] = [header + "\n"]

    total = 0
    changed = 0
    for raw in lines[1:]:
        total += 1
        line = raw.rstrip("\n")
        if not line:
            out_lines.append("\n")
            continue
        try:
            sample, ann_str = line.split(";", 1)
        except ValueError:
            # Preserve lines that do not match expected format
            out_lines.append(raw)
            continue

        try:
            anns = ast.literal_eval(ann_str)
            if isinstance(anns, tuple):
                anns = [anns]
            typed: List[Tuple[int, int, str]] = [(int(a), int(b), str(c)) for a, b, c in anns]
        except Exception:
            # If parsing fails, keep original line
            out_lines.append(raw)
            continue

        filtered = filter_out_o_spans(typed)
        if len(filtered) != len(typed):
            changed += 1
        out_lines.append(f"{sample};{repr(filtered)}\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fout:
        fout.writelines(out_lines)

    print(f"Processed: {total} lines, modified: {changed} lines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove spans labeled 'O' from CSV annotations.")
    parser.add_argument(
        "--source", "--исходный_файл", dest="source", required=False, help="Path to source CSV (sample;annotation)"
    )
    parser.add_argument("--dest", "--файл_назначения", dest="dest", required=False, help="Path to destination CSV")
    args = parser.parse_args()

    if not args.source or not args.dest:
        raise SystemExit("Please provide --source/--исходный_файл and --dest/--файл_назначения")

    src = Path(args.source)
    dst = Path(args.dest)

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    process_file(src, dst)


if __name__ == "__main__":
    main()
