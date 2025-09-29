#!/usr/bin/env python3
"""
Скрипт для сравнения submission_val.csv и data/submission.csv
"""

import pandas as pd
import ast
import sys
from pathlib import Path


def parse_annotation(annotation_str):
    """Парсит строку аннотации в список кортежей"""
    try:
        return ast.literal_eval(annotation_str)
    except (ValueError, SyntaxError):
        return None


def compare_annotations(ann1, ann2):
    """Сравнивает две аннотации"""
    if ann1 == ann2:
        return True

    # Нормализуем аннотации для более точного сравнения
    if ann1 is None or ann2 is None:
        return ann1 == ann2

    # Сортируем по начальной позиции для сравнения
    ann1_sorted = sorted(ann1, key=lambda x: x[0])
    ann2_sorted = sorted(ann2, key=lambda x: x[0])

    return ann1_sorted == ann2_sorted


def main():
    # Пути к файлам
    base_dir = Path(__file__).parent.parent
    val_file = base_dir / "submission_val.csv"
    data_file = base_dir / "data" / "submission.csv"

    print("Загрузка файлов...")

    # Загружаем файлы
    try:
        df_val = pd.read_csv(val_file, sep=";")
        df_data = pd.read_csv(data_file, sep=";")
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return 1

    print(f"Загружено строк из submission_val.csv: {len(df_val)}")
    print(f"Загружено строк из data/submission.csv: {len(df_data)}")

    # Проверяем, что количество строк совпадает
    if len(df_val) != len(df_data):
        print(f"⚠️  ВНИМАНИЕ: Разное количество строк!")
        print(f"   submission_val.csv: {len(df_val)}")
        print(f"   data/submission.csv: {len(df_data)}")

    # Проверяем заголовки
    if list(df_val.columns) != list(df_data.columns):
        print(f"⚠️  ВНИМАНИЕ: Разные заголовки!")
        print(f"   submission_val.csv: {list(df_val.columns)}")
        print(f"   data/submission.csv: {list(df_data.columns)}")

    # Сравниваем содержимое
    print("\nСравнение содержимого...")

    differences = []
    identical_count = 0
    total_comparisons = min(len(df_val), len(df_data))

    for i in range(total_comparisons):
        val_sample = df_val.iloc[i]["sample"]
        data_sample = df_data.iloc[i]["sample"]

        val_annotation = df_val.iloc[i]["annotation"]
        data_annotation = df_data.iloc[i]["annotation"]

        # Проверяем, что образцы одинаковые
        if val_sample != data_sample:
            differences.append(
                {
                    "row": i + 2,  # +2 потому что индексация с 0 и есть заголовок
                    "type": "sample_mismatch",
                    "val_sample": val_sample,
                    "data_sample": data_sample,
                    "val_annotation": val_annotation,
                    "data_annotation": data_annotation,
                }
            )
            continue

        # Парсим аннотации
        val_parsed = parse_annotation(val_annotation)
        data_parsed = parse_annotation(data_annotation)

        # Сравниваем аннотации
        if not compare_annotations(val_parsed, data_parsed):
            differences.append(
                {
                    "row": i + 2,
                    "type": "annotation_mismatch",
                    "sample": val_sample,
                    "val_annotation": val_annotation,
                    "data_annotation": data_annotation,
                }
            )
        else:
            identical_count += 1

    # Выводим результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print(f"   Всего строк сравнено: {total_comparisons}")
    print(f"   Идентичных строк: {identical_count}")
    print(f"   Различающихся строк: {len(differences)}")
    print(f"   Процент совпадения: {identical_count/total_comparisons*100:.2f}%")

    if differences:
        print(f"\n🔍 НАЙДЕННЫЕ РАЗЛИЧИЯ:")

        # Группируем по типам различий
        sample_mismatches = [d for d in differences if d["type"] == "sample_mismatch"]
        annotation_mismatches = [d for d in differences if d["type"] == "annotation_mismatch"]

        if sample_mismatches:
            print(f"\n   Несовпадающие образцы ({len(sample_mismatches)}):")
            for diff in sample_mismatches[:10]:  # Показываем первые 10
                print(f"     Строка {diff['row']}:")
                print(f"       val: '{diff['val_sample']}'")
                print(f"       data: '{diff['data_sample']}'")

        if annotation_mismatches:
            print(f"\n   Несовпадающие аннотации ({len(annotation_mismatches)}):")
            for diff in annotation_mismatches[:10]:  # Показываем первые 10
                print(f"     Строка {diff['row']} - '{diff['sample']}':")
                print(f"       val: {diff['val_annotation']}")
                print(f"       data: {diff['data_annotation']}")

        if len(differences) > 20:
            print(f"     ... и еще {len(differences) - 20} различий")

        # Сохраняем подробный отчет
        output_file = base_dir / "comparison_report.csv"
        print(f"\n💾 Сохраняю подробный отчет в {output_file}")

        df_report = pd.DataFrame(differences)
        df_report.to_csv(output_file, index=False, sep=";")
        print(f"   Отчет сохранен: {len(differences)} различий")

    else:
        print(f"\n✅ Файлы полностью идентичны!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
