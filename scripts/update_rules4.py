#!/usr/bin/env python3
"""
Скрипт для обновления правил в train_with_submission.csv
Находит строки где source='s', длина текста 4 символа, установлен B-BRAND,
и если первые 3 символа следующей строки не равны текущей строке,
устанавливает rules4
"""

import pandas as pd
import ast


def update_rules4():
    """
    Обновляет правила в файле train_with_submission.csv
    """
    input_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"
    output_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"

    print("Читаем файл train_with_submission.csv...")
    df = pd.read_csv(input_file, sep=";")
    print(f"Размер файла: {len(df)} строк")

    # Находим строки с source='s', длиной 4 символа и B-BRAND
    print("Ищем строки с source='s', длиной 4 символа и B-BRAND...")

    # Фильтруем по условиям
    s_mask = df["source"] == "s"
    length_4_mask = df["sample"].str.len() == 4
    b_brand_mask = df["annotation"].str.contains("B-BRAND")

    # Объединяем условия
    target_mask = s_mask & length_4_mask & b_brand_mask
    target_indices = df[target_mask].index.tolist()

    print(f"Найдено {len(target_indices)} строк с source='s', длиной 4 символа и B-BRAND")

    # Показываем найденные строки
    if len(target_indices) > 0:
        print("Найденные строки:")
        for idx in target_indices[:10]:  # Показываем первые 10
            sample = df.loc[idx, "sample"]
            annotation = df.loc[idx, "annotation"]
            print(f"  {idx}: '{sample}' - {annotation}")
        if len(target_indices) > 10:
            print(f"  ... и еще {len(target_indices) - 10} строк")

    # Счетчик обновлений
    updated_count = 0

    # Проходим по каждой найденной строке
    for idx in target_indices:
        # Проверяем, что это не последняя строка
        if idx < len(df) - 1:
            current_sample = df.loc[idx, "sample"]
            next_row = df.loc[idx + 1]

            # Проверяем условие: первые 3 символа следующей строки не равны текущей строке
            if len(next_row["sample"]) >= 3 and next_row["sample"][:3] != current_sample[:3]:
                # Устанавливаем rules4
                df.loc[idx, "rules"] = "rules4"
                updated_count += 1

                print(f"Обновлена строка {idx}: '{current_sample}' -> rules4 (следующая: '{next_row['sample']}')")

    print(f"\nОбновлено {updated_count} строк")

    # Сохраняем обновленный файл
    print(f"Сохраняем обновленный файл...")
    df.to_csv(output_file, sep=";", index=False)

    # Показываем статистику
    print("\nСтатистика после обновления:")
    print(f"Строк с rules1: {len(df[df['rules'] == 'rules1'])}")
    print(f"Строк с rules2: {len(df[df['rules'] == 'rules2'])}")
    print(f"Строк с rules3: {len(df[df['rules'] == 'rules3'])}")
    print(f"Строк с rules4: {len(df[df['rules'] == 'rules4'])}")
    print(f"Строк с source='s' и rules='rules4': {len(df[(df['source'] == 's') & (df['rules'] == 'rules4')])}")

    # Показываем примеры обновленных строк
    print("\nПримеры обновленных строк:")
    updated_rows = df[(df["source"] == "s") & (df["rules"] == "rules4")]
    if len(updated_rows) > 0:
        print(updated_rows[["sample", "source", "rules"]].head(10))

    return df


def main():
    try:
        df_result = update_rules4()
    except Exception as e:
        print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
