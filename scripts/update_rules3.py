#!/usr/bin/env python3
"""
Скрипт для обновления правил в train_with_submission.csv
Находит строки где source='s' и rules='rules1',
и если следующая строка имеет source='s' и первые 3 символа не совпадают,
заменяет rules1 на rules3
"""

import pandas as pd


def update_rules3():
    """
    Обновляет правила в файле train_with_submission.csv
    """
    input_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"
    output_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"

    print("Читаем файл train_with_submission.csv...")
    df = pd.read_csv(input_file, sep=";")
    print(f"Размер файла: {len(df)} строк")

    # Находим строки с source='s' и rules='rules1'
    print("Ищем строки с source='s' и rules='rules1'...")
    s_rules1_mask = (df["source"] == "s") & (df["rules"] == "rules1")
    s_rules1_indices = df[s_rules1_mask].index.tolist()
    print(f"Найдено {len(s_rules1_indices)} строк с source='s' и rules='rules1'")

    # Счетчик обновлений
    updated_count = 0

    # Проходим по каждой найденной строке
    for idx in s_rules1_indices:
        # Проверяем, что это не последняя строка
        if idx < len(df) - 1:
            current_sample = df.loc[idx, "sample"]
            next_row = df.loc[idx + 1]

            # Проверяем условия:
            # 1. Следующая строка имеет source='s'
            # 2. Первые 3 символа следующей строки не равны текущей строке
            if next_row["source"] == "s" and len(next_row["sample"]) >= 3 and next_row["sample"][:3] != current_sample:

                # Заменяем rules1 на rules3
                df.loc[idx, "rules"] = "rules3"
                updated_count += 1

                print(f"Обновлена строка {idx}: '{current_sample}' -> rules3 (следующая: '{next_row['sample']}')")

    print(f"\nОбновлено {updated_count} строк")

    # Сохраняем обновленный файл
    print(f"Сохраняем обновленный файл...")
    df.to_csv(output_file, sep=";", index=False)

    # Показываем статистику
    print("\nСтатистика после обновления:")
    print(f"Строк с rules1: {len(df[df['rules'] == 'rules1'])}")
    print(f"Строк с rules2: {len(df[df['rules'] == 'rules2'])}")
    print(f"Строк с rules3: {len(df[df['rules'] == 'rules3'])}")
    print(f"Строк с source='s' и rules='rules1': {len(df[(df['source'] == 's') & (df['rules'] == 'rules1')])}")
    print(f"Строк с source='s' и rules='rules2': {len(df[(df['source'] == 's') & (df['rules'] == 'rules2')])}")
    print(f"Строк с source='s' и rules='rules3': {len(df[(df['source'] == 's') & (df['rules'] == 'rules3')])}")

    # Показываем примеры обновленных строк
    print("\nПримеры обновленных строк:")
    updated_rows = df[(df["source"] == "s") & (df["rules"] == "rules3")]
    if len(updated_rows) > 0:
        print(updated_rows[["sample", "source", "rules"]].head(10))

    return df


def main():
    try:
        df_result = update_rules3()
    except Exception as e:
        print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
