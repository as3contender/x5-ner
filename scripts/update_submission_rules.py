#!/usr/bin/env python3
"""
Скрипт для обновления submission.csv
Находит строки с rules2 и rules3 в train_with_submission.csv,
находит точные совпадения в submission.csv и заменяет аннотации на 'O'
"""

import pandas as pd
import ast


def update_submission_with_rules():
    """
    Обновляет submission.csv на основе правил из train_with_submission.csv
    """
    # Пути к файлам
    train_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"
    submission_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/submission.csv"
    output_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/submission_updated.csv"

    print("Читаем train_with_submission.csv...")
    df_train = pd.read_csv(train_file, sep=";")
    print(f"Размер train_with_submission.csv: {len(df_train)} строк")

    print("Читаем submission.csv...")
    df_submission = pd.read_csv(submission_file, sep=";")
    print(f"Размер submission.csv: {len(df_submission)} строк")

    # Находим строки с rules2 и rules3
    print("Ищем строки с rules2 и rules3...")
    rules_mask = (df_train["rules"] == "rules2") | (df_train["rules"] == "rules3")
    rules_rows = df_train[rules_mask]
    print(f"Найдено {len(rules_rows)} строк с rules2/rules3")

    # Извлекаем тексты для поиска
    texts_to_find = rules_rows["sample"].tolist()
    print(f"Тексты для поиска: {texts_to_find[:10]}...")  # Показываем первые 10

    # Счетчик обновлений
    updated_count = 0

    # Проходим по submission.csv и ищем совпадения
    print("Ищем совпадения в submission.csv...")
    for idx, row in df_submission.iterrows():
        sample_text = row["sample"]

        # Проверяем, есть ли этот текст в списке для замены
        if sample_text in texts_to_find:
            # Заменяем аннотацию на 'O' для всего текста
            text_len = len(sample_text)
            new_annotation = f"[(0, {text_len}, 'O')]"

            df_submission.loc[idx, "annotation"] = new_annotation
            updated_count += 1

            print(f"Обновлена строка {idx}: '{sample_text}' -> 'O'")

    print(f"\nОбновлено {updated_count} строк в submission.csv")

    # Сохраняем обновленный файл
    print(f"Сохраняем обновленный submission.csv в: {output_file}")
    df_submission.to_csv(output_file, sep=";", index=False)

    # Показываем статистику
    print("\nСтатистика обновления:")
    print(f"Всего строк в submission.csv: {len(df_submission)}")
    print(f"Обновлено строк: {updated_count}")

    # Показываем примеры обновленных строк
    print("\nПримеры обновленных строк:")
    updated_rows = df_submission[df_submission["sample"].isin(texts_to_find)]
    if len(updated_rows) > 0:
        print(updated_rows[["sample", "annotation"]].head(10))

    return df_submission


def main():
    try:
        df_result = update_submission_with_rules()
    except Exception as e:
        print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
