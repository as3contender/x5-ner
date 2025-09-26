#!/usr/bin/env python3
"""
Скрипт для объединения train.csv и submission.csv
- Добавляет колонку source (t для train, s для submission)
- Добавляет колонку rules (rules1 если текст <=3 символа)
- Сортирует по алфавиту
"""

import pandas as pd
import os


def merge_train_submission():
    """
    Объединяет train.csv и submission.csv с дополнительными колонками
    """
    # Пути к файлам
    train_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train.csv"
    submission_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/submission.csv"
    output_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/train_with_submission.csv"

    print("Читаем train.csv...")
    # Читаем train.csv
    df_train = pd.read_csv(train_file, sep=";")
    print(f"Размер train.csv: {len(df_train)} строк")

    print("Читаем submission.csv...")
    # Читаем submission.csv
    df_submission = pd.read_csv(submission_file, sep=";")
    print(f"Размер submission.csv: {len(df_submission)} строк")

    # Добавляем колонку source
    print("Добавляем колонку source...")
    df_train["source"] = "t"
    df_submission["source"] = "s"

    # Объединяем файлы
    print("Объединяем файлы...")
    df_merged = pd.concat([df_train, df_submission], ignore_index=True)
    print(f"Размер объединенного файла: {len(df_merged)} строк")

    # Добавляем колонку rules
    print("Добавляем колонку rules...")
    df_merged["rules"] = df_merged["sample"].apply(lambda x: "rules1" if len(x) <= 3 else "")

    # Сортируем по алфавиту
    print("Сортируем по алфавиту...")
    df_sorted = df_merged.sort_values("sample").reset_index(drop=True)

    # Сохраняем результат
    print(f"Сохраняем результат в: {output_file}")
    df_sorted.to_csv(output_file, sep=";", index=False)

    print("Обработка завершена!")
    print(f"Создан файл: {output_file}")
    print(f"Размер итогового файла: {len(df_sorted)} строк")

    # Показываем статистику
    print("\nСтатистика:")
    print(f"Строк из train: {len(df_sorted[df_sorted['source'] == 't'])}")
    print(f"Строк из submission: {len(df_sorted[df_sorted['source'] == 's'])}")
    print(f"Строк с rules1: {len(df_sorted[df_sorted['rules'] == 'rules1'])}")

    # Показываем первые несколько строк
    print("\nПервые 5 строк:")
    print(df_sorted[["sample", "source", "rules"]].head())

    # Показываем последние несколько строк
    print("\nПоследние 5 строк:")
    print(df_sorted[["sample", "source", "rules"]].tail())

    return df_sorted


def main():
    try:
        df_result = merge_train_submission()
    except Exception as e:
        print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
