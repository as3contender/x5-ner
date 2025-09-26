#!/usr/bin/env python3
"""
Скрипт для обработки файла submission_final_fixed_large_chain_o_fixe_2.csv
- Создает копию файла
- Сортирует по возрастанию
- Добавляет признаки: text_len, text_type_len, text_brand_len
"""

import pandas as pd
import ast
import os
from pathlib import Path


def extract_annotation_lengths(annotation_str):
    """
    Извлекает длины различных типов аннотаций из строки аннотации
    """
    try:
        # Парсим строку аннотации как список кортежей
        annotations = ast.literal_eval(annotation_str)

        total_len = 0
        type_len = 0
        brand_len = 0

        for start, end, label in annotations:
            span_len = end - start
            total_len += span_len

            if label.startswith("B-TYPE") or label.startswith("I-TYPE"):
                type_len += span_len
            elif label.startswith("B-BRAND") or label.startswith("I-BRAND"):
                brand_len += span_len

        return total_len, type_len, brand_len
    except:
        return 0, 0, 0


def process_submission_file(input_file, output_file):
    """
    Обрабатывает файл submission и создает новую версию с дополнительными признаками
    """
    print(f"Читаем файл: {input_file}")

    # Читаем CSV файл
    df = pd.read_csv(input_file, sep=";")

    print(f"Исходный размер данных: {len(df)} строк")
    print(f"Колонки: {list(df.columns)}")

    # Добавляем новые признаки
    print("Добавляем признаки...")

    # Длина текста
    df["text_len"] = df["sample"].str.len()

    # Извлекаем длины аннотаций
    annotation_lengths = df["annotation"].apply(extract_annotation_lengths)
    df["text_annotation_len"] = [x[0] for x in annotation_lengths]
    df["text_type_len"] = [x[1] for x in annotation_lengths]
    df["text_brand_len"] = [x[2] for x in annotation_lengths]

    # Сортируем по алфавиту
    print("Сортируем по алфавиту...")
    df_sorted = df.sort_values("sample").reset_index(drop=True)

    # Сохраняем обработанный файл
    print(f"Сохраняем результат в: {output_file}")
    df_sorted.to_csv(output_file, sep=";", index=False)

    print("Обработка завершена!")
    print(f"Создан файл: {output_file}")
    print(f"Размер обработанных данных: {len(df_sorted)} строк")

    # Показываем статистику
    print("\nСтатистика:")
    print(f"Средняя длина текста: {df_sorted['text_len'].mean():.2f}")
    print(f"Средняя длина аннотаций: {df_sorted['text_annotation_len'].mean():.2f}")
    print(f"Средняя длина типов: {df_sorted['text_type_len'].mean():.2f}")
    print(f"Средняя длина брендов: {df_sorted['text_brand_len'].mean():.2f}")

    return df_sorted


def main():
    # Пути к файлам
    input_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/submission_final_fixed_large_chain_o_fixe_2.csv"
    output_file = "/Users/contender/Documents/GitHub/hakaton_10/x5-ner-skeleton/x5-ner/data/submission_processed.csv"

    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден!")
        return

    # Обрабатываем файл
    try:
        df_processed = process_submission_file(input_file, output_file)

        # Показываем первые несколько строк результата
        print("\nПервые 5 строк обработанного файла:")
        print(df_processed[["sample", "text_len", "text_type_len", "text_brand_len"]].head())

    except Exception as e:
        print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
