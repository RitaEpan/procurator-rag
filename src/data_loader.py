import os
import pandas as pd
from pandas import DataFrame
import pickle


def load_dataset(filepath: str) -> DataFrame:
    """
    Загружает датасет из CSV-файла.

    Args:
        filepath (str): Путь к файлу

    Returns:
        pd.DataFrame: Таблица с данными

    Raises:
       FileNotFoundError: Если файл не найден
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty at {filepath}")
    except pd.errors.ParserError:
        raise ValueError(f"Parse error at {filepath}")


def save_knowledge_base(data: dict, filepath: str) -> None:
    """
    Сохраняет базу знаний (данные + векторы) в pickle.

    Args:
        data (dict): Словарь с данными(df, embeddings)
        filepath (str): Путь к файлу

    Returns:
        None
    """
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    except (OSError, pickle.PickleError) as e:
        raise IOError(f"Failed to save knowledge base: {filepath}") from e
    print(f"Knowledge base saved: {filepath}")