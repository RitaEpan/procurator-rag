import os
import pandas as pd
from pandas import DataFrame


def load_dataset(filepath: str) -> DataFrame:
    """
    Загружает датасет из CSV-файла

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

