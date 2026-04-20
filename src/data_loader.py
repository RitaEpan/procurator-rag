import logging
import os
import pickle
import zipfile
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): File path.

    Returns:
        pd.DataFrame: Loaded data table.

    Raises:
        FileNotFoundError: If the file does not exist.
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
    Save the knowledge base (data + vectors) as a portable zip archive.

    Args:
        data (dict): Dictionary with df and embeddings.
        filepath (str): Output file path.

    Returns:
        None
    """
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    try:
        dataframe_json = data["df"].to_json(orient="table", force_ascii=False)
        embeddings_buffer = BytesIO()
        np.save(embeddings_buffer, data["embeddings"], allow_pickle=False)
        embeddings_buffer.seek(0)

        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("dataframe.json", dataframe_json)
            archive.writestr("embeddings.npy", embeddings_buffer.read())
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise IOError(f"Failed to save knowledge base: {filepath}") from e

    logger.info("Saved knowledge base: %s", filepath)


def load_knowledge_base(filepath: str) -> dict | None:
    """
    Load the knowledge base from a portable archive or a legacy pickle file.

    Args:
        filepath (str): File path.

    Returns:
        dict | None: Dictionary with df and embeddings, or None if unavailable.
    """
    if not os.path.isfile(filepath):
        return None

    try:
        with zipfile.ZipFile(filepath, "r") as archive:
            dataframe_json = archive.read("dataframe.json").decode("utf-8")
            embeddings_bytes = archive.read("embeddings.npy")

        df = pd.read_json(StringIO(dataframe_json), orient="table")
        embeddings = np.load(BytesIO(embeddings_bytes), allow_pickle=False)
        return {"df": df, "embeddings": embeddings}
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
        logger.warning("Falling back to legacy knowledge base format for %s: %s", filepath, e)

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or "df" not in data or "embeddings" not in data:
            logger.warning("Legacy knowledge base has unexpected structure: %s", filepath)
            return None

        return data
    except (OSError, pickle.UnpicklingError, AttributeError, ImportError, EOFError, NotImplementedError) as e:
        logger.warning("Failed to load knowledge base %s: %s", filepath, e)
        return None
