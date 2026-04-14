import os
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self):
        """
        Инициализация векторизатора.
        Загружает модель в память. При первом запуске скачивает её из интернета.
        """
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Embedding model loaded successfully")

    def encode_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Преобразует СПИСОК текстов в матрицу векторов.
        Используется для создания базы знаний (один раз при старте).

        Args:
            texts: Список строк (например, все жалобы из CSV).
            show_progress: Показывать прогресс-бар (полезно для больших данных).

        Returns:
            numpy.ndarray: Матрица размерности (N, D), где N - кол-во текстов, D - размер вектора.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Преобразует ОДИН текст в вектор.
        Используется для поиска похожей жалобы (при каждом запросе пользователя).

        Args:
            text: Одна строка (жалоба пользователя).

        Returns:
            numpy.ndarray: Вектор размерности (D,).
        """
        if not text:
            return np.array([])

        embeddings = self.model.encode([text])
        return embeddings[0]
