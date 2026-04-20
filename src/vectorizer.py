import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self):

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Embedding model loaded successfully")

    def encode_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert a list of texts into an embedding matrix.
        Used when building the knowledge base.

        Args:
            texts: List of input strings.
            show_progress: Whether to show the model progress bar.

        Returns:
            numpy.ndarray: Matrix with shape (N, D), where N is text count and D is vector size.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convert a single text into one embedding vector.
        Used to search for similar complaints.

        Args:
            text: One input string.

        Returns:
            numpy.ndarray: Vector with shape (D,).
        """
        if not text:
            return np.array([])

        embeddings = self.model.encode([text])
        return embeddings[0]
