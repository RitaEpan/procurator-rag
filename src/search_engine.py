import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import NUM_SIMILAR_EXAMPLES


class SearchEngine:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
        print(f"Search engine is ready. Records in the base: {len(embeddings)}")

    def find_similar(self, query_vector: np.ndarray, top_k: int = NUM_SIMILAR_EXAMPLES) -> list[dict]:
        similarities = cosine_similarity([query_vector], self.embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        top_indices = sorted_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'complaint': self.df.iloc[idx]['complaint'],
                'response': self.df.iloc[idx]['response'],
                'score': float(similarities[idx])
            })

        return results
