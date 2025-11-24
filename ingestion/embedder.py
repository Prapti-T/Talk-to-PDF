from typing import List
import numpy as np
from config import EMBEDDER


class Embedder:
    def __init__(self):
        self.model = EMBEDDER

    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
