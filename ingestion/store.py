from pinecone import Pinecone, ServerlessSpec
from config import CONFIG

class PineconeClient:
    """Handles Pinecone initialization, index management, and upserting embeddings/chunks."""

    def __init__(self):
        # Create Pinecone client instance
        self.pc = Pinecone(
            api_key=CONFIG["PINECONE_API_KEY"],
            environment=CONFIG["PINECONE_ENV"]
        )
        self.index_name = CONFIG["PINECONE_INDEX"]

    def ensure_index(self, dim: int):
        """Ensure the Pinecone index exists. If not, create it with given dimension. If index exists, check dimension and raise an error if mismatch."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            # Create index if it does not exist
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust region if needed
            )
        else:
            # Check existing index dimension
            index_info = self.pc.describe_index(self.index_name)
            existing_dim = index_info['dimension']
            if existing_dim != dim:
                raise ValueError(
                    f"Dimension mismatch: Pinecone index '{self.index_name}' expects {existing_dim}d, "
                    f"but your embeddings are {dim}d. Either change your embedding model or create a new index."
                )

    def upsert_chunks(self, vectors: list, chunks: list, doc_id: str):
        """Upload embeddings and corresponding text chunks to Pinecone index."""
        index = self.pc.Index(self.index_name)
        items = []

        if vectors is None or len(vectors) == 0 or len(chunks) == 0:
            print("No vectors or chunks to upsert.")
            return

        for i, (vec, text) in enumerate(zip(vectors, chunks)):
            uid = f"{doc_id}-{i}"
            meta = {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": text
            }
            items.append((uid, vec.tolist(), meta))

        index.upsert(vectors=items)

    def query(self, vector, top_k: int = 5):
        """Query Pinecone index with embedding vector and return top_k text chunks."""
        if not vector:
            return []

        index = self.pc.Index(self.index_name)
        
        # Ensure vector is properly formatted
        import numpy as np
        if isinstance(vector, np.ndarray):
            query_vector = vector.tolist()
        else:
            query_vector = vector
        
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        matches = results.get('matches', [])

        return [item['metadata']['text'] for item in matches]