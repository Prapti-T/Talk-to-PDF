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
        """Ensure the Pinecone index exists. If not, create it with given dimension."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust region if needed
            )

    def upsert_chunks(self, vectors: list, chunks: list, doc_id: str):
        """Upload embeddings and corresponding text chunks to Pinecone index. Each chunk gets a unique ID and metadata."""
        index = self.pc.Index(self.index_name)
        items = []

        for i, (vec, text) in enumerate(zip(vectors, chunks)):
            uid = f"{doc_id}-{i}"  # unique ID per chunk
            meta = {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": text
            }
            items.append((uid, vec.tolist(), meta))

        index.upsert(vectors=items)

    def query(self, vector: list, top_k: int = 5):
        index = self.pc.Index(self.index_name)
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        return [item['metadata']['text'] for item in results['matches']]
