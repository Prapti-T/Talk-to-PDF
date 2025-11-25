from config import TOKENIZER, EMBEDDER
from ingestion.store import PineconeClient

class Retriever:
    """Retrieves top chunks for a query using Pinecone embeddings"""

    def __init__(self):
        self.tokenizer = TOKENIZER
        self.embedder = EMBEDDER
        self.pc_client = PineconeClient()

    def retrieve(self, query: str, top_k: int = 5):
        """
        Embed query, query Pinecone, return top chunk texts as a list
        """
        query_embedding = self.embedder.encode([query])[0]
        
        query_vector = query_embedding.tolist()
        
        top_chunks = self.pc_client.query(query_vector, top_k=top_k)
        
        return top_chunks