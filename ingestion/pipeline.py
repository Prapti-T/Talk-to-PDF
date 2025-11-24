import pathlib
import uuid
from .parser import PDFParser
from .chunker import MarkdownChunker
from .embedder import Embedder
from .store import PineconeClient


class PDFIngestPipeline:
    """
    Orchestrates ingestion of a PDF into a RAG-friendly vector store:
    1. Parse PDF → Markdown
    2. Clean text
    3. Chunk into semantic + token-aware chunks
    4. Embed chunks
    5. Upsert embeddings + metadata into Pinecone
    """

    def __init__(self, pdf_path: str = None):
        if pdf_path is None:
            # Default PDF path in project files folder
            self.pdf_path = pathlib.Path(__file__).resolve().parents[1] / "files" / "input.pdf"
        else:
            self.pdf_path = pathlib.Path(pdf_path).resolve()

        self.doc_id = f"{self.pdf_path.stem}_{uuid.uuid4().hex[:6]}"

        # Initialize component classes
        self.parser = PDFParser(str(self.pdf_path))
        self.chunker = MarkdownChunker()
        self.embedder = Embedder()
        self.store = PineconeClient()

    def run(self):
        """Run the full pipeline and return number of chunks ingested."""

        # --- Step 1: Parse PDF to Markdown ---
        raw_md = self.parser.extract_to_markdown()
        if not raw_md:
            print("Failed to extract markdown from PDF.")
            return 0

        # --- Step 2: Clean Markdown ---
        cleaned_md = self.parser.clean_markdown()

        # --- Step 3: Chunk Markdown ---
        chunks = self.chunker.markdown_to_chunks(cleaned_md)
        if not chunks:
            print("No chunks extracted.")
            return 0

        # --- Step 4: Embed chunks ---
        vectors = self.embedder.embed_chunks(chunks)

        # --- Step 5: Store in Pinecone ---
        dim = vectors[0].shape[0] if len(vectors) > 0 else 0
        if dim == 0:
            print("Embeddings have zero dimension.")
            return 0

        self.store.ensure_index(dim)
        self.store.upsert_chunks(vectors, chunks, self.doc_id)

        print(f"Ingested {len(chunks)} chunks into Pinecone index '{self.store.index_name}' for doc_id '{self.doc_id}'.")
        return len(chunks)


def ingest_pdf_to_pinecone(pdf_path: str = 'files/input.pdf'):
    pipeline = PDFIngestPipeline(pdf_path)
    return pipeline.run()
