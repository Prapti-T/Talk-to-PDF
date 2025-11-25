import pathlib
import uuid
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from parser import PDFParser
from chunker import MarkdownChunker
from embedder import Embedder
from store import PineconeClient

class PDFIngestPipeline:
    """
    Orchestrates ingestion of a PDF into a vector store:
    1. Parse PDF -> Markdown
    2. Clean text
    3. Tokenize and chunk into semantic + token-aware chunks
    4. Embed chunks
    5. Upsert embeddings + metadata into Pinecone
    """

    def __init__(self, pdf_path: str = None):
        if pdf_path is None:
            self.pdf_path = pathlib.Path(__file__).resolve().parents[0] / "files" / "input.pdf"
        else:
            self.pdf_path = pathlib.Path(pdf_path).resolve()

        self.doc_id = f"{self.pdf_path.stem}_{uuid.uuid4().hex[:6]}"

        self.parser = PDFParser(str(self.pdf_path))
        self.chunker = MarkdownChunker()
        self.embedder = Embedder()
        self.store = PineconeClient()

    def run(self):
        """Run the full pipeline and return number of chunks ingested."""
        raw_md = self.parser.extract_to_markdown()
        if not raw_md:
            print("Failed to extract markdown from PDF.")
            return 0

        cleaned_md = self.parser.clean_markdown()
        chunks = self.chunker.markdown_to_chunks(cleaned_md)
        if not chunks:
            print("No chunks extracted.")
            return 0

        vectors = self.embedder.embed_chunks(chunks)
        dim = vectors[0].shape[0] if len(vectors) > 0 else 0
        if dim == 0:
            print("Embeddings have zero dimension.")
            return 0

        self.store.ensure_index(dim)
        self.store.upsert_chunks(vectors, chunks, self.doc_id)

        print(f"Ingested {len(chunks)} chunks into Pinecone index '{self.store.index_name}' for doc_id '{self.doc_id}'.")
        return len(chunks)


def ingest_pdf_to_pinecone(pdf_path: str = None):
    pipeline = PDFIngestPipeline(pdf_path)
    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a PDF into Pinecone.")
    parser.add_argument("pdf_path", nargs="?", default="files/input.pdf", help="Path to the PDF file to ingest.")
    args = parser.parse_args()

    ingested_count = ingest_pdf_to_pinecone(args.pdf_path)
    print(f"Total chunks ingested: {ingested_count}")
