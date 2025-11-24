import pymupdf4llm

class PDFParser:
    """Handles PDF ingestion: parsing to markdown and cleaning text."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.cleaned_text = ""

    def extract_to_markdown(self) -> str:
        """Converts PDF to markdown using pymupdf4llm."""
        try:
            self.raw_text = pymupdf4llm.to_markdown(self.pdf_path)
        except Exception as e:
            print(f"\nError processing PDF: {e}")
            self.raw_text = ""
        return self.raw_text

    def clean_markdown(self) -> str:
        """Cleans markdown by removing nulls and normalizing whitespace."""
        text = self.raw_text.replace("\x00", "")
        text = text.replace("   ", " ").replace("  ", " ")
        self.cleaned_text = text.strip()
        return self.cleaned_text
