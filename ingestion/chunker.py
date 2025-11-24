from typing import List
import re
from config import TOKENIZER, CONFIG

class MarkdownChunker:
    """Semantic splitting of markdown text into blocks (headings, tables, lists, paragraphs), Token-aware chunking of these blocks based on MAX_TOKENS and sliding overlap"""

    def __init__(self):
        self.tokenizer = TOKENIZER
        self.max_tokens = CONFIG["CHUNK_TOKENS"]
        self.overlap = CONFIG["CHUNK_OVERLAP"]

    def split_blocks(self, md_text: str) -> List[str]:
        """Split markdown text into semantic blocks:
        - headings (#, ##, ...)
        - markdown tables (| ... |)
        - bullet lists
        - paragraphs
        This is purely text-level, no tokenization happens here."""
        lines = md_text.splitlines()
        blocks = []
        cur = []

        def flush():
            """Flush current buffer as a block if not empty."""
            if cur:
                text = "\n".join(cur).strip()
                if text:
                    blocks.append(text)
                cur.clear()

        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("#"):  # heading
                flush()
                blocks.append(stripped)
            elif '|' in ln and re.search(r'\|.*\|', ln):  # table line
                cur.append(ln)
            elif stripped.startswith(("-", "*", "+", "•")):  # bullet list
                cur.append(ln)
            elif stripped == "":  # blank line → flush
                flush()
            else:
                cur.append(ln)
        flush()
        return blocks

    def chunk_block(self, block: str) -> List[str]:
        """
        - Convert block → token IDs using tokenizer
        - If block smaller than MAX_TOKENS, return as-is
        - Otherwise, split by sentence, if sentence too long, hard-split by tokens with sliding overlap
        """

        token_ids = self.tokenizer.encode(block, add_special_tokens=False)

        if len(token_ids) <= self.max_tokens:
            return [block]

        sentences = re.split(r'(?<=[.!?])\s+', block)
        chunks = []
        cur_tokens = []
        cur_text = ""

        for sent in sentences:
            sent_ids = self.tokenizer.encode(sent, add_special_tokens=False)

            if len(cur_tokens) + len(sent_ids) <= self.max_tokens:
                cur_tokens += sent_ids
                cur_text += (" " if cur_text else "") + sent
            else:
                if cur_text:
                    chunks.append(cur_text.strip())

                if len(sent_ids) > self.max_tokens:
                    start = 0
                    while start < len(sent_ids):
                        part_ids = sent_ids[start:start+self.max_tokens]
                        part_text = self.tokenizer.decode(part_ids, clean_up_tokenization_spaces=True)
                        chunks.append(part_text.strip())
                        start += self.max_tokens - self.overlap
                    cur_tokens = []
                    cur_text = ""
                else:
                    cur_tokens = sent_ids.copy()
                    cur_text = sent

        if cur_text:
            chunks.append(cur_text.strip())

        return chunks

    def markdown_to_chunks(self, md_text: str) -> List[str]:
        """
        - Split markdown → semantic blocks
        - Chunk each block token-wise
        - Ensure no chunk exceeds MAX_TOKENS using final token slicing pass
        """
        blocks = self.split_blocks(md_text)
        chunks = []

        for block in blocks:
            chunks.extend(self.chunk_block(block))

        final_chunks = []
        for c in chunks:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if len(ids) <= self.max_tokens:
                final_chunks.append(c)
            else:
                start = 0
                while start < len(ids):
                    part = ids[start:start+self.max_tokens]
                    part_text = self.tokenizer.decode(part, clean_up_tokenization_spaces=True).strip()
                    final_chunks.append(part_text)
                    start += self.max_tokens - self.overlap

        return final_chunks
