from typing import List
from config import TOKENIZER, CONFIG
import re

class MarkdownChunker:
    def __init__(self):
        self.tokenizer = TOKENIZER
        self.max_tokens = CONFIG["CHUNK_TOKENS"]
        self.overlap = CONFIG["CHUNK_OVERLAP"]

    def chunk_tokens(self, token_ids: List[int]) -> List[List[int]]:
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + self.max_tokens, len(token_ids))
            chunks.append(token_ids[start:end])
            start += self.max_tokens - self.overlap
        return chunks

    def split_blocks(self, md_text: str) -> List[str]:
        lines = md_text.split('\n')
        blocks = []
        cur_block = []
        in_table = False

        for line in lines:
            stripped = line.strip()

            # Detect table lines
            if '|' in line and re.search(r'\|\s*\S', line):
                if not in_table:
                    if cur_block:
                        blocks.append('\n'.join(cur_block).strip())
                        cur_block = []
                    in_table = True
                cur_block.append(line)
            else:
                if in_table:
                    blocks.append('\n'.join(cur_block).strip())
                    cur_block = []
                    in_table = False
                cur_block.append(line)

        if cur_block:
            blocks.append('\n'.join(cur_block).strip())
        return blocks

    def markdown_to_chunks(self, md_text: str) -> List[str]:
        blocks = self.split_blocks(md_text)
        all_chunks = []

        for block in blocks:
            if '|' in block:
                rows = block.split('\n')

                header_tokens = []
                start_idx = 0
                if rows:
                    first_cols = [c.strip() for c in rows[0].split('|')]
                    if "" in first_cols and len(rows) > 1:
                        header_rows = rows[:2]
                        start_idx = 2
                    else:
                        header_rows = [rows[0]]
                        start_idx = 1

                    for hr in header_rows:
                        header_tokens.extend(self.tokenizer.encode(hr, add_special_tokens=False))

                # Process remaining rows
                for row in rows[start_idx:]:
                    row_tokens = self.tokenizer.encode(row, add_special_tokens=False)
                    if len(header_tokens) + len(row_tokens) <= self.max_tokens:
                        chunk_ids = header_tokens + row_tokens
                        all_chunks.append(self.tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True).strip())
                    else:
                        start = 0
                        while start < len(row_tokens):
                            end = start + (self.max_tokens - len(header_tokens))
                            chunk_ids = header_tokens + row_tokens[start:end]
                            all_chunks.append(self.tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True).strip())
                            start += self.max_tokens - self.overlap
            else:
                block_tokens = self.tokenizer.encode(block, add_special_tokens=False)
                for chunk_ids in self.chunk_tokens(block_tokens):
                    all_chunks.append(self.tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True).strip())

        return all_chunks
