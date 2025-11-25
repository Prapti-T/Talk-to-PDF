import re
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
    
    def remove_page_headers(self, text: str) -> str:
        """
        Remove page headers/footers that repeat across pages.
        Pattern: Lines that are exactly "AI competency framework for teachers – **...**"
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if re.match(
                r'^AI competency framew?ork for teachers\s*[–-]\s*\*\*.+?\*\*\s*$',
                line_stripped,
                flags=re.IGNORECASE
            ):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def remove_toc_section(self, text: str) -> str:
        """
        Remove everything from '#### ~~Table of contents~~' 
        to '#### ~~List of acronyms and abbreviations~~' (exclusive).
        """
        toc_start_marker = "#### ~~Table of contents~~"
        toc_end_marker = "#### ~~List of acronyms and abbreviations~~"
        
        toc_start = text.find(toc_start_marker)
        
        if toc_start == -1:
            print("TOC start marker not found")
            return text
        
        toc_end = text.find(toc_end_marker, toc_start)
        
        if toc_end == -1:
            print("TOC end marker not found")
            return text
        
        print(f"Removing TOC section from position {toc_start} to {toc_end}")
        
        cleaned_text = text[:toc_start] + text[toc_end:]
        
        return cleaned_text
    
    def smart_paragraph_join(self, text: str) -> str:
        """
        Intelligently join sentences that were split across paragraph breaks.
        A line ending without proper punctuation (. ! ? : ;) likely continues on next line.
        """
        lines = text.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                result.append('')
                i += 1
                continue
            
            ends_sentence = re.search(r'[.!?:;]\s*$', current_line)
            ends_with_number = re.search(r'\d+\s*$', current_line)  # Page numbers
            is_heading = re.match(r'^#+\s+', current_line) or re.match(r'^\*\*.*\*\*\s*$', current_line)
            is_list_item = re.match(r'^[-*•]\s+', current_line) or re.match(r'^\d+\.\s+', current_line)
            
            if not ends_sentence and not ends_with_number and not is_heading and not is_list_item:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                if j < len(lines):
                    next_line = lines[j].strip()
                    
                    starts_lowercase = next_line and next_line[0].islower()
                    starts_continuation = re.match(r'^(and|or|but|which|that|who|where|when)\s+', next_line, re.IGNORECASE)
                    
                    if starts_lowercase or starts_continuation:
                        current_line = current_line + ' ' + next_line
                        for k in range(i + 1, j):
                            if not lines[k].strip():
                                result.append('')
                        i = j + 1
                        result.append(current_line)
                        continue
            
            result.append(current_line)
            i += 1
        
        return '\n'.join(result)
    
    def remove_page_numbers(self, text: str) -> str:
        """
        Remove standalone page numbers that are isolated between empty lines.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        i = 0
        while i < len(lines):
            line_stripped = lines[i].strip()
            
            is_page_number = re.match(r'^\*{1,2}\d+\*{1,2}$', line_stripped)
            
            if is_page_number:
                prev_empty = (i == 0) or (not lines[i-1].strip())
                
                next_empty = (i == len(lines) - 1) or (i + 1 < len(lines) and not lines[i+1].strip())
                
                if prev_empty and next_empty:
                    i += 1
                    continue
            
            cleaned_lines.append(lines[i])
            i += 1
        
        return '\n'.join(cleaned_lines)

    def clean_markdown(self) -> str:
        """Cleans markdown while preserving paragraphs, tables, and lists, and removing TOC."""
        text = self.raw_text.replace("\x00", "")
        
        text = self.remove_toc_section(text)
        text = self.remove_page_numbers(text)
        text = self.remove_page_headers(text)
        text = self.smart_paragraph_join(text)

        paragraphs = re.split(r'\n\s*\n+', text)

        cleaned_paragraphs = []
        for para in paragraphs:
            p = para.strip()

            if re.match(r'^[\*~#`\s]+$', p):
                continue
            
            p = re.sub(r'[ \t]+', ' ', p)
            
            if p:
                cleaned_paragraphs.append(p)
            
        self.cleaned_text = '\n\n'.join(cleaned_paragraphs)
        return self.cleaned_text