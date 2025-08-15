import re
from PyPDF2 import PdfReader
try:
    import docx
except ImportError:
    docx = None

def read_pdf(file) -> str:
    return "\n".join([page.extract_text() or "" for page in PdfReader(file).pages])

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def read_docx(file) -> str:
    if not docx:
        return ""
    return "\n".join(p.text for p in docx.Document(file).paragraphs)

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def trim_text_tokens(s: str, max_chars: int = 24000) -> str:
    return s if len(s) <= max_chars else s[:max_chars]