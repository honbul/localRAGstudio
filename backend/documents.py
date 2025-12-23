import os
from typing import Iterable, Tuple

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def iter_paths(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name)


def read_pdf(path: str) -> Tuple[str, dict]:
    reader = PdfReader(path)
    pages = []
    for idx, page in enumerate(reader.pages):
        pages.append({"page": idx + 1, "text": page.extract_text() or ""})
    text = "\n".join(p["text"] for p in pages)
    return text, {"pages": pages}


def read_docx(path: str) -> Tuple[str, dict]:
    doc = Document(path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text, {}


def read_text(path: str) -> Tuple[str, dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(), {}


def load_document(path: str) -> Tuple[str, dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type: {ext}")
