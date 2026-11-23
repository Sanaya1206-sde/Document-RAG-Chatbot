import os
from dataclasses import dataclass

import pdfplumber
from docx import Document


@dataclass
class Chunk:
    text: str
    source: str        # file name, used for citations
    page_number: int   # page (PDF) or section (DOCX/TXT)
    chunk_id: int


def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ''
            if text.strip():
                pages.append((number, text))
    return pages


def split_into_sections(text, section_size=3000):
    # DOCX and TXT files have no real pages, so we make
    # 'sections' of about 3000 characters to cite instead.
    sections, current, number = [], '', 1
    for line in text.split('\n'):
        if len(current) + len(line) > section_size and current.strip():
            sections.append((number, current))
            number += 1
            current = ''
        current += line + '\n'
    if current.strip():
        sections.append((number, current))
    return sections


def extract_docx(path):
    doc = Document(path)
    text = '\n'.join(p.text for p in doc.paragraphs)
    return split_into_sections(text)


def extract_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return split_into_sections(f.read())


def chunk_text(text, chunk_size=800, overlap=150):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def process_file(path, source_name=None, chunk_size=800, overlap=150):
    source = source_name or os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pdf':
        pages = extract_pdf(path)
    elif ext == '.docx':
        pages = extract_docx(path)
    elif ext == '.txt':
        pages = extract_txt(path)
    else:
        raise ValueError('Unsupported file type: ' + ext)

    chunks, chunk_id = [], 0
    for page_number, text in pages:
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append(Chunk(piece, source, page_number, chunk_id))
            chunk_id += 1

    if not chunks:
        raise ValueError('No readable text found (scanned PDFs need OCR).')
    return chunks


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python document_processor.py <file>')
        sys.exit(1)
    result = process_file(sys.argv[1])
    print('Chunks created:', len(result))
    print('First chunk is from page', result[0].page_number)
    print(result[0].text[:300])