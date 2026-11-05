"""Extract text from PDF files, one entry per page."""

import json

import pymupdf

from config import settings


def parse_pdf(pdf_path):
    """Extract text from a PDF, keeping one entry per non-empty page"""
    doc = pymupdf.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append(
                {
                    "doc_id": pdf_path.stem,
                    "page": i + 1,
                    "text": text,
                }
            )

    doc.close()
    return pages


def main():
    """Parse every PDF in the raw directory and write them as JSONL."""
    settings.pages_file.parent.mkdir(parents=True, exist_ok=True)
    all_pages = []

    for pdf_path in sorted(settings.raw_dir.glob("*.pdf")):
        pages = parse_pdf(pdf_path)
        print(f"{pdf_path.name}: {len(pages)} pages")
        all_pages += pages

    with settings.pages_file.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_pages)} pages written to {settings.pages_file}")


if __name__ == "__main__":
    main()
