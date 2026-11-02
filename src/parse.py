"""Extract text from PDF files, one entry per page."""

import json
from pathlib import Path

import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
OUT_FILE = ROOT_DIR / "data" / "processed" / "pages.jsonl"


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
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_pages = []

    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        pages = parse_pdf(pdf_path)
        print(f"{pdf_path.name}: {len(pages)} pages")
        all_pages += pages

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_pages)} pages written to {OUT_FILE}")


if __name__ == "__main__":
    main()
