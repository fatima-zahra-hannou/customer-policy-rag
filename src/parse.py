"""Extract text from PDF files, one entry per page."""

import json
import logging

import pymupdf

from config import settings

logger = logging.getLogger("parse")


def parse_pdf(pdf_path):
    """Extract text from a PDF, keeping one entry per non-empty page."""
    doc = pymupdf.open(pdf_path)
    pages = []
    empty = 0

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            empty += 1
            continue
        pages.append(
            {
                "doc_id": pdf_path.stem,
                "page": i + 1,
                "text": text,
            }
        )

    doc.close()

    if empty:
        logger.warning("%s empty pages skipped in %s", empty, pdf_path.name)

    return pages


def main():
    """Parse every PDF in the raw directory and write them as JSONL."""
    settings.pages_file.parent.mkdir(parents=True, exist_ok=True)
    all_pages = []

    pdf_files = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("no PDF found in %s", settings.raw_dir)
        return

    for pdf_path in pdf_files:
        pages = parse_pdf(pdf_path)
        logger.info("%s: %s pages extracted", pdf_path.name, len(pages))
        all_pages += pages

    with settings.pages_file.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    logger.info("%s pages written to %s", len(all_pages), settings.pages_file)


if __name__ == "__main__":
    main()
