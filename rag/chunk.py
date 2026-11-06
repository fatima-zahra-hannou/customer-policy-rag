"""Split pages into chunks, respecting article boundaries."""

import json
import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings

logger = logging.getLogger("chunk")

TITLE_PATTERN = re.compile(r"^(Article\s+\d+\s*[-–].+|\d+(?:\.\d+)+\s+.+)$", re.MULTILINE)


def clean(text):
    """Normalize whitespace so paragraph boundaries are detectable."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_pages():
    """Concatenate all pages into one text, tracking where each page starts."""
    full_text = ""
    page_starts = []

    with settings.pages_file.open(encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            page_starts.append((len(full_text), page["page"], page["doc_id"]))
            full_text += clean(page["text"]) + "\n"

    return full_text, page_starts


def skip_front_matter(full_text, page_starts):
    """Drop the table of contents, which is parsed as plain text.

    The PDF has no structured bookmarks, so every entry of the table of
    contents matches TITLE_PATTERN and would create an empty section.
    TODO: replace with a layout-aware parser to support arbitrary documents.
    """
    body_start = full_text.find(settings.body_marker)
    if body_start <= 0:
        logger.warning("body marker '%s' not found, keeping full text", settings.body_marker)
        return full_text, page_starts

    shifted = [(max(0, pos - body_start), page, doc_id) for pos, page, doc_id in page_starts]
    return full_text[body_start:], shifted


def find_source(page_starts, offset):
    """Return the page number and document id for a given character offset."""
    found = (None, None)
    for start, page, doc_id in page_starts:
        if start <= offset:
            found = (page, doc_id)
        else:
            break
    return found


def main():
    """Split the corpus into chunks and write them as JSONL."""
    full_text, page_starts = load_pages()
    full_text, page_starts = skip_front_matter(full_text, page_starts)

    titles = [(m.start(), m.group(0).strip()) for m in TITLE_PATTERN.finditer(full_text)]
    logger.info("%s titles detected", len(titles))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", ". ", "\n", " ", ""],
        add_start_index=True,
    )

    bounds = [pos for pos, _ in titles] + [len(full_text)]
    chunks = []

    for i in range(len(titles)):
        start, title = titles[i]
        end = bounds[i + 1]
        section_text = full_text[start:end]

        for doc in splitter.create_documents([section_text]):
            if len(doc.page_content.strip()) < settings.min_chunk_length:
                continue

            offset = start + doc.metadata["start_index"]
            page, doc_id = find_source(page_starts, offset)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{len(chunks)}",
                    "doc_id": doc_id,
                    "page": page,
                    "section": title,
                    "text": doc.page_content,
                }
            )

    settings.chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with settings.chunks_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in chunks]
    logger.info("%s chunks written to %s", len(chunks), settings.chunks_file)
    logger.info("size min %s, max %s, average %s", min(sizes), max(sizes), sum(sizes) // len(sizes))


if __name__ == "__main__":
    main()
