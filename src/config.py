"""Central configuration for the RAG pipeline."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


class Settings(BaseSettings):
    """Pipeline settings, overridable through environment variables."""

    data_dir: Path = ROOT_DIR / "data"

    chunk_size: int = 800
    chunk_overlap: int = 300
    min_chunk_length: int = 100
    body_marker: str = "Préambule"

    embedding_model: str = "intfloat/multilingual-e5-large"
    collection_name: str = "cgv"
    distance_metric: str = "cosine"
    batch_size: int = 32
    device: str = "cpu"

    top_k: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def pages_file(self) -> Path:
        return self.processed_dir / "pages.jsonl"

    @property
    def chunks_file(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
