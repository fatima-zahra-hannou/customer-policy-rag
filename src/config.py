"""Central configuration for the RAG pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Pipeline settings, overridable through environment variables."""

    data_dir: Path = ROOT_DIR / "data"

    chunk_size: int = 800
    chunk_overlap: int = 300
    min_chunk_length: int = 100
    body_marker: str = "Préambule"

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


settings = Settings()
