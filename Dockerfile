FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY rag/ ./rag/
COPY data/raw/ ./data/raw/
COPY eval/golden_set.jsonl ./eval/golden_set.jsonl

ENV HF_HUB_DISABLE_XET=1

RUN uv run python -m rag.parse && \
    uv run python -m rag.chunk && \
    uv run python -m rag.index

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
