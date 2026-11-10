"""Single entry point for LLM calls."""

import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.config import settings

logger = logging.getLogger("llm")


@lru_cache(maxsize=1)
def get_client():
    """Create the Gemini client once, from the API key in the environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def call_llm(prompt, response_schema=None):
    """Call the LLM with retries; return JSON text when a schema is given."""
    config = types.GenerateContentConfig(temperature=settings.temperature)
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    for attempt in range(settings.llm_retries):
        try:
            response = get_client().models.generate_content(
                model=settings.llm_model, contents=prompt, config=config
            )
            return response.text.strip()
        except Exception as exc:
            if attempt == settings.llm_retries - 1:
                raise
            wait = 2**attempt
            logger.warning("LLM call failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)
    return None
