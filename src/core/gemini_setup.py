import os

from src.core.env import load_env
from src.core.gemini_provider import GeminiProvider


def create_gemini_provider() -> GeminiProvider:
    """Create the configured Gemini provider from .env."""
    load_env()
    return GeminiProvider(
        model_name=os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )
