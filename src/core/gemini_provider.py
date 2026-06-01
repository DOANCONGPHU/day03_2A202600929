import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiProvider.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = self._post_generate_content(full_prompt)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = self._extract_content(response)
        usage_metadata = response.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0)
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        yield self.generate(prompt, system_prompt=system_prompt)["content"]

    def _post_generate_content(self, prompt: str) -> Dict[str, Any]:
        last_error = None
        for model_name in self._candidate_models():
            for attempt in range(2):
                try:
                    return self._post_to_model(model_name, prompt)
                except urllib.error.HTTPError as exc:
                    error_body = exc.read().decode("utf-8", errors="replace")
                    last_error = RuntimeError(
                        f"Gemini API error {exc.code} on {model_name}: {error_body}"
                    )
                    if exc.code not in (429, 503):
                        raise last_error from exc
                    time.sleep(1 + attempt)

        raise last_error or RuntimeError("Gemini API request failed.")

    def _post_to_model(self, model_name: str, prompt: str) -> Dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError:
            raise

    def _candidate_models(self) -> list:
        models = [self.model_name.removeprefix("models/")]
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.0-flash,gemini-flash-latest")
        models.extend(model.strip().removeprefix("models/") for model in fallbacks.split(","))
        return list(dict.fromkeys(model for model in models if model))

    def _extract_content(self, response: Dict[str, Any]) -> str:
        candidates = response.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()
