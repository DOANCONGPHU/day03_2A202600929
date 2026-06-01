from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


MUSIC_CHATBOT_PROMPT = """
Ban la AI Music Chatbot. Hay tu van y tuong am nhac cho nguoi dung.
Ban khong co cong cu de tao file .mid hoac .wav, nen chi duoc tra loi bang van ban.
Neu nguoi dung yeu cau tao file am thanh, hay mo ta ban se tao gi va noi ro rang
rang baseline chatbot khong the truc tiep tao artifact.
"""


class Chatbot:
    """
    Minimal chatbot baseline for comparing against the ReAct agent.

    It does not receive tools and cannot act on the filesystem. This makes it useful
    for showing the baseline limitation in artifact-generation tasks.
    """

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt or MUSIC_CHATBOT_PROMPT
        self.history: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> str:
        logger.log_event(
            "CHATBOT_START",
            {"input": user_input, "model": self.llm.model_name},
        )

        result = self.llm.generate(
            user_input,
            system_prompt=self.system_prompt,
        )
        content = result.get("content", "").strip()
        self.history.append({"input": user_input, "response": content})

        logger.log_event(
            "CHATBOT_RESPONSE",
            {
                "content": content,
                "usage": result.get("usage", {}),
                "latency_ms": result.get("latency_ms"),
                "provider": result.get("provider"),
            },
        )
        logger.log_event("CHATBOT_END", {"status": "completed"})

        return content
