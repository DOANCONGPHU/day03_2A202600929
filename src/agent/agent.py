import ast
import json
import re
from typing import Callable, List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

MUSIC_AGENT_PROMPT = """
Ban la AI Music Agent, mot agent tao file am thanh cho nguoi dung.

Muc tieu chinh:
- Khi nguoi dung yeu cau tao nhac, hay tao file .wav that bang tool duoc cap.
- Khong chi mo ta y tuong, khong in code, khong noi rang ban khong the tao file.
- Ket qua cuoi cung phai la duong dan file .wav that nam trong thu muc outputs/.

Quy trinh bat buoc:
- Ban phai lam viec theo vong lap Thought - Action - Observation.
- Moi luot chi duoc viet toi da mot Action.
- Sau khi viet Action, dung lai ngay. Chuong trinh se tu chay tool va cung cap Observation.
- Khong bao gio tu viet Observation.
- Khong bao gio viet Final Answer cung luot voi Action.
- Khong bao gio tu bia duong dan file.
- Chi su dung duong dan file xuat hien trong Observation that.

Chon tool:
- Neu can lam nhanh, dung create_music_wav(...) de tao MIDI va WAV trong mot action.
- Neu can trinh bay dung quy trinh hai buoc, goi create_midi(...) truoc, doi Observation tra ve .mid,
  sau do goi midi_to_wav(midi_path="...") o luot tiep theo.
- Neu nguoi dung da yeu cau xuat .wav, Final Answer chi duoc dua ra sau khi Observation co duong dan .wav.

Chuan hoa tham so:
- title: dat ngan gon, khong dau, phu hop yeu cau; vi du "calm_music", "drill_track".
- mood: uu tien mot trong happy, sad, calm, epic, energetic, drill, dark, lofi.
- key: neu nguoi dung khong noi, dung "C"; neu nhac toi/drill, co the dung "A minor" hoac "C".
- tempo: neu nguoi dung khong noi, chon theo mood; calm 80-90, lofi 75-90, drill 120-145, epic 105-130.
- bars: neu nguoi dung khong noi, dung 4; neu co noi, dung dung so bars.
- waveform: chi dung sine, square, hoac soft_square neu can; neu khong chac thi bo qua.

Dinh dang phan hoi:
- Khi can tool:
  Thought: ly do ngan gon
  Action: tool_name(arg1="value", arg2=123)
- Khi da co .wav trong Observation:
  Final Answer: outputs/ten_file.wav
"""


class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
        role_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.role_prompt = role_prompt
        self.history = []

    def get_system_prompt(self) -> str:
        """Build a prompt that teaches the model the available tools and loop format."""
        tool_descriptions = "\n".join(
            [
                f"- {tool['name']}: {tool.get('description', 'No description provided.')}"
                for tool in self.tools
            ]
        )
        role_prompt = self.role_prompt or "You are an intelligent assistant that solves tasks with ReAct reasoning."
        return f"""
        {role_prompt.strip()}

        You have access to these tools:
        {tool_descriptions or "- No tools available."}

Rules:
- Use a tool only when it is needed to answer accurately.
- For music requests, create a .mid file first, then convert it to .wav.
- Do not print source code as the answer.
- Do not write Observation yourself. Observation is produced only by the program after a real tool call.
- Do not invent file paths. Only use file paths returned in Observation.
- If you write an Action, do not write Final Answer in the same response.
- The final answer for a generated audio request must be exactly the final .wav path or one short sentence that includes it.
- When using a tool, write exactly one Action line in this format:
  Action: tool_name(arguments)
- Arguments may be a plain string, a JSON object, a JSON array, or key=value pairs.
- After an Observation is provided, continue reasoning from that observation.
        - When you know the answer, stop using tools and write:
        Final Answer: your final response.
        """

    def run(self, user_input: str) -> str:
        """
        Implement the ReAct loop logic.
        1. Generate Thought + Action.
        2. Parse Action and execute Tool.
        3. Append Observation to prompt and repeat until Final Answer.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        self.history = []
        current_prompt = f"Question: {user_input}\n"
        final_answer = ""

        for step in range(1, self.max_steps + 1):
            result = self.llm.generate(
                current_prompt,
                system_prompt=self.get_system_prompt(),
            )
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )
            content = result.get("content", "").strip()
            self.history.append({"step": step, "llm_response": content})

            logger.log_event(
                "LLM_RESPONSE",
                {
                    "step": step,
                    "content": content,
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms"),
                    "provider": result.get("provider"),
                },
            )

            action = self._parse_action(content)
            if action is not None:
                tool_name, args = action
                observation = self._execute_tool(tool_name, args)
                logger.log_event(
                    "TOOL_CALL",
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": args,
                        "observation": observation,
                    },
                )
            else:
                final_answer = self._extract_final_answer(content)
                if final_answer:
                    logger.log_event("AGENT_END", {"steps": step, "status": "final_answer"})
                    return final_answer

                observation = (
                    "Parser error: no valid Action line found. Use "
                    "'Action: tool_name(arguments)' or provide 'Final Answer: ...'."
                )
                logger.log_event(
                    "PARSER_ERROR",
                    {"step": step, "content": content, "observation": observation},
                )

            self.history[-1]["observation"] = observation
            current_prompt = (
                f"{current_prompt}{self._remove_hallucinated_tool_continuation(content)}\n"
                f"Observation: {observation}\n"
            )

        logger.log_event(
            "AGENT_END",
            {"steps": self.max_steps, "status": "max_steps_exceeded"},
        )
        return (
            "I could not produce a final answer within "
            f"{self.max_steps} steps. Last observation: "
            f"{self.history[-1].get('observation', 'none') if self.history else 'none'}"
        )

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        for tool in self.tools:
            if tool["name"] == tool_name:
                func = self._get_tool_callable(tool)
                if func is None:
                    return f"Tool {tool_name} has no callable function configured."

                try:
                    parsed_args = self._parse_tool_args(args)
                    if isinstance(parsed_args, dict):
                        result = func(**parsed_args)
                    elif isinstance(parsed_args, list):
                        result = func(*parsed_args)
                    elif parsed_args in ("", None):
                        result = func()
                    else:
                        result = func(parsed_args)
                    return str(result)
                except Exception as exc:
                    logger.log_event(
                        "TOOL_ERROR",
                        {"tool": tool_name, "args": args, "error": str(exc)},
                    )
                    return f"Tool {tool_name} failed: {exc}"
        return f"Tool {tool_name} not found."

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract an Action line from model output."""
        cleaned = self._strip_code_fences(text)
        pattern = r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:\n|$)"
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if not match:
            return None
        return match.group(1), match.group(2).strip()

    def _extract_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer\s*:\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _get_tool_callable(self, tool: Dict[str, Any]) -> Optional[Callable[..., Any]]:
        for key in ("func", "function", "callable", "run"):
            candidate = tool.get(key)
            if callable(candidate):
                return candidate
        return None

    def _parse_tool_args(self, args: str) -> Any:
        args = self._strip_code_fences(args).strip()
        if not args:
            return ""

        if self._looks_like_json(args):
            return json.loads(args)

        try:
            return ast.literal_eval(args)
        except (SyntaxError, ValueError):
            pass

        keyword_args = self._parse_keyword_args(args)
        if keyword_args is not None:
            return keyword_args

        return args.strip("\"'")

    def _parse_keyword_args(self, args: str) -> Optional[Dict[str, Any]]:
        if "=" not in args:
            return None

        parsed = {}
        for part in self._split_args(args):
            if "=" not in part:
                return None
            key, value = part.split("=", 1)
            key = key.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                return None
            parsed[key] = self._parse_single_value(value.strip())
        return parsed

    def _parse_single_value(self, value: str) -> Any:
        if self._looks_like_json(value):
            return json.loads(value)
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("\"'")

    def _split_args(self, args: str) -> List[str]:
        parts = []
        current = []
        depth = 0
        quote = None

        for char in args:
            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                continue

            if char in ("'", '"'):
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue

            current.append(char)

        if current:
            parts.append("".join(current).strip())
        return parts

    def _looks_like_json(self, value: str) -> bool:
        return (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        )

    def _strip_code_fences(self, text: str) -> str:
        return re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text.strip())

    def _remove_hallucinated_tool_continuation(self, text: str) -> str:
        return re.split(
            r"\n\s*(Observation|Final Answer)\s*:",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
