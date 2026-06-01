import argparse
import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from src.agent.agent import MUSIC_AGENT_PROMPT, ReActAgent
from src.chatbot.chatbot import Chatbot
from src.core.gemini_setup import create_gemini_provider
from src.tools.music_tools import MUSIC_TOOLS


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
HTML_PATH = ROOT / "demo.html"


class DemoHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/outputs/"):
            self._serve_output(parsed.path, include_body=False)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/demo.html"):
            self._send_file(HTML_PATH, "text/html; charset=utf-8")
            return

        if parsed.path.startswith("/outputs/"):
            self._serve_output(parsed.path)
            return

        if parsed.path == "/api/health":
            self._send_json({"status": "ok"})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chatbot":
            self._handle_chatbot()
            return

        if parsed.path == "/api/agent":
            self._handle_agent()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_chatbot(self):
        try:
            user_input = self._read_user_input()
            chatbot = Chatbot(create_gemini_provider())
            response = chatbot.run(user_input)
            self._send_json({"response": response, "history": chatbot.history})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_agent(self):
        try:
            user_input = self._read_user_input()
            agent = ReActAgent(
                llm=create_gemini_provider(),
                tools=MUSIC_TOOLS,
                max_steps=5,
                role_prompt=MUSIC_AGENT_PROMPT,
            )
            response = agent.run(user_input)
            wav_path = _find_wav_path(response, agent.history)
            payload = {
                "response": response,
                "history": agent.history,
                "wav_path": wav_path,
                "wav_url": _path_to_url(wav_path) if wav_path else None,
            }
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_user_input(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            body = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            body = raw_body.decode("cp1258", errors="replace")
        payload = json.loads(body or "{}")
        user_input = str(payload.get("message", "")).strip()
        if not user_input:
            raise ValueError("Message is required.")
        return user_input

    def _serve_output(self, url_path, include_body=True):
        relative = unquote(url_path.removeprefix("/outputs/"))
        target = (OUTPUT_DIR / relative).resolve()
        if not _is_inside(target, OUTPUT_DIR.resolve()) or not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Output file not found")
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_file(target, content_type, include_body=include_body)

    def _send_file(self, path, content_type, include_body=True):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def _find_wav_path(response, history):
    candidates = re.findall(r"outputs[/\\][\w .-]+\.wav", response or "")
    for item in history:
        observation = str(item.get("observation", ""))
        candidates.extend(re.findall(r"outputs[/\\][\w .-]+\.wav", observation))

    for candidate in reversed(candidates):
        normalized = candidate.replace("\\", os.sep).replace("/", os.sep)
        path = (ROOT / normalized).resolve()
        if _is_inside(path, OUTPUT_DIR.resolve()) and path.exists():
            return str(path.relative_to(ROOT)).replace("\\", "/")
    return None


def _path_to_url(path):
    if not path:
        return None
    return "/" + quote(path.replace("\\", "/"))


def _is_inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def run_server(port):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for candidate_port in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate_port), DemoHandler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"No available port found from {port} to {port + 19}.")

    host, active_port = server.server_address
    print(f"Demo server running at http://{host}:{active_port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Run the AI Music Agent demo server.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
