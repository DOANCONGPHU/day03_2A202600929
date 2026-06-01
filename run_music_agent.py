import sys

from src.agent.agent import MUSIC_AGENT_PROMPT, ReActAgent
from src.core.gemini_setup import create_gemini_provider
from src.tools.music_tools import MUSIC_TOOLS


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    user_input = " ".join(sys.argv[1:]).strip()
    if not user_input:
        user_input = input("Nhap yeu cau tao nhac cho agent: ").strip()

    if not user_input:
        print("Vui long nhap yeu cau.")
        return

    agent = ReActAgent(
        llm=create_gemini_provider(),
        tools=MUSIC_TOOLS,
        max_steps=5,
        role_prompt=MUSIC_AGENT_PROMPT,
    )
    print(agent.run(user_input))


if __name__ == "__main__":
    main()
