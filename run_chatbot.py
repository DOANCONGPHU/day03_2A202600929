import sys

from src.chatbot.chatbot import Chatbot
from src.core.gemini_setup import create_gemini_provider


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    user_input = " ".join(sys.argv[1:]).strip()
    if not user_input:
        user_input = input("Nhap yeu cau cho chatbot: ").strip()

    if not user_input:
        print("Vui long nhap yeu cau.")
        return

    chatbot = Chatbot(create_gemini_provider())
    print(chatbot.run(user_input))


if __name__ == "__main__":
    main()
