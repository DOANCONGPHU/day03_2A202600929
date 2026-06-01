import os

from src.agent.agent import MUSIC_AGENT_PROMPT, ReActAgent
from src.chatbot.chatbot import Chatbot
from src.core.llm_provider import LLMProvider
from src.tools.music_tools import MUSIC_TOOLS


class FakeChatLLM(LLMProvider):
    def __init__(self):
        super().__init__("fake-chat")

    def generate(self, prompt, system_prompt=None):
        return {
            "content": "Toi co the mo ta y tuong nhac, nhung khong tao file wav.",
            "usage": {},
            "latency_ms": 1,
            "provider": "fake",
        }

    def stream(self, prompt, system_prompt=None):
        yield ""


class FakeMusicAgentLLM(LLMProvider):
    def __init__(self):
        super().__init__("fake-agent")
        self.calls = 0

    def generate(self, prompt, system_prompt=None):
        self.calls += 1
        if self.calls == 1:
            content = (
                "Thought: Can tao MIDI truoc.\n"
                "Action: create_midi(title=\"test_music_agent\", mood=\"calm\", key=\"C\", bars=1)"
            )
        elif self.calls == 2:
            content = (
                "Thought: Da co MIDI, bay gio chuyen sang WAV.\n"
                "Action: midi_to_wav(midi_path=\"outputs/test_music_agent.mid\")"
            )
        else:
            content = "Thought: Da co file wav.\nFinal Answer: outputs/test_music_agent.wav"

        return {
            "content": content,
            "usage": {},
            "latency_ms": 1,
            "provider": "fake",
        }

    def stream(self, prompt, system_prompt=None):
        yield ""


class FakeHallucinatingAgentLLM(LLMProvider):
    def __init__(self):
        super().__init__("fake-hallucinating-agent")
        self.calls = 0

    def generate(self, prompt, system_prompt=None):
        self.calls += 1
        if self.calls == 1:
            content = (
                "Thought: I should call the tool.\n"
                "Action: create_music_wav(title='hallucinated_path_test', mood='drill', key='Am', tempo=120, bars=1)\n"
                "Observation: File created. /tmp/fake.wav\n"
                "Final Answer: /tmp/fake.wav"
            )
        else:
            content = "Thought: I will use the real observation.\nFinal Answer: outputs/hallucinated_path_test.wav"

        return {
            "content": content,
            "usage": {},
            "latency_ms": 1,
            "provider": "fake",
        }

    def stream(self, prompt, system_prompt=None):
        yield ""


def test_chatbot_baseline_returns_text_only():
    chatbot = Chatbot(FakeChatLLM())
    response = chatbot.run("Tao mot file wav")
    assert "khong tao file wav" in response


def test_music_agent_creates_wav_file():
    agent = ReActAgent(
        FakeMusicAgentLLM(),
        MUSIC_TOOLS,
        max_steps=3,
        role_prompt=MUSIC_AGENT_PROMPT,
    )
    response = agent.run("Tao mot file wav calm")

    assert response == "outputs/test_music_agent.wav"
    assert os.path.exists("outputs/test_music_agent.mid")
    assert os.path.exists("outputs/test_music_agent.wav")


def test_music_agent_ignores_hallucinated_observation_and_final_answer():
    agent = ReActAgent(
        FakeHallucinatingAgentLLM(),
        MUSIC_TOOLS,
        max_steps=3,
        role_prompt=MUSIC_AGENT_PROMPT,
    )
    response = agent.run("Tao mot ban drill 1 bar")

    assert response == "outputs/hallucinated_path_test.wav"
    assert "/tmp/fake.wav" not in response
    assert os.path.exists("outputs/hallucinated_path_test.wav")
