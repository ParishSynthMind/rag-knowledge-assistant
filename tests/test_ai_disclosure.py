"""EU AI Act Art. 50: the assistant's reply opens by saying it is an AI. Offline: no model, no key."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402

DEFAULT = "I'm an AI assistant for Orbitly."
CHUNKS = [{"score": 0.9, "source": "pricing.md", "heading": "Plans", "text": "Pro is $12."}]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AI_DISCLOSURE", "AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST", "BUSINESS_NAME"):
        monkeypatch.delenv(name, raising=False)


def run_ask(question="How much is Pro?"):
    lines = []
    rag.ask(question, out=lines.append, retrieve_fn=lambda q: CHUNKS,
            generate_fn=lambda q, c: "Pro costs $12 a month [pricing.md].")
    return "\n".join(lines)


def test_reply_opens_with_the_disclosure_once():
    text = run_ask()
    reply = text.split("Answer:\n", 1)[1]
    assert reply.startswith(DEFAULT) and "Pro costs $12" in reply
    assert text.count(DEFAULT) == 1


def test_every_new_conversation_discloses_again():
    assert run_ask().count(DEFAULT) == 1
    assert run_ask("Can I get a refund?").count(DEFAULT) == 1


def test_blank_text_without_written_request_keeps_the_default(monkeypatch):
    monkeypatch.setenv("AI_DISCLOSURE", "  ")
    assert DEFAULT in run_ask()
    monkeypatch.setenv("AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST", "   ")
    assert DEFAULT in run_ask()


def test_custom_text_and_business_name(monkeypatch):
    monkeypatch.setenv("AI_DISCLOSURE", "You are talking to an AI for {business}.")
    monkeypatch.setenv("BUSINESS_NAME", "Acme")
    assert "You are talking to an AI for Acme." in run_ask()


def test_removed_only_with_a_recorded_written_request(monkeypatch):
    monkeypatch.setenv("AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST", "2026-11-17, email from Acme ops")
    text = run_ask()
    assert "AI assistant" not in text and "Pro costs $12" in text


def test_system_prompt_says_ai():
    assert "AI support assistant" in rag.SYSTEM_PROMPT
