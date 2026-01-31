import asyncio

from core.models import LeftBrainModel


class DummyLlmClient:
    async def complete(self, prompt: str, *, system=None, temperature: float = 0.7) -> str:
        assert "Internal collaborator notes" in prompt
        assert system is not None and "Do not mention internal orchestration" in system
        return "やあ！元気？"


def test_integrate_info_async_hides_notes():
    model = LeftBrainModel()
    model._llm_client = DummyLlmClient()

    out = asyncio.run(
        model.integrate_info_async(
            question="やあ",
            draft="こんにちは！どうしましたか？",
            info="- Add a warm, informal tone (e.g., '元気？').",
            temperature=0.25,
        )
    )
    assert out == "やあ！元気？"
