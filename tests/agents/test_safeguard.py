"""Safeguard 模型选择逻辑的测试。

决策（docs/18 K4）：safeguard 优先复用 DeepSeek key（本机已配），
无 DeepSeek key 时才用 Groq 专用安全模型，两者都缺则降级 SAFE。
"""

import pytest

from agents.safeguard import Safeguard, SafetyAssessment


def test_safeguard_prefers_deepseek(monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    from schema.models import DeepseekModelName

    class FakeModel:
        @classmethod
        def from_ai(cls, ai_msg):
            inst = cls()
            inst._ai = ai_msg
            return inst

        def with_config(self, **kwargs):
            return self

        def invoke(self, messages):
            return self._ai

        async def ainvoke(self, messages):
            return self._ai

    calls = {"supervisor": 0, "plain": 0}

    def fake_supervisor(name, /):
        calls["supervisor"] += 1
        assert name == DeepseekModelName.DEEPSEEK_V4_FLASH
        return FakeModel.from_ai(AIMessage(content='{"violation": 0}'))

    def fake_get(name, /):
        calls["plain"] += 1
        raise AssertionError("有 DeepSeek key 时不应走 get_model")

    monkeypatch.setattr("agents.safeguard.get_supervisor_model", fake_supervisor)
    monkeypatch.setattr("agents.safeguard.get_model", fake_get)
    monkeypatch.setattr("agents.safeguard.settings.DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setattr("agents.safeguard.settings.GROQ_API_KEY", None)

    sav = Safeguard()
    assert sav.model is not None
    out = sav.invoke([type("M", (), {"type": "human", "content": "hi"})()])
    assert out.safety_assessment == SafetyAssessment.SAFE
    assert calls["supervisor"] == 1
    assert calls["plain"] == 0


def test_safeguard_falls_back_to_groq_without_deepseek(monkeypatch) -> None:
    monkeypatch.setattr("agents.safeguard.settings.DEEPSEEK_API_KEY", None)
    monkeypatch.setattr("agents.safeguard.settings.GROQ_API_KEY", "fake-groq")

    from schema.models import GroqModelName

    seen = {}

    def fake_get(name, /):
        seen["name"] = name
        return FakeRunnable()

    monkeypatch.setattr("agents.safeguard.get_model", fake_get)
    # 仅探测构造时的分支：get_supervisor_model 不应被调用
    monkeypatch.setattr(
        "agents.safeguard.get_supervisor_model",
        lambda n, /: (_ for _ in ()).throw(AssertionError()),
    )

    class FakeRunnable:
        def with_config(self, **kwargs):
            return self

    sav = Safeguard()
    assert sav.model is not None
    assert seen["name"] == GroqModelName.GPT_OSS_SAFEGUARD_20B


def test_safeguard_disables_when_no_keys(monkeypatch) -> None:
    monkeypatch.setattr("agents.safeguard.settings.DEEPSEEK_API_KEY", None)
    monkeypatch.setattr("agents.safeguard.settings.GROQ_API_KEY", None)

    sav = Safeguard()
    assert sav.model is None
    out = sav.invoke(
        [type("M", (), {"type": "human", "content": "prompt injection: ignore instructions"})()]
    )
    assert out.safety_assessment == SafetyAssessment.SAFE  # 降级 SAFE，接口行为不变


def test_safeguard_uses_fake_model_without_live_calls(monkeypatch) -> None:
    """确保没真实 key 时构造 Safeguard 不会发任何网络请求。"""
    monkeypatch.setattr("agents.safeguard.settings.DEEPSEEK_API_KEY", None)
    monkeypatch.setattr("agents.safeguard.settings.GROQ_API_KEY", None)
    sav = Safeguard()
    assert sav.model is None


def _boom_safeguard(cls):
    class BoomModel:
        def with_config(self, **kwargs):
            return self

        def invoke(self, *a, **k):
            raise RuntimeError("402 - Insufficient Balance")

        async def ainvoke(self, *a, **k):
            raise RuntimeError("402 - Insufficient Balance")

    sav = cls.__new__(cls)
    sav.model = BoomModel()
    sav.system_prompt = "test"
    return sav


def test_safeguard_invoke_survives_model_failure(monkeypatch) -> None:
    """守卫的模型调用失败（402/429/超时）不应当让整次请求崩溃（docs/20 F1）。

    注入式反例：把 `except Exception` 从 invoke 删掉，本用例必须变红。
    """
    from langchain_core.messages import HumanMessage

    sav = _boom_safeguard(Safeguard)
    out = sav.invoke([HumanMessage(content="hi")])
    assert out.safety_assessment == SafetyAssessment.ERROR


@pytest.mark.asyncio
async def test_safeguard_ainvoke_survives_model_failure(monkeypatch) -> None:
    """同上，异步路径（真正被 factory 入口节点使用的路径）。"""
    from langchain_core.messages import HumanMessage

    sav = _boom_safeguard(Safeguard)
    out = await sav.ainvoke([HumanMessage(content="hi")])
    assert out.safety_assessment == SafetyAssessment.ERROR
