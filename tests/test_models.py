"""cognia.models 模型路由层单元测试。

全部使用 fake LLM 替换 ChatDeepSeek，离线、快速、不依赖 DEEPSEEK_API_KEY。
"""

import pytest

from cognia import models
from cognia.schemas import CognitiveState, Confidence, Diagnosis


def _install_fake_llm(monkeypatch):
    """替换 models.ChatDeepSeek 为记录构造参数的 fake，返回捕获容器。"""
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def with_structured_output(self, schema):
            captured["schema"] = schema
            return self

    monkeypatch.setattr(models, "ChatDeepSeek", FakeLLM)
    return captured


def test_planner_model_default(monkeypatch):
    """planner 默认 deepseek-v4-flash，temperature 对齐角色策略。"""
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    captured = _install_fake_llm(monkeypatch)
    models.get_planner_model()
    assert captured["kwargs"]["model"] == "deepseek-v4-flash"
    assert captured["kwargs"]["temperature"] == models.PLANNER_TEMPERATURE


def test_teacher_model_default(monkeypatch):
    """teacher 默认 deepseek-v4-flash。"""
    monkeypatch.delenv("TEACHER_MODEL", raising=False)
    captured = _install_fake_llm(monkeypatch)
    models.get_teacher_model()
    assert captured["kwargs"]["model"] == "deepseek-v4-flash"


def test_diagnoser_model_low_temperature(monkeypatch):
    """diagnoser 默认 deepseek-v4-pro + 低 temperature 锁死确定性。"""
    monkeypatch.delenv("DIAGNOSER_MODEL", raising=False)
    captured = _install_fake_llm(monkeypatch)
    models.get_diagnoser_model()
    assert captured["kwargs"]["model"] == "deepseek-v4-pro"
    assert captured["kwargs"]["temperature"] == 0.0


def test_env_var_overrides_model_id(monkeypatch):
    """环境变量可覆盖模型 ID（模型无关护栏，宪法 §7）。"""
    monkeypatch.setenv("PLANNER_MODEL", "custom-flash")
    captured = _install_fake_llm(monkeypatch)
    models.get_planner_model()
    assert captured["kwargs"]["model"] == "custom-flash"


def test_structured_diagnoser_binds_diagnosis(monkeypatch):
    """结构化诊断模型必须绑定 Diagnosis schema。"""
    captured = _install_fake_llm(monkeypatch)
    models.get_structured_diagnoser()
    assert captured["schema"] is Diagnosis


def test_structured_diagnoser_mock_invoke(monkeypatch):
    """模拟一次结构化诊断调用，验证完整链路（输入消息 → 输出 Diagnosis 实例）。"""
    class FakeStructuredLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            self._schema = schema
            return self

        def invoke(self, _messages):
            return Diagnosis(
                point_id="aop-proxy",
                state=CognitiveState.PARTIAL,
                confidence=Confidence.MEDIUM,
                evidence=["用户说 AOP 就是切面，但说不清代理机制"],
            )

    monkeypatch.setattr(models, "ChatDeepSeek", FakeStructuredLLM)
    structured = models.get_structured_diagnoser()
    result = structured.invoke([("human", "AOP 就是切面")])

    assert isinstance(result, Diagnosis)
    assert result.point_id == "aop-proxy"
    assert result.state == CognitiveState.PARTIAL
    assert result.confidence == Confidence.MEDIUM
    assert result.evidence == ["用户说 AOP 就是切面，但说不清代理机制"]
