"""cognia.models 模型路由层单元测试。

全部使用 fake LLM 替换 ChatDeepSeek，离线、快速、不依赖 DEEPSEEK_API_KEY。
"""

import pytest

from cognia import models


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
    """diagnoser 默认 deepseek-v4-flash + 低 temperature 锁死确定性。"""
    monkeypatch.delenv("DIAGNOSER_MODEL", raising=False)
    captured = _install_fake_llm(monkeypatch)
    models.get_diagnoser_model()
    assert captured["kwargs"]["model"] == "deepseek-v4-flash"
    assert captured["kwargs"]["temperature"] == 0.0


def test_env_var_overrides_model_id(monkeypatch):
    """环境变量可覆盖模型 ID（模型无关护栏，宪法 §7）。"""
    monkeypatch.setenv("PLANNER_MODEL", "custom-flash")
    captured = _install_fake_llm(monkeypatch)
    models.get_planner_model()
    assert captured["kwargs"]["model"] == "custom-flash"
