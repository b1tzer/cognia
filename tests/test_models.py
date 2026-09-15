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


# ---- JSON 提取容错（LLM 输出常见语法错误）----

def test_extract_json_trailing_comma():
    """尾逗号 `[...,]` 应被自动修复后正确解析。"""
    obj = models._extract_json('[{"id": "a", "name": "A",},]')
    assert obj == [{"id": "a", "name": "A"}]


def test_extract_json_missing_comma_between_objects():
    """相邻对象漏写逗号（"Expecting ',' delimiter"）应被自动补齐。"""
    obj = models._extract_json('[{"id": "a"} {"id": "b"}]')
    assert obj == [{"id": "a"}, {"id": "b"}]


def test_extract_json_preserves_comma_inside_string():
    """字符串内容里的 `,}` / `}` 不应被误判为结构字符而破坏。"""
    obj = models._extract_json('[{"id": "a", "name": "x,}y"}]')
    assert obj == [{"id": "a", "name": "x,}y"}]


# ---- 知识模型构建重试 ----

class _Msg:
    def __init__(self, content):
        self.content = content


class _ScriptedPlanner:
    """按队列返回预设 content 的假 planner，记录调用次数。"""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        assert self._contents, "planner 响应队列耗尽"
        return _Msg(self._contents.pop(0))


def test_build_knowledge_model_retries_on_invalid_json():
    """首次输出无法解析为 JSON 时回喂错误重试一次，避免终止整个 agent 运行。"""
    from cognia.learning_engine import build_knowledge_model

    planner = _ScriptedPlanner([
        '[{"id": "a", "name": "A",',  # 截断、无法修复
        '[{"id": "a", "name": "A", "description": "d", "prerequisites": []}]',
    ])
    km = build_knowledge_model(planner, "Spring AOP")
    assert planner.calls == 2
    assert km.goal == "Spring AOP"
    assert [p.id for p in km.points] == ["a"]
