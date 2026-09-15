"""cognia.conversation_agent 单元测试。

聚焦基座能力：
1. 对话管理：build_agent_messages 注入最近对话历史（上下文连贯性）。
2. 记忆：build_agent_messages 注入用户画像（长期偏好）。
"""

from cognia.conversation_agent import build_agent_messages


def test_build_agent_messages_injects_history():
    """最近对话历史被注入 human 消息，Agent 具备上下文连贯性。"""
    state = {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，想学点什么？"},
        ],
        "user_message": "我想学 Spring AOP",
    }
    msgs = build_agent_messages(state)
    assert msgs[0][0] == "system"
    assert msgs[1][0] == "human"

    human = msgs[1][1]
    assert "你好" in human          # 历史 user 内容
    assert "想学点什么" in human     # 历史 assistant 内容
    assert "我想学 Spring AOP" in human  # 当前输入


def test_build_agent_messages_injects_profile():
    """用户画像被注入 human 消息，Agent 感知长期偏好。"""
    state = {"user_message": "hi"}
    profile = {"language": "zh", "communication_style": "简洁"}
    msgs = build_agent_messages(state, profile)

    human = msgs[1][1]
    assert "用户画像" in human
    assert "language" in human
    assert "zh" in human
    assert "communication_style" in human
    assert "简洁" in human


def test_build_agent_messages_without_profile_no_section():
    """未提供画像时不注入画像区块（不产生空标题）。"""
    msgs = build_agent_messages({"user_message": "hi"})
    assert "用户画像" not in msgs[1][1]
