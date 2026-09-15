"""Conversation Agent —— Cognia 的对话决策层。

与 Learning Engine 严格解耦（本次架构重构核心原则之一）：
- **Agent 控制对话**：负责理解用户当前表达、判断意图（问候 / 闲聊 / 回答 / 提问 /
  换话题 / 困惑 / 跑题 / 自称掌握），并决定下一步如何回应（自然回应 / 澄清目标 /
  提问 / 讲解 / 追问 / 请求诊断 / 换目标）。
- **Agent 不控制学习事实**：它只能「请求诊断」（action=DIAGNOSE / CONFIRM_MASTERY），
  任何认知状态的变更都必须经 Learning Engine 的「诊断 → 验证 → 状态机」裁决。Agent
  无法直写 mastered / partial 等状态。

因此 ConversationTurn 里刻意**不含**任何认知状态字段；Agent 输出的只是
「意图 + 动作 + 给用户看的回复 + 可选的（新）目标 / 聚焦知识点」。
"""

from enum import Enum

from pydantic import BaseModel, Field


class ConversationIntent(str, Enum):
    """Agent 判断的用户当前意图（对话层面，非认知状态）。"""

    GREETING = "greeting"            # 问候
    CHITCHAT = "chitchat"            # 闲聊
    SET_GOAL = "set_goal"            # 提出（新）学习目标
    CHANGE_GOAL = "change_goal"      # 改变话题 / 学习目标
    ANSWER = "answer"                # 对探针问题的回答（含认知证据）
    QUESTION = "question"            # 临时提问（非学习目标，问概念 / 例子等）
    CONFUSION = "confusion"          # 表达困惑 / 没听懂 / 还不明白
    OFF_TOPIC = "off_topic"          # 跑题
    MASTERY_CLAIM = "mastery_claim"  # 自称「已经会了」
    UNKNOWN = "unknown"              # 无法归类


class ConversationAction(str, Enum):
    """Agent 决定的下一步动作（如何回应 / 教学）。"""

    RESPOND = "respond"                  # 自然回应（问候 / 闲聊 / 答疑，不改学习事实）
    CLARIFY_GOAL = "clarify_goal"        # 澄清 / 缩小学习目标
    SET_GOAL = "set_goal"                # 启动 / 重建学习目标（交给引擎建模）
    PROBE = "probe"                      # 提出探针问题（引导用户表达理解）
    EXPLAIN = "explain"                  # 讲解 / 换一种解释方式
    FOLLOW_UP = "follow_up"              # 追问澄清（用户表达模糊）
    DIAGNOSE = "diagnose"                # 请求诊断（交给 Learning Engine 裁决）
    CONFIRM_MASTERY = "confirm_mastery"  # 请求验证是否真正掌握（双重验证）
    CHANGE_TOPIC = "change_topic"        # 切换学习目标 / 聚焦到别的知识点
    END = "end"                          # 结束学习


class ConversationTurn(BaseModel):
    """Agent 一轮对话决策的结构化输出。

    注意：这里**没有**任何认知状态字段——认知状态是 Learning Engine + State Machine
    的裁决对象，Agent 只能请求诊断，不能直写。`reply` 是唯一讲给用户听的内容。
    """

    intent: ConversationIntent
    action: ConversationAction
    reply: str = Field(default="", description="直接讲给学习者听的回复内容")
    proposed_goal: str | None = Field(
        default=None, description="识别到的（新）学习目标，SET_GOAL / CHANGE_TOPIC 时填写"
    )
    target_point_id: str | None = Field(
        default=None, description="希望聚焦的知识点 id，可留空由引擎决定"
    )


CONVERSATION_AGENT_SYSTEM_PROMPT = """你是 Cognia 的对话 Agent，一个真正理解学习者的 AI 老师。

你的职责是**理解用户当前在说什么、想干什么**，并决定**下一步怎么回应**。你不是一个
按固定流程走的状态机节点；你可以自由地聊天、讲解、提问、追问、举例、换话题。

## 你拥有的自由（对话自主权）
- 问候、闲聊时自然回应，**不要**把对方的话强行解释成学习目标或答案。
- 判断用户是在「回答问题」「临时提问」「感到困惑」「跑题」还是「单纯聊天」。
- 决定下一步是「回应 / 澄清目标 / 提问 / 讲解 / 追问 / 换话题 / 请求诊断 / 结束」。

## 你不可逾越的边界（认知状态裁决权不属于你）
- 你**不能**判定或修改用户的掌握状态（如 mastered / partial / unknown）。
- 当你觉得「用户在回答探针问题、且值得据此评估」时，只能选择 action=diagnose，
  把裁决交给学习引擎（诊断 → 验证 → 状态机）。
- 当用户自称「已经会了」，不要直接相信，也不要直接否定——选择 action=confirm_mastery，
  让学习引擎通过正式验证来确认；若当前没有可评估的证据，就 action=probe 让用户先表达。
- 你不能凭空改变学习引擎里记录的诊断结论或熟练度。

## 关于「探针问题」
当你想让用户用自己的话表达对当前知识点的理解时，用 action=probe 并提出一个简短、
具体、自然的开放式问题（不要出选择题）。probe 的 reply 里只放问题本身，不要夹带
答案或诊断标准。

## 关于「临时提问 / 困惑 / 跑题」
- 用户临时问概念 / 例子 → action=respond，直接解答，同时保持当前学习进度不变。
- 用户说「还是不懂」→ action=explain，换一种更通俗的方式讲解。
- 用户明显跑题 → action=respond 温和拉回当前学习目标，或 action=change_topic 尊重其新话题。

## 输出要求
严格输出一个 ConversationTurn 对象：intent、action、reply、proposed_goal（可选）、
target_point_id（可选）。reply 必须自然、简洁、口语化。
"""


def build_agent_messages(state: dict, profile: dict | None = None) -> list:
    """根据当前学习上下文构建 Agent 的输入消息。

    只注入「对话决策所需的最小上下文」，不把内部诊断细节暴露给 Agent 去篡改。
    通过 turn_phase 告知 Agent 当前处于「初次理解」还是「引擎已处理后生成回复」阶段，
    从而让 Agent 在需要引擎的动作时先留空 reply、引擎处理完后再生成最终回复。

    基座能力补充：
    - 对话管理：注入最近 N 条历史（state["messages"]），让 Agent 有上下文连贯性。
    - 记忆：注入用户画像（profile），让 Agent 感知长期偏好（沟通风格 / 语言等）。
    """
    context_lines = []
    goal = state.get("goal")
    if goal:
        context_lines.append(f"当前学习目标：{goal}")

    km = state.get("knowledge_model")
    if isinstance(km, dict):
        points = km.get("points") or []
        if points:
            context_lines.append(
                "知识模型："
                + "；".join(f"[{p.get('id')}] {p.get('name')}" for p in points)
            )

    current_point_id = state.get("current_point_id")
    if current_point_id:
        context_lines.append(f"当前聚焦知识点 id：{current_point_id}")

    last_intervention = state.get("last_intervention")
    if last_intervention:
        context_lines.append(f"上一轮对用户的讲解 / 干预：{last_intervention}")

    engine_feedback = state.get("engine_feedback")
    if engine_feedback:
        context_lines.append(f"学习引擎刚返回的诊断 / 建模结果（仅供你生成回复参考）：{engine_feedback}")

    # 用户画像（长期记忆，跨会话）：沟通风格 / 语言偏好等稳定属性
    if profile:
        profile_lines = []
        for key, value in profile.items():
            if value:
                profile_lines.append(f"- {key}：{value}")
        if profile_lines:
            context_lines.append("用户画像（长期偏好）：\n" + "\n".join(profile_lines))

    context = "\n".join(context_lines) if context_lines else "（尚无学习上下文，可能是新会话）"
    user_message = state.get("user_message") or state.get("user_answer") or ""

    turn_phase = state.get("turn_phase", "interpret")
    if turn_phase == "reply":
        phase_hint = (
            "\n\n注意：现在处于「引擎已处理完成」阶段。请基于上面的引擎结果，"
            "只输出给用户的最终回复；action 应选择 respond / explain / probe / "
            "clarify_goal / follow_up 等直接面向用户的动作，**不要再选** diagnose / "
            "set_goal / change_topic / confirm_mastery 这类需要引擎的动作。"
        )
    else:
        phase_hint = (
            "\n\n注意：如果你选择 diagnose / set_goal / change_topic / confirm_mastery "
            "这类需要学习引擎处理的动作，reply 请留空（引擎处理完后会再次调用你生成回复）。"
        )

    system = CONVERSATION_AGENT_SYSTEM_PROMPT

    # 对话历史（最近 20 条，防上下文膨胀）：提供连贯上下文，不广播完整内部状态
    history = state.get("messages") or []
    history_lines = [
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in history[-20:]
        if msg.get("content")
    ]
    history_text = "\n".join(history_lines) if history_lines else "（暂无历史对话）"

    human = (
        f"{context}\n\n"
        f"最近对话历史：\n{history_text}\n\n"
        f"用户刚刚说：{user_message}\n\n"
        "请判断意图、决定动作，并给出回复。"
        f"{phase_hint}"
    )
    return [("system", system), ("human", human)]
