"""Learning Engine —— Cognia 的学习事实层。

职责（与 Conversation Agent 解耦，本次架构重构核心原则之一）：
- 知识模型构建：把学习目标拆解为知识点 + 依赖关系（build_knowledge_model）。
- 认知诊断：基于用户表达判定五态 + 置信度 + 证据（run_diagnosis）。

诊断（Diagnosis）只产出「候选观察值」，权威认知状态由系统用 BKT 算法融合
观察历史后算出（见 proficiency_engine / memory.query_proficiency），AI 无权
直接改写长期状态。

本模块**不负责「怎么和用户交流」**（那是对话 Agent 的职责），只负责生成
结构化的诊断候选与知识模型。
"""

from cognia import models
from cognia.schemas import Diagnosis, KnowledgeModel, KnowledgePoint
from cognia.prompts.learning_engine import (
    DIAGNOSER_SYSTEM_PROMPT,
    KNOWLEDGE_MODELER_SYSTEM_PROMPT,
)


# ---- 知识模型构建 ----

def build_knowledge_model(planner, goal: str, config=None) -> KnowledgeModel:
    """构建知识模型：普通文本调用 + 扁平 JSON 数组解析。

    背景：adapter 上游工蜂 Gateway 对带嵌套对象引用（$defs/$ref）的 JSON Schema 的
    function calling 支持有缺陷，KnowledgeModel 改用「普通 invoke + 输出扁平 JSON
    数组 + 手动解析」。

    兼容测试桩：ScriptedLLM 的 invoke 直接返回 KnowledgeModel 对象（而非字符串），
    isinstance 命中后原样返回。
    """
    messages = [
        ("system", KNOWLEDGE_MODELER_SYSTEM_PROMPT),
        ("human", (
            f"学习目标：{goal}\n\n"
            "请只输出一个 JSON 数组，每个元素是一个知识点对象，格式如下"
            "（不要输出任何解释、注释或 markdown 代码块）：\n"
            '[{"id": "唯一标识", "name": "知识点名称", "description": "一句话描述", '
            '"prerequisites": ["依赖的知识点id"]}]'
        )),
    ]
    result = planner.invoke(messages, config)
    if isinstance(result, KnowledgeModel):
        return result
    text = result.content if hasattr(result, "content") else str(result)
    try:
        data = models._extract_json(text)
    except ValueError as exc:
        # 解析失败（截断 / 语法错误等）：把错误回喂给模型，让其修正后重试一次，
        # 避免一次坏 JSON 直接让 LangGraph 工具节点抛错、终止整个 agent 运行。
        retry_result = planner.invoke([
            *messages,
            ("human", (
                f"你上一次的输出无法解析为合法 JSON，错误信息：{exc}\n"
                "请重新只输出一个合法 JSON 数组，不要输出任何解释、注释或 markdown 代码块。"
            )),
        ], config)
        if isinstance(retry_result, KnowledgeModel):
            return retry_result
        retry_text = retry_result.content if hasattr(retry_result, "content") else str(retry_result)
        data = models._extract_json(retry_text)
    points = [KnowledgePoint.model_validate(p) for p in data]
    return KnowledgeModel(goal=goal, points=points)


# ---- 诊断（LLM 判定，产出候选观察值）----

def run_diagnosis(diagnoser, point: KnowledgePoint, question: str, user_answer: str, config=None) -> Diagnosis:
    """基于用户表达诊断五态 + 置信度 + 证据（独立严格 prompt）。

    产出的是「候选观察值」，仅作为 Observation 提交；最终权威状态由系统 BKT
    算法融合观察历史后算出（memory.query_proficiency），AI 无权直接改写。
    """
    diagnosis = models.structured_output(diagnoser, Diagnosis, [
        ("system", DIAGNOSER_SYSTEM_PROMPT),
        ("human", (
            f"知识点：{point.name}（{point.description}）\n"
            f"探针问题：{question}\n"
            f"用户回答：{user_answer}\n\n请诊断。"
        )),
    ], config=config)
    diagnosis.point_id = point.id
    return diagnosis
