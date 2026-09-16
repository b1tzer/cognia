"""Cognia 认知模型工具层（Skills）。

把认知模型的核心操作封装为 LangChain `@tool`，回归宪法 §1「有副作用的操作只允许
放在 tool 节点」。工具分三类：

1. **读**（无副作用，Agent 可自由调用）：读熟练度、读知识模型。
2. **写**（有副作用，内部强制走三层闸门）：`propose_diagnosis` —— LLM 只能「提议
   诊断」，最终状态迁移由「诊断 → 双重验证 → 状态机」在工具内部裁决，Agent 无法旁路。
3. **教学**（生成型，无长期副作用）：生成探针问题、生成讲解。

工具通过工厂 `build_cognia_tools()` 构建，依赖（diagnoser / planner / teacher / store）
以闭包注入，工具签名只暴露 LLM 能填写的简单参数（str / list / dict）。

安全边界（不可破坏）：
- `propose_diagnosis` 内部完整复用 `run_diagnosis` → `run_verification` →
  `resolve_migration` 三层闸门；中 / 低置信度一律不迁移；mastered 必须双重验证全过。
- 产生的 Proficiency Delta 直接增量写入 store（副作用隔离在 tool），幂等
  （key = point_id:timestamp）。
"""

import json
import os
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from cognia.learning_engine import (
    build_knowledge_model,
    resolve_migration,
    run_diagnosis,
    run_verification,
)
from cognia.memory import (
    append_proficiency_delta,
    get_current_proficiency,
    get_knowledge_model,
    normalize_goal,
    put_knowledge_model,
)
from cognia.prompts.teacher import (
    EXPLAINER_SYSTEM_PROMPT,
    PROBE_GENERATOR_SYSTEM_PROMPT,
)
from cognia.schemas import (
    CognitiveState,
    Confidence,
    KnowledgeModel,
    KnowledgePoint,
)
from cognia.state_machine import is_mastered_migration_allowed


def _model_to_lines(km: KnowledgeModel) -> list[str]:
    """知识模型 → 便于 LLM 阅读的单行描述列表。"""
    return [f"[{p.id}] {p.name}（{p.description}）" for p in km.points]

def _search_web(query: str, max_results: int = 5) -> list[dict]:
    """调用本地 SearXNG 元搜索引擎，返回 [{title, url, snippet}] 列表。

    这是 web_search 工具与 explain 工具共用的底层检索实现；任何联网搜索都
    汇聚到这里，保证检索逻辑唯一、可单点加固（超时、限流、域名过滤等）。
    """
    max_results = max(1, min(int(max_results), 10))
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    req = urllib.request.Request(
        f"{searxng_url}/search?{params}",
        headers={"User-Agent": "cognia-agent"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        # 检索失败不阻断讲解，返回空列表，由上层决定降级策略。
        return []

    results = []
    for item in (data.get("results") or [])[:max_results]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        if not title and not url:
            continue
        results.append({
            "title": title,
            "url": url,
            "snippet": snippet[:200],
        })
    return results

def build_cognia_tools(diagnoser=None, planner=None, teacher=None, store=None, user_id=None):
    """构建认知模型工具集（闭包注入模型与存储依赖）。

    - diagnoser：诊断 + 双重验证（run_diagnosis / run_verification）
    - planner：知识模型构建
    - teacher：探针问题 / 讲解生成
    - store：长期 Store（熟练度 / 知识模型持久化），None 时跳过持久化（测试 / 纯推理）
    - user_id：当前用户标识，**由工厂闭包注入，不暴露给 LLM**（宪法 §5：user_id 来自
      runtime context，LLM 只应填写业务参数，不能伪造身份）。

    模型为**惰性初始化**：只在对应工具真正被调用时才获取，避免构造工具集（含纯读
    工具如 read_learner_state）时强制要求 LLM API 凭据，也便于测试只注入需要的假模型。
    """

    def _get_diagnoser():
        nonlocal diagnoser
        if diagnoser is None:
            from cognia import models as _models
            diagnoser = _models.get_diagnoser_model()
        return diagnoser

    def _get_planner():
        nonlocal planner
        if planner is None:
            from cognia import models as _models
            planner = _models.get_planner_model()
        return planner

    def _get_teacher():
        nonlocal teacher
        if teacher is None:
            from cognia import models as _models
            teacher = _models.get_teacher_model()
        return teacher

    @tool
    def read_learner_state(point_id: str) -> str:
        """读取当前用户对某知识点的认知状态（五态之一）。

        Args:
            point_id: 知识点唯一 id。
        """
        state = (
            get_current_proficiency(store, user_id, point_id)
            if (store and user_id)
            else None
        )
        # 返回 JSON 结构（而非纯文本），便于 CopilotKit Inspector 解析工具结果展示
        return json.dumps({"state": state or "unassessed"}, ensure_ascii=False)

    @tool
    def build_learning_goal(goal: str) -> str:
        """为当前用户构建 / 复用某学习目标的知识模型（知识点列表 + 前置依赖）。

        load-or-build：同一 (user_id, goal) 会复用已冻结的知识模型，保证知识点 id
        跨会话稳定。

        Args:
            goal: 学习目标（如「Spring AOP」）。
        """
        goal = (goal or "").strip()
        goal_key = normalize_goal(goal)

        km_dict = get_knowledge_model(store, user_id, goal_key) if (store and user_id) else None
        if km_dict is not None:
            km = KnowledgeModel.model_validate(km_dict)
            verb = "复用"
        else:
            km = build_knowledge_model(_get_planner(), goal)
            km.goal = goal
            if store and user_id:
                put_knowledge_model(store, user_id, goal_key, km.model_dump(mode="json"))
            verb = "构建"

        lines = _model_to_lines(km)
        return json.dumps({
            "action": verb,
            "goal": km.goal,
            "point_count": len(km.points),
            "points": lines,
            "first_point_id": km.points[0].id if km.points else None,
        }, ensure_ascii=False)

    @tool
    def propose_diagnosis(
        point_id: str,
        point_name: str,
        point_description: str,
        question: str,
        user_answer: str,
        current_state: str,
    ) -> str:
        """提议对用户回答做一次认知诊断，并让系统裁决是否发生状态迁移。

        重要：本工具只「提议」，内部会强制经过「诊断 → 双重验证 → 状态机」三层
        闸门裁决，不会因为本工具被调用就无条件改变认知状态。中 / 低置信度不迁移；
        判 mastered 必须概念 + 场景双重验证全过。

        Args:
            point_id: 知识点唯一 id。
            point_name: 知识点名称。
            point_description: 知识点一句话描述。
            question: 最近一次向用户提出的探针问题。
            user_answer: 用户对探针问题的回答原文。
            current_state: 该知识点当前五态（unassessed/unknown/partial/misconception/mastered）。
        """
        point = KnowledgePoint(
            id=point_id,
            name=point_name,
            description=point_description,
        )
        diagnosis = run_diagnosis(_get_diagnoser(), point, question, user_answer)

        # 当前长期状态：unassessed 视为 None（首次诊断）
        from_state = None
        if current_state and current_state != CognitiveState.UNASSESSED.value:
            from_state = CognitiveState(current_state)

        verification = None
        migrated = False
        final_state = current_state or CognitiveState.UNASSESSED.value

        if diagnosis.confidence == Confidence.HIGH:
            if diagnosis.state == CognitiveState.MASTERED:
                # mastered 候选：必须双重验证
                verification = run_verification(_get_diagnoser(), point, user_answer, question, diagnosis)
                if is_mastered_migration_allowed(verification):
                    entry = resolve_migration(diagnosis, from_state, verification)
                    if entry is not None:
                        if store and user_id:
                            append_proficiency_delta(store, user_id, entry)
                        migrated = True
                        final_state = entry.to_state.value
                else:
                    # 验证失败：诊断降级 partial，不迁移
                    diagnosis = diagnosis.model_copy(update={"state": CognitiveState.PARTIAL})
                    final_state = current_state or CognitiveState.UNASSESSED.value
            else:
                # 非 mastered 候选：直接走状态机裁决（无需双重验证）
                entry = resolve_migration(diagnosis, from_state, None)
                if entry is not None:
                    if store and user_id:
                        append_proficiency_delta(store, user_id, entry)
                    migrated = True
                    final_state = entry.to_state.value
        # 中 / 低置信度：不迁移，保持原状态

        result = {
            "diagnosed_state": diagnosis.state.value,
            "confidence": diagnosis.confidence.value,
            "evidence": diagnosis.evidence,
            "migrated": migrated,
            "final_state": final_state,
            "verification": (
                {
                    "concept": verification.concept.value,
                    "scenario": verification.scenario.value,
                }
                if verification is not None
                else None
            ),
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def generate_probe(point_name: str, point_description: str) -> str:
        """生成一个针对某知识点的开放式探针问题，引导用户用自己的话表达理解。

        Args:
            point_name: 知识点名称。
            point_description: 知识点一句话描述。
        """
        result = _get_teacher().invoke([
            ("system", PROBE_GENERATOR_SYSTEM_PROMPT),
            ("human", f"知识点：{point_name}（{point_description}）"),
        ])
        content = result.content if hasattr(result, "content") else str(result)
        return content.strip()

    @tool
    def explain(point_name: str, point_description: str, user_state: str) -> str:
        """针对用户当前认知状态，用通俗方式讲解一个知识点。

        本工具会**先强制联网检索官方 / 权威资料**，再基于检索结果讲解，
        确保技术事实准确，并在结尾附「参考来源」链接。

        Args:
            point_name: 知识点名称。
            point_description: 知识点一句话描述。
            user_state: 用户当前认知状态（partial 用引导式；misconception 用颠覆式；unknown 从零建立）。
        """
        style_hint = {
            CognitiveState.PARTIAL.value: "引导式：渐进补全细节，追问澄清边界",
            CognitiveState.MISCONCEPTION.value: "颠覆式：先指出认知冲突，再重建正确模型",
            CognitiveState.UNKNOWN.value: "从零建立：先用生活化类比建立直觉",
        }.get(user_state, "自然讲解")

        # 强制检索：用「知识点 + 官方文档」引导优先命中官方 / 权威站点。
        search_results = _search_web(f"{point_name} 官方文档 official docs", max_results=5)
        refs = "\n".join(
            f"- {r['title']}（{r['url']}）" for r in search_results
        )
        if not refs:
            refs = "（本次未检索到外部资料，请仅基于已确定的技术事实谨慎讲解，不确定处明确说明。）"

        result = _get_teacher().invoke([
            ("system", EXPLAINER_SYSTEM_PROMPT),
            ("human", (
                f"知识点：{point_name}（{point_description}）\n"
                f"教学策略：{style_hint}\n\n"
                f"检索资料：\n{refs}"
            )),
        ])
        content = result.content if hasattr(result, "content") else str(result)
        return content.strip()

    @tool
    def web_search(query: str, max_results: int = 8) -> str:
        """通过本地 SearXNG 元搜索引擎联网搜索，返回相关网页的标题、链接与摘要。

        当需要获取最新信息、事实核查、外部资料，或教学讲解需要补充实时内容时使用。

        Args:
            query: 搜索查询词（支持中文或英文）。
            max_results: 返回结果数量上限（默认 8，范围 1-10）。
        """
        results = _search_web(query, max_results)
        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False)

    return {
        "read_learner_state": read_learner_state,
        "build_learning_goal": build_learning_goal,
        "propose_diagnosis": propose_diagnosis,
        "generate_probe": generate_probe,
        "explain": explain,
        "web_search": web_search,
    }
