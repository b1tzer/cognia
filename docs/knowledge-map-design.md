# Cognia 知识版图子系统设计（Knowledge Map）

> 本文档定义「知识版图」这个**独立子系统**的能力边界、数据契约与关键决策。
> 它是 plan.md 中「认知模型」部分的**演进方向**：把「AI 诊断 = 最终结论」重构为
> 「AI 诊断 = 观察样本 → 系统算法融合 = 权威结论」。
> 本文档先冻结**能力方案**，代码落地另开任务，不在本文档内改代码。

## 0. 一句话定位

知识版图（Knowledge Map）是一个**独立于教学对话的子系统**，它拥有两类权威：

1. **知识结构**：知识点 + 上下级/依赖关系 + 属性（这是「知识本体」，与谁学无关）。
2. **熟练度**：每个用户对每个知识点的「最终理解程度」，由算法（BKT/贝叶斯）融合
   AI 提交的观察样本计算得出，AI 只能查询、无权直接改写。

教学对话 Agent 只是知识版图的**使用者**：通过 skills（工具）知道版图有哪些能力，
**具体何时调用、怎么组合，由 AI 自己判断**（Bound, Don't Script）。

## 1. 背景与动机

### 1.1 现有实现的问题（为什么要拆）

当前 `propose_diagnosis`（`cognia/tools.py`）把「AI 判断」当成了**最终结论**：

```
AI 诊断（五态 + 置信度）
   → 三层闸门过滤（诊断 → 双重验证 → 状态机）
   → 直接落库 ProficiencyEntry（AI 判断即最终认知状态）
```

这带来的问题：

- AI 单次诊断直接决定了用户长期认知状态，缺乏跨次观察的统计融合。
- 「一次还是两次才能定级」这条规则被硬塞在教学流程的 mastered 分支里，属于
  教学流程，不属于版图能力。
- 用户答得再多、AI 判得再准，只要没被「直接采纳」，证据就丢了（表现为
  `kafka_what` 那条 partial 被静默丢弃）——本质是「AI 判断被当结论却又不完整采纳」
  的架构错位。

### 1.2 目标架构

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│   教学对话 Agent（AI）        │        │   知识版图子系统（System）          │
│                             │        │                                  │
│  职责：语义判断              │  提交   │  职责：确定性事实 + 算法融合        │
│  · 判断"用户理解程度"         │ ─观察─▶ │  · 记录日志（不可变，只追加）       │
│    （这只是它的观察值）       │        │  · BKT/贝叶斯等算法处理            │
│  · 决定问什么/讲什么          │  查询   │  · 产出"最终理解"权威结果          │
│  · 管理知识结构（插点/关联）  │ ◀结果─  │                                  │
│                             │        │  不能：被 AI 直接改写结论           │
│  不能：直接写熟练度结论       │        │                                  │
└─────────────────────────────┘        └──────────────────────────────────┘
```

**核心一句话**：AI 只提交「观察值」（observation），系统把观察值当数据点，用算法
算出「权威状态」（latent proficiency），AI 只能查询这个结果、无权直接改写。

## 2. 能力清单（四个能力域）

### 能力域 A：知识结构管理（Knowledge Structure）

管理「知识本体」本身，与「谁学得怎么样」完全无关。知识结构是 DAG（有向无环图），
点 = 知识点，边 = 前置依赖（上下级关系）。

| 能力 | 说明 |
|------|------|
| `插入知识点` | 新增一个知识节点 |
| `更新知识点` | 改名称 / 描述 |
| `删除知识点` | 删节点，级联处理其依赖边 |
| `建立关联` | 定义前置依赖边（`A 依赖 B`，即 B 是 A 的前置） |
| `解除关联` | 删一条依赖边 |
| `登记属性` | 给节点挂元数据（见 §4.1 PointAttributes） |
| `查询结构` | 读整图 / 单点 / 其依赖（前置）/ 其被依赖（后继） |

### 能力域 B：观察记录（Observation Log）

AI 每判断一次「用户对某知识点的理解程度」，系统**只记日志，不直接改结论**。

| 能力 | 说明 |
|------|------|
| `记录观察值` | 追加一条**不可变**日志：`point_id + AI 判定五态 + 置信度 + 证据(用户原话) + 时间戳` |
| `查询观察历史` | 按 point 读回全部历史观察，按时间升序，可审计 |

**关键约束**：观察值是「过程数据」，只 append、永不覆盖、永不被算法结果篡改。
它呼应宪法 §5 的「增量 Delta」精神，但语义从「状态迁移」升级为「一条观测样本」。

### 能力域 C：熟练度引擎（Proficiency Engine）

子系统最核心、AI 完全碰不到的部分。把某 point 的全部观察值喂给算法，算出
「最终理解程度」。

| 能力 | 说明 |
|------|------|
| `计算权威状态` | 融合观察值 → BKT 后验 → 连续值 + 离散档位 |
| `离散化` | 连续值 `P(learned)` 映射回五态（见 §5.2） |
| `产出解释` | 结论 + 后验概率 + 不确定性 + 还需几次观察 |

**核心裁决权**：「这个点需要几次独立观察才能定到某档」完全属于本引擎，由知识点
属性 `bloom_level` 派生（见 §5.3），AI 无权干预，只负责不断提交观察值。

### 能力域 D：查询（Read API，供 AI 与 UI 消费）

AI 唯一能拿到的是系统计算结果。

| 能力 | 说明 |
|------|------|
| `查询单点权威状态` | AI 查「系统认为他对这个点掌握到什么程度」 |
| `查询整图状态` | UI 渲染知识版图（现有 `knowledge-map` 端点即此类） |
| `查询学习进度` | 哪些点已掌握 / 盲区 / 在学 |

## 3. 子系统与教学 Agent 的交互方式

知识版图的所有能力以 **tools / skills** 形式暴露给教学 Agent（LangChain `@tool`），
Agent 通过 system prompt 得知「有哪些能力」，但**何时调用哪个、如何组合，由 AI 运行时
自行判断**（对齐宪法「Bound, Don't Script」与铁律「指定 What/Why，拒绝指定 How」）。

划分规则（对齐记忆中的「决策点 vs 记忆点」判据）：

- **记忆点（确定性存储，代码守）**：DAG 结构、观察值追加、BKT 数值更新、离散化映射。
  这些永远不被 AI 控制流直接篡改。
- **决策点（语义判断，AI 定）**：要不要探测、探测哪个点、怎么讲解、要不要切换教学模式。
  这些交还 LLM。

## 4. 数据契约（三个核心对象）

### 4.1 Point（知识结构）

```python
class BloomLevel(str, Enum):
    REMEMBER = "remember"      # 记忆
    UNDERSTAND = "understand"  # 理解
    APPLY = "apply"            # 应用
    ANALYZE = "analyze"        # 分析
    EVALUATE = "evaluate"      # 评价
    CREATE = "create"          # 创造

class PointType(str, Enum):
    CONCEPT = "concept"        # 概念
    FACT = "fact"              # 事实
    SKILL = "skill"            # 技能
    PRINCIPLE = "principle"    # 原理
    PROCESS = "process"        # 流程

class PointAttributes(BaseModel):
    type: PointType                 # 类型
    difficulty: int                 # 难度 1-5（影响 BKT 参数）
    importance: int                 # 重要度 1-5（决定教学优先级 / 版图视觉权重）
    bloom_level: BloomLevel         # 认知层级（派生 verification_depth）

class KnowledgePoint(BaseModel):
    id: str
    name: str
    description: str
    prerequisites: list[str]        # 前置依赖知识点 id
    attributes: PointAttributes     # 新增：本体属性
```

### 4.2 Observation（AI 的输入，只追加）

```python
class Observation(BaseModel):
    point_id: str
    observed_state: CognitiveState  # AI 判定的五态（观察值，非结论）
    confidence: Confidence          # AI 的置信度
    evidence: list[str]             # 用户原话片段，严禁脑补（spec §6）
    observer: Literal["ai"] = "ai"  # 观察者身份（预留多源）
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### 4.3 Proficiency（系统的产出，AI 只读）

```python
class Proficiency(BaseModel):
    point_id: str
    latent_value: float             # 连续值 P(learned)，BKT 后验 ∈ [0,1]
    mapped_state: CognitiveState    # 离散化后的五态（唯一对外权威状态）
    uncertainty: float              # 不确定性（如 1 - max(p, 1-p)）
    source_algorithm: str           # "bkt"
    observation_count: int          # 已融合的观察次数
    last_updated: datetime
```

**关键区分**：`Observation.observed_state`（AI 判了什么）≠ `Proficiency.mapped_state`
（系统认为他真正会什么）。用户答得再多、AI 判得再准，没进算法或没到阈值，权威状态
就不变——这是系统故意为之，不是 bug。

## 5. 关键设计决策（已拍板）

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | 观察值形态 | AI 提交**离散五态 + 置信度**；连续值由系统算法算出 |
| 2 | 算法选型 | **BKT**（教育领域最成熟、参数可解释、天然增量更新） |
| 3 | 定级次数 | 由知识点 `bloom_level` 派生 `verification_depth`（见 §5.3） |
| 4 | 属性体系 | MVP 只落 4 个本体属性：`type / difficulty / importance / bloom_level` |

### 5.1 为什么让 AI 提交离散五态、而非连续分

- LLM 强在**语义判断**（"partial 还是 misconception"），弱在**精确打分**（"0.73 vs 0.68"）。
- 让 LLM 打连续分 = 引入假精度，BKT 会被带偏。
- 连续 `P(learned)` 是**统计推断的产物**，不是 LLM 该干的事。

### 5.2 观察值 → BKT 二元观测的映射

BKT 需要「答对/答错」二元观测，五态观察值映射如下（MVP 简化，后续可扩展为多级观测模型）：

| observed_state | BKT 观测 |
|----------------|----------|
| `mastered` / `partial` | 答对（correct） |
| `misconception` / `unknown` | 答错（incorrect） |
| `unassessed` | 不产生观测（跳过） |

BKT 四参数：`P(L0)` 先验已掌握概率、`P(T)` 学习转移概率、`P(G)` 猜测概率、`P(S)` 失误概率。
MVP 用全局默认值 + 按 `difficulty` 查表（`difficulty` 越高 `P(T)` 越低），不逐点手工调参。

### 5.3 verification_depth（「一次还是两次」的落点）

「这个点需要几次独立观察才能定到某档」由 `bloom_level` 派生，属子系统内部规则：

| bloom_level | verification_depth | 含义 |
|-------------|-------------------|------|
| remember / understand | 1 次 | 一次独立观察即可定级 |
| apply / analyze / evaluate / create | 2 次 | 概念 + 场景两次独立观察才能定 mastered |

离散化规则（连续值 → 五态）：

- `mastered`：`P(learned) ≥ mastery_threshold` **且** `observation_count ≥ verification_depth`
- `partial`：`P(learned)` 未达 mastery 阈值，但存在正确性证据
- `misconception` / `unknown`：以最近一次观察值 + 后验共同判定
- `unassessed`：`observation_count == 0`

（离散化的精确阈值留待实现阶段调参，本文档只锁行为边界。）

## 6. 与现有实现的差异（演进路径）

| 现在（plan.md 现状） | 目标（本文档） |
|----------------------|----------------|
| `KnowledgePoint` 仅 id/name/description/prerequisites | + `attributes` 字段（4 个本体属性） |
| `ProficiencyEntry`（状态迁移 Delta） | 语义升级为 `Observation`（观测样本） |
| `CognitiveState` 五态被 AI 直接定 | 五态仅作 `Proficiency.mapped_state` 映射输出，连续值才是底层 |
| `propose_diagnosis` 三层闸门 + 直接落库 | AI 只提交 Observation，`proficiency_engine` 单独算 Proficiency |
| 无算法层 | 新增 `proficiency_engine`（BKT） |

**迁移兼容性**：现有 `proficiency.<user_id>` namespace 下的历史 Delta 可视为「观测样本」，
BKT 引擎可将其作为冷启动历史数据回放（具体数据迁移方案在落地任务中定，不在本文档展开）。

## 7. 接口签名（草案，落地时以实现为准）

```python
# 能力域 A：知识结构管理
insert_point(point: KnowledgePoint) -> None
update_point(point_id: str, name: str, description: str) -> None
delete_point(point_id: str) -> None
add_prerequisite(point_id: str, prerequisite_id: str) -> None
remove_prerequisite(point_id: str, prerequisite_id: str) -> None
set_attributes(point_id: str, attributes: PointAttributes) -> None
query_structure(goal_key: str) -> KnowledgeModel
query_point(point_id: str) -> KnowledgePoint
query_prerequisites(point_id: str) -> list[KnowledgePoint]
query_dependents(point_id: str) -> list[KnowledgePoint]

# 能力域 B：观察记录（memory 层能力，由 propose_diagnosis 内部调用，非 LLM 暴露工具）
record_observation(observation: Observation) -> None
query_observations(point_id: str) -> list[Observation]

# 能力域 C：熟练度引擎（纯函数，无 IO）
compute_proficiency(point_id: str) -> Proficiency
#   内部：query_observations → BKT 融合 → 离散化

# 能力域 D：查询
query_proficiency(point_id: str) -> Proficiency
query_map() -> 整图状态（points + proficiencies）
```

## 8. 验收标准

- 知识版图能力与教学对话流程**解耦**：脱离对话也能独立插点 / 建边 / 登记属性 / 计算熟练度。
- AI 无法直接改写 `Proficiency.mapped_state`；它只能提交 `Observation` 并查询结果。
- 观察值只追加、不可覆盖、可审计（按 point 回读完整历史）。
- `verification_depth` 由 `bloom_level` 派生，`mastered` 必须满足「阈值 + 最少观察次数」双条件。
- BKT 纯数学，零 LLM 依赖，单测可直接用 InMemoryStore 驱动。
- 与宪法 §5 一致：观察值 / 熟练度均为增量更新，`user_id` 走 runtime context 隔离。
