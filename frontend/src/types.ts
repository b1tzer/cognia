// 与后端 schemas.py 对应的前端类型定义

export type CognitiveState = 'understood' | 'partial' | 'misconceived' | 'insufficient'
export type TutorAction = 'probe' | 'explain' | 'correct' | 'backtrack' | 'advance'

export interface Concept {
  id: string
  name: string
  summary: string
  why_matters: string
  prerequisites: string[]
  common_misconceptions: string[]
}

export interface KnowledgeModel {
  goal: string
  root_concepts: string[]
  concepts: Concept[]
}

export interface ConceptMastery {
  concept_id: string
  concept_name: string
  mastery: number
  state: CognitiveState
  evidence_count: number
  consecutive_failures: number
  last_evidence: string
}

export interface CognitiveModel {
  goal: string
  concepts: ConceptMastery[]
  updated_at: string
}

export interface ClarifyInfo {
  candidates: string[]
  question: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  action?: TutorAction | null
  diagnosis?: DiagnosticResult | null
  decision?: ActionDecision | null
  trace?: LLMTrace[] | null
  clarify?: ClarifyInfo | null
}

// 一次 LLM 调用的思考轨迹（用于「思考过程」折叠面板展示）
export interface LLMTrace {
  label: string           // 层中文标签，如「认知诊断」
  model: string           // 实际使用的模型
  system: string          // 发给 LLM 的 system prompt
  user: string            // 发给 LLM 的 user prompt
  output: string          // LLM 原始输出
  note?: string           // 异常/降级标注（如 reasoning 截断回退）
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export interface DiagnosticResult {
  state: CognitiveState
  confidence: number
  concept_ids: string[]
  evidence: string
  misconception: string
  missing: string[]
}

export interface ActionReason {
  evidence_cited: string
  criterion_used: string
  pedagogical_intent: string
  confidence: number
}

export interface ActionDecision {
  chosen_action: TutorAction
  reasons: ActionReason
}

export type SessionStage = 'clarifying' | 'active' | 'completed'

export interface Session {
  id: string
  goal: string
  created_at: string
  updated_at: string
  messages: Message[]
  cognitive: CognitiveModel | null
  knowledge: KnowledgeModel | null
  status: 'active' | 'completed'
  stage: SessionStage
  focus_concept?: Concept | null
}

export interface ChatResponse {
  session_id: string
  action: TutorAction
  reply: string
  diagnosis: DiagnosticResult | null
  decision: ActionDecision | null
  cognitive: CognitiveModel | null
  status: 'active' | 'completed'
  stage?: SessionStage
  focus_concept: Concept | null
  trace?: LLMTrace[]
  clarify?: ClarifyInfo | null
}

// ---------- 认知版图（Cognitive Atlas） ----------

export type RelationType = 'is-a' | 'related' | 'prerequisite'

export interface AtlasConcept {
  id: string
  name: string
  summary: string
  mastery: number
  state: CognitiveState
}

export interface AtlasRelation {
  from: string
  to: string
  relation_type: RelationType
}

export interface AtlasBridge {
  bridge: string
  bridge_name: string
  neighbors: Array<{ id: string; name: string }>
  betweenness: number
}

export interface AtlasView {
  concepts: AtlasConcept[]
  relations: AtlasRelation[]
  bridges: AtlasBridge[]
}

export interface AtlasNeighbor {
  from: string
  to: string
  relation_type: RelationType
  depth: number
  node: string
}

export interface AtlasNeighborsResponse {
  concept_id: string
  neighbors: AtlasNeighbor[]
}
