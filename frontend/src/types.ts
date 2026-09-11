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
  last_evidence: string
}

export interface CognitiveModel {
  goal: string
  concepts: ConceptMastery[]
  updated_at: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  action?: TutorAction | null
  diagnosis?: DiagnosticResult | null
}

export interface DiagnosticResult {
  state: CognitiveState
  confidence: number
  concept_ids: string[]
  evidence: string
  misconception: string
  missing: string[]
}

export interface Session {
  id: string
  goal: string
  created_at: string
  updated_at: string
  messages: Message[]
  cognitive: CognitiveModel | null
  knowledge: KnowledgeModel | null
  status: 'active' | 'completed'
}

export interface ChatResponse {
  session_id: string
  action: TutorAction
  reply: string
  diagnosis: DiagnosticResult
  cognitive: CognitiveModel
  status: 'active' | 'completed'
  focus_concept: Concept | null
}
