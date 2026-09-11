import { useState } from 'react'

interface Props {
  onStart: (goal: string) => void
  busy: boolean
  error: string | null
}

export default function GoalInput({ onStart, busy, error }: Props) {
  const [goal, setGoal] = useState('')

  const submit = () => {
    const g = goal.trim()
    if (!g || busy) return
    onStart(g)
  }

  return (
    <div className="hero">
      <div className="hero-inner">
        <h1 className="hero-title">
          你想真正<span className="accent">理解</span>什么？
        </h1>
        <p className="hero-sub">
          Cognia 不是问答机器人，也不是题库。它会主动拆解你的学习目标，诊断你
          <b> 懂了什么、没懂什么、哪里理解错了</b>，再用追问引导你建立完整、可迁移的知识体系。
        </p>
        <div className="goal-box">
          <input
            className="goal-input"
            placeholder="例如：深入理解 HTTP 协议 / Git 版本控制 / 机器学习的本质…"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            autoFocus
            disabled={busy}
          />
          <button className="btn btn-primary" onClick={submit} disabled={busy || !goal.trim()}>
            {busy ? '构建知识模型中…' : '开始学习 →'}
          </button>
        </div>
        {error && <div className="error-tip">{error}</div>}
        <div className="hero-hints">
          <span>🔍 主动诊断认知状态</span>
          <span>🧠 构建概念依赖图谱</span>
          <span>💬 苏格拉底式追问</span>
        </div>
      </div>
    </div>
  )
}
