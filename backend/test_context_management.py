"""需求 #71 需求2：长会话上下文管理（滑动窗口 + 摘要）的测试。

覆盖：
1. _session_history 组装「摘要 + 最近 KEEP_RECENT 轮」滑动窗口
2. _maybe_summarize 摘要触发条件（covered_upto 前进）
3. db 的 summary 存取（summary_json 列）
4. _summarize 离线模式安全降级

强制 AI_ENABLED=False，摘要走离线降级（不调 LLM），保证可复现。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "ctx_test.db")

import db
import main


def _make_messages(n: int) -> list[dict]:
    """构造 n 条交替 user/assistant 消息（user 开头）。"""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append({
                "role": "user", "content": f"用户第{i}条",
                "action": None, "diagnosis": {"state": "partial"},
            })
        else:
            msgs.append({
                "role": "assistant", "content": f"导师第{i}条", "action": "probe",
            })
    return msgs


class TestSessionHistory(unittest.TestCase):
    def test_empty_session(self):
        s = {"messages": [], "summary": None}
        history, summary = main._session_history(s)
        self.assertEqual(history, [])
        self.assertEqual(summary, "")

    def test_recent_window_bounded(self):
        # 消息数远超 KEEP_RECENT 时，history 只保留滑动窗口内的轮次
        n = config.CONTEXT_KEEP_RECENT + 8
        s = {"messages": _make_messages(n), "summary": None}
        history, summary = main._session_history(s)
        # 每 2 条消息配对成 1 轮；窗口内轮数不会超过 KEEP_RECENT/2 加少量边界
        self.assertLessEqual(len(history), config.CONTEXT_KEEP_RECENT // 2 + 1)

    def test_summary_text_returned(self):
        s = {
            "messages": _make_messages(4),
            "summary": {"text": "已学概念A", "covered_upto": 2},
        }
        history, summary = main._session_history(s)
        self.assertEqual(summary, "已学概念A")

    def test_recent_pairs_user_and_assistant(self):
        # 最近一轮应是 user+assistant 配对
        s = {"messages": _make_messages(4), "summary": None}
        history, _ = main._session_history(s)
        self.assertGreaterEqual(len(history), 1)
        last = history[-1]
        self.assertIn("用户", last.get("user_text", ""))
        self.assertIn("导师", last.get("ai_reply", ""))


class TestMaybeSummarize(unittest.TestCase):
    def test_below_threshold_no_summary(self):
        s = {"messages": _make_messages(10), "summary": None}
        result = main._maybe_summarize(s)
        self.assertEqual(result, {})  # 未达阈值，原样返回空 summary

    def test_at_threshold_triggers(self):
        n = config.CONTEXT_SUMMARY_THRESHOLD
        s = {"messages": _make_messages(n), "summary": None}
        result = main._maybe_summarize(s)
        self.assertIsNotNone(result)
        # covered_upto 前进到 len - KEEP_RECENT
        self.assertEqual(result["covered_upto"], n - config.CONTEXT_KEEP_RECENT)

    def test_already_covered_no_resummarize(self):
        # covered_upto 已覆盖到窗口边界时，不再摘要
        n = config.CONTEXT_SUMMARY_THRESHOLD
        s = {
            "messages": _make_messages(n),
            "summary": {"text": "已有摘要", "covered_upto": n - config.CONTEXT_KEEP_RECENT},
        }
        result = main._maybe_summarize(s)
        self.assertEqual(result["covered_upto"], n - config.CONTEXT_KEEP_RECENT)
        self.assertEqual(result["text"], "已有摘要")


class TestSummarizeOffline(unittest.TestCase):
    def test_offline_returns_old_summary(self):
        # 离线模式不调 LLM，_summarize 返回原摘要
        result = main._summarize("旧摘要", [{"role": "user", "content": "新内容"}])
        self.assertEqual(result, "旧摘要")


class TestSummaryPersistence(unittest.TestCase):
    def setUp(self):
        db.init_db()

    def test_update_and_get_summary(self):
        sid = db.create_session("测试目标")["id"]
        db.update_summary(sid, {"text": "摘要内容", "covered_upto": 5})
        s = db.get_session(sid)
        self.assertEqual(s["summary"]["text"], "摘要内容")
        self.assertEqual(s["summary"]["covered_upto"], 5)

    def test_no_summary_returns_none(self):
        sid = db.create_session("测试目标")["id"]
        s = db.get_session(sid)
        self.assertIsNone(s["summary"])


if __name__ == "__main__":
    unittest.main()
