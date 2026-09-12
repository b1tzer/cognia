"""概念级对话栈端到端装配测试（Phase 1 · Task 3）。

验证「进入取栈 → 注入诊断/回复 → 结束压栈 → 下一轮取到」的完整链路，
模拟连续 probe 追问场景下的对话栈装配。不依赖真实 db，用 _push_dialogue
模拟 _persist 的压栈动作。
"""
from __future__ import annotations

import unittest

import config

config.AI_ENABLED = False

import domain_model
import main


def _make_session() -> dict:
    km = domain_model.build_knowledge_model("理解 HTTP")
    cognitive = main._build_cognitive("理解 HTTP", km.concepts)
    return {
        "knowledge": km.model_dump(),
        "cognitive": cognitive,
        "status": "active",
    }


class TestDialogueStackE2E(unittest.TestCase):
    def test_first_round_has_empty_history(self):
        # 新概念首轮无历史，等价降级到现状
        s = _make_session()
        ctx = main._process_turn(s, "HTTP 是无状态的应用层协议，用于客户端和服务器通信")
        self.assertEqual(ctx["history"], [])

    def test_second_round_inherits_history(self):
        # 连续 probe 追问：第二轮应能取到第一轮压入的轨迹
        s = _make_session()
        ctx1 = main._process_turn(s, "HTTP 是无状态协议，客户端发请求服务器响应")
        self.assertEqual(ctx1["history"], [])
        main._push_dialogue(
            s["cognitive"], ctx1["focus_id"], ctx1["content"],
            "那请求里包含哪些关键部分？", ctx1["action"], ctx1["diagnosis"].state,
        )
        ctx2 = main._process_turn(s, "就是请求行、请求头和请求体")
        self.assertGreaterEqual(len(ctx2["history"]), 1)
        last = ctx2["history"][-1]
        self.assertEqual(last["user_text"], "HTTP 是无状态协议，客户端发请求服务器响应")
        self.assertEqual(last["ai_reply"], "那请求里包含哪些关键部分？")

    def test_history_stays_within_capacity(self):
        # 多轮后对话栈不超过 3 轮
        s = _make_session()
        for i in range(5):
            ctx = main._process_turn(s, f"这是第 {i} 轮的补充回答，关于 HTTP 的一些理解")
            main._push_dialogue(
                s["cognitive"], ctx["focus_id"], ctx["content"],
                f"追问 {i}", ctx["action"], ctx["diagnosis"].state,
            )
        ctx = main._process_turn(s, "总结一下")
        self.assertLessEqual(len(ctx["history"]), 3)


if __name__ == "__main__":
    unittest.main()
