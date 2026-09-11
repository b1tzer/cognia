"""optimizer.py 在线 Prompt 优化反馈回路的单元测试。

覆盖：规则存取、各层样本提取、单层优化循环的质量门（采纳/回滚）。

用标准库 unittest + mock，不引入额外依赖。强制 AI_ENABLED=False，
并通过 mock 隔离 LLM 与文件系统，保证测试确定性、不污染真实规则文件。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

# 强制离线 + 临时数据库（避免污染真实 cognia.db）
config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "optimizer_test.db")

import optimizer
import prompt_rules


def _make_session() -> dict:
    """构造一个含真实对话结构的会话（用于样本提取测试）。"""
    diag = {
        "state": "partial",
        "confidence": 0.7,
        "concept_ids": ["http-basics"],
        "evidence": "方向对",
        "misconception": "",
        "missing": [],
    }
    return {
        "id": "s1",
        "goal": "理解 HTTP 协议",
        "knowledge": {
            "goal": "理解 HTTP 协议",
            "root_concepts": ["tls"],
            "concepts": [
                {"id": "http-basics", "name": "HTTP 基础", "summary": "无状态协议",
                 "why_matters": "", "prerequisites": [], "common_misconceptions": []},
                {"id": "tls", "name": "TLS", "summary": "加密",
                 "why_matters": "", "prerequisites": ["http-basics"], "common_misconceptions": []},
            ],
        },
        "cognitive": {
            "goal": "理解 HTTP 协议",
            "concepts": [
                {"concept_id": "http-basics", "concept_name": "HTTP 基础",
                 "mastery": 0.7, "evidence_count": 1, "consecutive_failures": 0,
                 "state": "partial", "last_evidence": ""},
            ],
        },
        "messages": [
            {"role": "assistant", "content": "开场白", "action": "probe", "diagnosis": None},
            {"role": "user", "content": "HTTP 是无状态协议，通过请求响应工作",
             "action": None, "diagnosis": diag},
            {"role": "assistant", "content": "那你能说说无状态意味着什么吗？", "action": "probe",
             "diagnosis": diag, "decision": {"chosen_action": "probe"}},
        ],
    }


class TestPromptRules(unittest.TestCase):
    def setUp(self):
        # 把规则文件路径指向临时目录，避免污染真实 eval/*.json
        tmp = tempfile.mkdtemp()
        self._paths = {
            layer: Path(tmp) / f"{layer}.json"
            for layer in prompt_rules.LAYER_RULES_PATH
        }
        self._patcher = mock.patch.dict(prompt_rules.LAYER_RULES_PATH, self._paths)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_save_and_load_rules(self):
        prompt_rules.save_rules("diagnosis", ["规则A", "规则B", "规则A"])  # 含重复
        self.assertEqual(prompt_rules.load_rules("diagnosis"), ["规则A", "规则B"])

    def test_load_missing_returns_empty(self):
        self.assertEqual(prompt_rules.load_rules("tutor"), [])

    def test_rules_suffix_empty(self):
        self.assertEqual(prompt_rules.rules_suffix("decision_action"), "")

    def test_rules_suffix_contains_rules(self):
        prompt_rules.save_rules("domain_model", ["原则1"])
        suffix = prompt_rules.rules_suffix("domain_model")
        self.assertIn("原则1", suffix)

    def test_save_clips_to_max(self):
        rules = [f"r{i}" for i in range(prompt_rules.MAX_RULES_PER_LAYER + 5)]
        prompt_rules.save_rules("tutor", rules)
        self.assertEqual(
            len(prompt_rules.load_rules("tutor")), prompt_rules.MAX_RULES_PER_LAYER
        )


class TestExtractSamples(unittest.TestCase):
    @mock.patch.object(optimizer.db, "recent_sessions")
    def test_extract_diagnosis(self, recent):
        recent.return_value = [_make_session()]
        samples = optimizer.extract_samples("diagnosis", 10)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["predicted_state"], "partial")
        self.assertEqual(samples[0]["user_text"], "HTTP 是无状态协议，通过请求响应工作")

    @mock.patch.object(optimizer.db, "recent_sessions")
    def test_extract_decision_action(self, recent):
        recent.return_value = [_make_session()]
        samples = optimizer.extract_samples("decision_action", 10)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["action"], "probe")
        self.assertEqual(samples[0]["state"], "partial")

    @mock.patch.object(optimizer.db, "recent_sessions")
    def test_extract_tutor(self, recent):
        recent.return_value = [_make_session()]
        samples = optimizer.extract_samples("tutor", 10)
        self.assertEqual(len(samples), 1)
        self.assertIn("无状态", samples[0]["reply"])

    @mock.patch.object(optimizer.db, "recent_sessions")
    def test_extract_domain_model(self, recent):
        recent.return_value = [_make_session()]
        samples = optimizer.extract_samples("domain_model", 10)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0]["concepts"]), 2)


class TestRunCycle(unittest.TestCase):
    def _sample(self):
        return {"layer": "diagnosis", "goal": "g", "concepts": [], "user_text": "x"}

    @mock.patch.object(prompt_rules, "save_rules")
    @mock.patch.object(prompt_rules, "load_rules", return_value=[])
    @mock.patch.object(optimizer, "reevaluate", return_value={"predicted_state": "understood"})
    @mock.patch.object(optimizer, "generate_rules", return_value=["新规则"])
    @mock.patch.object(optimizer, "extract_samples")
    def test_optimized(self, extract, gen, reev, load, save):
        # 6 个样本，1 个错误（首个 judge=False），复评后修正（第 7 次 judge=True）
        samples = [self._sample() for _ in range(6)]
        extract.return_value = samples
        with mock.patch.object(
            optimizer, "judge_sample",
            side_effect=[False, True, True, True, True, True, True],
        ):
            result = optimizer.run_cycle("diagnosis", limit=10)
        self.assertEqual(result["status"], "optimized")
        self.assertEqual(result["added"], ["新规则"])
        self.assertEqual(result["correct_before"], 5)
        self.assertEqual(result["correct_after"], 6)
        # 采纳：只保存一次（不回滚）
        self.assertEqual(save.call_count, 1)

    @mock.patch.object(prompt_rules, "save_rules")
    @mock.patch.object(prompt_rules, "load_rules", return_value=[])
    @mock.patch.object(optimizer, "reevaluate", return_value={"predicted_state": "understood"})
    @mock.patch.object(optimizer, "generate_rules", return_value=["新规则"])
    @mock.patch.object(optimizer, "extract_samples")
    def test_rolled_back(self, extract, gen, reev, load, save):
        samples = [self._sample() for _ in range(6)]
        extract.return_value = samples
        # 复评后仍错误（第 7 次 judge=False）→ 质量门不通过，回滚
        with mock.patch.object(
            optimizer, "judge_sample",
            side_effect=[False, True, True, True, True, True, False],
        ):
            result = optimizer.run_cycle("diagnosis", limit=10)
        self.assertEqual(result["status"], "rolled_back")
        # 回滚：先注入保存一次，再回滚保存一次
        self.assertEqual(save.call_count, 2)

    @mock.patch.object(optimizer, "judge_sample", return_value=True)
    @mock.patch.object(optimizer, "extract_samples")
    def test_no_error(self, extract, judge):
        extract.return_value = [self._sample() for _ in range(6)]
        result = optimizer.run_cycle("diagnosis", limit=10)
        self.assertEqual(result["status"], "no_error")

    @mock.patch.object(optimizer, "extract_samples")
    def test_skipped_insufficient(self, extract):
        extract.return_value = [self._sample() for _ in range(2)]  # < MIN_SAMPLES(5)
        result = optimizer.run_cycle("diagnosis", limit=10)
        self.assertEqual(result["status"], "skipped")


class TestWatermark(unittest.TestCase):
    def test_count_new_user_turns(self):
        sessions = [
            {"id": "a", "messages": [
                {"role": "assistant"}, {"role": "user"},
                {"role": "assistant"}, {"role": "user"},
            ]},
        ]
        wm = {"a": 2}  # 已消费前 2 条（assistant, user）
        self.assertEqual(optimizer.count_new_user_turns(sessions, wm), 1)

    def test_count_new_user_turns_no_watermark(self):
        sessions = [
            {"id": "a", "messages": [
                {"role": "user"}, {"role": "assistant"}, {"role": "user"},
            ]},
        ]
        self.assertEqual(optimizer.count_new_user_turns(sessions, {}), 2)

    def test_advance_watermark(self):
        sessions = [{"id": "a", "messages": [1, 2, 3, 4]}]
        wm = {"a": 2, "b": 5}
        new = optimizer.advance_watermark(sessions, wm)
        self.assertEqual(new["a"], 4)
        self.assertEqual(new["b"], 5)  # 未出现的会话水位不动


class TestRunAllCycles(unittest.TestCase):
    @mock.patch.object(optimizer, "save_state")
    @mock.patch.object(optimizer, "run_cycle")
    @mock.patch.object(optimizer, "count_new_user_turns", return_value=3)
    @mock.patch.object(optimizer, "_recent_sessions_safe", return_value=[])
    @mock.patch.object(optimizer, "load_state", return_value={"watermark": {}, "last_optimized_at": ""})
    def test_waiting_for_data(self, load, recent, count, run, save):
        # 新增对话轮次 < 阈值 → 不执行任何优化、不推进水位
        result = optimizer.run_all_cycles(limit=10)
        self.assertEqual(result["status"], "waiting_for_data")
        self.assertEqual(result["new_turns"], 3)
        run.assert_not_called()
        save.assert_not_called()

    @mock.patch.object(optimizer, "save_state")
    @mock.patch.object(optimizer, "advance_watermark", return_value={"sid": 5})
    @mock.patch.object(optimizer, "run_cycle", return_value={"status": "optimized", "samples": 5})
    @mock.patch.object(optimizer, "count_new_user_turns", return_value=12)
    @mock.patch.object(optimizer, "_recent_sessions_safe", return_value=[{"id": "sid", "messages": []}])
    @mock.patch.object(optimizer, "load_state", return_value={"watermark": {}, "last_optimized_at": ""})
    def test_executed_advance_watermark(self, load, recent, count, run, advance, save):
        # 新增对话轮次达标 → 执行并推进水位
        result = optimizer.run_all_cycles(limit=10)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["optimized"], 4)  # 4 层均 optimized
        advance.assert_called_once()
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
