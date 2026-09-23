#!/usr/bin/env python3
"""把一份「设计 / 需求拆分」自动落为 GitHub 三层 issue 结构并建立 Sub-issue 关联。

固化的是「流程」，不是具体内容：具体内容写在一个规格文件里，脚本只负责
创建 issue + 建立父子关系。以后任何需求拆分，只需写一份规格文件然后跑本脚本。

规格文件格式（.py 定义 SPEC 字典，或 .json 等价结构）：

    SPEC = {
        "repo": "b1tzer/cognia",          # 可选，缺省用默认仓库
        "requirement": {
            "title": "Cognia 接入 Langfuse 可观测",
            "body": "## 背景\n...",
            "labels": ["kind/requirement", "status/todo"],  # 缺省会自补 kind/ + status/todo
        },
        "use_cases": [                     # 全部挂到 requirement 下
            {"title": "完整对话 trace 树", "body": "...", "labels": ["kind/use-case"]},
        ],
        "tasks": [                         # parent: "requirement" 或 use_cases 的整数下标
            {"title": "T1 开关层", "body": "...", "labels": ["kind/task", "area/backend", "assignee/pm"]},
            {"title": "T2 注入层", "body": "...", "labels": [...], "parent": 0},  # 挂到第 0 个用例
        ],
    }

标题前缀 [需求]/[用例]/[任务] 缺省自动补全；kind/* 与 status/todo 标签缺省自动补。

用法：
    uv run python scripts/create_three_layer_issues.py --spec scripts/examples/langfuse_issues_spec.py
    uv run python scripts/create_three_layer_issues.py --spec spec.json --repo b1tzer/cognia
    uv run python scripts/create_three_layer_issues.py --spec spec.py --dry-run   # 只打印计划不创建

依赖：仅 Python 标准库 + 已登录的 gh CLI（token 需含 repo 权限）。
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "b1tzer/cognia"

# 类型 -> 标题前缀 / 对应 kind 标签
PREFIX = {"requirement": "[需求]", "use-case": "[用例]", "task": "[任务]"}
KIND_LABEL = {
    "requirement": "kind/requirement",
    "use-case": "kind/use-case",
    "task": "kind/task",
}


def run_gh(args: list[str], check: bool = True) -> str:
    """调用 gh CLI，返回 stdout 文本。check=True 时非零退出即终止。"""
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0 and check:
        sys.exit(f"[ERROR] gh {' '.join(args)}\n{r.stderr.strip()}")
    return r.stdout


def ensure_gh() -> None:
    """预检 gh 可执行且已登录。"""
    try:
        out = run_gh(["auth", "status", "-t"], check=False)
    except FileNotFoundError:
        sys.exit("[ERROR] 未找到 gh CLI，请先安装并登录：gh auth login")
    if "Logged in to" not in out and "github.com" not in out:
        sys.exit("[ERROR] gh 未登录，请先执行 gh auth login")


def load_spec(path: str) -> dict:
    """加载 .py（含 SPEC 变量）或 .json 规格文件。"""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] 规格文件不存在：{path}")
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("spec_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "SPEC"):
            sys.exit(f"[ERROR] {path} 必须定义顶层变量 SPEC = {{...}}")
        return mod.SPEC
    sys.exit(f"[ERROR] 不支持的规格格式：{p.suffix}，请用 .py 或 .json")


def ensure_prefix(kind: str, title: str) -> str:
    pre = PREFIX[kind]
    return title if title.startswith(pre) else f"{pre}{title}"


def ensure_labels(kind: str, labels: list[str] | None) -> list[str]:
    labels = list(labels or [])
    if KIND_LABEL[kind] not in labels:
        labels.append(KIND_LABEL[kind])
    if "status/todo" not in labels:
        labels.append("status/todo")
    return labels


def create_issue(repo: str, kind: str, title: str, body: str, labels: list[str]) -> dict:
    """POST 一个 issue，返回 {number, node_id, html_url}。"""
    cmd = [
        "api", "--method", "POST", f"repos/{repo}/issues",
        "-f", f"title={ensure_prefix(kind, title)}",
        "-f", f"body={body}",
    ]
    for lab in ensure_labels(kind, labels):
        cmd += ["-f", f"labels[]={lab}"]
    return json.loads(run_gh(cmd))


def add_sub_issue(parent_id: str, child_id: str) -> None:
    """GraphQL addSubIssue 建立父子关联。用 -F 变量传 node_id，规避引号转义坑。"""
    query = (
        "mutation($p:ID!,$c:ID!){"
        "addSubIssue(input:{issueId:$p,subIssueId:$c}){clientMutationId}}"
    )
    out = run_gh(["api", "graphql", "-f", f"query={query}", "-F", f"p={parent_id}", "-F", f"c={child_id}"])
    try:
        data = json.loads(out)
        if data.get("errors"):
            print(f"  [WARN] addSubIssue 返回错误（可能已关联）：{data['errors']}", file=sys.stderr)
    except json.JSONDecodeError:
        pass


def build_and_link(spec: dict, repo: str, dry_run: bool) -> None:
    req = spec.get("requirement")
    if not req:
        sys.exit("[ERROR] SPEC 缺少 requirement")
    use_cases = spec.get("use_cases", []) or []
    tasks = spec.get("tasks", []) or []

    print(f"仓库: {repo}")
    print(f"需求: {ensure_prefix('requirement', req['title'])}")
    print(f"用例: {len(use_cases)} 个 | 任务: {len(tasks)} 个")

    if dry_run:
        print("\n[dry-run] 计划如下（不实际创建）：")
        print(f"  [需求] {req['title']}  labels={ensure_labels('requirement', req.get('labels'))}")
        for i, uc in enumerate(use_cases):
            print(f"  [用例#{i}] {uc['title']}  -> 挂到 需求")
        for t in tasks:
            parent = t.get("parent", "requirement")
            pdesc = "需求" if parent == "requirement" else f"用例#{parent}"
            print(f"  [任务] {t['title']}  -> 挂到 {pdesc}")
        return

    # 1) 需求
    req_obj = create_issue(repo, "requirement", req["title"], req["body"], req.get("labels"))
    req_id = req_obj["node_id"]
    print(f"  创建需求 #{req_obj['number']}")

    # 2) 用例，逐个挂到需求
    uc_objs = []
    for i, uc in enumerate(use_cases):
        o = create_issue(repo, "use-case", uc["title"], uc["body"], uc.get("labels"))
        add_sub_issue(req_id, o["node_id"])
        uc_objs.append(o)
        print(f"  创建用例 #{o['number']} -> 挂到 #{req_obj['number']}")

    # 3) 任务，按 parent 挂到需求或某个用例
    for t in tasks:
        o = create_issue(repo, "task", t["title"], t["body"], t.get("labels"))
        parent = t.get("parent", "requirement")
        if parent == "requirement":
            pid, pnum = req_id, req_obj["number"]
        else:
            try:
                idx = int(parent)
                target = uc_objs[idx]
            except (ValueError, IndexError):
                sys.exit(f"[ERROR] 任务「{t['title']}」的 parent={parent!r} 非法（需为 'requirement' 或用例整数下标）")
            pid, pnum = target["node_id"], target["number"]
        add_sub_issue(pid, o["node_id"])
        print(f"  创建任务 #{o['number']} -> 挂到 #{pnum}")

    print("\n=== 创建完成 ===")
    print(f"需求: #{req_obj['number']} {req_obj['html_url']}")
    for o in uc_objs:
        print(f"用例: #{o['number']} {o['html_url']}")
    for t in tasks:
        # tasks 未保留 obj，简单打印标题
        print(f"任务: {t['title']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将设计/需求拆分自动落为 GitHub 三层 issue（需求→用例→任务）并建立 Sub-issue 关联。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--spec", required=True, help="规格文件：.py（定义 SPEC 字典）或 .json")
    ap.add_argument("--repo", default=None, help=f"目标仓库，缺省取规格内 repo 或 {DEFAULT_REPO}")
    ap.add_argument("--dry-run", action="store_true", help="只打印创建计划，不实际创建/关联")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    repo = args.repo or spec.get("repo") or DEFAULT_REPO

    if not args.dry_run:
        ensure_gh()

    build_and_link(spec, repo, args.dry_run)


if __name__ == "__main__":
    main()
