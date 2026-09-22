"""个人 Wiki 的 Git 版本层。

用系统 git CLI（subprocess）为每个 user 维护一个独立 git 仓库，正文 markdown
文件为版本真相：git commit = 版本，git log = 历史，git show/checkout = 溯源与回滚。
零第三方依赖（不引入 GitPython / pygit2，系统必有 git）。

仓库布局：``{WIKI_REPO_ROOT}/{user_id}/``，每个 user 一个 git 仓库，天然多租户隔离。
正文文件：``{page_id}.md``（页面树关系存元数据层，不靠文件目录）。

安全边界：所有 git 操作失败统一抛 :class:`WikiRepoError`，由上层（router）捕获后
降级返回 503，绝不 500、绝不静默。本模块不负责 user_id / page_id 校验（上层保证）。
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = "data/wiki"

_AUTHOR_MAP = {
    "ai": ("cognia-ai", "ai@cognia.local"),
    "user": ("cognia-user", "user@cognia.local"),
}


class WikiRepoError(Exception):
    """wiki 仓库操作失败（git 不可用 / 磁盘异常等）。"""


def _root() -> Path:
    """wiki 仓库根目录（环境变量 WIKI_REPO_ROOT 覆盖，默认 data/wiki）。"""
    return Path(os.getenv("WIKI_REPO_ROOT", DEFAULT_ROOT))


def repo_path(user_id: str) -> Path:
    """某 user 的仓库目录路径。"""
    return _root() / user_id


def _run_git(
    user_id: str,
    args: list[str],
    *,
    check: bool = True,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess:
    """在指定 user 的仓库目录下执行 git 命令。"""
    cmd = ["git", "-C", str(repo_path(user_id)), *args]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WikiRepoError(f"git 执行失败：{exc}") from exc
    if check and proc.returncode != 0:
        raise WikiRepoError(f"git {' '.join(args)} 失败：{proc.stderr.strip()}")
    return proc


def ensure_repo(user_id: str) -> None:
    """确保该 user 的仓库存在（不存在则 git init + 配置本地提交身份）。"""
    path = repo_path(user_id)
    if (path / ".git").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    _run_git(user_id, ["init", "-q"])
    _run_git(user_id, ["config", "user.name", "cognia"])
    _run_git(user_id, ["config", "user.email", "wiki@cognia.local"])


def _author_env(author: str) -> dict:
    """按逻辑作者（ai / user）映射 git 提交者身份。"""
    name, email = _AUTHOR_MAP.get(author, _AUTHOR_MAP["ai"])
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def head_commit(user_id: str) -> str:
    """当前仓库 HEAD 的完整 commit hash。"""
    return _run_git(user_id, ["rev-parse", "HEAD"]).stdout.strip()


def write_page(user_id: str, page_id: str, content: str, author: str, summary: str = "") -> str:
    """写入（或更新）一页正文并 commit，返回新的 commit hash。

    content 与上一版相同时 git commit 无变更，此时返回当前 HEAD（不算错误）。
    """
    ensure_repo(user_id)
    file = repo_path(user_id) / f"{page_id}.md"
    file.write_text(content, encoding="utf-8")
    _run_git(user_id, ["add", f"{page_id}.md"], extra_env=_author_env(author))
    msg = summary or f"update {page_id}"
    proc = _run_git(user_id, ["commit", "-m", msg], check=False, extra_env=_author_env(author))
    if proc.returncode != 0:
        # git 的 "nothing to commit" 输出在 stdout（非 stderr），两者都查。
        if "nothing to commit" not in (proc.stdout + proc.stderr):
            raise WikiRepoError(f"git commit 失败：{proc.stderr.strip() or proc.stdout.strip()}")
    return head_commit(user_id)


def read_page(user_id: str, page_id: str) -> str | None:
    """读一页正文（工作区即最新版本），不存在返回 None。"""
    file = repo_path(user_id) / f"{page_id}.md"
    if not file.exists():
        return None
    return file.read_text(encoding="utf-8")


def delete_page(user_id: str, page_id: str, author: str) -> str:
    """删除一页并 commit，返回新的 commit hash。"""
    file = repo_path(user_id) / f"{page_id}.md"
    if not file.exists():
        return head_commit(user_id)
    _run_git(user_id, ["rm", f"{page_id}.md"], extra_env=_author_env(author))
    _run_git(user_id, ["commit", "-m", f"delete {page_id}"], extra_env=_author_env(author))
    return head_commit(user_id)


def history(user_id: str, page_id: str) -> list[dict]:
    """读一页的版本历史（git log，按时间降序，最新在前）。

    返回 ``[{commit_hash, author, timestamp, summary}]``。
    """
    file = repo_path(user_id) / f"{page_id}.md"
    if not file.exists():
        return []
    proc = _run_git(
        user_id,
        ["log", "--format=%H%x09%an%x09%at%x09%s", "--", f"{page_id}.md"],
    )
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        commit_hash, author_name, ts, summary = parts
        author = "user" if author_name == "cognia-user" else "ai"
        try:
            ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            ts_iso = ""
        out.append({
            "commit_hash": commit_hash,
            "author": author,
            "timestamp": ts_iso,
            "summary": summary,
        })
    return out


def rollback(user_id: str, page_id: str, commit_hash: str, author: str) -> str:
    """把一页回滚到指定 commit 的内容，产生新 commit，返回新 commit hash。"""
    file = repo_path(user_id) / f"{page_id}.md"
    if not file.exists():
        raise WikiRepoError(f"页面不存在：{page_id}")
    proc = _run_git(user_id, ["show", f"{commit_hash}:{page_id}.md"], check=False)
    if proc.returncode != 0:
        raise WikiRepoError(f"commit 不存在或页面不在该版本：{commit_hash}")
    file.write_text(proc.stdout, encoding="utf-8")
    _run_git(user_id, ["add", f"{page_id}.md"], extra_env=_author_env(author))
    _run_git(user_id, ["commit", "-m", f"rollback to {commit_hash}"], extra_env=_author_env(author))
    return head_commit(user_id)
