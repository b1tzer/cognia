#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录（脚本位于 scripts/ 子目录）
cd "$(dirname "$0")/.."

# 创建日志目录
mkdir -p logs

# 启动 Chainlit：stdout/stderr 同时输出到终端和 logs/cognia.log，
# 之后任何报错都能在 logs/cognia.log 里看到完整 traceback。
exec uv run chainlit run cognia/app.py 2>&1 | tee -a logs/cognia.log
