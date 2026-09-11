I'll start by reading the required skill file, then do web research on building AI Agent web applications.



Now let me research the topic using public web search.



Now I have enough research material. Let me compose the full technical document.

---

# 构建 AI Agent Web 应用：技术方案与最小可运行示例

## 1. 前端方案

### 1.1 聊天界面架构

AI Agent 的 Web 前端核心是一个**消息列表 + 输入框**的聊天界面，与普通聊天 UI 的区别在于需要处理**流式输出**和**富文本渲染**。

**核心组件：**

| 组件 | 职责 |
|---|---|
| 消息列表（MessageList） | 按时间顺序展示用户消息和 AI 回复 |
| 输入框（InputBox） | 用户输入文本，支持 Enter 发送 |
| 流式渲染器（StreamRenderer） | 实时追加 SSE 推送的 token |
| Markdown 渲染器 | 将 AI 回复中的 Markdown 转为 HTML |
| 代码高亮 | 对代码块进行语法着色 |

### 1.2 流式渲染：SSE vs WebSocket

| 特性 | SSE (Server-Sent Events) | WebSocket |
|---|---|---|
| 通信方向 | 服务器 → 客户端（单向） | 双向 |
| 协议 | HTTP（标准） | ws:// 自定义协议 |
| 浏览器原生支持 | `EventSource` API | `WebSocket` API |
| 自动重连 | 内置 | 需手动实现 |
| 消息格式 | `text/event-stream`（文本） | 二进制或文本帧 |
| 适用场景 | AI 流式输出、通知推送 | 实时游戏、协作编辑 |

**推荐 SSE**，原因：
- AI Agent 的核心需求是**服务器推送文本流**，无需客户端频繁上行
- 浏览器 `EventSource` 原生支持自动重连
- 实现简单，在 FastAPI 中只需 `yield` + `EventSourceResponse`
- 兼容标准 HTTP 代理和负载均衡

**SSE 数据格式：**

```
data: {"token": "你好"}

data: {"token": "，我"}

data: {"token": "是AI助手"}

data: {"type": "done"}
```

每条消息以 `data:` 开头，双换行符分隔。客户端通过 `EventSource.onmessage` 接收。

### 1.3 消息历史管理

消息历史需要持久化，以便：
- 刷新页面后恢复对话
- 作为 LLM 上下文传递给 Agent

**存储方案：**

| 存储 | 适用场景 |
|---|---|
| 内存 dict | 开发/单用户原型 |
| SQLite | 单机部署，零依赖 |
| PostgreSQL + pgvector | 生产环境，支持向量检索 |
| Redis | 高速缓存 + 消息队列 |

每条消息的结构：

```json
{
  "id": "msg_xxx",
  "role": "user" | "assistant",
  "content": "消息正文（Markdown 格式）",
  "timestamp": "2026-09-11T10:00:00Z"
}
```

### 1.4 Markdown / 代码块渲染

前端使用两个成熟库：

- **[marked.js](https://marked.js.org/)** — 将 Markdown 文本解析为 HTML，支持 GFM（表格、列表、链接等）
- **[highlight.js](https://highlightjs.org/)** — 对代码块进行语法高亮，支持 190+ 编程语言

**渲染流程：**

```
AI 回复文本 (Markdown)
    → marked.parse(text)   → HTML
    → 注入 message 容器 DOM
    → hljs.highlightAll()  → 代码块着色
```

**安全注意：** `marked.js` 默认会渲染 HTML 标签，需设置 `sanitize` 或使用 DOMPurify 防止 XSS。推荐做法：用 `marked.parse(text, { breaks: true })` 后，通过 `textContent` / `innerHTML` 安全注入。

---

## 2. 最小可运行端到端示例

以下是一个完整的 AI Agent Web 应用，使用 **Python FastAPI 后端 + 原生 JavaScript 前端**，支持 SSE 流式输出、消息历史、Markdown 渲染和代码高亮。

### 2.1 项目结构

```
ai-agent-web/
├── requirements.txt      # Python 依赖
├── main.py               # FastAPI 后端（含 Agent 逻辑）
├── static/
│   └── index.html        # 单页前端（HTML + CSS + JS 合一）
└── chat_history.db       # SQLite 数据库（运行时自动创建）
```

### 2.2 后端代码：`main.py`

```python
"""
AI Agent Web 应用后端
技术栈：FastAPI + SSE 流式输出 + SQLite 消息持久化
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timezone
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# 数据库层：SQLite 存储对话历史
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "chat_history.db"


def init_db():
    """初始化数据库表结构"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session
        ON messages(session_id, id)
    """)
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    """保存一条消息到数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    """获取指定会话的历史消息"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows[-limit:]]


# ---------------------------------------------------------------------------
# AI Agent 核心：模拟 LLM 流式输出
# 实际项目中替换为 OpenAI / Anthropic / Ollama 等 API 调用
# ---------------------------------------------------------------------------

async def agent_stream(prompt: str, history: list[dict]) -> AsyncGenerator[str, None]:
    """
    模拟 AI Agent 的流式响应生成器。
    实际使用时替换为 LLM 的 stream() 调用。例如：
      async with client.chat.stream(messages=messages) as stream:
          async for chunk in stream:
              yield chunk.delta
    """
    # 构造系统提示词，让 Agent 知道自己的身份
    system_prompt = (
        "你是一个有用的 AI 助手。请用中文回答用户问题。"
        "回答中可以使用 Markdown 格式，包括 **加粗**、`代码`、```代码块```、列表、表格等。"
    )

    # 组装完整消息列表（系统提示 + 历史 + 当前问题）
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # ---- 模拟流式输出 ----
    # 实际项目中，此处调用 LLM 的流式 API
    full_response = (
        f"你好！你问的是：**{prompt}**\n\n"
        "这是一个流式输出的示例。在实际项目中，这里会接入真实的 LLM API。\n\n"
        "### 支持的功能\n\n"
        "- **Markdown 渲染**：标题、列表、表格、链接\n"
        "- **代码高亮**：支持多种编程语言\n"
        "- **流式输出**：逐 token 推送，实时显示\n\n"
        "```python\n"
        "def hello():\n"
        "    print(\"Hello, AI Agent!\")\n"
        "```\n\n"
        "| 功能 | 状态 |\n"
        "|------|------|\n"
        "| SSE 流式输出 | ✅ |\n"
        "| Markdown 渲染 | ✅ |\n"
        "| 代码高亮 | ✅ |\n"
        "| 消息历史 | ✅ |\n"
    )

    # 逐字模拟流式输出（每 30ms 输出一个字符）
    for char in full_response:
        yield json.dumps({"token": char})
        await asyncio.sleep(0.03)

    # 发送结束标记
    yield json.dumps({"type": "done"})


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库"""
    init_db()
    yield


app = FastAPI(title="AI Agent Web", lifespan=lifespan)

# 挂载静态文件目录，提供前端 HTML
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """获取会话历史"""
    return get_history(session_id)


@app.post("/api/chat/{session_id}", response_class=EventSourceResponse)
async def chat(session_id: str, prompt: str = Form(...)):
    """
    聊天接口：接收用户输入，返回 SSE 流式响应

    请求方式：POST
    请求体：Content-Type: application/x-www-form-urlencoded
            prompt=用户消息
    响应：text/event-stream 格式的 SSE 流
    """
    # 保存用户消息
    save_message(session_id, "user", prompt)

    # 获取历史上下文
    history = get_history(session_id)

    async def event_generator():
        """SSE 事件生成器：逐 token 推送"""
        full_response = ""
        async for chunk in agent_stream(prompt, history):
            data = json.loads(chunk)
            if "token" in data:
                full_response += data["token"]
            yield {"data": chunk}

        # 保存 AI 完整回复到数据库
        save_message(session_id, "assistant", full_response)

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### 2.3 前端代码：`static/index.html`

这是一个**单页 HTML**，包含全部 CSS 样式和 JavaScript 逻辑，无需构建工具。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent Chat</title>

    <!-- marked.js: Markdown → HTML 渲染 -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- highlight.js: 代码语法高亮 -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>

    <style>
        /* ---- 全局样式 ---- */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; height: 100vh;
            display: flex; flex-direction: column;
        }

        /* ---- 头部 ---- */
        .header {
            padding: 16px 24px; background: #1e293b;
            border-bottom: 1px solid #334155; text-align: center;
        }
        .header h1 { font-size: 18px; font-weight: 600; }
        .header .subtitle { font-size: 12px; color: #94a3b8; margin-top: 4px; }

        /* ---- 消息列表 ---- */
        .chat-container { flex: 1; overflow-y: auto; padding: 20px 0; }
        .messages { max-width: 800px; margin: 0 auto; padding: 0 20px; }
        .message {
            margin-bottom: 20px; display: flex; gap: 12px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { flex-direction: row-reverse; }
        .avatar {
            width: 32px; height: 32px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; flex-shrink: 0;
        }
        .message.user .avatar { background: #3b82f6; }
        .message.assistant .avatar { background: #8b5cf6; }
        .bubble {
            max-width: 75%; padding: 12px 16px; border-radius: 12px;
            line-height: 1.6; font-size: 14px;
        }
        .message.user .bubble { background: #3b82f6; color: #fff; border-bottom-right-radius: 4px; }
        .message.assistant .bubble { background: #1e293b; border: 1px solid #334155; border-bottom-left-radius: 4px; }

        /* ---- Markdown 渲染样式 ---- */
        .bubble h1, .bubble h2, .bubble h3 { margin: 12px 0 6px; color: #f1f5f9; }
        .bubble h1 { font-size: 18px; } .bubble h2 { font-size: 16px; } .bubble h3 { font-size: 15px; }
        .bubble p { margin: 6px 0; }
        .bubble ul, .bubble ol { margin: 6px 0; padding-left: 20px; }
        .bubble li { margin: 3px 0; }
        .bubble code:not(pre code) {
            background: #334155; padding: 2px 6px; border-radius: 4px;
            font-size: 13px; color: #fbbf24;
        }
        .bubble pre {
            margin: 10px 0; border-radius: 8px; overflow: hidden;
            position: relative;
        }
        .bubble pre code {
            display: block; padding: 14px 16px; overflow-x: auto;
            font-size: 13px; line-height: 1.5;
        }
        .bubble table { border-collapse: collapse; margin: 10px 0; width: 100%; }
        .bubble th, .bubble td {
            border: 1px solid #334155; padding: 8px 12px; text-align: left;
        }
        .bubble th { background: #1e293b; font-weight: 600; }
        .bubble blockquote {
            border-left: 3px solid #8b5cf6; padding-left: 12px;
            color: #94a3b8; margin: 8px 0;
        }
        .bubble a { color: #60a5fa; text-decoration: none; }
        .bubble a:hover { text-decoration: underline; }

        /* ---- 输入区域 ---- */
        .input-area {
            border-top: 1px solid #334155; padding: 16px 24px;
            background: #1e293b;
        }
        .input-wrapper {
            max-width: 800px; margin: 0 auto;
            display: flex; gap: 8px;
        }
        .input-wrapper textarea {
            flex: 1; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #334155; background: #0f172a;
            color: #e2e8f0; font-size: 14px; resize: none;
            font-family: inherit; outline: none; min-height: 48px;
            line-height: 1.5;
        }
        .input-wrapper textarea:focus { border-color: #3b82f6; }
        .input-wrapper button {
            padding: 12px 24px; border-radius: 8px; border: none;
            background: #3b82f6; color: #fff; font-size: 14px;
            cursor: pointer; font-weight: 500; transition: background 0.2s;
            white-space: nowrap;
        }
        .input-wrapper button:hover { background: #2563eb; }
        .input-wrapper button:disabled { background: #475569; cursor: not-allowed; }

        /* ---- 加载指示器 ---- */
        .typing-indicator {
            display: flex; gap: 4px; align-items: center; padding: 4px 0;
        }
        .typing-indicator span {
            width: 8px; height: 8px; border-radius: 50%;
            background: #8b5cf6; animation: bounce 1.4s ease infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.6); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>🤖 AI Agent Chat</h1>
        <div class="subtitle">SSE 流式输出 · Markdown 渲染 · 代码高亮 · 消息历史</div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="messages" id="messages"></div>
    </div>

    <div class="input-area">
        <div class="input-wrapper">
            <textarea id="input" rows="1" placeholder="输入消息，Enter 发送..."></textarea>
            <button id="sendBtn" onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        // ---- 会话管理 ----
        // 每个浏览器标签页使用一个随机 session_id
        const SESSION_ID = 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8-dom.getEl);
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('sendBtn');
        const chatContainer = document.getElementById('chatContainer');

        // 当前正在流式输出的消息元素（用于追加 token）
        let currentAssistantMsg = null;
        let isStreaming = false;

        // 配置 marked.js
        marked.setOptions({
            breaks: true,           // 支持换行符
            gfm: true,              // 启用 GitHub Flavored Markdown
        });

        // ---- 辅助函数 ----

        /** 自动调整 textarea 高度 */
        function autoResize() {
            inputEl.style.height = 'auto';
            inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
        }
        inputEl.addEventListener('input', autoResize);

        /** Enter 发送，Shift+Enter 换行 */
        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        /** 滚动到底部 */
        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        /** 创建消息 DOM 元素 */
        function createMessageEl(role) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="avatar">${role === 'user' ? '🧑' : '🤖'}</div>
                <div class="bubble"></div>
            `;
            messagesEl.appendChild(div);
            scrollToBottom();
            return div.querySelector('.bubble');
        }

        /** 渲染 Markdown 并高亮代码块 */
        function renderMarkdown(text) {
            // 将 Markdown 转为 HTML
            const html = marked.parse(text);
            // 创建一个临时容器来应用高亮
            const container = document.createElement('div');
            container.innerHTML = html;
            // 对每个代码块应用 highlight.js
            container.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
            return container.innerHTML;
        }

        // ---- 核心功能 ----

        /** 发送消息 */
        async function sendMessage() {
            const text = inputEl.value.trim();
            if (!text || isStreaming) return;

            // 清空输入框
            inputEl.value = '';
            autoResize();

            // 显示用户消息
            const userBubble = createMessageEl('user');
            userBubble.textContent = text;

            // 创建 AI 消息占位（显示打字指示器）
            currentAssistantMsg = createMessageEl('assistant');
            currentAssistantMsg.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

            // 禁用发送按钮
            isStreaming = true;
            sendBtn.disabled = true;

            try {
                // 发起 SSE 请求
                const response = await fetch(`/api/chat/${SESSION_ID}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `prompt=${encodeURIComponent(text)}`,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                // 读取 SSE 流
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let fullText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    // 解析 SSE 数据行
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';  // 保留不完整的行

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6).trim();
                            if (!data) continue;
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.type === 'done') {
                                    // 流式输出结束
                                    break;
                                }
                                if (parsed.token) {
                                    fullText += parsed.token;
                                    // 实时更新气泡内容
                                    currentAssistantMsg.innerHTML = renderMarkdown(fullText);
                                    scrollToBottom();
                                }
                            } catch (e) {
                                // 忽略解析错误
                            }
                        }
                    }
                }

                // 最终渲染（确保所有代码块高亮）
                currentAssistantMsg.innerHTML = renderMarkdown(fullTexthed);

            } catch (err) {
                currentAssistantMsg.innerHTML = `<p style="color: #f87171;">⚠️ 请求失败：${err.message}</p>`;
            } finally {
                isStreaming = false;
                sendBtn.disabled = false;
                scrollToBottom();
            }
        }

        // ---- 页面加载时恢复历史消息 ----
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const res = await fetch(`/api/history/${SESSION_ID}`);
                const history = await res.json();
                for (const msg of history) {
                    const bubble = createMessageEl(msg.role);
                    if (msg.role === 'user') {
                        bubble.textContent = msg.content;
                    } else {
                        bubble.innerHTML = renderMarkdown(msg.content);
                    }
                }
            } catch (e) {
                console.log('无历史消息');
            }
        });
    </script>

</body>
</html>
```

### 2.4 依赖文件：`requirements.txt`

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
```

### 2.5 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python main.py

# 3. 打开浏览器访问
# http://localhost:8000
```

### 2.6 接入真实 LLM

将 `main.py` 中的 `agent_stream()` 函数替换为真实 LLM 调用即可。以下是接入 OpenAI 兼容 API 的示例：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",  # 或任何兼容的 API 地址
)

async def agent_stream(prompt: str, history: list[dict]) -> AsyncGenerator[str, None]:
    messages = [{"role": "system", "content": "你是一个有用的 AI 助手。"}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield json.dumps({"token": chunk.choices[0].delta.content})

    yield json.dumps({"type": "done"})
```

---

## 3. 部署方案

### 3.1 本地开发

```bash
# 开发模式（热重载）
python main.py
# 或
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

开发时建议：
- 使用 `--reload` 实现代码修改后自动重启
- 前端用 CDN 加载 marked.js / highlight.js，无需构建工具
- SQLite 文件自动创建，零配置

### 3.2 容器化（Docker）

**`Dockerfile`：**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY main.py .
COPY static/ ./static/

# 暴露端口
EXPOSE 8000

# 启动（生产模式，多 worker）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**`docker-compose.yml`（含反向代理）：**

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chat_history.db:/app/chat_history.db  # 持久化 SQLite
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - app
```

**`nginx.conf`（SSE 需要禁用缓冲）：**

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://app:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 必须禁用缓冲
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }
}
```

### 3.3 上云要点

| 维度 | 注意点 |
|---|---|
| **SSE 与负载均衡** | 确保 LB 不缓冲响应体。AWS ALB / NLB、Nginx 需设置 `proxy_buffering off` |
| **WebSocket 备用** | 部分云 LB 对长连接有超时（如 60s），SSE 一般无此问题；若遇超时，可降级为 WebSocket |
| **数据库** | SQLite 不适合多进程写入。生产环境换 PostgreSQL，或用 Redis 做消息队列 + 异步写入 |
| **会话管理** | 当前用随机 session_id，生产环境应接入 JWT / Session 中间件 |
| **CORS** | 前后端分离部署时，FastAPI 需配置 `CORSMiddleware` |
| **日志与监控** | 接入结构化日志（structlog）和 Prometheus 指标 |
| **速率限制** | 对 `/api/chat` 接口做限流，防止滥用 |
| **LLM API Key** | 通过环境变量注入，不要硬编码 |
| **静态资源** | 生产环境将 marked.js / highlight.js 从 CDN 下载到本地部署，避免外网依赖 |

### 3.4 生产架构参考

```
[用户浏览器] → CDN → Nginx (反向代理) → FastAPI (Uvicorn workers)
                                              ├── LLM API (OpenAI / 自建)
                                              ├── PostgreSQL (消息持久化)
                                              └── Redis (缓存 / 速率限制)
```

对于高并发场景，将 FastAPI 水平扩展（多容器 + LB），数据库换 PostgreSQL，引入 Celery 做异步任务队列（参考第一节的 SSE + Celery 架构）。

---

## 总结

本文档覆盖了构建 AI Agent Web 应用的三个核心层面：

1. **前端方案**：SSE 流式输出（优于 WebSocket 的简单场景）、marked.js + highlight.js 实现 Markdown/代码渲染、SQLite 消息持久化
2. **最小可运行示例**：约 120 行 Python 后端 + 单页 HTML 前端，零构建工具，pip install 即可运行
3. **部署方案**：从 `uvicorn --reload` 开发到 Docker 容器化再到生产上云的全链路要点

整个示例的核心设计原则是**最小依赖、可替换**——后端 `agent_stream()` 是唯一的 LLM 接入点，替换它即可对接任何 AI 模型或 Agent 框架，前端和后端其他部分无需改动。