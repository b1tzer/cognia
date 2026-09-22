"""消息反馈路由（/feedback：点赞 / 点踩持久化）。

与 AG-UI 接入层解耦，属独立 HTTP 端点。权威数据在 PostgreSQL 的
`message_feedback` 表。前端经 Next.js 代理转发，user_id 显式传参。

语义（对齐 ChatGPT）：点赞 / 点踩 / 取消 / 切换。
- PUT    /feedback  → 写入或切换反馈（up / down）。
- DELETE /feedback  → 取消反馈。
- GET    /feedback  → 读某会话全部反馈，供前端刷新后恢复高亮。
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cognia import feedback

router = APIRouter()


class UpsertFeedbackRequest(BaseModel):
    """写入 / 切换反馈请求体。"""

    thread_id: str
    message_id: str
    user_id: str
    feedback: str


class DeleteFeedbackRequest(BaseModel):
    """取消反馈请求体。"""

    thread_id: str
    message_id: str
    user_id: str


def _feedback_available(request: Request) -> bool:
    """Postgres 降级时关闭反馈持久化，返回 False。"""
    return bool(getattr(request.app.state, "threads_available", False))


@router.put("/feedback")
async def upsert_feedback_endpoint(
    payload: UpsertFeedbackRequest, request: Request
):
    """写入 / 切换一条反馈（up / down）。"""
    if not _feedback_available(request):
        raise HTTPException(status_code=503, detail="反馈持久化暂不可用（Postgres 未连接）")
    try:
        return await feedback.upsert_feedback(
            request.app.state.pool,
            payload.thread_id,
            payload.message_id,
            payload.user_id,
            payload.feedback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/feedback")
async def delete_feedback_endpoint(
    payload: DeleteFeedbackRequest, request: Request
):
    """取消一条反馈（幂等）。"""
    if not _feedback_available(request):
        raise HTTPException(status_code=503, detail="反馈持久化暂不可用（Postgres 未连接）")
    await feedback.delete_feedback(
        request.app.state.pool,
        payload.thread_id,
        payload.message_id,
        payload.user_id,
    )
    return {"deleted": True}


@router.get("/feedback")
async def list_feedback_endpoint(request: Request, thread_id: str, user_id: str):
    """读某会话全部反馈，返回 `{message_id: feedback}`。"""
    if not _feedback_available(request):
        return {}
    if not thread_id or not user_id:
        raise HTTPException(status_code=400, detail="缺少 thread_id / user_id 参数")
    return await feedback.list_feedback(
        request.app.state.pool, thread_id, user_id
    )
