import { NextRequest, NextResponse } from "next/server";

// 个人 Wiki API 走 Python FastAPI 后端（cognia/server.py 的 /wiki CRUD + summarize）。
// 浏览器不能直接跨域访问 8123，由 Next.js 服务端统一代理转发。
const BACKEND_URL = process.env.AGENT_URL || "http://localhost:8123";

async function proxy(req: NextRequest): Promise<NextResponse> {
  // 把 /api/wiki[/{page_id}[/history|/rollback|/summarize]] 映射为后端 /wiki[...]
  const path = req.nextUrl.pathname.replace(/^\/api\/wiki/, "/wiki");
  const target = new URL(path, BACKEND_URL);
  target.search = req.nextUrl.search;

  const headers: Record<string, string> = {};
  const contentType = req.headers.get("content-type");
  if (contentType) headers["content-type"] = contentType;

  let body: string | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    body = await req.text();
  }

  const resp = await fetch(target, {
    method: req.method,
    headers,
    body,
    cache: "no-store",
  });

  const text = await resp.text();
  const responseHeaders: Record<string, string> = {};
  const ct = resp.headers.get("content-type");
  if (ct) responseHeaders["content-type"] = ct;

  return new NextResponse(text, { status: resp.status, headers: responseHeaders });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const DELETE = proxy;
