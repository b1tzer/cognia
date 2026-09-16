import { NextRequest, NextResponse } from "next/server";

// 知识版图 API 走 Python FastAPI 后端（cognia/server.py 的 GET /knowledge-map）。
// 浏览器不能直接跨域访问 8123，由 Next.js 服务端统一代理转发。
const BACKEND_URL = process.env.AGENT_URL || "http://localhost:8123";

export async function GET(req: NextRequest) {
  const user_id = req.nextUrl.searchParams.get("user_id");
  const target = new URL("/knowledge-map", BACKEND_URL);
  if (user_id) target.searchParams.set("user_id", user_id);

  const resp = await fetch(target, { cache: "no-store" });
  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: {
      "content-type": resp.headers.get("content-type") || "application/json",
    },
  });
}
