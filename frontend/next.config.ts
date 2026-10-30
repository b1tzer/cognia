import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 允许 docker 浏览器容器（remote-browser，通过网桥 IP 172.23.0.1）访问 dev
  // server 的 HMR/WebSocket。否则浏览器经非 localhost Origin 访问时会被
  // Next.js 的 allowedDevOrigins 校验拒绝，导致客户端 hydration 失败、页面卡在
  // 「加载中…」。
  allowedDevOrigins: ["172.23.0.1"],
};

export default nextConfig;
