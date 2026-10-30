// 匿名 user_id：localStorage 持久化（clarifications Q6），首次生成 UUID，
// localStorage 不可用时降级为会话内临时标识（模块级单例，保证会话内稳定）。
const USER_ID_KEY = "cognia:userId";

let fallbackUserId: string | null = null;

// 生成 UUID v4。crypto.randomUUID 仅在安全上下文（HTTPS / localhost）可用，
// 通过 IP 访问（如 docker 网桥 172.23.0.1）时会抛 not a function，故降级用
// getRandomValues（Web Crypto，所有上下文可用）手动拼 UUID v4。
export function generateUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const buf = new Uint8Array(16);
  crypto.getRandomValues(buf);
  buf[6] = (buf[6] & 0x0f) | 0x40; // version 4
  buf[8] = (buf[8] & 0x3f) | 0x80; // variant 10xx
  const hex = Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function getOrCreateUserId(): string {
  if (typeof window === "undefined") return "";
  try {
    let id = window.localStorage.getItem(USER_ID_KEY);
    if (!id) {
      id = generateUuid();
      window.localStorage.setItem(USER_ID_KEY, id);
    }
    return id;
  } catch {
    if (!fallbackUserId) fallbackUserId = generateUuid();
    return fallbackUserId;
  }
}
