// 匿名 user_id：localStorage 持久化（clarifications Q6），首次生成 UUID，
// localStorage 不可用时降级为会话内临时标识（模块级单例，保证会话内稳定）。
const USER_ID_KEY = "cognia:userId";

let fallbackUserId: string | null = null;

export function getOrCreateUserId(): string {
  if (typeof window === "undefined") return "";
  try {
    let id = window.localStorage.getItem(USER_ID_KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem(USER_ID_KEY, id);
    }
    return id;
  } catch {
    if (!fallbackUserId) fallbackUserId = crypto.randomUUID();
    return fallbackUserId;
  }
}
