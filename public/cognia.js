// Cognia 匿名 user_id 持久化（clarifications Q6）。
// 读 localStorage 的 cognia_user_id，没有则生成 UUID 写入；同步写 cookie，
// 让后端从 HTTP_COOKIE 读回，实现 New Chat 之间保持同一 user_id（session 与 user 分离）。
(function () {
  var KEY = "cognia_user_id";
  var uid = null;
  try {
    uid = localStorage.getItem(KEY);
  } catch (e) {
    // localStorage 不可用（隐私模式等）时降级为仅 cookie
  }
  if (!uid) {
    uid = (window.crypto && crypto.randomUUID && crypto.randomUUID()) ||
      "uid-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
    try {
      localStorage.setItem(KEY, uid);
    } catch (e) {
      // 写 localStorage 失败不阻断，cookie 仍会写入（本次会话内有效）
    }
  }
  // 同步写 cookie（path=/，长期有效），后端从 session.environ["HTTP_COOKIE"] 读回
  document.cookie = KEY + "=" + encodeURIComponent(uid) +
    "; path=/; max-age=31536000; SameSite=Lax";
})();
