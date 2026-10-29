import { test, expect } from "@playwright/test";
import { mockApi } from "./helpers/mock-api";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
});

test("应用启动后首页关键内容可见", async ({ page }) => {
  // 侧边栏品牌区
  await expect(page.getByRole("heading", { name: "Cognia" })).toBeVisible();

  // 三个主视图入口（已下沉到侧边栏功能导航）
  await expect(page.getByRole("button", { name: "新建对话" })).toBeVisible();
  await expect(page.getByRole("button", { name: "知识版图" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Wiki" })).toBeVisible();

  // 会话侧边栏
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();

  // 对话输入区
  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toBeVisible();
});

test("输入内容后发送按钮由禁用变为可用", async ({ page }) => {
  const send = page.getByRole("button", { name: "发送" });
  await expect(send).toBeDisabled();

  await page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行").fill("你好");
  await expect(send).toBeEnabled();
});

test("侧边栏导航在知识版图 / Wiki 间切换，会话项回到对话", async ({ page }) => {
  // 知识版图：显示 mock 的学习目标与星系图
  await page.getByRole("button", { name: "知识版图" }).click();
  await expect(
    page.getByRole("heading", { name: "Java 并发基础" }),
  ).toBeVisible();
  await expect(page.getByRole("img", { name: /径向星系/ })).toBeVisible();

  // Wiki：空列表显示引导文案
  await page.getByRole("button", { name: "Wiki" }).click();
  await expect(page.getByText("暂无 wiki 页面", { exact: true })).toBeVisible();
  await expect(page.getByText("选择或新建一个 wiki 页面")).toBeVisible();

  // 回到对话：点击侧边栏会话项
  await page.getByText("测试会话", { exact: true }).click();
  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toBeVisible();
});

test("空态建议卡片点击后预填输入框", async ({ page }) => {
  // 初始空态（mock 会话无历史消息）即显示快速开始建议卡片
  await expect(
    page.getByRole("button", { name: "学习 Java 线程与并发" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "学习 Java 线程与并发" }).click();

  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toHaveValue("学习 Java 线程与并发");
});
