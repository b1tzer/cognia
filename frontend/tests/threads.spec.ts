import { test, expect } from "@playwright/test";
import { mockApi } from "./helpers/mock-api";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();
});

test("新建会话在侧边栏新增一条", async ({ page }) => {
  await page.getByRole("button", { name: "+ 新建会话" }).click();
  await expect(page.getByText("新对话", { exact: true })).toBeVisible();
});

test("重命名会话更新侧边栏标题", async ({ page }) => {
  // hover 到会话行文本，触发操作按钮显示（Tailwind .group-hover:flex）
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "重命名" }).click();

  const input = page.getByPlaceholder("输入会话标题");
  await input.fill("改名后的会话");
  await input.press("Enter");

  await expect(page.getByText("改名后的会话", { exact: true })).toBeVisible();
});

test("删除会话需二次确认，可取消或确认删除", async ({ page }) => {
  // 先取消删除，会话仍在
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "删除" }).click();
  await expect(page.getByRole("button", { name: "取消", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();

  // 再确认删除，会话从侧边栏移除
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "删除" }).click();
  await page.getByRole("button", { name: "删除", exact: true }).click();
  await expect(page.getByText("测试会话", { exact: true })).toHaveCount(0);
});
