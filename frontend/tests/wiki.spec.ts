import { test, expect } from "@playwright/test";
import { mockApi } from "./helpers/mock-api";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Wiki" }).click();
  await expect(page.getByText("暂无 wiki 页面", { exact: true })).toBeVisible();
});

test("新建页面时校验标题与正文不能为空", async ({ page }) => {
  await page.getByRole("button", { name: "+ 新建页面" }).click();

  // 空标题 → 提示标题不能为空
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("标题不能为空", { exact: true })).toBeVisible();

  // 有标题、空正文 → 提示正文不能为空
  await page.getByPlaceholder("标题").fill("我的第一篇 Wiki");
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("正文不能为空", { exact: true })).toBeVisible();
});

test("填写完整后保存并显示页面内容", async ({ page }) => {
  await page.getByRole("button", { name: "+ 新建页面" }).click();
  await page.getByPlaceholder("标题").fill("我的第一篇 Wiki");
  await page.getByPlaceholder("用 Markdown 撰写正文…").fill("# Hello Cognia");
  await page.getByRole("button", { name: "保存" }).click();

  // 保存后回到查看模式，标题出现在详情头部
  await expect(
    page.getByRole("heading", { name: "我的第一篇 Wiki" }),
  ).toBeVisible();
  // 左侧列表同步出现该页面标题
  await expect(
    page.locator("aside").getByText("我的第一篇 Wiki", { exact: true }),
  ).toBeVisible();
  // 保存的 Markdown 正文被渲染（# Hello Cognia → h1），确保正文真正落库并展示，
  // 而非仅标题保存成功、正文丢失。
  await expect(
    page.getByRole("heading", { name: "Hello Cognia" }),
  ).toBeVisible();
});

test("取消编辑返回查看模式且不保存", async ({ page }) => {
  await page.getByRole("button", { name: "+ 新建页面" }).click();
  await page.getByPlaceholder("标题").fill("不应保存的页面");
  await page.getByRole("button", { name: "取消" }).click();

  // 回到空态引导文案，且标题没有落库
  await expect(page.getByText("选择或新建一个 wiki 页面")).toBeVisible();
  await expect(
    page.getByText("不应保存的页面", { exact: true }),
  ).toHaveCount(0);

  // 重新加载页面（重新拉取服务端列表），确认取消操作没有把草稿持久化到后端。
  await page.reload();
  await page.getByRole("button", { name: "Wiki" }).click();
  await expect(page.getByText("暂无 wiki 页面", { exact: true })).toBeVisible();
});
