"""用真实浏览器对已部署的 Streamlit 应用进行端到端冒烟测试。

加载应用，发送一条聊天消息，并验证 assistant 回复会流式返回。
Streamlit 由 websocket 驱动，所以单纯的 HTTP 检查只能证明页面外壳
加载了——此项驱动实际的聊天往返（浏览器 -> Streamlit -> agent
服务 -> LLM -> 返回）。

用法：
    uv run --with playwright python scripts/smoke_live_app.py [URL]

URL 默认为已部署的应用，或设置 LIVE_APP_URL。本地测试：
    uv run --with playwright python scripts/smoke_live_app.py http://localhost:8501

需要 Playwright 能找到的 Chromium：要么 `playwright install chromium`，
要么通过 PLAYWRIGHT_BROWSERS_PATH 预置浏览器（如 Claude Code
云环境中那样）。通过时退出码 0，失败时退出码 1，并在失败时于
CWD 旁写入 smoke_live_app_failure.png 以供诊断。

Note: 针对已部署的应用，这会发送一条真实消息，消耗一次
（廉价的）LLM 调用。已休眠的 Streamlit Community Cloud 应用需
点击唤醒屏幕来唤醒；请为此预留几分钟。
"""

import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://agent-service-toolkit.streamlit.app/"
# 指向 Claude Code 云环境中预置浏览器的稳定符号链接，
# 当已安装的 playwright 自带的浏览器构建缺失时用作回退。
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"
TEST_MESSAGE = "Reply with the single word: pong"
CHAT_INPUT_SELECTOR = '[data-testid="stChatInput"] textarea'
WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4


def log(msg: str) -> None:
    print(f"[smoke_live_app] {msg}", flush=True)


def _dump_testids(ctx, label: str) -> None:
    try:
        ids = ctx.locator("[data-testid]").evaluate_all(
            "els => Array.from(new Set(els.map(e => e.getAttribute('data-testid')))).slice(0, 40)"
        )
        log(f"data-testids in {label} ({len(ids)}): {ids}")
    except Exception as e:
        log(f"data-testids in {label} unavailable: {e}")


def dump_diagnostics(page) -> None:
    """针对失败运行的文本诊断信息，可直接从 CI 日志中读取。"""
    try:
        log(f"page title: {page.title()!r}  url: {page.url}")
    except Exception:
        pass
    n_iframes = page.locator("iframe").count()
    log(f"iframes present: {n_iframes}")
    _dump_testids(page, "top frame")
    for i in range(n_iframes):
        _dump_testids(page.frame_locator("iframe").nth(i), f"iframe[{i}]")
    try:
        body = (page.locator("body").inner_text(timeout=2_000) or "").strip()
        log(f"top-frame visible text ({len(body)} chars): {body[:600]!r}")
    except Exception:
        pass


def fail(page, reason: str) -> None:
    log(f"FAIL: {reason}")
    try:
        dump_diagnostics(page)
    except Exception as e:
        log(f"diagnostics unavailable: {e}")
    try:
        page.screenshot(path="smoke_live_app_failure.png", full_page=True)
        log("screenshot saved to smoke_live_app_failure.png")
    except Exception:
        pass
    sys.exit(1)


def wake_if_sleeping(page) -> None:
    """Streamlit Community Cloud 会为已休眠的应用显示唤醒屏幕。"""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # 未休眠
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def find_app_root(page, timeout_s: int):
    """返回持有应用聊天输入框的页面或 iframe 上下文。

    本地运行会在顶层渲染应用；Streamlit Community Cloud 会将其
    包裹在 page.locator() 无法看到的 iframe 中，所以也要探测 iframe。
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if page.locator(CHAT_INPUT_SELECTOR).count():
            return page
        for i in range(page.locator("iframe").count()):
            frame = page.frame_locator("iframe").nth(i)
            if frame.locator(CHAT_INPUT_SELECTOR).count():
                return frame
        time.sleep(1)
    return None


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    log(f"target: {url}")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
            if not os.path.exists(executable):
                raise
            log(f"bundled browser missing - falling back to {executable}")
            browser = p.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        wake_if_sleeping(page)

        # 聊天输入框出现意味着 Streamlit 已启动、websocket 已就绪，
        # 且应用脚本已运行（它在 agent/model 初始化后渲染）。判断
        # 它位于顶层（本地）还是 Community Cloud 的 iframe 内，
        # 并针对该上下文运行所有应用交互。
        root = find_app_root(page, WAKE_TIMEOUT_S)
        if root is None:
            fail(page, f"chat input never appeared within {WAKE_TIMEOUT_S}s")
        log(f"app loaded, chat input visible ({'top-level' if root is page else 'iframe'})")

        chat_input = root.locator(CHAT_INPUT_SELECTOR)
        messages = root.locator('[data-testid="stChatMessage"]')
        # 欢迎消息仅在空 thread 上渲染，并在发送后的重跑中消失，
        # 因此检测基于文本，而非计数。
        pre_send_last = (messages.last.inner_text() or "").strip() if messages.count() else ""

        chat_input.fill(TEST_MESSAGE)
        chat_input.press("Enter")
        log("message sent, waiting for assistant response")

        # 预期我们的消息出现在 thread 中，随后是一条最终的
        # assistant 消息，非空、新增且稳定（流式已完成）。
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        last_text, stable_since = "", None
        while time.monotonic() < deadline:
            texts = [(t or "").strip() for t in messages.all_inner_texts()]
            if texts and any(TEST_MESSAGE in t for t in texts[:-1]):
                text = texts[-1]
                if text and text != TEST_MESSAGE and text != pre_send_last:
                    if text == last_text:
                        if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                            log(f"PASS: assistant responded ({len(text)} chars): {text[:120]!r}")
                            browser.close()
                            return
                    else:
                        last_text, stable_since = text, time.monotonic()
            time.sleep(1)

        fail(page, f"no stable assistant response within {RESPONSE_TIMEOUT_S}s")


if __name__ == "__main__":
    main()
