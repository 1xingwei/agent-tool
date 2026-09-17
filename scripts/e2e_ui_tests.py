"""Streamlit 应用 UI 的浏览器端到端场景。

扩展 ``scripts/smoke_live_app.py``（单次聊天往返），加入一小组
覆盖关键用户旅程的用例，这些旅程曾因 Streamlit 版本升级或客户端 /
schema 变更而被悄悄破坏。它像用户一样驱动真实浏览器走完
应用，因此能捕获 pytest 套件（mock 了传输层）和 docker CI 健康检查
无法发现的破坏。

用法：
    uv run --with playwright python scripts/e2e_ui_tests.py [URL] [scenario ...]

运行全部（默认），或仅运行指定场景：
    uv run --with playwright python scripts/e2e_ui_tests.py            # 全部，针对 localhost
    uv run --with playwright python scripts/e2e_ui_tests.py chat feedback
    uv run --with playwright python scripts/e2e_ui_tests.py https://my-app.example.com
    uv run --with playwright python scripts/e2e_ui_tests.py --list

场景不硬编码应用 URL——而是将其作为参数传入——因此同一套件
可用于任何部署。默认指向本地运行的应用
（http://localhost:8501）；传入不同 URL（或设置 ``LIVE_APP_URL``）即可指向
已部署的实例。因此一套套件两用：PR 前针对本地
``USE_FAKE_MODEL=true`` 服务 + ``streamlit run`` 运行，针对
已部署 URL 运行以检查生产环境。

默认场景使用 ``fake`` 模型，因此本地运行无需 API key
也不会发起真实 LLM 调用。要端到端验证真实模型，请运行
可选的 ``live_model`` 场景并传入 ``--model=<name>``（或 ``E2E_LIVE_MODEL``）；
它会在 Settings 中选择该模型并发送一条简短 prompt（一次廉价的 LLM 调用）：
    uv run --with playwright python scripts/e2e_ui_tests.py --model=gpt-5-nano live_model

注意：
  - 场景只发送简短 prompt，因此即使实时运行也很廉价。
  - feedback 场景验证组件能渲染且可交互（即
    Streamlit 升级会破坏的部分），但不提交评分——点击星标会通过后端
    写入 LangSmith，这会在每次监控运行时污染生产项目。（套件中
    其他内容不触碰 LangSmith：普通聊天/历史不会追踪，除非显式启用追踪。）

需要 Playwright 能找到的 Chromium：要么 ``playwright install chromium``，
要么通过 PLAYWRIGHT_BROWSERS_PATH 预置浏览器（如 Claude Code
云环境中）。若所有选中场景通过则退出码为 0，否则为 1，并
对任何失败场景在 CWD 旁写入 ``e2e_<scenario>_failure.png``。
"""

import os
import sys
import time
import urllib.parse

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

DEFAULT_URL = "http://localhost:8501"
# 指向 Claude Code 云环境中预置浏览器的稳定符号链接，
# 当已安装的 playwright 自带浏览器构建缺失时用作回退。
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"

# 一个简单的无工具 agent 使消息 thread 保持确定性（每轮恰好一条
# assistant 消息），因此基于计数的等待无论背后是哪个模型都可靠。
# 专门测试 agent 选择的场景会覆盖此设置。
CHAT_AGENT = "chatbot"

# resume 场景需要完整的多轮历史才能在 checkpoint 往返后存活。
# @entrypoint 风格的 chatbot 仅通过 /history 暴露其最后一条回复，因此
# 改用 StateGraph agent（其 messages 通道累积每一轮）。
HISTORY_AGENT = "research-assistant"

WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4

CHAT_INPUT = '[data-testid="stChatInput"] textarea'
CHAT_MESSAGE = '[data-testid="stChatMessage"]'


class E2EError(Exception):
    """场景断言失败。"""


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 浏览器 / 应用辅助函数
# --------------------------------------------------------------------------- #
def launch_browser(p) -> Browser:
    try:
        return p.chromium.launch()
    except Exception:
        executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
        if not os.path.exists(executable):
            raise
        log(f"bundled browser missing - falling back to {executable}")
        return p.chromium.launch(executable_path=executable)


def build_url(base_url: str, **params: str) -> str:
    """将查询参数合并到 base_url，保留其已有的参数。"""
    parts = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    query.update({k: v for k, v in params.items() if v is not None})
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def wake_if_sleeping(page: Page) -> None:
    """Streamlit Community Cloud 会为休眠的应用显示唤醒界面。"""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # 未休眠
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def open_app(browser: Browser, url: str, agent: str | None = None) -> Page:
    """在应用上打开全新的浏览器上下文，并等待其可交互。"""
    if agent:
        url = build_url(url, agent=agent)
    page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    wake_if_sleeping(page)
    # 聊天输入框出现意味着 Streamlit 已启动、websocket 已连接、
    # 且应用脚本已运行（它在 agent/model 初始化后渲染）。
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=WAKE_TIMEOUT_S * 1_000)
    return page


def query_param(page: Page, key: str) -> str | None:
    query = urllib.parse.urlsplit(page.url).query
    return dict(urllib.parse.parse_qsl(query)).get(key)


def message_texts(page: Page) -> list[str]:
    return [(t or "").strip() for t in page.locator(CHAT_MESSAGE).all_inner_texts()]


def send_message(page: Page, text: str) -> None:
    chat_input = page.locator(CHAT_INPUT)
    chat_input.fill(text)
    chat_input.press("Enter")


def wait_for_response(
    page: Page,
    prompt: str,
    min_count: int,
    timeout_s: int = RESPONSE_TIMEOUT_S,
) -> str:
    """等待 `prompt` 的新 assistant 回复出现并停止流式输出。

    使用消息计数（而非文本）来检测新一轮，因为 fake 模型
    每轮回复相同文本。当最后一条消息非空、不是 prompt 本身、
    且持续 STREAM_SETTLE_S 不再变化时，视为回复完成。
    """
    deadline = time.monotonic() + timeout_s
    last_text, stable_since = "", None
    while time.monotonic() < deadline:
        texts = message_texts(page)
        # 在开始信任最后一条消息之前，prompt 必须已作为更早的消息落地，
        # 且 assistant 回复在其之后。
        if len(texts) >= min_count and any(prompt in t for t in texts[:-1]):
            text = texts[-1]
            if text and text != prompt:
                if text == last_text:
                    if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                        return text
                else:
                    last_text, stable_since = text, time.monotonic()
        time.sleep(1)
    raise E2EError(f"no stable assistant reply to {prompt!r} (>= {min_count} msgs) in {timeout_s}s")


def open_settings(page: Page) -> None:
    """打开 Settings 弹出框。它在 rerun 之间保持打开，因此打开一次并
    在关闭前完成所有设置交互——再次点击会将其关闭。"""
    page.get_by_role("button", name="Settings").first.click()
    page.locator('[data-testid="stSelectbox"]').first.wait_for(state="visible", timeout=15_000)


def selectbox_value(page: Page, label: str) -> str | None:
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
    return box.get_by_role("combobox").get_attribute("value")


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
def scenario_chat(browser: Browser, base_url: str) -> None:
    """基线：发送一条消息，一条 assistant 回复流式返回并稳定。"""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"assistant replied ({len(reply)} chars): {reply[:80]!r}")


def scenario_multi_turn_resume(browser: Browser, base_url: str) -> None:
    """两轮对话，然后在新会话中通过 Share 链接恢复它。

    覆盖 thread 持久化、恢复时按 agent 感知的 /history 拉取，以及
    Share/resume 对话框——#330 修复的对话框回归问题会在此失败，
    因为构建分享 URL 时会报错而不是渲染出链接。
    """
    turn1 = "First turn: remember the number 7"
    turn2 = "Second turn: what number did I mention?"
    page = open_app(browser, base_url, agent=HISTORY_AGENT)
    thread_id = query_param(page, "thread_id")
    if not thread_id:
        raise E2EError("thread_id was not published to the URL on load")

    send_message(page, turn1)
    wait_for_response(page, turn1, min_count=2)
    send_message(page, turn2)
    wait_for_response(page, turn2, min_count=4)

    # 打开 Share/resume 对话框并读取它构建出的可分享 URL。对话框
    # 框架会先于 Streamlit 流式写入其 markdown 出现，所以要等待
    # 代码块的文本，而不是在对话框一打开就读取它。
    page.get_by_role("button", name="Share/resume chat").first.click()
    dialog = page.locator('[role="dialog"]')
    dialog.wait_for(state="visible", timeout=15_000)
    code = dialog.locator('[data-testid="stCode"] code')
    share_url = ""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if code.count() and (share_url := code.first.inner_text().strip()):
            break
        page.wait_for_timeout(500)
    if not share_url:
        raise E2EError("Share/resume dialog rendered no chat URL (share_chat_dialog broken?)")
    if thread_id not in share_url or "agent=" not in share_url:
        raise E2EError(f"share URL missing thread_id/agent: {share_url!r}")
    log(f"share URL: {share_url}")

    # 在一个全新会话中恢复（无共享状态），并确认 thread 已从
    # history 重新水合：StateGraph agent 会持久化每一轮，所以我们的
    # 两条 prompt 都应重放。（全新/空 thread 则只会显示
    # agent 的欢迎消息。）
    resumed = open_app(browser, share_url)
    if query_param(resumed, "thread_id") != thread_id:
        raise E2EError("resumed session did not carry the original thread_id from the share URL")
    joined = "\n".join(message_texts(resumed))
    for needle in (turn1, turn2):
        if needle not in joined:
            raise E2EError(f"resumed thread did not replay {needle!r} - history not restored")
    log(f"resumed thread {thread_id} replayed both prior turns from history")


def scenario_settings_selectors(browser: Browser, base_url: str) -> None:
    """Settings 弹出框：model + agent 选择框渲染出来，并且将
    agent 切换到非默认项会同步到 ?agent= URL 参数中。"""
    page = open_app(browser, base_url)  # 默认 agent，所以 ?agent= 初始时不存在
    open_settings(page)

    model = selectbox_value(page, "LLM to use")
    if not model:
        raise E2EError("LLM selectbox rendered without a selected model")
    log(f"model selectbox shows: {model!r}")

    default_agent = selectbox_value(page, "Agent to use")
    agents = page.locator('[data-testid="stSelectbox"]').filter(has_text="Agent to use")
    agents.get_by_role("combobox").click()
    options = page.locator('[role="option"]')
    options.first.wait_for(state="visible", timeout=10_000)
    all_agents = [options.nth(i).inner_text().strip() for i in range(options.count())]
    if len(all_agents) < 2:
        raise E2EError(f"expected multiple agents to choose from, saw {all_agents}")
    # 选择任意非默认的 agent；默认项会从 URL 中移除，
    # 所以切换到非默认项才能证明查询参数绑定生效。
    target = next(a for a in all_agents if a != default_agent)
    options.filter(has_text=target).first.click()
    page.wait_for_timeout(1_500)

    if selectbox_value(page, "Agent to use") != target:
        raise E2EError(f"agent selectbox did not switch to {target!r}")
    if query_param(page, "agent") != target:
        raise E2EError(f"?agent= URL param is {query_param(page, 'agent')!r}, expected {target!r}")
    log(f"agent switched {default_agent!r} -> {target!r} and synced to the URL")


def scenario_feedback(browser: Browser, base_url: str) -> None:
    """回复之后，星标反馈组件会渲染出来且可交互。

    断言该组件的结构——即 Streamlit 升级会破坏的部分——而不
    提交评分：点击星标会通过后端写入 LangSmith，而在每次运行时
    都这样做会在监控期间污染生产 LangSmith 项目（并且会在
    后端无法访问 LangSmith 时挂起）。我们验证星标渲染出来、
    带有预期的 aria-label，并且处于启用/可点击状态。
    """
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)

    widget = page.locator('[data-testid="stFeedback"]').last
    widget.wait_for(state="visible", timeout=15_000)
    stars = widget.locator('[data-testid="stFeedbackButton"]')
    if stars.count() != 5:
        raise E2EError(f"expected a 5-star feedback widget, found {stars.count()} stars")
    labels = [stars.nth(i).get_attribute("aria-label") for i in range(5)]
    if labels != [f"{i} out of 5 stars" for i in range(1, 6)]:
        raise E2EError(f"feedback stars have unexpected aria-labels: {labels}")
    if not stars.first.is_enabled() or not stars.last.is_enabled():
        raise E2EError("feedback stars rendered but are not interactive")
    log("5-star feedback widget rendered with expected labels and is interactive")


def scenario_streaming_toggle(browser: Browser, base_url: str) -> None:
    """关闭「Stream results」后仍会通过非流式（ainvoke）路径产生
    回复——这是默认流式运行从不触及的代码路径。"""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    open_settings(page)
    toggle = page.locator('[data-testid="stCheckbox"]').filter(has_text="Stream results")
    toggle.wait_for(state="visible", timeout=10_000)
    toggle.click()  # 默认开启 -> 将其关闭
    page.keyboard.press("Escape")  # 关闭弹出框，以便能够到聊天输入框
    page.wait_for_timeout(500)

    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"non-streaming reply rendered ({len(reply)} chars)")


def scenario_new_chat(browser: Browser, base_url: str) -> None:
    """「New Chat」会开启一个全新 thread：URL 中出现新的 thread_id，
    且对话被清空。"""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)
    old_thread = query_param(page, "thread_id")

    page.get_by_role("button", name="New Chat").first.click()
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=30_000)

    deadline = time.monotonic() + 15
    while query_param(page, "thread_id") == old_thread and time.monotonic() < deadline:
        page.wait_for_timeout(500)
    new_thread = query_param(page, "thread_id")
    if not new_thread or new_thread == old_thread:
        raise E2EError(f"thread_id did not change on New Chat (still {old_thread!r})")
    if any(prompt in t for t in message_texts(page)):
        raise E2EError("previous conversation was not cleared after New Chat")
    log(f"New Chat reset thread {old_thread} -> {new_thread} and cleared the conversation")


def scenario_live_model(browser: Browser, base_url: str) -> None:
    """可选启用：在 Settings 中选择一个真实模型，并确认它能端到端
    作答。

    与其他场景（运行在 ``fake`` 模型上）不同，此项会调用真实 LLM，
    因此需要一个提供该模型的后端及其凭据。用 ``--model=<name>``
    或 ``E2E_LIVE_MODEL`` 指定模型名。它发送一条简短 prompt——
    单次廉价的 LLM 调用——并检查回复是真实的，而非 fake 模型的
    占位内容。
    """
    model = os.environ.get("E2E_LIVE_MODEL", "").strip()
    if not model:
        raise E2EError("live_model needs a model - pass --model=<name> or set E2E_LIVE_MODEL")
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    open_settings(page)
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="LLM to use")
    box.get_by_role("combobox").click()
    option = page.get_by_role("option", name=model, exact=True)
    if not option.count():
        available = page.locator('[role="option"]').all_inner_texts()
        raise E2EError(f"model {model!r} is not offered by this app; available: {available}")
    option.first.click()
    page.keyboard.press("Escape")  # 关闭弹出框，以便能够到聊天输入框
    page.wait_for_timeout(500)

    prompt = "Reply with only the word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    if "fake model" in reply.lower():
        raise E2EError(f"expected a reply from {model!r} but got the fake-model placeholder")
    log(f"live model {model!r} replied ({len(reply)} chars): {reply[:80]!r}")


# 默认套件运行在 fake 模型上，无需 API key；live_model 为
# 可选启用（会调用真实 LLM），仅在指定时或给出 --model 时运行。
SCENARIOS = {
    "chat": scenario_chat,
    "multi_turn_resume": scenario_multi_turn_resume,
    "settings_selectors": scenario_settings_selectors,
    "feedback": scenario_feedback,
    "streaming_toggle": scenario_streaming_toggle,
    "new_chat": scenario_new_chat,
    "live_model": scenario_live_model,
}
DEFAULT_SCENARIOS = [name for name in SCENARIOS if name != "live_model"]


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join(SCENARIOS))
        return

    url = os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    names = []
    for arg in args:
        if arg in SCENARIOS:
            names.append(arg)
        elif arg.startswith("--model="):
            os.environ["E2E_LIVE_MODEL"] = arg.split("=", 1)[1]
        elif arg.startswith(("http://", "https://")):
            url = arg
        else:
            print(f"unknown argument: {arg!r} (scenarios: {', '.join(SCENARIOS)})")
            sys.exit(2)
    if not names:
        # 默认运行：fake 模型套件，仅当设置了模型时才加上 live_model。
        names = list(DEFAULT_SCENARIOS)
        if os.environ.get("E2E_LIVE_MODEL"):
            names.append("live_model")

    log(f"target: {url}")
    log(f"scenarios: {', '.join(names)}")

    results: dict[str, str] = {}
    with sync_playwright() as p:
        browser = launch_browser(p)
        for name in names:
            log(f"--- {name} ---")
            start = time.monotonic()
            try:
                SCENARIOS[name](browser, url)
                results[name] = "PASS"
                log(f"PASS: {name} ({time.monotonic() - start:.0f}s)")
            except Exception as e:
                results[name] = f"FAIL: {e}"
                log(f"FAIL: {name}: {e}")
                _screenshot_failure(browser, name)
        browser.close()

    log("=" * 60)
    for name in names:
        log(f"{results[name].split(':')[0]:<4} {name}: {results[name]}")
    failed = [n for n, r in results.items() if not r.startswith("PASS")]
    if failed:
        log(f"{len(failed)} of {len(names)} scenario(s) failed: {', '.join(failed)}")
        sys.exit(1)
    log(f"all {len(names)} scenario(s) passed")


def _screenshot_failure(browser: Browser, name: str) -> None:
    """为失败场景保存最后打开页面的截图。"""
    path = f"e2e_{name}_failure.png"
    try:
        contexts = browser.contexts
        if contexts and contexts[-1].pages:
            contexts[-1].pages[-1].screenshot(path=path, full_page=True)
            log(f"screenshot saved to {path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
