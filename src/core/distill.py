"""会话蒸馏：长会话的摘要压缩。

## 为什么放在 core 而不是 agents 或 service

蒸馏是**跨 agent 的通用能力**——它作用于「送给模型的对话视图」，与 agent 图
怎么建、节点怎么连无关。放在 `agents` 下会让 12 个 agent 各 import 一次；
放在 `service` 下则等于承认「只有 HTTP 调用才需要蒸馏」，而 `langgraph dev`
之类的直接调用路径同样会顶窗口。`core` 已有 `settings` / `llm` / `embeddings`
三个同类基础设施，蒸馏是第四个。

因此本模块**不 import 任何 agents 或 service 符号**，只依赖 `core.settings`
与 `core.llm`。

## 语义口径（三种「蒸馏」里选哪一种）

docs/15 §4.1 把业界做法拆成三种，工程含义差一个数量级：

| 实现 | 做什么 | 可审计性 |
|---|---|---|
| dsh compaction | 只改模型视图，日志全量保留 | 纯函数重放，最强 |
| OpenAI compaction | 清空并重写存储的历史 | 有损改写 |
| Bedrock Summary | 异步派生新的长期记忆记录 | 最终一致，20–40 秒可见 |

**本项目采用第一种**（语义等同 OpenAI compaction 的省 token 效果，
但**不清空存储**）：

- `checkpointer` 里的 `messages` **一条都不删**，`/history` 仍返回全量；
- 摘要是**留在 state 里的独立字段**，不进 `messages`，因此不会被误当成对话轮次；
- 替换只发生在**送进模型的那一份视图**上（`apply_distillation`）。

这是本模块最重要的不变量。任何一次重构如果让 `messages` 被裁剪，
就退化成 OpenAI 那种有损改写，`/history` 与 `threads` 的语义会跟着一起坏掉。

## 不致命原则

蒸馏是**可选优化**，不是功能前提。摘要模型调用失败、配置非法、
消息形状意外，全部只记 warning 并退回「不蒸馏」的旧行为——
绝不让一次流式请求因为压缩失败而 500。
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from core import settings
from core.llm import get_model

logger = logging.getLogger(__name__)


def _count_turn_pairs(messages: list[BaseMessage]) -> int:
    """按「人类发言」计数对话轮次。

    轮次不用 `len(messages)` 数，因为一条人类发言后面可能跟着
    AIMessage + ToolMessage + AIMessage 任意条，条数与轮次不成比例。
    人类发言是唯一稳定的锚点。
    """
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _message_text(message: BaseMessage) -> str:
    """把消息正文压成一行纯文本，供摘要输入使用。

    工具结果可能极长（一个 PDF 片段就几 KB），而且本身往往是噪声，
    因此截断到 `DISTILL_SNIPPET_CHARS`。
    """
    content = message.content
    if isinstance(content, list):
        parts = [
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, (str, dict))
        ]
        content = " ".join(parts)
    text = " ".join(str(content).split())
    limit = settings.DISTILL_SNIPPET_CHARS
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _role_label(message: BaseMessage) -> str:
    match message:
        case HumanMessage():
            return "用户"
        case ToolMessage():
            return f"工具({message.name or 'unknown'})"
        case AIMessage():
            return "助手"
        case SystemMessage():
            return "系统"
        case _:
            return "其他"


# ---------------------------------------------------------------------------
# 写入路径
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillDecision:
    """一次蒸馏判定：是否值得摘要，以及摘要哪一段。

    把判定与执行分开，是为了让「该不该蒸馏」成为可以单独断言的纯函数——
    它是这个模块里唯一有分支逻辑的部分，也是最容易写错的部分。
    """

    should_distill: bool
    reason: str
    to_summarize: list[BaseMessage]


def plan_distillation(
    messages: list[BaseMessage],
    summary_so_far: str = "",
) -> DistillDecision:
    """判定是否要生成摘要，以及摘要覆盖哪些消息。

    三条拒绝理由，全部是「不蒸馏」而非报错：

    - `disabled`：`DISTILL_ENABLED=False`，功能被配置关掉；
    - `below_threshold`：消息数没到 `DISTILL_TRIGGER_MESSAGES`；
    - `nothing_new`：超阈值，但超出部分不足 `DISTILL_MIN_NEW_MESSAGES`
      ——避免每次请求都重摘一遍同一段，摘要调用是有成本的。

    摘要范围 = 全部消息去掉最后 `DISTILL_KEEP_TURNS` 轮，
    且**必须**留下至少 `DISTILL_KEEP_MESSAGES` 条。两者取更保守的那个
    （留得多的一方），因为裁掉最近上下文对连贯性的伤害远大于多花点 token。
    """
    if not settings.DISTILL_ENABLED:
        return DistillDecision(False, "disabled", [])

    if len(messages) < settings.DISTILL_TRIGGER_MESSAGES:
        return DistillDecision(
            False,
            f"below_threshold({len(messages)}<{settings.DISTILL_TRIGGER_MESSAGES})",
            [],
        )

    cut = _cut_index(messages, settings.DISTILL_KEEP_TURNS)
    # 保守修正：无论轮次怎么算，最近 N 条一律不碰。
    cut = min(cut, max(0, len(messages) - settings.DISTILL_KEEP_MESSAGES))

    to_summarize = list(messages[:cut])

    if len(to_summarize) < settings.DISTILL_MIN_NEW_MESSAGES:
        return DistillDecision(
            False,
            f"nothing_new({len(to_summarize)}<{settings.DISTILL_MIN_NEW_MESSAGES})",
            [],
        )

    already = "（已有摘要）" if summary_so_far else "（首次蒸馏）"
    return DistillDecision(True, f"ok{already}", to_summarize)


def _cut_index(messages: list[BaseMessage], keep_turns: int) -> int:
    """返回下标：该下标之前的消息属于待摘要区间。

    从尾部往前数够 `keep_turns` 个人类发言，切点落在**那个人类发言之前**。
    这样保留区间永远以人类发言开头，不会把一轮对话从中间劈开
    （劈开会留下孤立的工具结果，某些 provider 直接 400）。
    """
    if keep_turns <= 0:
        return len(messages)
    seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            seen += 1
            if seen > keep_turns:
                return index
    return 0


def build_summary_prompt(
    to_summarize: list[BaseMessage],
    previous_summary: str = "",
) -> str:
    """拼出摘要请求的用户消息正文。

    分两段：已有摘要（增量合并）+ 新增对话。模型要做的是**合并**而不是
    重写——否则每蒸馏一轮，最早的上下文就再丢失一次。
    """
    lines: list[str] = []
    if previous_summary:
        lines.append("## 已有摘要（来自更早的对话，请与新内容合并）")
        lines.append(previous_summary)
        lines.append("")
    lines.append("## 新增对话")
    for message in to_summarize:
        text = _message_text(message)
        if text:
            lines.append(f"{_role_label(message)}: {text}")
    return "\n".join(lines)


def summarize_messages(
    model: BaseChatModel,
    to_summarize: list[BaseMessage],
    previous_summary: str = "",
) -> str:
    """调用模型产出摘要。

    这里**不吞异常**——由调用方（`distill_history` 节点）决定失败怎么办，
    因为「失败就放弃」是节点层的策略，不是这个函数的职责。
    """
    prompt = build_summary_prompt(to_summarize, previous_summary)
    instruction = settings.DISTILL_SUMMARY_PROMPT
    response = model.invoke([SystemMessage(content=instruction), HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, (str, dict))
        )
    return str(content).strip()


def get_distill_model() -> BaseChatModel:
    """返回用于摘要的模型。

    默认复用 `settings.DISTILL_MODEL`（留空则 `DEFAULT_MODEL`），
    与其他 agent 共享 `@cache` 的实例，不额外构造。

    这里**不需要** `get_supervisor_model` 那套 thinking 关闭处理：
    摘要请求是单轮 `invoke`，历史里没有 content-only 的 assistant 消息，
    不会触发 DeepSeek「必须回传 reasoning_content」的约束。
    """
    name: Any = settings.DISTILL_MODEL or settings.DEFAULT_MODEL
    return get_model(name)  # type: ignore[return-value]


def distill_history(state: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
    """图节点：超阈值时为**视图**生成摘要，但不碰 checkpoint 里的原文。

    ## 这里为什么不返回 `RemoveMessage`

    直觉做法是返回 `[RemoveMessage(id=...) for m in to_summarize]`，让
    `add_messages` 把旧消息从 state 里删掉——这是 LangGraph 文档里的标准写法，
    也是 OpenAI compaction 那种「重写存储」的做法。

    **但那条路与本项目的要求相反。** 实测（`var/probe_p07_e2e.py`，12 轮对话）：

    ```text
    模型历次收到的消息数: [1, 3, 5, 7, 9, 5, 7, 9, 5, 7, 9, 5]
    checkpoint 里的消息数: 6      ← 历史被真的删了
    ```

    `RemoveMessage` 由 reducer 在本节点返回后立即执行，写进 checkpoint 的
    就是裁剪后的短列表。后果有两个：`/history` 再也拿不到原文（违反 docs/15
    「原消息可查」），而且**摘要里的信息在存储里没有对应的原文可回溯**，
    一旦摘要写错就永久丢失（有损改写）。

    所以本节点**只写 `distilled_summary`**，一条消息都不删。摘要与全量原文
    一起躺在 checkpoint 里；「折叠」这件事推迟到读取时由
    `apply_distillation` 对**送给模型的那一份视图**做。这样：

    - `/history` 与 `threads` 语义不变，仍看到全量；
    - 摘要可回溯、可重放、可换提示词重摘，因为它依赖的原文一直还在；
    - 代价是存储不省——这是有意付的，省 token 与省存储本来就不是一回事。

    返回 part 里**必须**含 `messages` 键——`AgentState` 继承 `MessagesState`，
    缺键会被类型检查拦下（本项目已踩过的坑，见 `code_reviewer._recalled`）。
    """
    summary_so_far = str(state.get("distilled_summary") or "")
    messages: list[BaseMessage] = list(state.get("messages") or [])

    decision = plan_distillation(messages, summary_so_far)
    if not decision.should_distill:
        logger.debug(f"跳过会话蒸馏：{decision.reason}")
        return {"messages": []}

    try:
        model = get_distill_model()
        summary = summarize_messages(model, decision.to_summarize, summary_so_far)
    except Exception as e:  # noqa: BLE001 —— 蒸馏失败不能影响主流程
        logger.warning(f"会话蒸馏失败，本轮退回未压缩行为：{e}")
        return {"messages": []}

    if not summary:
        logger.warning("会话蒸馏产出了空摘要，放弃本轮压缩")
        return {"messages": []}

    logger.info(
        f"会话蒸馏：已将 {len(decision.to_summarize)} 条历史消息概括为 "
        f"{len(summary)} 字摘要；原文全部保留在 checkpoint 中"
    )
    return {"messages": [], "distilled_summary": summary}


# ---------------------------------------------------------------------------
# 读取路径
# ---------------------------------------------------------------------------

SUMMARY_HEADER = "以下是你与用户**更早对话**的摘要（原文已从上下文中折叠掉，但服务器仍保留全量）。"


def build_summary_message(summary: str) -> HumanMessage:
    """把摘要包成一条 `HumanMessage`。

    用 HumanMessage 而不是 SystemMessage 的原因：`apply_distillation` 要把它
    放在对话最前面，而多数 provider 对「非首位 SystemMessage」的处理各不相同；
    人类消息则任何位置都合法。措辞上明确标注是摘要，避免模型误以为用户刚说了这些。
    """
    return HumanMessage(content=f"{SUMMARY_HEADER}\n\n{summary}")


def apply_distillation(
    messages: list[BaseMessage],
    summary: str,
    keep_messages: int | None = None,
) -> list[BaseMessage]:
    """构造送给模型的视图：摘要 + 最近若干条原文。

    这是**纯函数**，不改动入参，也不碰 checkpointer。四条不变量：

    1. 无摘要 → 原样返回（未蒸馏过的会话行为完全不变）；
    2. `keep_messages` 为 0 或负数 → 原样返回；
    3. 消息数不超过 `keep_messages` → 原样返回，此时摘要若存在说明
       消息被外部改过，宁可给全量也不给一个缺了正文的视图；
    4. 结果**一定以 Human 开头**——切点会前移直到落在一个人类发言上，
       否则孤立工具结果会让部分 provider 直接 400。

    原消息不在这里被删除，只是不出现在返回值里。
    """
    if not summary:
        return messages

    keep = settings.DISTILL_KEEP_MESSAGES if keep_messages is None else keep_messages
    if keep <= 0 or len(messages) <= keep:
        return messages

    cut = len(messages) - keep
    # 规则 4：把切点往前挪到最近的人类发言上。
    while cut > 0 and not isinstance(messages[cut], HumanMessage):
        cut -= 1

    tail = list(messages[cut:])
    if not tail:
        return messages
    return [build_summary_message(summary), *tail]
