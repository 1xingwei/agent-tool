# 代码库审查 Agent（code-reviewer）

Date: 2026-09-16 · 现状修订: 2026-09-18 · Status: 已落地
Project: agent-service-toolkit（面试展示）

> **本文口径**：原稿是 09-16 的**设计提议**，已被实现取代。2026-09-18 **按当前代码重写** ——
> 每条描述都能在仓库里指到出处；原设计中被实现推翻或始终没落地的地方，统一收进
> §9「与设计稿的差异」。涉及会话蒸馏的现状（已实现、但读取侧未接通）见
> `docs/15-记忆与检索层-对标核实与行动清单.md` §7.2.3，本文只描述接线本身。

## Goal

面向 git 仓库的**变更审查 + 代码问答** agent，用真实 git 历史做数据源，
替换项目里缺乏真实性的玩具 demo，构成面试主叙事。

## 范围与排除

- 两个技能：变更审查（主叙事）+ 代码问答（工具多样性）。
- 严格只读工具集：agent 只能读和分析。注意「只读」指**代码库** —— 它确实会往 store
  写审查结论，见 §4；这两件事不冲突。
- 排除：技术债扫描、issue 生成、自动写报告文件、代码向量检索（RAG）。

> ⚠️ **排除 RAG 的理由已经过时。** 原稿的理由是「`tools.py` 的 `OpenAIEmbeddings()`
> 要 `OPENAI_API_KEY`、`./chroma_db` 也不存在，一调用即崩」。该前提已于 2026-09-18 消失
> （本地 embedding 可用，见 `docs/15` §7.2.1）。**现状仍未给本 agent 接 RAG**，
> 但这是**未复核的遗留决策**，不是重新论证过的结论 —— 需要时再议。

## 1. 架构

注册在 `src/agents/agents.py:38`：`"code-reviewer"` → `Agent(graph_like=code_reviewer)`，
复用 FastAPI `/stream` + Streamlit UI + checkpointer/store 全套基础设施，与其它 agent 同构。

**当前图**（`src/agents/code_reviewer.py:183-239`）：

```text
guard_input(entry) ──check_safety──┬─ unsafe → block_unsafe_content → END
                                   └─ safe   → recall_reviews → distill_history → model
                                                                                  │
                                                     pending_tool_calls ──────────┤
                                                       ├─ tools → model（回环）   │
                                                       └─ remember_review → END   ┘
```

即：**安全闸 → 记忆读 → 会话蒸馏 → 模型**，模型侧带工具回环，收尾写记忆。
图构造仍是 `StateGraph + RemainingSteps + ToolNode + pending_tool_calls`，
与 `loop_agent.py` 同一套写法。

### 1.1 状态（`AgentState`，`code_reviewer.py:21-28`）

| 字段 | 类型 | 谁写 | 为什么存在 |
|---|---|---|---|
| `messages` | `add_messages` 归约 | 全部节点 | `MessagesState` 的必填键 |
| `safety` | `SafeguardOutput` | `guard_input` | 安全判定结果，条件边据此分流 |
| `remaining_steps` | `RemainingSteps`（managed） | 运行时 | 步数预算，兜底逻辑在 `acall_model` |
| `recalled_reviews` | `str` | `recall_reviews` | 已格式化的历史结论，拼进 system prompt |
| `distilled_summary` | `str` | `distill_history` | 会话摘要。**放在 state 而不是塞进 `messages`，正是为了让 `/history` 仍能返回全量原文** |

> `MessagesState` 把 `messages` 声明为必填：节点即使不改动消息，也必须显式返回
> `{"messages": []}`，否则 pyrefly 报 `Missing required key 'messages' for TypedDict`
> （`_recalled` 的 docstring 记了这条）。
>
> 另一条硬约束：**LangGraph 会静默丢弃「state schema 里没有的键」**。给 state 加字段
> 必须在 `AgentState` 里声明同名通道，否则写入永远蒸发，且不报错、不打日志。

## 2. 节点逐个说明

| 节点 | 位置 | 做什么 | 边界与取舍 |
|---|---|---|---|
| `guard_input` | `:171` | `Safeguard().ainvoke(messages)` → `safety` | **可选**：`GROQ_API_KEY` 为空时 `Safeguard.model=None` 并直接返回 `SAFE`（`safeguard.py:92-93,112-113`），本机走的就是这条降级路径 |
| `check_safety` | `:197` | 条件边：`UNSAFE` → `unsafe`，其余 → `safe` | 用 `match` 而非 `if`，未知取值落 `safe` |
| `block_unsafe_content` | `:177` | 拼一条显式拒答 `AIMessage` → END | 不进入模型，也就不消耗模型调用 |
| `recall_reviews` | `:101` | 读路径：`store.asearch(namespace, query=最后一条人类消息, limit=3)` | 见 §4；`score is None` 一律跳过 |
| `distill_history` | `core/distill.py` | 阈值触发 LLM 摘要，产出 `distilled_summary` | **同步节点**（内部 `model.invoke`）；关闭时只做一次阈值判断并返回 `{"messages": []}` |
| `model` | `:73` | `wrap_model(model).ainvoke(state)`，即 `[SystemMessage] + state["messages"]` | `remaining_steps < 2` 且仍有 `tool_calls` → 回「Sorry, need more steps to process this request.」 |
| `tools` | `:185` | `ToolNode(tools)` | 固定 `tools → model`，回环由 `pending_tool_calls` 控制 |
| `remember_review` | `:149` | 把最终结论 `aput` 进 store | 三重守卫，见 §4 |

### 2.1 提示词组装（`wrap_model`，`:49-63`）

```text
[SystemMessage(instructions [+ 召回历史结论块])] + state["messages"]
```

`instructions`（`:34-46`）是**英文**的，末尾一句 `Chat content in the user's language`
把回答语言交给用户语言决定。这一点是刻意的：`docs/14` 的中文化只覆盖注释与
docstring，**不译提示词**。召回到历史结论时，会在 system 后追加一段中文说明，
要求模型区分「哪些是历史结论、哪些是本次新发现」。

## 3. 工具集（全部只读，`src/agents/code/tools.py`）

| 工具 | 签名 | 实现要点 |
|---|---|---|
| `git_log` | `(repo_path=rc, max_count=20)` | `git log -N --date=short --pretty=format:<四项占位>` + `--shortstat`；按空行切块后把 stat 并到同一行，每提交一行：hash、作者日期、主题、变更文件数 |
| `git_diff` | `(repo_path=rc, ref="HEAD")` | 实为 `git show`：拿某次提交的完整 diff |
| `file_search` | `(repo_path=rc, name_pattern="", content_pattern="", max_results=20)` | `Path.rglob` + 正则；跳过 `.git` / `.venv` / `__pycache__` / `node_modules`；带内容匹配时输出「路径:命中行」（单行截断 120 字符） |
| `read_file` | `(repo_path=rc, path="", max_chars=8000)` | 拒绝 `..`；`resolve()` 后再校验 `is_relative_to(root)`；超长截断并附「...[truncated N chars]」 |

三条共用的实现约束：

1. **`_git` 显式指定编码**（`:22-23`）：`encoding="utf-8", errors="replace"`。不写这两项时，
   文本模式按 `locale.getpreferredencoding()` 解码，Windows 服务进程上就是 cp936 ——
   非 ASCII 提交信息会在 reader 线程里解码失败 → `stdout` 变 `None` → 调用方 `.strip()`
   抛 `AttributeError`。完整案例见 `docs/10-审核与修复总账.md`。另有 `timeout=30`
   与 `check=False`（失败转成可读字符串而非抛异常）。
2. **`_repo_root` 带 `@cache`**（`:32-43`）：把 `repo_path`（文件或目录）向上找到含
   `.git` 的目录，找不到则 `ValueError`。缓存是因为同一轮里工具会被反复调用。
3. git 参数固定、无写操作；`read_file` 的目录穿越校验有两道（`..` 与 `is_relative_to`）。

## 4. 记忆契约（`namespace = ("code-reviewer", user_id)`）

| | 读（`recall_reviews`） | 写（`remember_review`） |
|---|---|---|
| 落点 | `store.asearch(ns, query=…, limit=3)` | `store.aput(ns, key, {"conclusion": …})` |
| key | — | `review-YYYYMMDD`（**同一天多次审查会互相覆盖**） |
| 触发 | 每轮回答前 | 仅当最后一条是 `AIMessage` 且无 `tool_calls` |
| 失败姿态 | 异常只 `logger.warning`，返回空串，**不影响本次审查** | `store is None` 时告警并跳过（`langgraph dev`、`run_agent.py`、单测都会是 None） |

读侧的**判据**值得单独写下来：`score is None` 表示 store 未开启语义检索，此时
`asearch` 不报错、仍返回结果，但排序是主键序 —— 那种结果不可信。因此代码
**按 score 是否有效过滤，而不是按「结果是否为空」判断**（`code_reviewer.py:136-140`）。
`user_id` 缺失时落 `"anonymous"`。

## 5. 会话蒸馏的接线

`distill_history` 插在 `recall_reviews → model` 之间（`:214,217`），语义是
**只折叠「送给模型的那份视图」**：checkpoint 原文一条不删，`/history` 永远是全量。

- 默认 `DISTILL_ENABLED=False`，不配置即零行为变化。
- 摘要写在 `state["distilled_summary"]`，**不进 `messages`**，否则 `/history` 会被污染。
- ⚠️ **该折叠当前没有到达 `model` 节点**：`acall_model` 直接用 `state["messages"]`，
  并未调用 `apply_distillation`。实测摘要长度 7、模型历次收到 `[1,3,…,23]` 全量。
  根因与修法见 `docs/15` §7.2.3 —— 本文只声明接线，不把它记成「已生效」。

## 6. 交互契约

- `/code-reviewer/stream`：流式消息 + 工具调用过程。
- 审查指令示例：「review 最近 5 次提交，输出每个变更的文件、风险点、建议」。
- 回答必须引用真实 git 输出，不编造（`instructions` 第一条规则）。
- Streamlit 欢迎语（`streamlit_app.py:306-307`）：「我是代码库审查助手：用只读工具分析
  git 历史与源码。试试让我『review 最近几次提交』或『定位某个函数在哪里实现』？」

## 7. 测试（现状）

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/agents/test_code_reviewer.py` | 9 | 图编译断言、工具只读断言、`tmp_path` 真建 git 仓库跑 `git_log`/`git_diff`、**非 ASCII 提交解码回归**、`file_search`/`read_file`、`..` 穿越拦截、`remember_review` 写入与无 store no-op |
| `tests/agents/test_memory_read_path.py` | 12 | store 传不传 `index` 的 score 对照、`recall_reviews` 命中/跳过/无 store/namespace 隔离/空输入/异常不炸、**召回内容确实进了 system prompt**、model 前存在 `distill_history` 的边 |
| `tests/core/test_distill.py` | 26 | `plan_distillation` / `apply_distillation` / `distill_history` / `_distill_input` 的判定、改写与不变量 |

- 全部走 `tmp_path` **自建**仓库，不依赖 CI 的 git 状态（CI 无 git 上下文也能跑）。
- 编译断言现在是 `{"model", "tools", "remember_review"} <= nodes`，另有一条专门断言
  model 前存在 `distill_history` —— 因为「图能编译」并不等于「接线正确」，
  详见 `docs/10-审核与修复总账.md` §7.4。

## 8. 风险与对策

| 风险 | 现状对策 |
|---|---|
| `/invoke` 吞中间消息（只取最后一个事件） | 引导走 `/stream`；Streamlit 默认 stream |
| 子进程慢或挂死 | `_git` 固定 `timeout=30`，超时转成可读错误字符串 |
| `file_search` 在大仓库上 `rglob` 全树 | 排除四个目录 + `max_results` 上限；未做索引，大仓库仍会慢 |
| 步数耗尽 | `remaining_steps < 2` 且要调工具时返回提示，避免图抛异常 |
| 记忆读失败 | 只告警不中断；`score is None` 时宁可不召回 |
| 无 `GROQ_API_KEY` | 安全闸自动跳过并判 `SAFE`，接口行为不变 |

## 9. 与设计稿的差异（原稿 → 现状）

| # | 09-16 设计稿 | 现状 | 说明 |
|---|---|---|---|
| 1 | 图：`guard_input → model` | 中间多了 `recall_reviews`、`distill_history` | P0-2 补读路径、P0-7 接蒸馏 |
| 2 | `MessagesState` | `AgentState(MessagesState, total=False)` | 新增 `safety` / `recalled_reviews` / `distilled_summary` 三个通道 |
| 3 | 工具含 `git_show` | 只有 4 个工具，无 `git_show` | `git_diff` 内部走 `git show`，覆盖了该用途 |
| 4 | 函数名 `_subprocess_git` | `_git`（带 `encoding`/`errors`/`timeout`） | 编码缺陷修复后重写，见 `docs/10` |
| 5 | namespace 三元组（含 `repo_name`） | 二元组 `("code-reviewer", user_id)` | 未按仓库再分层；同一天跨仓库会互相覆盖 |
| 6 | 读路径 `store.aget`，命中则免跑 git | `store.asearch` 语义召回，只注入提示词 | 不做「命中即跳过工具」的短路 |
| 7 | `aput` 前先查重 | 无查重，同日 key 直接覆盖 | 未做 |
| 8 | 测试三项（编译/工具/store） | 三个文件共 47 条用例 | 含非 ASCII 解码、记忆读路径、蒸馏四组 |
| 9 | 文件清单 4 项 | 另含 `agents/code/tools.py`、`tests/agents/test_memory_read_path.py`，并依赖 `core/distill.py` | — |

## 10. 文件清单（现状）

- `src/agents/code_reviewer.py` —— 图、`AgentState`、instructions、记忆读写、蒸馏接线
- `src/agents/code/tools.py` —— `_git` / `_repo_root` + 4 个只读工具
- `src/agents/agents.py:38` —— 注册一行
- `src/streamlit_app.py:306` —— 欢迎语一行
- `src/core/distill.py` —— 共享的会话蒸馏（非本 agent 专属）
- `tests/agents/test_code_reviewer.py`、`tests/agents/test_memory_read_path.py`

## YAGNI 排除

多仓库抽象、RAG 索引、技术债扫描、git 写操作、跨 agent 共享状态。
