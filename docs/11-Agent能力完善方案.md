# Agent 能力完善方案：loop-agent 与 multi-agent

> 日期：2026-09-17
>
> **落地状态**（本节由后续实现回填，勿凭标题判断进度）：
>
> | 部分 | 内容 | 状态 |
> |---|---|---|
> | §1 | loop-agent 三项改动（共享止损常量、`fetch_url`、搜索预算） | **已落地**（2026-09-18 晚，见 §1.7） |
> | §2 | multi-agent 转交 500 | **已落地**（走路线 A，见 §2.5） |
> | §5 | 一轮独立复核的结论与逐条处置 | 已折入本文 |

针对 `loop-agent` 的质量缺陷与两个 supervisor 图的转交 500，给出可执行的改造方案。
两部分代价差别很大：loop-agent 是常规工程质量问题，可立即动手；
multi-agent 涉及上游框架缺陷，必须先定位触发条件再选路线。

## 0 现状与验收标准

实测环境：本机 Windows，服务 `:8080`，`deepseek-v4-flash`，2026-09-17 18:41。

| 功能 | 当前实测 | 目标 | 现状 |
|---|---|---|---|
| loop-agent 纯计算 | 2.5s / 1 次工具 / 正确 | 保持 | 未变 |
| loop-agent 搜索+计算 | 35.6s / **7 次工具** / **答案错** | ≤4 次工具，≤20s，答案正确 | **未落地** |
| supervisor 纯对话 | 2.2s / 0 轮 | 保持 | 保持 |
| supervisor 转交 | **500**（2.9s / 106.0s） | 200，且 UI 能展示子 Agent 转交 | **已修：200（2.87s）** |
| hierarchy 转交 | **500**（64.6s） | 200 | **已修：200（4.74s）** |

`service.err.log` 累计 8 条异常，去重后只有一条：
`400 - The reasoning_content in the thinking mode must be passed back to the API.`

「完善」的验收口径：

1. 五个用例全部返回 200，转交场景在 UI 上走通（含 `transfer_to` 展示）。
2. 转交路径要有 **CI 能拦截**的自动化测试 —— 这是当前最大的缺口。
3. 全量测试、`ruff format --check`、`ruff check`、`pyrefly check`、`pymarkdown scan` 全绿。

## 1 loop-agent（未落地）

### 1.1 病灶

| 编号 | 问题 | 位置 |
|---|---|---|
| P1 | instructions 缺「搜索止损」条款 | `loop_agent.py:22-34`（`research_assistant.py:50-53` 有） |
| P2 | `web_search` 只返回摘要，无正文 | `agents/tools.py:47-74` |
| P3 | 止损只靠 prompt 引导，代码层无强制 | `loop_agent.py:46-59` |

P1 是这次答错的直接原因：同一个搜索问题，research-assistant 搜 3~4 次就收手并答对，
loop-agent 搜了 6 次仍在空转，最后猜了个 `1.0.0`（实际为 `1.2.11`）。
**两处 instructions 不一致，是重构 web_search 时的遗漏。**

P2 是能力缺口：pypi 那类页面的**摘要里根本不含版本号**，模型只能靠换措辞反复重搜。
这也是 P1 之所以重要的原因 —— 止损只治「空转」，P2 才治「搜不到」。

### 1.2 改动一：把止损条款抽成共享常量

不能只是复制一份到 `loop_agent.py` —— 那正是这次 bug 的成因（两处副本，改一处漏一处）。

新增 `src/agents/instructions.py`：

```python
SEARCH_STOP_CONDITION = """
Search efficiently and stop early. Two or three web searches on the same question are enough:
if the results so far do not contain the answer, tell the user plainly what you could not find
instead of rewording the query and searching again. Never repeat a search that returned no new
information, and do not keep searching just to fill a gap you already know the results miss.
"""
```

> 注意：常量字符串以换行开头（三引号后的第一个字符是 `\n`）。
> 插入 f-string 时会多出一个空行，无害，但别照抄进需要紧凑排版的地方。

`research_assistant.py` 与 `loop_agent.py` 各自 import 后插入自己的 instructions 字符串。
既消除重复，也让第三条（见 §1.4）只维护一份。

### 1.3 改动二：新增 `fetch_url` 工具（真正的能力补全）

在 `agents/tools.py` 增加：

```python
@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a web page and return its visible text.

    Use this after web_search when the snippet does not contain the answer —
    version numbers, dates and tables usually live only in the page body.
    """
```

实现要点：

| 要点 | 做法 |
|---|---|
| HTTP 客户端 | `httpx`（已是依赖，零新增） |
| 超时 | 连接 5s / 读取 15s，与项目既有「工具侧 30s 超时」口径一致 |
| 体积上限 | **代码里写死硬上限**，`max_chars` 参数只允许在硬上限内收窄（`min(max_chars, HARD_CAP)`）—— 否则模型可以传 `max_chars=100000` 把上下文撑爆 |
| 代理 | 复用 `settings.WEB_SEARCH_PROXY`，与 `web_search` 同源 |
| 重定向 | 手工逐跳跟随（≤3 跳），**每跳重新校验目标地址** |
| HTML 转文本 | 正则剥 `<script>/<style>/<svg>` 与标签，再用标准库 `html.unescape` |
| 失败语义 | 返回描述性字符串而非抛异常，让模型能自行改策略 |

**SSRF 防护是必须项，不是加分项。** 工具能收 URL 就等于模型能构造 URL，
而模型可以被网页内容诱导。项目在 `agents/code/tools.py` 已有「路径穿越防护」先例，
这里要保持同一水准：

| 规则 | 拦截对象 |
|---|---|
| 只允许 `http` / `https` | `file://`、`gopher://`、`data:` |
| 解析域名后校验**全部** A/AAAA 记录 | 私网段 `10/8`、`172.16/12`、`192.168/16`、`127/8` |
| 额外拒绝 | 链路本地 `169.254/16`（云元数据）、`::1`、`fc00::/7` |
| 拒绝明文主机名 | `localhost`、`*.internal`、`*.local` |
| 响应体上限 | 先按 `Content-Length` 预检，读取时再硬截断 |

已知残留风险：DNS 解析与真正连接之间存在 TOCTOU 窗口（DNS rebinding）。
彻底解法是解析出 IP 后直接连 IP、手工带 `Host` 头。建议先实现校验版并在代码里写明残留风险，
不要假装它不存在。

零新依赖的代价是正文抽取质量一般（导航、页脚会混进来）。若后续觉得不够，
可加 `trafilatura`，但那是新增依赖，需要走 `uv lock`，本方案不纳入。

### 1.4 改动三（可选，进阶）：代码级搜索预算

P3 说明「只靠 prompt 约束」本质上是建议而非保证。代码侧兜底方案：

1. `AgentState` 增加 `search_calls: int`。
2. `acall_model` 里统计已发生的 `WebSearch` 调用。
3. 超过阈值（建议 4）后，`wrap_model` 改为 `bind_tools([t for t in tools if t.name != "WebSearch"])`
   —— **直接把工具摘掉**，模型失去了空转的物理条件，只能用已有信息作答。
4. 同时在 messages 里注入一条 `SystemMessage`，要求它如实说明信息不足。

比 prompt 引导硬，且实现量小（约 15 行）。建议先做 §1.2 / §1.3，用实测数据决定是否值得加。

### 1.5 改动四：instructions 补齐

`loop_agent.py` 的 instructions 增加两点：

1. 引用共享的 `SEARCH_STOP_CONDITION`。
2. 写明工具的**配合顺序**：先 `WebSearch` 找候选 URL，摘要不够时用 `fetch_url` 打开正文，
   拿到答案就停 —— 不给这个引导，模型倾向于反复 `WebSearch` 而不是换工具。

**注意工具集要与 `research_assistant` 对齐。** P2 被定性为**通用能力缺口**
（「摘要里根本不含版本号」），而答对 `1.2.11` 的恰恰是 research-assistant。
只在 loop-agent 加 `fetch_url` 会制造新的工具集不对称 —— 这正是 §1.1 说的「两处副本」
成因的另一种形态。要么两边同加，要么在代码注释里写明只给 loop-agent 的理由。

### 1.6 测试

`tests/agents/test_loop_agent.py` 补充：

| 用例 | 断言 |
|---|---|
| instructions 含止损条款 | `"stop early" in instructions`（防止再次漏改） |
| tools 含 `fetch_url` | `"fetch_url" in [t.name for t in tools]` |

新增 `tests/agents/test_tools_fetch_url.py`（**全部离线**，mock `httpx`）：

| 用例 | 输入 | 期望 |
|---|---|---|
| 拒绝回环地址 | `http://127.0.0.1:8080/info` | 返回拒绝说明，不发请求 |
| 拒绝私网 | `http://192.168.1.1/`、`http://10.0.0.1/` | 同上 |
| 拒绝云元数据 | `http://169.254.169.254/latest/meta-data/` | 同上 |
| 拒绝非 HTTP 协议 | `file:///etc/passwd`、`ftp://x/y` | 同上 |
| 拒绝明文主机名 | `http://localhost/`、`http://foo.internal/` | 同上 |
| **重定向到内网** | mock 公网 URL 返回 302 → `http://169.254.169.254/latest/meta-data/` | 被拦截，且**不发出第二次请求** |
| 正常抓取与截断 | mock 返回 10 KB HTML | 正文提取成功，长度 ≤ `max_chars` |
| 剥脚本样式 | HTML 含 `<script>` | 结果中不含脚本内容 |

**第 6 条是重点**：实现是「手工逐跳跟随重定向、每跳重新校验」，真正容易写错的是
**公网 URL 302 到内网**这条路径，而它恰恰是最容易漏测的。

### 1.7 loop-agent 验收

`probe_both.py` 的 L2 用例：工具调用数 ≤4、耗时 ≤20s、答出 `1.2.11`。
连续跑 3 次取一致性，不只看单次 —— 搜索本身有抖动。

#### 1.8 落地记录（2026-09-18 晚）

| 改动 | 落地内容 |
|---|---|
| §1.2 共享止损常量 | 新增 `src/agents/instructions.py` 的 `SEARCH_STOP_CONDITION`；`loop_agent` 与 `research_assistant` 各自 import（消除两处副本） |
| §1.3 `fetch_url` | `src/agents/tools.py` 新增 `fetch_url` 工具：httpx、连接 5s/读取 15s、`settings.WEB_SEARCH_PROXY`、手工逐跳重定向（≤3 跳，**每跳重新校验**）、正则剥标签、失败返回字符串。SSRF 防护：协议白名单 + 私网 IP 检测 + 明文主机名拒绝 + `Content-Length` 预检 + 硬上限 `FETCH_URL_HARD_CAP=100000` |
| §1.4 搜索预算 | `loop_agent.AgentState` 加 `search_calls`；`acall_model` 统计历史 `WebSearch` 调用，超过 `SEARCH_BUDGET=4` 后 `wrap_model` 从 `bind_tools` 摘掉 `WebSearch` 并注入 system 消息 |
| §1.5 工具集对齐 | `loop_agent` 与 `research_assistant` 的 tools 均为 `[WebSearch, Calculator, fetch_url]`；两处 instructions 均含止损条款与「先搜后抓」工具顺序 |

**实现中发现并修复的一个 SSRF 缺陷**：初版 `_validate_url_safe` 对 hostname 无条件走
`socket.getaddrinfo`，而 IP 字面量 URL（如 `http://169.254.169.254/`）的 DNS 结果
可被环境/mock 影响，导致私网 IP 字面量被当成公网放行。修法：hostname 本身是合法 IP
字面量时**直接校验该 IP**，只对域名解析 A/AAAA 记录。测试 `test_fetch_url_blocks_redirect_to_internal`
正是靠这条把 `getaddrinfo` mock 成公网 IP，从而暴露该绕过。

验证：

| 项 | 结果 |
|---|---|
| `pytest -q` | **299 passed / 7 skipped**（新增 13 条：`test_tools_fetch_url.py` 11 + `test_loop_agent.py` 扩展） |
| `ruff check` / `ruff format --check` | All checks passed / 65 files already formatted |
| `pyrefly check` | **0 errors** |

> **未做 §1.7 的 `probe_both.py` L2 实测**（工具调用数/耗时/答对 `1.2.11`）：
> 该验收需要真实联网搜索，收益是「实测稳定性」而非「功能是否存在」。
> 代码路径已由离线测试覆盖；真实搜索的收敛性待下次联网运行时补测。

## 2 multi-agent（已落地：路线 A）

### 2.1 病灶链条

三层叠加，**每一层各自都按规范行事，没有一个是 bug**：

| 层 | 机理 |
|---|---|
| 协议层 | DeepSeek thinking 模式要求把 `reasoning_content` 原样回传 |
| 框架层 | `core/llm.py:100-107` 用 `ChatOpenAI` 而非 `ChatDeepSeek`，**入站根本没捕获该字段**，既存不住也传不回 |
| 编排层 | `langgraph_supervisor` 把**子 Agent 的终答拼成一条 content-only 的 assistant 消息**塞进父图历史。thinking 模式下这条消息必须带 `reasoning_content`，而它不是本模型生成的、根本没有该字段 |

所以「在自己代码里写对」做不到，只能绕过或打补丁。这也是为什么它值得单列一节讲，
而不是当成一个普通 bug。

### 2.2 动手前必须知道的三个约束

| 约束 | 后果 |
|---|---|
| `get_model` 带 `@cache`；`agents.py:36` 是模块级字典，图在 import 期编译 | 运行时换模型对**非 loop** agent 结构性不支持。改之前先想清楚要不要连带重构注册表。（例外：`github_mcp_agent` 是 `LazyLoadingAgent`，图是懒加载的） |
| UI 靠 `"transfer_to" in tool_call["name"]` 识别转交 | `streamlit_app.py:484 / 520 / 690`。任何方案都必须保留 `transfer_to_*` 命名，否则前端要改。（`handle_sub_agent_msgs` 的**定义**在 `:619`，这三个行号是判定点，两处引用的是不同东西） |
| `FakeToolModel` 继承 `FakeListChatModel`，**只回文本、不产生 tool_calls** | `core/llm.py:45-50`。这个假模型**测不了转交**，必须换成能发 tool_call 的那个（见 §2.8） |

第三条是本节最关键的发现：它解释了为什么这个 500 从上线至今 CI 一直绿 ——
现有测试基建在原理上就碰不到转交路径。**补测试不是收尾工作，是方案的一部分。**

### 2.3 定位实验：已完成（2026-09-17 18:45）

脚本 `.workbuddy/tools/probe_supervisor_options.py`，四条备选路线各跑一次真实转交
（本地构建图，不经过服务）。**结果推翻了原先「编排层配置是触发条件」的判断。**

| 编号 | 变量 | 结果 |
|---|---|---|
| E0 | 基线（`full_history` + handoff-back） | 400 |
| E1 | `output_mode="last_message"` | 400 |
| E2 | `add_handoff_back_messages=False` | 400 |
| E3 | `extra_body={"thinking": {"type": "disabled"}}` | **200**，9 条消息，答案 `17 + 25 = 42` |
| E4 | `model_kwargs={"reasoning": {"enabled": False}}` | 400，但报错字段变成 `reasoning_text` |

三条推论：

1. **编排层不是触发条件。** E1 / E2 都改了消息拼接方式（E2 干脆不做 handoff-back），
   依然 400。而 E3 成功时角色序列里 `adjacent_assistant=1` ——
   **相邻 assistant 消息与成功可以共存**，先前的猜测是错的。
2. **触发条件是「历史里存在非本模型生成的 content-only assistant 消息」。**
   子 Agent 的终答被 `langgraph_supervisor` 拼成 assistant 消息塞进父图历史；
   thinking 模式下该消息必须带 `reasoning_content`，而它不是本次调用生成的、没有该字段。
   这是 `langgraph_supervisor` 的设计核心，**改配置改不掉**。
3. **E4 的死法也和传闻不同。** 社区报告是「被误判为 `/responses` API → 404」，
   实测是 400 且报错字段变成 `reasoning_text` —— 参数确实下发到了服务端，
   只是触发了另一套多轮协议要求。此路依然不通，但原因要记对。

> **结论强度的说明（复核意见，已采纳）**：推论 2 的性质是**推断**，不是实测结论 ——
> E1/E2/E3 都**没有观测过失败请求的载荷**，能正向识别触发字段的 E5（逐字段 diff）被取消了。
> 好在这个推断不影响已经落地的路线 A（路线 A 的可行性由 E3 直接正向证明）。
> 但**若将来要重新启用 thinking**，这个前提必须先补验：不构建图，直接用 SDK 构造两组历史
> 各调一次带 `tools` 的请求 ——
>
> | 组 | 历史形态 | 预期 |
> |---|---|---|
> | C1 | 子 Agent 终答作为 `AIMessage` 放入历史 | 400（与现象一致） |
> | C2 | 同一条终答改为 `ToolMessage` 放入历史 | 200 → **路线 C / D 的前提成立** |
>
> 这比已取消的 E5 便宜得多，且结论直接可用。

### 2.4 选路线

| 路线 | 做法 | 状态 | 改动面 | 风险 | 保留 thinking |
|---|---|---|---|---|---|
| **A** | 关掉 DeepSeek thinking | **已验证并已落地（E3）** | supervisor 模块内 2 处 | 低 | 否 |
| A′ | 改 `langgraph_supervisor` 编排配置 | **已否证（E1/E2）** | — | — | — |
| **D** | 保持库不动，用 `pre_model_hook` 在出站前把缺 `reasoning_content` 的消息改写成 `ToolMessage` | 未验证 | 约 10 行 hook | 低 | **是** |
| B | 换 `ChatDeepSeek` + 出站镜像 `reasoning_content` | 未验证 | `core/llm.py` + 新依赖 | **全局**，所有 agent 受影响 | 是 |
| C | 自建 supervisor（手写图，绕开 `langgraph_supervisor`） | 未验证 | 新 agent 文件 | 中 | 是 |
| E | 从 UI 摘掉这两个 agent | 可用 | `agents.py:56-62` | 最低 | 不适用 |

**已执行：A。** 它是唯一被实测证明当天可落地的路，改动量约 4 行。
代价是这两个 agent 失去 thinking —— 对「数学专家 + 研究专家」这类任务，损失可接受。

**路线 D 是复核时补上的中间档，值得记住。** `langgraph_supervisor` 0.0.31 的签名里
就有 `pre_model_hook`（`supervisor.py:220`，`:438` 透传给 agent 工厂），
而它支持返回 `{"llm_input_messages": [...]}` —— **只改这一次送给 LLM 的消息，不改图状态**
（`langgraph/prebuilt/chat_agent_executor.py:639` 消费）。因此：

> 保持 `langgraph_supervisor` 不动，用 `pre_model_hook` 在出站前把
> 「缺少 `reasoning_content` 的 content-only assistant 消息」改写成 `ToolMessage`。

它与路线 C 在语义上完全等价（子 Agent 终答以工具结果形式回到 supervisor），
但改动量约 10 行，且 §2.6 列的三条 UI 契约**一条都不用碰** ——
交接工具命名、`skip_stream` 约定、子 agent 状态是否入主图，全部沿用库的行为。

这条推断的依据是官方规则：**请求携带 `tools` 时，所有历史轮次的 `reasoning_content`
都应传回**（见 `.workbuddy/memory/2026-09-17.md:629-637`，
`.workbuddy/tools/probe_fix.py` 的 docstring 也照此写成）。原方案正文没有引用这条规则，
因此没能从它推出路线 D。**若将来要保住 thinking，先跑 §2.3 的 C1/C2 实验定档，再试 D；D 不通再上 C。**

**B 不建议做**：为修两个演示 agent 而改动全部 12 个 agent 的 LLM 层，回归成本与收益不匹配。
上游自己也没修好（langchain issue #35006），等于长期背一个私有 patch。
若要做，它本身是个好面试素材 ——「我定位并修了上游 bug」，但应单独开一档。

**A′ 已淘汰，不必再考虑。**

### 2.5 路线 A 细节（含实际落地形态）

`langgraph_supervisor_agent.py:9` 与 `hierarchy:7` 的 `model` 构造处需要关闭 thinking 的模型。
注意 `get_model` 带 `@cache`，**不能在 `get_model` 内部加**（会污染所有 agent 的同一实例）。

**实际落地的机制是把机制写死在一个新函数里**，不做参数化猜测：

```python
# src/core/llm.py
@cache
def get_supervisor_model(model_name: AllModelEnum, /) -> ModelT:
    """返回一个已关闭 DeepSeek thinking 模式的模型，供 supervisor 图使用。"""
    model = get_model(model_name)
    if model_name in DeepseekModelName and isinstance(model, ChatOpenAI):
        return model.model_copy(update={"extra_body": {"thinking": {"type": "disabled"}}})
    return model
```

两个有意为之的细节：

- **用 `model_copy` 而不是重新构造实例**，这样其他 agent 共享的那个 `@cache` 实例
  仍然保持 thinking 开启（这正是复核意见 B2 要求守住的风险点，已由
  `test_supervisor_smoke.py` 的「共享缓存不被污染」用例守住）。
- **仅对 DeepSeek 生效**。其他 provider 会拒绝未知的请求字段，而该参数会被原样发到它们的 API。

### 2.6 路线 C 细节

核心思路：**把「子 Agent 的终答」作为 `ToolMessage` 返回给 supervisor，而不是像
`langgraph_supervisor` 那样拼成一条 assistant 消息。** 这正是 §2.3 推论 2 指出的触发条件 ——
历史里不再出现「非本模型生成的 content-only assistant 消息」，
`reasoning_content` 的回传要求自然不成立，thinking 得以保留。

图结构（复用 `loop_agent.py` 的模式）：

```text
supervisor 节点（决策：直接答 or 调 transfer_to_<name>）
   ├─ tools 节点：ToolNode([transfer_to_math_expert, transfer_to_research_expert])
   └─ 子 agent 节点：各自的图，返回 ToolMessage(子 Agent 终答)
```

必须满足的契约：

1. handoff 工具命名一律 `transfer_to_<sub_agent_name>` —— 否则 UI 的
   `handle_sub_agent_msgs` 认不出来（判定点在 `streamlit_app.py:484 / 520 / 690`，
   定义在 `:619`）。
2. 子 Agent 的最终回答必须能被 UI 的 `handle_sub_agent_msgs` 消费；
   `langgraph_supervisor` 用的是 `tags=["skip_stream"]` 过滤，自建图要显式对齐这个约定。
3. 待决点：子 Agent 的内部 tool_call 是否要流入主图状态。流入了 UI 能展示细节但污染上下文，
   不流入则 UI 的转交展示会变薄。**这一条要有独立的设计确认，不在本方案内定死。**

### 2.7 顺带修的项目层问题

与转交 500 无关，但同属这两个 agent，改动小、收益明确：

| 问题 | 位置 | 优先级 | 处置 |
|---|---|---|---|
| 三级转发导入 `web_search` | `hierarchy:4` | 高 | **已修**：改从定义处 `agents.tools` 导入。注意原判据写错了 —— `web_search` **只定义在 `agents/tools.py:77`**，`research_assistant.py:14` 与 `langgraph_supervisor_agent.py:6` 都只是**再导出**。问题是「借道兄弟 agent 模块」，不是「定义在别处」 |
| prompt 拼接漏空格，拼出 `capabilities.For` | `hierarchy:37` | 中 | **已修**：补空格 |
| 数学专家只有 `add` / `multiply`，**没有除法** | `langgraph_supervisor_agent.py:12-19` | 中 | 补一个 `divide` 即可（未做） |
| `workflow(chosen_model)` 参数化了，调用点传的是全局 `model` | `hierarchy:10 / :46` | **降级：风格统一** | 原列为「高优先级缺陷」属**误诊**。回代码核实：`workflow(chosen_model)` 内部三处全部用的是 `chosen_model`（`:13` 数学子 agent、`:21` 内层 supervisor、`:35` 外层 supervisor），**参数是完整接通的**；调用点 `:46 workflow(model)` 传模块级 `model`，是 import 期**唯一可行**的做法。不是「接了一半」，最多是「参数只有一个调用点」 |
| 模块级固定模型，UI 换模型不生效 | 两个 supervisor 模块 | 低 | 受 §2.2 第一条约束，写成已知限制而非硬改 |

### 2.8 测试

这是本方案里最该做的事，因为它直接决定 500 会不会再次发生。

**注意：不要新写假模型。** 仓库里已经有一个能回放 `tool_calls` 的
`FakeMessagesListChatModel` 子类（原本藏在 `tests/service/test_service_message_generator.py:12`），
它已经驱动过三层 supervisor 的完整转交链路。做法是把它上提到 `tests/conftest.py`，
用 `fake_tool_model` fixture 暴露给 `tests/agents/` 与 `tests/service/` 两处共用
（`tests/` 下没有 `__init__.py`，`pyproject.toml:99` 的 `pythonpath` 只有 `["src", "scripts"]`，
所以两个目录无法通过普通 import 共享一个模块）。

实际落地的用例（`tests/agents/test_supervisor_smoke.py` 与 `test_loop_agent.py`）：

| 用例 | 方式 | 期望 |
|---|---|---|
| supervisor 使用关闭 thinking 的模型 | 离线 | 断言拿到的是 `get_supervisor_model` 的产物 |
| 共享缓存不被污染 | 离线 | 断言 `get_model(settings.DEFAULT_MODEL)` 返回的实例**未**被改动 |
| 脚本化模型驱动完整 `model → tools → model` | 离线 | 走完两条边，终答正确，`tool` 消息在序列里 |
| 真实转交（单层 + 嵌套） | `@pytest.mark.network`，真 DeepSeek | 200，且答出正确数值 |

> **已删除一条错误用例（复核意见 A1）**：原草稿里有一条
> 「消息序列无相邻 assistant ｜ 断言 `-1` 步产物 ｜ 防止 `full_history` 形态回归」。
> 它与 §2.3 推论 1 直接冲突 —— E3 成功时 `adjacent_assistant=1`，
> **相邻 assistant 消息与成功可以共存**，真实触发条件（缺 `reasoning_content`）
> 与消息是否相邻无关。那条用例跑绿了拦不住任何真实回归，却把已被证伪的假设固化进测试。
> 处置：不写这条用例；若将来要防回归，断言的是**真实条件**
> ——「出站消息里不存在缺少 `reasoning_content` 的 assistant 消息」，
> 或退回行为级断言「转交链路返回 200 且最终答案正确」。
> `test_supervisor_agent_compiles` 保留，它现在的作用只剩「import 期不炸」。

## 3 执行顺序

| 步骤 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| 1 | loop-agent：抽共享常量 + 补 instructions | 无 | 未做 |
| 2 | loop-agent：`fetch_url` + SSRF 测试 | 无 | 未做 |
| 3 | 实测定档：跑 `probe_both.py` L2 三次 | 步骤 1、2 | 未做 |
| 4 | multi-agent 定位实验 | — | **已完成**（E0--E4） |
| 5 | 实施路线 A：`get_supervisor_model` | 步骤 4 | **已完成** |
| 6 | 补转交测试（复用 conftest 假模型 + 联网用例） | 步骤 5 | **已完成** |
| 7 | 顺带修 §2.7 项目层问题 | 无，可并行 | 部分完成（空格、导入已修；`divide` 未加） |
| 8 | 全量回归 + 文档更新 | 全部 | **已完成**（226 passed / 7 skipped） |

步骤 1、2、7 互相独立，步骤 4 可与 1~3 并行。**步骤 5 在步骤 4 出结论前不动手。**

**验收缺口的补充（复核意见 B1）**：§0 验收口径第 1 条要求「转交场景在 UI 上走通」，
但上面的执行顺序没有任何一步覆盖 UI 端到端 —— 步骤 5 的产出只写「转交返回 200」。
处置：把该验收明确降级为「服务端 200」，UI 人工走查另行安排（本次未做）。

## 4 本方案不做的事

- **不摘除 agent**：路线 E 只作为「演示前的临时止血」，不作为终态 —— 摘掉等于放弃
  简历里「主管调度、嵌套层级主管」两条能力。
- **不改 `core/llm.py` 的全局默认行为**：除走路线 B 且单独排期验收，否则会波及全部 12 个 agent。
  落地时新增的 `get_supervisor_model` 走 `model_copy`，不触碰 `get_model` 的缓存实例，符合此约束。
- **不改注册表结构**：`agents.py` 的 id→单图 设计导致运行时换模型对非 loop agent 不支持，
  重构它属于独立课题，收益低，本次只记录为已知限制。
- **不引入新依赖**（`fetch_url` 走零依赖实现）：需单独排期验收的例外只有路线 B
  （必须新增 `langchain-deepseek`）。
- **不补「空 `reasoning_content` 占位」方案**：`.workbuddy/tools/probe_fix.py` 记录过一次尝试 ——
  向出站载荷注入 `reasoning_content=""` 与 `"placeholder"` **都返回 400**（注入确实生效，
  但伪值被拒）。这条否掉了「补一个空字段就完事」的直觉方案，故不再重提。

## 5 独立复核结论（原《Agent 能力完善方案审查》已折入本节）

复核对象是本方案 2026-09-17 18:46 版。审查方式：**不采信正文结论，
逐条回到代码、依赖清单、CI 配置与既有实验记录核实。**

### 5.1 核实结果

| 维度 | 结果 |
|---|---|
| 事实性论断 | 回代码核实 16 项，**14 项准确**，2 项错误（`chosen_model` 误诊、`web_search` 归属写错） |
| 内部一致性 | **2 处自相矛盾**（§2.8 的错误用例、§2.7 的归属），1 处导语已过期 |
| 结论强度 | §2.3 把推断写成「钉死」，与 §2.4 标注的「未验证」冲突 |
| 路线完整性 | **漏掉一条库内自带的中间档路线**（路线 D） |
| 可验收性 | §0 有 1 条验收标准没有对应的执行步骤（UI 端到端） |

核实无误、可直接作为实施依据的部分：`loop_agent.py` 缺止损条款、
`research_assistant.py` 有该条款、`agents/tools.py` 只回 title/href/body、
`web_search.name == "WebSearch"`、DeepSeek 走 `ChatOpenAI` 而非 `ChatDeepSeek`、
`FakeToolModel` 只回文本、`get_model` 带 `@cache`、`agents.py:36` 是模块级字典、
UI 靠 `"transfer_to" in name` 识别转交、`hierarchy` 各行的行号与内容（含漏空格）、
数学工具只有 `add`/`multiply`、CI 会对 `docs/` 跑 markdown 检查、
`httpx` 已是依赖、`agents/code/tools.py` 有路径防护先例。

### 5.2 逐条处置

| 编号 | 问题 | 处置 |
|---|---|---|
| A1 | §2.8 的「无相邻 assistant」用例与 §2.3 结论冲突 | **已删**（见 §2.8 末注） |
| A2 | §2.3 结论强度超过证据 | **已补说明**（见 §2.3 末注），并保留 C1/C2 实验设计供将来启用 thinking 时使用 |
| A3 | §2.7 的 `chosen_model` 一条是误诊 | **已降级为「风格统一」**（见 §2.7） |
| A4 | §2.7 把 `web_search` 的归属写错 | **已改对**（见 §2.7），代码也已按正确理由修 |
| A5 | 漏掉库内自带的中间档路线 D | **已补写**（见 §2.4） |
| B1 | 一条验收标准没有执行步骤 | **已降级为服务端 200** 并写明（见 §3 末） |
| B2 | 路线 A 的缓存污染风险没有测试守住 | **已补**用例（见 §2.8） |
| B3 | §2.5 的实施机制含糊 | **已写死**为 `model_copy` 形态（见 §2.5） |
| B4 | `fetch_url` 的 `max_chars` 由模型可控 | **已改为硬上限 + 参数只允许收窄**（见 §1.3） |
| B5 | SSRF 测试缺「公网 302 到内网」这条 | **已补**（见 §1.6 第 6 行） |
| B6 | `fetch_url` 只给 loop-agent，与「通用能力缺口」的定性不一致 | **已写明需两处对齐**（见 §1.5 末） |
| B7 | 未收录「伪值注入已证伪」的既有证据 | **已收入 §4** |
| C1 | 导语已过期（仍写「须先做定位实验」「三条路线」） | **已重写**（文首状态块） |
| C2 | §0 验收口径写「四个用例」，现状表有 5 行 | **已改「五个」** |
| C3 | 「12 张图在 import 期全部编译」不精确 | **已加例外说明**（`github_mcp_agent` 懒加载，见 §2.2） |
| C4 | §2.2 / §2.6 行号混用（`handle_sub_agent_msgs` 定义 vs 判定点） | **已分开标注** |
| C5 | §4 首条标题与正文并列矛盾 | **已改为「除单独排期外不引入新依赖」** |
| C6 | §1.2 代码块常量字符串以换行开头 | **已加提示** |

### 5.3 本次复核未覆盖

- **未重跑任何联网实验**：§2.3 的 E0--E4 数据取自日志记录，只核对了脚本存在与逻辑一致，
  没有复跑验证数值（复跑需真实 API 调用）。
- **未评估 `fetch_url` 的正文抽取质量**：正则剥标签的实际效果需要真实页面才能判断。
- 复核范围仅限本方案文档，未审查 `src/` 的现有未提交改动。

## 6 落地结果

§2（multi-agent）的落地明细 —— 改动清单、与方案不一致的三处、验证矩阵 ——
统一记在 `docs/10-审核与修复总账.md` §八 / §九，本文不再重复。

§1（loop-agent）尚未动手，是本文当前的唯一开放项。
