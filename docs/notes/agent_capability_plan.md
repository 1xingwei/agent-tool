# Agent 能力完善方案：loop-agent 与 multi-agent

针对 `loop-agent` 的质量缺陷与两个 supervisor 图的转交 500，给出可执行的改造方案。

方案分两部分：loop-agent（可立即动手，风险低）、multi-agent（**须先做定位实验再选路线**，
因为它涉及上游框架缺陷，三条路线代价差别很大）。

## 0 现状与验收标准

实测环境：本机 Windows，服务 `:8080`，`deepseek-v4-flash`，2026-09-17 18:41。

| 功能 | 当前实测 | 目标 |
|---|---|---|
| loop-agent 纯计算 | 2.5s / 1 次工具 / 正确 | 保持 |
| loop-agent 搜索+计算 | 35.6s / **7 次工具** / **答案错** | ≤4 次工具，≤20s，答案正确 |
| supervisor 纯对话 | 2.2s / 0 轮 | 保持 |
| supervisor 转交 | **500**（2.9s / 106.0s） | 200，且 UI 能展示子 Agent 转交 |
| hierarchy 转交 | **500**（64.6s） | 200 |

`service.err.log` 累计 8 条异常，去重后只有一条：
`400 - The reasoning_content in the thinking mode must be passed back to the API.`

「完善」的验收口径：

1. 四个用例全部返回 200，转交场景在 UI 上走通（含 `transfer_to` 展示）。
2. 转交路径要有 **CI 能拦截**的自动化测试 —— 这是当前最大的缺口。
3. 全量测试、`ruff format --check`、`ruff check`、`pyrefly check`、`pymarkdown scan` 全绿。

## 1 loop-agent

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

`research_assistant.py` 与 `loop_agent.py` 各自 import 后插入自己的 instructions 字符串。
既消除重复，也让第三条（见 1.4）只维护一份。

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
| 体积上限 | 工具侧截断到 `max_chars`（默认 4000），**不能省** —— 一次抓取就能撑爆上下文 |
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

比 prompt 引导硬，且实现量小（约 15 行）。建议先做 1.2 / 1.3，用实测数据决定是否值得加。

### 1.5 改动四：instructions 补齐

`loop_agent.py` 的 instructions 增加两点：

1. 引用共享的 `SEARCH_STOP_CONDITION`。
2. 写明工具的**配合顺序**：先 `WebSearch` 找候选 URL，摘要不够时用 `fetch_url` 打开正文，
   拿到答案就停 —— 不给这个引导，模型倾向于反复 `WebSearch` 而不是换工具。

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
| 正常抓取与截断 | mock 返回 10 KB HTML | 正文提取成功，长度 ≤ `max_chars` |
| 剥脚本样式 | HTML 含 `<script>` | 结果中不含脚本内容 |

### 1.7 loop-agent 验收

`probe_both.py` 的 L2 用例：工具调用数 ≤4、耗时 ≤20s、答出 `1.2.11`。
连续跑 3 次取一致性，不只看单次 —— 搜索本身有抖动。

## 2 multi-agent

### 2.1 病灶链条

三层叠加，**每一层各自都按规范行事，没有一个是 bug**：

| 层 | 机理 |
|---|---|
| 协议层 | DeepSeek thinking 模式要求把 `reasoning_content` 原样回传 |
| 框架层 | `core/llm.py:100-107` 用 `ChatOpenAI` 而非 `ChatDeepSeek`，**入站根本没捕获该字段**，既存不住也传不回 |
| 编排层 | `langgraph_supervisor` 把**子 Agent 的终答拼成一条 content-only 的 assistant 消息**塞进父图历史。thinking 模式下这条消息必须带 `reasoning_content`，而它不是本模型生成的、根本没有该字段（§2.3 实测钉死） |

所以「在自己代码里写对」做不到，只能绕过或打补丁。这也是为什么它值得单列一节讲，
而不是当成一个普通 bug。

### 2.2 动手前必须知道的三个约束

| 约束 | 后果 |
|---|---|
| `get_model` 带 `@cache`；`agents.py:36` 是模块级字典，12 张图在 import 期全部编译 | 运行时换模型对**非 loop** agent 结构性不支持。改之前先想清楚要不要连带重构注册表 |
| UI 靠 `"transfer_to" in tool_call["name"]` 识别转交 | `streamlit_app.py:484 / 520 / 690`。任何方案都必须保留 `transfer_to_*` 命名，否则前端要改 |
| `FakeToolModel` 继承 `FakeListChatModel`，**只回文本、不产生 tool_calls** | `core/llm.py:45-50`。现有假模型**测不了转交**，必须另造一个会发 tool_call 的假模型 |

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
2. **真正的触发条件是「历史里存在非本模型生成的 content-only assistant 消息」。**
   子 Agent 的终答被 `langgraph_supervisor` 拼成 assistant 消息塞进父图历史；
   thinking 模式下该消息必须带 `reasoning_content`，而它不是本次调用生成的、没有该字段。
   这是 `langgraph_supervisor` 的设计核心，**改配置改不掉**。
3. **E4 的死法也和传闻不同。** 社区报告是「被误判为 `/responses` API → 404」，
   实测是 400 且报错字段变成 `reasoning_text` —— 参数确实下发到了服务端，
   只是触发了另一套多轮协议要求。此路依然不通，但原因要记对。

原计划的 E5（逐字段 diff）**不再需要**：触发条件已由 E3 反证钉死。

### 2.4 选路线

| 路线 | 做法 | 状态 | 改动面 | 风险 | 保留 thinking |
|---|---|---|---|---|---|
| **A** | 关掉 DeepSeek thinking | **已验证可行（E3）** | supervisor 模块内 2 处 | 低 | 否 |
| A′ | 改 `langgraph_supervisor` 编排配置 | **已否证（E1/E2）** | — | — | — |
| B | 换 `ChatDeepSeek` + 出站镜像 `reasoning_content` | 未验证 | `core/llm.py` + 新依赖 | **全局**，所有 agent 受影响 | 是 |
| C | 自建 supervisor（手写图，绕开 `langgraph_supervisor`） | 未验证 | 新 agent 文件 | 中 | 是 |
| E | 从 UI 摘掉这两个 agent | 可用 | `agents.py:56-62` | 最低 | 不适用 |

**推荐：先走 A 止血，C 作为后续演进。**

- A 已被实测证明可行，改动量约 4 行，是唯一能在当天让「转交可用」落地的路。
  代价是这两个 agent 失去 thinking —— 对「数学专家 + 研究专家」这类任务，损失可接受。
- C 是唯一能同时保留 thinking、又不依赖上游修复的路，也把简历里「落地多智能体编排框架」
  讲实（自研编排 > 调库）。但它要自己处理 handoff 语义与 UI 契约（§2.6），需独立排期。
- B 不建议现在做：为修两个演示 agent 而改动全部 12 个 agent 的 LLM 层，回归成本与收益不匹配。
  上游自己也没修好（langchain issue #35006），等于长期背一个私有 patch。
  若要做，它本身是个好面试素材 ——「我定位并修了上游 bug」，但应单独开一档。
- A′ 已淘汰，不必再考虑。

### 2.5 路线 A 细节

`langgraph_supervisor_agent.py:9` 与 `hierarchy:7` 的 `model` 构造处传入关闭 thinking 的参数。
注意 `get_model` 带 `@cache`，**不能在 `get_model` 内部加**（会污染所有 agent 的同一实例）；
要在 supervisor 模块里基于 `get_model(...)` 再构造，或给 `get_model` 加一个显式参数并纳入 cache key。

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
   `handle_sub_agent_msgs` 认不出来（`streamlit_app.py:484 / 520 / 690`）。
2. 子 Agent 的最终回答必须能被 UI 的 `handle_sub_agent_msgs` 消费；
   `langgraph_supervisor` 用的是 `tags=["skip_stream"]` 过滤，自建图要显式对齐这个约定。
3. 待决点：子 Agent 的内部 tool_call 是否要流入主图状态。流入了 UI 能展示细节但污染上下文，
   不流入则 UI 的转交展示会变薄。**这一条要有独立的设计确认，不在本方案内定死。**

### 2.7 顺带修的项目层问题

与转交 500 无关，但同属这两个 agent，改动小、收益明确：

| 问题 | 位置 | 优先级 |
|---|---|---|
| `workflow(chosen_model)` 参数化了，调用点传的是全局 `model` —— 改到一半没接上 | `hierarchy:10 / :46` | 高 |
| 三级转发导入 `web_search`（实际定义在 `research_assistant.py`） | `hierarchy:4` → 改为 `from agents.tools import web_search` | 高 |
| prompt 拼接漏空格，拼出 `capabilities.For` | `hierarchy:37` | 中 |
| 数学专家只有 `add` / `multiply`，**没有除法** | `langgraph_supervisor_agent.py:12-19` | 中，补一个 `divide` 即可 |
| 模块级固定模型，UI 换模型不生效 | 两个 supervisor 模块 | 低，受 §2.2 第一条约束，建议写成已知限制而非硬改 |

### 2.8 测试

这是本方案里最该做的事，因为它直接决定 500 会不会再次发生。

新增 `tests/agents/test_supervisor_handoff.py`：

| 用例 | 方式 | 期望 |
|---|---|---|
| 转交链路离线跑通 | 自造会发 tool_call 的假模型，驱动 转交→子Agent→回supervisor | 无异常，消息序列符合预期 |
| 消息序列无相邻 assistant | 同上，断言 `-1` 步产物 | 防止 `full_history` 形态回归 |
| 真实转交 | `@pytest.mark.network`，真 DeepSeek | 200 |

假模型要点：`FakeToolModel` 不能用（见 §2.2）。需新写一个继承 `BaseChatModel` 的
`SequencedToolModel`，按调用次数依次返回预设的 `AIMessage`（第 1 次返回带
`transfer_to_*` 的 tool_call，第 2 次返回子 Agent 的工具调用，第 3 次返回终答），
并实现 `bind_tools` 返回自身。放在 `tests/agents/` 下，不进 `src/`。

`tests/agents/test_supervisor_smoke.py` 的 `test_supervisor_agent_compiles` 保留，
它现在的作用只剩「import 期不炸」。

## 3 执行顺序

| 步骤 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| 1 | loop-agent：抽共享常量 + 补 instructions | 无 | 改动小，可立即验收 |
| 2 | loop-agent：`fetch_url` + SSRF 测试 | 无 | 能力补全，离线测试全覆盖 |
| 3 | 实测定档：跑 `probe_both.py` L2 三次 | 步骤 1、2 | 决定是否做 §1.4 搜索预算 |
| 4 | multi-agent 定位实验：**已完成**，结论走路线 A | — | E3 验证可行，A′ 淘汰 |
| 5 | 实施路线 A：两个 supervisor 模块内关 thinking | 步骤 4 | 转交返回 200 |
| 6 | 补 `SequencedToolModel` 与转交测试 | 步骤 5 | CI 能拦住回归 |
| 7 | 修 §2.7 项目层问题 | 无，可并行 | 代码整洁 |
| 8 | 全量回归 + 文档更新 | 全部 | 测试 / lint / 文档全绿 |

步骤 1、2、7 互相独立，步骤 4 可与 1~3 并行。**步骤 5 在步骤 4 出结论前不动手**，
否则有白改的风险。

## 4 本方案不做的事

- **不摘除 agent**：路线 E 只作为「演示前的临时止血」，不作为终态 —— 摘掉等于放弃
  简历里「主管调度、嵌套层级主管」两条能力。
- **不改 `core/llm.py` 的全局默认行为**：除走路线 B 且单独排期验收，否则会波及全部 12 个 agent。
- **不改注册表结构**：`agents.py` 的 id→单图 设计导致运行时换模型对非 loop agent 不支持，
  重构它属于独立课题，收益低，本次只记录为已知限制。
- **不引入新依赖**（`fetch_url` 走零依赖实现）：除路线 B 必须新增 `langchain-deepseek`。
