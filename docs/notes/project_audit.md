# 全项目审核报告

审核时间：2026-09-17 19:45 -- 20:05
审核基线：`main` @ `883f1ab`，工作区干净
审核方式：静态走查 + 动态实测（服务在线，真实 API 调用）

结论一句话：**工程质量闸门全绿，但 12 个 agent 里有 4 条核心路径在本机实际不可用，
且 CI 对此零感知。**

> 本报告是 20:05 的**快照**。P0 四项与 P1 若干项已于 20:10 修完并对在线服务复验通过，
> 明细见文末「七、修复记录」；正文保留原样以便对照。

## 一、健康面（先说做对的）

| 项目 | 实测结果 |
|---|---|
| `pytest` 全量 | 212 passed, 5 skipped（38.3s） |
| 覆盖率 | 2633 stmts / 755 miss / **71%** |
| `ruff format --check` | 91 files already formatted |
| `ruff check` | All checks passed |
| `pyrefly check` | 0 errors |
| `pymarkdown scan README.md docs/ -r` | 通过 |
| 鉴权 | `8080` 无 token = 401，带 token = 200 |
| UI | `8501` = 200 |
| 敏感文件 | `.env` 已被 `.gitignore:147` 覆盖，未被跟踪 |
| 运行时产物 | `logs/` `var/` 均被忽略，根目录无残留 |

安全设计上确有几处真做的：`AUTH_SECRET` 空值告警（`service.py:95-99`）、
`agent_config` 保留键冲突拒绝（`service.py:166-170`）、
`read_file` 的目录穿越与越界双重校验（`code/tools.py:144-148`）、
git 参数固定 + `timeout=30`（`code/tools.py:12-20`）、
`remaining_steps` 兜底（`research_assistant.py:78-86`）。

## 二、P0：已实测复现的功能性缺陷

### P0-1　`code-reviewer` 首次被记录：git 工具在本机必崩（新发现）

**实测**：`POST /code-reviewer/invoke` → **500**（0.99s）。
`logs/service.err.log` 落盘：

```text
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa1 in position 64
ERROR:service.service:An exception occurred: 'NoneType' object has no attribute 'strip'
```

**根因**（`src/agents/code/tools.py:13-23`）：

```python
result = subprocess.run(cmd, cwd=repo_path, capture_output=True,
                        text=True, timeout=30, check=False)   # ← 没指定 encoding
...
return result.stdout.strip()
```

`text=True` 且未给 `encoding` 时，Python 用 `locale.getpreferredencoding()` 解码。
服务进程（计划任务 → `cmd.exe` 启动）拿到的是 **GBK/cp936**，
而 git 输出的是 UTF-8。本仓库**每个 commit 信息都是中文**，于是：

1. subprocess 的 reader 线程解码失败抛 `UnicodeDecodeError`（在子线程里，不冒泡）；
2. `result.stdout` 被置为 `None`；
3. `result.stdout.strip()` → `AttributeError` → `/invoke` 返回 500。

**影响面**：`git_log` 和 `git_diff` 两个工具全废，
即这个 agent 的**核心能力（代码库审查）完全不可用**。
`file_search` / `read_file` 不受影响（它们显式用 `encoding="utf-8"`，
见 `code/tools.py:117,149`）——这也反证了问题就出在 `_git()` 漏了编码参数。

**为什么 CI 是绿的**：`tests/agents/test_code_reviewer.py` 确实跑了真 git，
但两重错位让它碰不到这个 bug：

- 测试仓库的提交信息全是 ASCII（`f"add {name}"`，见 `:20`）；
- pytest 从 Git Bash 启动，继承 `PYTHONUTF8=1`，解码根本不会走 GBK。

**修法**（一处）：`_git()` 加 `encoding="utf-8", errors="replace"`。
改后仍需一条**含非 ASCII 提交信息**的用例，否则回归照样拦不住。

### P0-2　两个 supervisor agent 转交必 500（已复现，与既有记录一致）

| agent | 实测 |
|---|---|
| `langgraph-supervisor-agent` | **500**（3.21s） |
| `langgraph-supervisor-hierarchy-agent` | **500**（1.88s） |

`service.err.log` 去重后只有一条：
`400 - The reasoning_content in the thinking mode must be passed back to the API.`

纯对话（不触发转交）正常，一转交就崩 —— 这两个 agent 在演示里就是「点进去 500」。
成因与候选路线已完整记录在 `docs/notes/agent_capability_plan.md`，此处不重复。

### P0-3　语音功能静默失效，且原因比记录的更精确

`README.md:59` 宣称「voice input and output」。实测：`VoiceManager.from_env()` 恒返回 `None`。

**精确成因**（本轮新验，比此前记录深一层）：
`python-dotenv` 并不把 `KEY=   # 注释` 解析成空串，而是**把注释本身当成值**：

```text
VOICE_STT_PROVIDER = "# Speech-to-text provider (only 'openai' supported currently)"
VOICE_TTS_PROVIDER = "# Text-to-speech provider (only 'openai' supported currently)"
```

这个字符串被 `SpeechToText.from_env()` / `TextToSpeech.from_env()` 当作 provider 名，
抛出 `ValueError: Unknown STT provider: # Speech-to-text provider...`，
异常被吞（`voice/tts.py:149` 附近）后 `return None` → 语音功能整体关闭，
UI 上表现为「语音按钮灰着」，无任何报错。

涉及 `.env:118-119` 与 `.env.example:114-115` 四行。修法：**去掉行内注释**（或注释另起一行）。

### P0-4　唯一的联网用例本身是错的

`tests/agents/test_supervisor_smoke.py:12-14`：

```python
result = await web_search.ainvoke("langgraph agent framework")
assert "snippet" in result
```

**实测 `--run-network` → FAILED**：工具返回的是 `title / href / body` 三段格式
（`agents/tools.py:68-71`），**永远不会出现 "snippet" 这个词**。

它被 `@pytest.mark.network` 挡在默认测试之外（conftest 自动 skip），
CI 也不传 `--run-network`，所以这条断言错了多久没人知道。
它是全项目**唯一**的联网测试 —— 等于联网路径实际零覆盖。

## 三、P1：工程与配置

### P1-1　`pyproject.toml` 指向一个已被删除的文件

```toml
# Policy and override procedure: .claude/skills/dependency-refresh/SKILL.md
```

`.claude/` 目录不存在（在 `a7c2b62 Remove maintainer scaffolding from template instance`
里被删了，但这条指针没同步清掉）。**指向死路径的注释比没有注释更糟** ——
下一个人按它去找会扑空。

### P1-2　依赖冗余与过期

| 现象 | 证据 |
|---|---|
| 同时声明 `duckduckgo-search>=8.1.1` 与 `ddgs>=9.14.4` | `pyproject.toml:21,60`；代码只用 `ddgs`（`tools.py:57`），前者是它的前身包 |
| `langchain-community` 已在 sunset | 导入即发 `DeprecationWarning`，官方明示不再维护；本项目只用它做 OpenWeatherMap |

### P1-3　天气工具实际上从未注册

`research_assistant.py:32-36` 用 `if settings.OPENWEATHERMAP_API_KEY:` 守卫，
而本机该 key **NOT SET** → 工具不进 `tools` 列表。
这不是 bug（守卫是对的），但意味着 `langchain-community` 这条 sunset 依赖
目前是**为一个不生效的功能背着的**。

### P1-4　`remember_review` 在无 store 时崩（潜在）

`code_reviewer.py:86` 直接 `await store.aput(...)`，假定注入的 `store` 非空。
实测：服务内经 `lifespan` 注入后正常（无工具路径返回 200）；
但**脱离服务直接调用图**（如 `langgraph dev`、单测、`run_agent.py`）会得到
`AttributeError: 'NoneType' object has no attribute 'aput'`。

`tests/agents/test_code_reviewer.py:64-75` 用手工传入的 `InMemoryStore` 绕过了这一点，
所以测试是绿的 —— **测的是「有 store 时的行为」，而风险恰恰在「没有 store 时」**。

### P1-5　`hierarchy` 的 prompt 拼接漏空格

`langgraph_supervisor_hierarchy_agent.py:37-38` 两个字符串字面量相邻拼接：

```python
"...with math capabilities." "For current events, use research_agent. "
```

拼出 `capabilities.For`。不影响运行，但会被模型读到，属低级瑕疵。

### P1-6　跨模块再导出引发归属混乱

`hierarchy:4` 从 `agents.langgraph_supervisor_agent` 导入 `web_search`，
而 `web_search` 的真正定义在 `agents/tools.py:77`，
`langgraph_supervisor_agent.py` 本身也是从 `research_assistant` 转发来的。
`test_supervisor_smoke.py:3` 又照着这条链导入 —— 一层的再导出被当成了源头。

## 四、P2：文档与一致性

| 项 | 现状 | 应改为 |
|---|---|---|
| `docs/notes/resume_jd_alignment.md:18` | 「**209** 个单测用例全绿」 | 212 passed / 5 skipped（数字已漂移） |
| 同上 `:11` | 「内置内容安全过滤（safeguard）」 | 代码在，但 `GROQ_API_KEY` 未配时 `Safeguard` 恒返回 SAFE（`safeguard.py:92-94,112-113`），**本机是空转的** |
| `README.md:59` | 「including voice input and output」 | 见 P0-3，本机实际关闭 |
| 弱测试 | `test_loop_agent.py:5-6`、`test_supervisor_smoke.py:7` 仅断言「图能编译 / 有 ainvoke」 | 编译通过 ≠ 能跑通；这两个 agent 的 500 都从这类断言的缝里漏过去了 |

**一条共性**：`safeguard_input` 在每个请求上跑 `Safeguard()`，
未配 Groq 时是**同步 `print()` 到 stdout**（`safeguard.py:93`），
在服务里应走 logger。属实现瑕疵，不计入缺陷。

## 五、建议动手顺序

按「投入产出比 + 风险」排：

| # | 动作 | 改动量 | 风险 | 收益 |
|---|---|---|---|---|
| 1 | `_git()` 加 `encoding="utf-8"` + 补一条非 ASCII 提交的用例 | 1 行 + 1 用例 | 极低 | **恢复 code-reviewer 的核心能力** |
| 2 | 清掉 `.env` / `.env.example` 那 4 行行内注释 | 4 行 | 极低 | 恢复语音功能 |
| 3 | 修 `test_supervisor_smoke.py:14` 的断言（改判 `title`/`href`） | 1 行 | 极低 | 让唯一联网用例变成真测试 |
| 4 | 删掉 `pyproject.toml` 的死路径注释、去掉 `duckduckgo-search` | 2 行 | 低 | 消除误导 |
| 5 | supervisor 转交 500（走 `agent_capability_plan.md` 路线 A） | ~4 行 | 低 | 恢复 2 个 agent |
| 6 | `remember_review` 加 store 空值兜底 | 2 行 | 极低 | 消除潜在崩溃 |
| 7 | 同步 `resume_jd_alignment.md` 的数字与措辞 | 文档 | 无 | 面试不被追问打脸 |

第 1--4、6 项互相独立，可以一次做完再统一验证；
第 5 项需要真实 API 调用，建议单独一轮。
第 7 项在简历投出前必须做 —— 那份文档就是面试时的自证材料。

## 六、审核未覆盖

- 未做压测复验（历史数据见 `load_testing_and_redis.md`）
- 未做 Docker 端到端（本机未跑 `--run-docker`）
- 未验证 Postgres / MongoDB / Redis 三种后端的真实连接
- 未审查 `src/voice/` 与 `src/client/` 的逐行实现（仅覆盖配置解析路径）

## 七、修复记录（2026-09-17 20:10）

全部改动都对**在线服务**重发过真请求，不只跑单测。

| 项 | 修法 | 验证 |
|---|---|---|
| P0-1 git 工具编码 | `_git()` 加 `encoding="utf-8", errors="replace"` | `/code-reviewer/invoke` **500 → 200**（21.3s），且读到真实中文提交历史 |
| P0-2 supervisor 转交 | 新增 `core.llm.get_supervisor_model()`，以 `model_copy` 关掉 DeepSeek thinking | 两个 agent **500 → 200**（2.87s / 4.74s），`12×7=84`、`9×8=72` |
| P0-3 语音静默失效 | `.env` / `.env.example` 四行行内注释另起一行 | `dotenv_values` 解析为 `''`，走 `logger.debug("... disabled")` 的干净关闭路径 |
| P0-4 联网用例断言 | 改判 `title / href / body` 的真实格式 | `--run-network` 由 FAILED 转 **2 passed** |
| P1-1 死路径注释 | 删掉 `.claude/skills/dependency-refresh/SKILL.md` 指针 | — |
| P1-2 冗余依赖 | 去掉 `duckduckgo-search` 并 `uv lock` | `uv.lock` 仅 −16 行，无版本漂移 |
| P1-4 store 空值 | `remember_review` 加空值守卫 + warning | 新增用例覆盖 `store=None` |
| P1-5 prompt 漏空格 | `capabilities.` 补空格 | — |
| P1-6 跨模块再导出 | `web_search` 改从定义处 `agents.tools` 导入 | — |
| P2 简历数字 | 单测数改为实测值 | — |

两道关键设计，避免修一处伤一片：

- **`get_supervisor_model` 用 `model_copy` 而非重构造**，所以 `get_model` 的 `@cache`
  实例（其余 10 个 agent 共用）不受影响；且**仅对 DeepSeek 生效** —— 其他供应商会拒绝
  未知请求字段，这个参数会原样发到它们 API。
- **`_git()` 的编码无法用纯行为断言守住**：pytest 继承 `PYTHONUTF8=1`、CI 跑在 UTF-8
  locale，两处都能正常解码。所以新用例额外**钉住 `encoding` 参数本身**。

**未修**：P1-3（天气工具未注册 —— 守卫本身是对的，属依赖取舍）、P2 的弱测试
（`test_loop_agent.py` 与 `test_supervisor_smoke.py:6` 仍只断言「图能编译」）。

**验证**：`pytest` **214 passed / 5 skipped**（基线 212 + 本轮新增 2 条）、
`ruff format --check` / `ruff check` / `pyrefly 0 errors` / `pymarkdown scan -r` 全过；
`service.err.log` 在三个真请求之后**零异常**（修复前累计 8 条 `reasoning_content`）。
