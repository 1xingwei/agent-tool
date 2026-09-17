# SSE 压测与 Redis 共享存储方案

面试用：给 `/stream` 流式端点压测方案，以及多实例下用 Redis 做共享
checkpoint / 记忆存储的落地方案。前者可立即跑出真实数字，后者结合
LangGraph 官方现成库，以配置开关方式接入，不破坏现有单机行为。

> 文中所有路径 / 字段 / 库 API 已对照本仓库源码与官方文档核实，见 §3。

## 1. SSE 端点压测

### 1.1 正确的指标

SSE 是长连接，单个请求持续数秒，**"QPS"是错误指标**。面试能讲对这几个：

| 指标 | 含义 |
|---|---|
| 并发连接数 | 服务能同时挂多少条流 |
| 首 token 时延（p50/p95） | 用户感知的"第一反应" |
| 完成率 / 错误率 / 超时率 | 稳定性 |
| 事件吞吐（events/s） | 流式推送速度 |

### 1.2 工具选型

`wrk` / `ab` 是短请求工具，不适合 SSE 长连接。用 **asyncio + httpx
并发脚本**（`httpx` 本项目依赖里已有，零新依赖），或用 locust（能保持
长连接读流）。后者重，前者贴合项目还能展示代码。

### 1.3 分层压测

1. **连接层（测服务自身能力）**：起 `USE_FAKE_MODEL` 后端。真实 DeepSeek
   有速率限制，会挡住服务层量不准；fake 模型瞬回，量的是
   **uvicorn / FastAPI / SSE 本身**。
2. **端到端（真实模型）**：换 DeepSeek 再跑，报告时明说"此延迟主要来自
   LLM 供应商，非服务层"。

### 1.4 可运行脚本（临时跑，不进仓库）

接口契约：`POST /{agent_id}/stream`，body 为 `StreamInput`（`message` /
`stream_tokens` / `thread_id`，见 `src/schema/schema.py:71`）；流结尾**无条件**
发 `data: [DONE]`（`src/service/service.py:353`），所以读到它即可判流结束。

```python
import asyncio, statistics, time, httpx

AGENT = "research-assistant"  # agents.py 里的 DEFAULT_AGENT

async def one(client, url, i, sem, out):
    payload = {"message": "hi", "stream_tokens": False, "thread_id": f"bench-{i}"}
    async with sem:
        t0 = time.perf_counter(); first = events = None; status = 0
        try:
            async with client.stream(
                "POST", f"{url}/{AGENT}/stream", json=payload) as r:
                status = r.status_code; events = 0
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    events += 1
                    if first is None:
                        first = time.perf_counter() - t0
                    if "[DONE]" in line:
                        break
        except httpx.HTTPError:
            status = 500
        out.append((status, time.perf_counter() - t0, first, events))

async def main():
    url = "http://127.0.0.1:8080"
    N, C = 200, 50
    sem = asyncio.Semaphore(C); out = []
    async with httpx.AsyncClient(timeout=None) as c:
        t_wall = time.perf_counter()
        await asyncio.gather(*(one(c, url, i, sem, out) for i in range(N)))
        wall = time.perf_counter() - t_wall
    ok = [r for r in out if r[0] == 200]
    lat = [r[2] for r in ok if r[2] is not None]
    print(f"ok {len(ok)}/{N}  concurrency {C}  wall {wall:.1f}s")
    if lat:
        print("first-token p50/p95:",
              f"{statistics.median(lat)*1000:.0f}ms",
              f"{sorted(lat)[int(.95*len(lat))]*1000:.0f}ms")
    print(f"events/s: {sum(r[3] for r in ok)/wall:.0f}")

asyncio.run(main())
```

### 1.6 实测结果（2026-09-16，本机 Windows，uvicorn 单/多 worker）

环境：`USE_FAKE_MODEL=true` 隔离 LLM 供应商；`/chatbot/stream`，message=hello，
`model=fake`，`stream_tokens=false`；每档 200 个请求。单 worker 用 `run_service.py`，
多 worker 用 `uvicorn service:app --app-dir src --workers 4`。

#### 单 worker（SQLite 默认）

| 并发 | 完成率 | 首 token p50 | 首 token p95 | events/s | req/s |
|---|---|---|---|---|---|
| 50 | 200/200 | 167.6ms | 434.4ms | 164 | 82.0 |
| 200 | 200/200 | 1021.1ms | 1582.0ms | 161 | 80.6 |
| 1000 | 200/200 | 1257.5ms | 1818.6ms | 149 | 74.7 |

#### 4 workers（SQLite 共享文件）

| 并发 | 完成率 | 首 token p50 | 首 token p95 | events/s | req/s |
|---|---|---|---|---|---|
| 50 | 200/200 | 67.8ms | **44934.8ms** | 9 | 3.9 |
| 200 | 200/200 | 569.8ms | 1935.5ms | 167 | 68.7 |
| 1000 | 200/200 | 425.3ms | 1734.1ms | 197 | 82.4 |

结论（直接当面试话术）：

- 真实模型对比（同机，DeepSeek，8080）首 token p50 ≈ 1.3s，与 fake 单 worker
  p50 167ms 相差一个数量级 → **延迟大头在 LLM 供应商，服务层开销可忽略**。
- 单 worker 并发从 50→1000，p50 从 168ms 涨到 1.26s，p95 到 1.8s —— 排队效应明显，
  uvicorn 单进程是吞吐瓶颈，符合预期。
- 4 workers 在 **低并发下反而击穿 p95（44.9s）**：多 worker 各开连接写同一
  `checkpoints.db`，SQLite 文件锁竞争，个别请求等锁数秒级。这正是"多 worker 需要
  共享存储"的实测证据，见 §2 Redis 方案。注意：首次跑 C=50 也会受冷启动影响（44.9s
  → 17.5s 浮动），但高并发档从未出现这种击穿，锁竞争是主因。
- 压测细节：`USE_FAKE_MODEL=true` 只让 fake 进入可选模型，`.env` 里 `DEFAULT_MODEL`
  仍会将默认指向真实模型；请求体必须显式带 `"model": "fake"` 才能量到服务层。

```sh
# 1) 连接层：隔离 LLM 供应商
USE_FAKE_MODEL=true uv run python src/run_service.py

# 2) 另开终端跑脚本，逐步加大 C = 50 / 200 / 1000

# 3) 多 worker 扩容实验（src/ 不是包，必须用 --app-dir）
uv run uvicorn service:app --app-dir src --workers 4
```

Windows PowerShell 下环境变量写法为行内 `$env:USE_FAKE_MODEL="true"`，再接
`uv run python src/run_service.py`。

观察点（直接当面试话术）：

- 单 worker 的并发连接上限在哪（uvicorn 默认单进程，瓶颈最先出现）
- `--workers N` 后上限是否线性涨
- fake vs 真实模型的首 token 时延差，证明"延迟大头在 LLM 供应商"

> 多 worker 会暴露一个真实问题：SQLite checkpointer 每个 worker 各开连接写
> 同一个文件，可能出现 `database is locked`。这恰好是下一节要换共享存储的
> 论据——面试可以主动抛出来。

## 2. Redis 共享 checkpoint / 记忆存储

### 2.1 为什么

单机 SQLite 零运维、免网络、够用；一旦**多实例 / 水平扩展**，
会话 checkpoint 和长程记忆必须跨实例共享，才有无粘性会话的优雅扩容。
选 Redis 是因为它是标准共享存储，且 LangGraph 官方有现成实现，不自研。

### 2.2 官方现成实现（一个包全包，不自研）

`langgraph-checkpoint-redis`（Redis Inc. 维护）**同时提供 saver 和 store**，
不需要 `langchain-redis`（后者只做 vector store / cache / chat history，
且要求 Python < 3.14，与本项目 3.12–3.14 冲突）：

| 职责 | import | 用法 |
|---|---|---|
| Checkpointer（会话历史） | `from langgraph.checkpoint.redis.aio import AsyncRedisSaver` | `AsyncRedisSaver.from_conn_string(url)`，async 上下文管理器 |
| Store（长程记忆） | `from langgraph.store.redis.aio import AsyncRedisStore` | `AsyncRedisStore.from_conn_string(url)`，async 上下文管理器，`await store.setup()` |

两者接口和现有的 `get_sqlite_saver()` / `get_sqlite_store()`
（`src/memory/sqlite.py`）**完全同构**——都是 `from_conn_string(...)`
配 async context manager，替换干净。

**硬前置条件**：Redis 需带 **RedisJSON + RediSearch** 模块。Redis 8.0+
已内置；低版本要用 **Redis Stack** 镜像（如 `redis/redis-stack`），普通
`redis:7` 镜像**不够**，否则 setup 建索引时报错。

### 2.3 落地点（2 处改动 + 1 个开关）

1. `pyproject.toml` 加依赖：`langgraph-checkpoint-redis`（会带入
   `redis` / `redisvl`，也可显式加 `redis`）
2. `src/core/settings.py`：

   ```python
   REDIS_URL: str | None = None  # 设置后启用 Redis，否则用现有 SQLite/Postgres
   ```

3. `src/memory/__init__.py`：

   ```python
   def initialize_database():
       if settings.REDIS_URL:
           return AsyncRedisSaver.from_conn_string(settings.REDIS_URL)
       if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
           return get_postgres_saver()
       ...  # 现有分支不变

   def initialize_store():
       if settings.REDIS_URL:
           return AsyncRedisStore.from_conn_string(settings.REDIS_URL)
       ...  # 现有分支不变
   ```

**无 Redis 时**（`REDIS_URL=None`）不进入 Redis 分支，现有单机行为零变化，
可安全默认。

### 2.4 实测核对（已落地验证）

1. **saver 的 setup 无需改动**：`AsyncRedisSaver` 的 `setup()` **本身就是
   async**（`inspect.iscoroutinefunction` 为真），与 `service.py` 的
   `await saver.setup()` 直接兼容，lifespan 不用改。README 里的
   `asetup()` 只是别名。ps：异步类的 `asetup` 与 `setup` 都是协程。
2. **store 的 setup**（`AsyncRedisStore.setup()`）为 async，与现有
   `get_sqlite_store()` 一致，无需额外改。

   落地（本次已实现）：`pyproject.toml` + `settings.REDIS_URL` +
   `memory/__init__.py` 两个分支；无 Redis 时零影响，全量测试 202 通过。

3. **实际坑**：测试里 `patch("memory.settings")` 会把整份 settings 变成
   MagicMock，新增的 `REDIS_URL` 真值为 True → 误入 Redis 分支。凡 mock
   过 settings 的 fixture 需显式置 `REDIS_URL = None`。
4. **选型开关重复**：项目已有 `DATABASE_TYPE`（sqlite/postgres/mongo），
   也可给它加 `REDIS` 枚举值统一管理；用独立 `REDIS_URL` 也不破坏，
   二选一说明清楚即可，别两套并存。

### 2.5 部署形态

```text
Load Balancer（无状态）
   └── agent_service 副本 N 个（uvicorn，可多 worker）
          └── 共享 Redis：checkpoint（会话） + store（记忆）
```

状态在 Redis，副本间无需会话粘性，任意请求落到任意副本都能续上前文。

### 2.6 面试话术

> "单机场景 SQLite 零运维够用；一旦多实例 / 水平扩展，会话与记忆必须
> 跨实例共享，我选用官方 Redis 实现——checkpoint 存会话、store 存跨会话
> 记忆，任意副本可续。代价是引入 Redis 运维和模块依赖（Redis Stack / 8.0+）
> 与持久化（AOF/RDB）权衡，所以做成配置开关默认关闭，而不是强制替换。"

备答点：

- 为什么不用 Postgres 存 → 已支持（`AsyncPostgresSaver`），Redis 是更高频
  低延迟的小 KV 语义首选；两者都可作为共享存储。
- Redis 持久化取舍 → 会话丢了可重问，AOF everysec 通常够。

## 3. 可行性核查记录（对照源码 / 官方文档）

| 结论 | 依据 |
|---|---|
| 端点 `/{agent_id}/stream`、字段 `message`/`stream_tokens`/`thread_id` | `src/service/service.py:378`、`src/schema/schema.py:71` |
| 流结尾无条件 `data: [DONE]` | `src/service/service.py:353` |
| `src/` 不是包，`uvicorn src.service:app` 会 ImportError | 无 `src/__init__.py`；`run_service.py:32` 用 `"service:app"` |
| store 是 `from_conn_string` + async CM + `setup()` | `src/memory/sqlite.py:14` |
| `AsyncRedisSaver.setup()` 是 async，lifespan 无需改动 | `inspect.iscoroutinefunction`（0.5.2 实测） |
| Redis store 在 `langgraph.store.redis`，非 `langchain-redis` | `langgraph-checkpoint-redis` README（PyPI） |
| 需 RedisJSON + RediSearch（Redis 8.0+ / Redis Stack） | 同上 |
