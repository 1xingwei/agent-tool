# AG-UI 协议支持

服务把每一个 agent 都通过 [AG-UI 协议](https://docs.ag-ui.com) 暴露出来——这是一个开放的、基于事件的标准，用于把 agent 接入面向用户的应用，被 [CopilotKit](https://docs.copilotkit.ai) 以及越来越多框架采用。它让你能为 agent 构建生产级 React/Next.js 前端，同时保留 Streamlit 应用用于开发调试。

繁重的工作（把 LangGraph 的执行过程翻译成 AG-UI 事件）由官方 [`ag-ui-langgraph`](https://pypi.org/project/ag-ui-langgraph/) 包完成。服务层只加了一层薄适配器（`src/service/agui.py`），把它接进 agent 注册表、Bearer 鉴权与 Langfuse 追踪。

## 接口

| 接口 | 说明 |
| --- | --- |
| `POST /agui/{agent_id}/run` | 运行指定 agent，通过 SSE 流式返回 AG-UI 事件 |
| `POST /agui/run` | 同上，使用默认 agent |

请求体是标准的 AG-UI `RunAgentInput`。鉴权与其余接口一致，使用同一个 `AUTH_SECRET` Bearer 令牌。

## 接入前端

生产环境的标准做法是 CopilotKit 架构：前端对接 [CopilotKit runtime](https://docs.copilotkit.ai)（例如一个 Next.js API 路由），由该 runtime 在服务端用 AG-UI 的 `HttpAgent` 连接本服务：

```ts
import { HttpAgent } from "@ag-ui/client";

const agent = new HttpAgent({
  url: "http://your-service:8080/agui/research-assistant/run",
  headers: { Authorization: `Bearer ${process.env.AUTH_SECRET}` },
});
```

runtime 持有 Bearer 令牌，充当浏览器与 agent 服务之间的可信层——角色等同于 Streamlit 应用之于原生 API。浏览器也可以直接访问该接口用于实验，但你需要自行给服务加上 CORS 中间件，而且 `AUTH_SECRET` 会暴露给浏览器——建议优先采用 runtime 模式。

## 试跑

仓库内含一个使用官方 SDK 的最小参考客户端：

```sh
# 在一个终端启动服务（或 docker compose watch）
python src/run_service.py

# 在另一个终端
cd scripts/agui-client
npm install
node client.mjs "Tell me a joke!" chatbot
```

用 `THREAD_ID` 继续已有对话，按需设置 `AUTH_SECRET` / `AGENT_URL`：

```sh
THREAD_ID=my-thread node client.mjs "And another one" chatbot
```

## 行为说明

- **thread 与原生 API 共享。** 两种协议使用同一个 checkpointer，以 thread ID 为键，所以在 `/stream` 上开始的对话可以在 AG-UI 上继续，反之亦然。有一点需要注意：消息按 ID 去重，因此不要用新生成的 ID 重放历史消息（规范实现的 AG-UI 客户端会保留 ID，不受影响）。
- **按请求传入配置**放在 `forwardedProps.configurable` 中——它对应原生 API 的 `model` / `user_id` / `agent_config` 字段，例如 `{"forwardedProps": {"configurable": {"model": "gpt-5.2"}}}`。由协议管理的键（`thread_id`、`checkpoint_id`、`checkpoint_ns`）会被拒绝。`model` 会对照 `AVAILABLE_MODELS` 校验（不允许则返回 400），与原生 API 一致。
- **中断**（human-in-the-loop）以一个名为 `on_interrupt` 的 `CUSTOM` 事件呈现。恢复方式是用同一个 thread 再跑一次，并带上 `{"forwardedProps": {"command": {"resume": <answer>}}}`。
- **图状态对客户端可见。** AG-UI 的共享状态特性会发送包含完整图状态的 `STATE_SNAPSHOT` 事件，因此不要把密钥或仅限内部的数据放进 agent 状态。
- **`RAW` 透传事件被适配器过滤掉了。** 标准 AG-UI 客户端会忽略它们，而它们会把服务端内部实现（包括完整渲染后的 prompt）暴露给调用方。如果需要在可信层之后拿到完整事件流做调试（例如 AG-UI Event Inspector），删掉 `src/service/agui.py` 里的过滤即可。
- **`/feedback` 与 `/history` 未做桥接。** AG-UI 的 `runId` 由客户端生成，不作为 LangSmith 的 run ID 使用，因此星级反馈对 AG-UI 运行无效。AG-UI 客户端自行从事件流管理消息历史。
- `ag-ui-langgraph` 包尚未到 1.0，因此按此固定版本。若将来 LangGraph 的某个大版本与它冲突，应当放弃或滞后该集成，而不是拖住核心升级。
