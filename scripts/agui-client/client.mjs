// 用于手动验证服务 /agui 端点的最小 AG-UI 客户端，
// 基于官方 @ag-ui/client SDK 构建。详见 docs/02-AG-UI协议支持.md。
//
// 用法：
//   cd scripts/agui-client
//   npm install
//   node client.mjs [message] [agent]
//
// 环境变量：
//   AGENT_URL   - agent 服务的基础 URL（默认：http://localhost:8080）
//   AUTH_SECRET - bearer token，如果服务配置了的话
//   THREAD_ID   - 复用 thread 以继续对话（默认：随机）

import { randomUUID } from "crypto";
import { HttpAgent } from "@ag-ui/client";

const message = process.argv[2] ?? "Tell me a joke!";
const agentId = process.argv[3] ?? "chatbot";
const baseUrl = process.env.AGENT_URL ?? "http://localhost:8080";
const threadId = process.env.THREAD_ID ?? randomUUID();

const agent = new HttpAgent({
  url: `${baseUrl}/agui/${agentId}/run`,
  threadId,
  headers: process.env.AUTH_SECRET
    ? { Authorization: `Bearer ${process.env.AUTH_SECRET}` }
    : {},
});

console.log(`agent: ${agentId}  thread: ${threadId}`);
console.log(`user: ${message}\n`);

agent.messages = [{ id: randomUUID(), role: "user", content: message }];

const eventTypes = new Set();
let printedPrefix = false;
try {
  await agent.runAgent(
    { runId: randomUUID() },
    {
      onEvent({ event }) {
        eventTypes.add(event.type);
      },
      onTextMessageContentEvent({ event }) {
        // 每个增量到达时立即打印，而非重新渲染整个缓冲区——
        // 通过 \r 进行原地覆盖依赖于终端支持，而这一点并不
        // 一致（例如通过 `docker compose logs` 管道输出）。
        if (!printedPrefix) {
          process.stdout.write("assistant: ");
          printedPrefix = true;
        }
        process.stdout.write(event.delta);
      },
      onCustomEvent({ event }) {
        if (event.name === "on_interrupt") {
          console.log(`[interrupted] ${JSON.stringify(event.value)}`);
          console.log(
            "resume by running the same thread with forwardedProps {command: {resume: <answer>}} - see docs/02-AG-UI协议支持.md"
          );
        }
      },
    }
  );
} catch (err) {
  console.error(`\nrun failed: ${err.message ?? err}`);
  if (String(err.message).includes("401")) {
    console.error("hint: set AUTH_SECRET to the service's bearer token");
  }
  process.exit(1);
}

console.log(`\n\nevent types received: ${[...eventTypes].join(", ")}`);
console.log(`continue this thread with: THREAD_ID=${threadId} node client.mjs '<message>' ${agentId}`);
