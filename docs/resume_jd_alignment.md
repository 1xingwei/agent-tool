# 简历 / JD 对齐

面向 Agent 开发岗位的简历条目，以及对照企业 JD 的能力核查。

## 简历条目

**2026.05 -- 至今　　　Agent 服务中台（agent-service-toolkit）　　　开发人员**

本项目基于 LangGraph、FastAPI、LangChain、Streamlit，自研了一套面向多智能体的 AI Agent 服务中台，提供统一的智能体接入、编排、对话与记忆能力，支持多轮对话的会话持久化、跨会话长期记忆、工具调用与人工介入。

**项目功能：** 平台内置 12 个职能各异的智能体，覆盖研究助手（联网搜索+计算）、代码库审查、RAG 知识问答、主管编排、人工审批、GitHub 仓库操作（MCP）与后台任务调度等场景；统一基于 FastAPI 的 SSE 流式接口与 AG-UI 协议对外提供服务，内置内容安全过滤（safeguard）与 AUTH_SECRET 鉴权；支持 DeepSeek / OpenAI / Anthropic / Gemini / 本地模型的多供应商切换。

**项目成果：**
1、基于 LangGraph 状态图 + ReAct / Tool Calling 的设计落地多智能体编排框架，覆盖单智能体、ReAct 循环、主管调度、嵌套层级主管等编排模式；通过 interrupt 机制实现敏感操作的人工审批（HITL）流，支撑工具调用审计闭环；
2、工具层实现完整执行链路：联网搜索、命令行执行、代码库只读审查（git log/diff、文件检索）等真实工具；对接 MCP 协议（基于 langchain-mcp-adapters 构建 GitHub MCP 客户端，动态加载远端工具）；工具侧做路径穿越防护、固定 git 参数防注入、30s 超时与最大步骤数兜底，防范 Agent 死循环与成本失控；
3、实现 Agent 记忆与上下文管理：checkpointer 存储会话状态、Store 存储跨会话长程记忆，支持 SQLite / Postgres / MongoDB / Redis 多后端一键切换，任意请求可续上下文；中断-恢复场景下依赖 checkpointer 保证状态一致性；
4、服务端针对高并发做了分层压测（50/200/1000 并发），量化服务层与 LLM 供应商的延迟占比，发现并定位多 worker 下 SQLite checkpoint 文件锁竞争导致的 P95 击穿（44.9s），据此设计并验证 Redis 共享存储迁移方案，给出多副本部署路径；
5、工程质量：209 个单测用例全绿，pytest + pytest-cov 覆盖统计、ruff 静态检查、GitHub Actions CI、pre-commit 全链路接入；对接 LangSmith / Langfuse 观测平台，可追踪 Agent 决策链路与工具调用轨迹。

## JD 能力核查

2026 年 Agent 岗位门槛（来源：腾讯 / 拼多多 / 京东 / 海康 / 阿里云等真实 JD）：

| 能力 | 状态 | 备注 |
|---|---|---|
| LangGraph / LangChain 主导项目 | 达成 | 12 个 agent 架构 |
| Python + FastAPI 后端 | 达成 | SSE 流式 + AG-UI |
| 任务拆解 / 工具调用 / 记忆 | 达成 | ReAct、Tool Calling、checkpointer + store |
| HITL 人工介入 | 达成 | interrupt 机制 |
| 可观测性（LangSmith/Langfuse） | 达成 | 决策链路 + 工具轨迹追踪 |
| 压测 / 稳定性数据 | 达成 | 分层压测 + SQLite 锁击穿定位 |
| RAG 全链路 | **未达成** | rag_assistant 用 Chroma，本机无 key 跑不通（见 RAG_Assistant.md） |
| MCP Server 开发 | 部分 | 目前仅 client 消费端，非自建 Server |
| Redis 集成上线 | 部分 | 代码就位+单测过，集成验证未做（见 load_testing_and_redis.md） |
| 模型本地部署 / 微调 | 未达成 | 简历不提 |

## 面试须知

- 压测与 SQLite 击穿是有实测数据的真故事，是全场最硬的资历。
- RAG、MCP Server、Redis 集成三处是追问风险点：答不上细节就不要在简历亮出来，或先补上再过面。