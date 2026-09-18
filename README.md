# 🧰 AI Agent Service Toolkit

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml) [![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

A full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit.

It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

This project offers a template for you to easily build and run your own agents using the LangGraph framework. It demonstrates a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects by providing a full, robust toolkit.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Overview

### [Try the app!](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600" alt="App screenshot"></a>

### Quickstart

Run directly in python

```sh
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is the recommended way to install agent-service-toolkit, but "pip install ." also works
# For uv installation options, see: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh

# Install dependencies. "uv sync" creates .venv automatically
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600" alt="Agent architecture diagram">

See [docs/01-架构说明.md](docs/01-架构说明.md) for a written walkthrough of the same
picture: layer responsibilities, the request lifecycle, the two-part memory system,
how graphs are wired at startup, and the design trade-offs.

### Documentation

The Chinese docs live in `docs/`, flat and numbered `01`–`17` in reading order.
`01` is the architecture overview, `02`–`07` are provider and feature guides,
`08`–`09` are design specs, and `10`–`17` are engineering records:

| # | Doc | What it covers |
|---|---|---|
| 01 | [架构说明](docs/01-架构说明.md) | Layer responsibilities, request lifecycle, memory wiring, trade-offs |
| 02 | [AG-UI协议支持](docs/02-AG-UI协议支持.md) | Serving every agent over the AG-UI protocol |
| 03 | [RAG检索助手](docs/03-RAG检索助手.md) | The Chroma-backed RAG agent and how to build its index |
| 04 | [GitHub-MCP助手](docs/04-GitHub-MCP助手.md) | The GitHub MCP client agent |
| 05 | [基于文件的凭证管理](docs/05-基于文件的凭证管理.md) | Working with `privatecredentials/` |
| 06 | [谷歌模型接入](docs/06-谷歌模型接入.md) | Setting up VertexAI |
| 07 | [Ollama本地模型接入](docs/07-Ollama本地模型接入.md) | Running against a local model |
| 08 | [面试加固设计](docs/08-面试加固设计.md) | Design spec: ingest pipeline, SQLite store, real search |
| 09 | [代码库审查Agent设计](docs/09-代码库审查Agent设计.md) | The code-review agent as implemented: graph, read-only tools, memory contract, tests |
| 10 | [审核与修复总账](docs/10-审核与修复总账.md) | Route matrix, audit findings, fix plan, landing + verification record |
| 11 | [Agent能力完善方案](docs/11-Agent能力完善方案.md) | loop-agent fixes, the supervisor handoff fix, and its independent review |
| 12 | [压测与Redis共享存储](docs/12-压测与Redis共享存储.md) | Load testing `/stream`; Redis shared checkpoint/store |
| 13 | [面试材料与岗位对照](docs/13-面试材料与岗位对照.md) | Resume bullets and the JD capability check |
| 14 | [注释中文化与审校](docs/14-注释中文化与审校.md) | The comment/docstring Chinese migration and its audit |
| 15 | [记忆与检索层-对标核实与行动清单](docs/15-记忆与检索层-对标核实与行动清单.md) | Memory/retrieval industry baseline and the P0/P1 action list |
| 16 | [精简与重构方案](docs/16-精简与重构方案.md) | Landing plan for the 8-item audit: dedup, scaffolding factory, schema/UI layering |
| 17 | [审查台账](docs/17-审查台账.md) | The audit ledger: per-item evidence behind the 8 findings |

### Key Features

1. **LangGraph Agent and latest features**: A customizable agent built using the LangGraph framework. Implements the latest LangGraph v1.0 features including human in the loop with `interrupt()`, flow control with `Command`, long-term memory with `Store`, and `langgraph-supervisor`.
1. **FastAPI Service**: Serves the agent with both streaming and non-streaming endpoints.
1. **Advanced Streaming**: A novel approach to support both token-based and message-based streaming.
1. **AG-UI Protocol Support**: Every agent is also served over the [AG-UI protocol](https://docs.ag-ui.com) for connecting AG-UI compatible frontends like CopilotKit - see [docs](docs/02-AG-UI协议支持.md).
1. **Streamlit Interface**: Provides a user-friendly chat interface for interacting with the agent, including voice input and output (requires an OpenAI API key plus `VOICE_STT_PROVIDER` / `VOICE_TTS_PROVIDER`; both are unset by default, which disables the feature).
1. **Multiple Agent Support**: Run multiple agents in the service and call by URL path. Available agents and models are described in `/info`
1. **Asynchronous Design**: Utilizes async/await for efficient handling of concurrent requests.
1. **Content Moderation**: Implements Safeguard for content moderation (requires Groq API key).
1. **RAG Agent**: A basic RAG agent implementation using ChromaDB - see [docs](docs/03-RAG检索助手.md).
1. **Chat History**: Lists a user's previous conversations per agent via `/threads`, with a "Previous Chats" sidebar in the Streamlit app.
1. **Feedback Mechanism**: Includes a star-based feedback system integrated with LangSmith.
1. **Docker Support**: Includes Dockerfiles and a docker compose file for easy development and deployment.
1. **Testing**: Includes robust unit and integration tests for the full repo.

### Key Files

The repository is structured as follows:

- `src/agents/`: Defines several agents with different capabilities
- `src/schema/`: Defines the protocol schema
- `src/core/`: Core modules including LLM definition and settings
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface
- `tests/`: Unit and integration tests
- `docs/`: Chinese docs, flat and numbered `01`–`17` in reading order — `01` architecture overview, `02`–`07` provider and feature guides, `08`–`09` design specs, `10`–`17` engineering notes
- `scripts/`: Setup, smoke-test and local-start helpers
- `docker/`: Dockerfiles and the optional MongoDB compose file
- `data/`: Sample corpus used to build the Chroma index
- `logs/`, `var/`: Runtime output and local state (gitignored, recreated on demand)

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/JoshuaC215/agent-service-toolkit.git
   cd agent-service-toolkit
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory. At least one LLM API key or configuration is required. See the [`.env.example` file](./.env.example) for a full list of available environment variables, including a variety of model provider API keys, header-based authentication, LangSmith tracing, testing and development modes, and OpenWeatherMap API key.

3. You can now run the agent service and the Streamlit app locally, either with Docker or just using Python. The Docker setup is recommended for simpler environment setup and immediate reloading of the services when you make changes to your code.

### Additional setup for specific AI providers

- [Setting up Ollama](docs/07-Ollama本地模型接入.md)
- [Setting up VertexAI](docs/06-谷歌模型接入.md)
- [Setting up RAG with ChromaDB](docs/03-RAG检索助手.md)

### Building or customizing your own agent

To customize the agent for your own use case:

1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Handling Private Credential files

If your agents or chosen LLM require file-based credential files or certificates, the `privatecredentials/` has been provided for your development convenience. All contents, excluding the `.gitkeep` files, are ignored by git and docker's build process. See [Working with File-based Credentials](docs/05-基于文件的凭证管理.md) for suggested use.

### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose (>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2. Create a `.env` file from the `.env.example`. At minimum, you need to provide an LLM API key (e.g., OPENAI_API_KEY).

   ```sh
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

   This will automatically:
   - Start a PostgreSQL database service that the agent service connects to
   - Start the agent service with FastAPI
   - Start the Streamlit app for the user interface

4. The services will now automatically update when you make changes to your code:
   - Changes in the relevant python files and directories will trigger updates for the relevant services.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

5. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

6. The agent service API will be available at `http://0.0.0.0:8080`. You can also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

7. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

### Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
print(response.content)
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."
```

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

`langgraph-cli[inmem]` is installed with `uv sync`. You can simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

#### Starting both processes on Windows

`scripts/start.ps1` launches the service and the Streamlit app in the background, redirects their
output into `logs/`, then waits until both ports answer before returning:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Runtime output is kept out of the source tree:

| Path | Contents |
|---|---|
| `logs/` | `service*.log` and `streamlit*.log`, written by `scripts/start.ps1` |
| `var/` | SQLite checkpoints, the long-term memory store, and the Chroma index — see `SQLITE_DB_PATH`, `SQLITE_STORE_PATH` and `CHROMA_DIR` |

Both directories are gitignored and are recreated on demand, so a fresh clone does not need them.

## Projects built with or inspired by agent-service-toolkit

The following are a few of the public projects that drew code or inspiration from this repo.

- **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
- **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
- **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request editing the README or open a discussion with any new ones to be added!** Would love to include more projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**A note on how this repo is maintained:** this is a solo-maintainer project, and issues, PRs, and discussions are triaged on a roughly biweekly cycle. Thanks for your patience if responses take a week or two — I will do my best to respond to truly urgent issues (vulnerability reports, etc.) or in-progress PRs within a few days.

Currently the tests need to be run using the local development without Docker setup. To run the tests for the agent service:

1. Ensure you're in the project root directory and have activated your virtual environment.

2. Install the development dependencies and pre-commit hooks:

   ```sh
   uv sync --frozen
   pre-commit install
   ```

3. Run the tests using pytest:

   ```sh
   pytest
   ```

### Smoke testing optional dependencies

Some integrations aren't exercised by the unit suite or the default CI run because they
need real infrastructure: the Postgres and MongoDB checkpointers, the AG-UI endpoint, and
LangFuse tracing. `scripts/smoke_test.sh` spins up each dependency in Docker, runs the
service against it, verifies the integration end-to-end (including a check that the
intended backend was actually used, not a silent SQLite fallback), and tears it down.

```sh
./scripts/smoke_test.sh                 # default: postgres, mongo, agui
./scripts/smoke_test.sh mongo           # a single target
./scripts/smoke_test.sh langfuse        # heavy: starts LangFuse's full self-host stack
./scripts/smoke_test.sh all             # everything, including langfuse
```

These are opt-in confidence checks for a maintainer or agent — not part of CI. Run the
target that matches what you changed rather than the whole set. The optional add-on
compose files live in `docker/` (e.g. `docker/compose.mongo.yaml`), layered on top of the
default `compose.yaml` so the default stack stays lightweight.

### Cleaning up pytest temp directories on Windows

Git marks loose objects under `.git/objects` as read-only. On Windows that breaks pytest's
own `tmp_path` cleanup: `shutil.rmtree` raises `PermissionError [WinError 5]` on them,
pytest swallows the error, and leaves one `garbage-<uuid>` directory behind per session.
The system temp folder therefore accumulates a few multi-megabyte leftovers over time.
`scripts/clean_pytest_tmp.py` clears the read-only bit first, then deletes them:

```sh
uv run python scripts/clean_pytest_tmp.py --dry-run        # report only
uv run python scripts/clean_pytest_tmp.py                  # clean the system temp dir
uv run python scripts/clean_pytest_tmp.py .pytest_tmp_run  # also a local --basetemp
```

It only touches pytest scratch directories, never project data. Linux and macOS are
unaffected, so CI never needs it — this is a local convenience, not part of the suite.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
