# 使用 Ollama

⚠️ _**注意：** agent-service-toolkit 中的 Ollama 支持属于实验性质，可能无法按预期工作。下述步骤已在 MacBook Pro 上的 Docker Desktop 环境下验证。遇到任何问题请提 issue。_

你也可以用 [Ollama](https://ollama.com) 运行驱动 agent 服务的 LLM。

1. 按 <https://github.com/ollama/ollama> 的说明安装 Ollama
1. 拉取你想用的模型，例如 `ollama pull llama3.2`，并把 `OLLAMA_MODEL` 环境变量设为你用的模型，例如 `OLLAMA_MODEL=llama3.2`

如果你在本机运行服务（例如 `python src/run_service.py`），到这里就可以用了。

如果你在 Docker 中运行服务，还需要：

1. [按此配置 Ollama 服务端](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)，例如在 MacOS 上执行 `launchctl setenv OLLAMA_HOST "0.0.0.0"` 并重启 Ollama。
1. 把 `OLLAMA_BASE_URL` 环境变量设为 Ollama 服务端的基地址，例如 `OLLAMA_BASE_URL=http://host.docker.internal:11434`
1. 另一种做法是在 Docker 中运行 `ollama/ollama` 镜像并采用类似配置（不过某些情况下会更慢）。
