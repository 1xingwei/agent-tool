"""针对每个已配置的 LLM provider 的轻量级实时冒烟测试。

向 schema.models 中所有 provider 的每个模型发送一个最小的单 token 提示词，
这些 provider 的凭据需存在于环境变量中，并按模型报告通过/失败。
这是维护者用于定期刷新模型目录的工具（参见
model-refresh skill）——它不属于 pytest 测试套件，因为它会对 provider API
发起真实网络调用并产生少量真实费用。

用法（从仓库根目录运行；需要 PYTHONPATH=src，与 src/run_service.py 相同）：
    PYTHONPATH=src uv run python scripts/check_live_models.py
    PYTHONPATH=src uv run python scripts/check_live_models.py --provider anthropic google

如果 ANTHROPIC_API_KEY 本身在你的环境中无法设置，可将密钥放在
其他变量名下，并用 --anthropic-api-key-env 指向它：
    PYTHONPATH=src uv run python scripts/check_live_models.py --anthropic-api-key-env MY_VAR
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable


def _remap_anthropic_api_key(argv: list[str]) -> None:
    """若提供了 --anthropic-api-key-env 变量，则将其复制到 ANTHROPIC_API_KEY。

    在导入 core.settings 之前运行，因为 Settings 在构造时读取环境变量
    ——所以不能等到下面主 argparse 解析时再处理。
    仅在显式传入该标志时生效；否则 ANTHROPIC_API_KEY
    保持已设置（或未设置）的原状。
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_var = None
    for i, arg in enumerate(argv):
        if arg == "--anthropic-api-key-env" and i + 1 < len(argv):
            env_var = argv[i + 1]
            break
        if arg.startswith("--anthropic-api-key-env="):
            env_var = arg.split("=", 1)[1]
            break
    if env_var and os.environ.get(env_var):
        os.environ["ANTHROPIC_API_KEY"] = os.environ[env_var]


_remap_anthropic_api_key(sys.argv[1:])

from core.llm import get_model  # noqa: E402
from core.settings import settings  # noqa: E402
from schema.models import (  # noqa: E402
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)

PROMPT = "Reply with exactly one word: OK"

# 仅需 API key/标志即可冒烟测试的 provider。Ollama、OpenAI 兼容
# 槽位以及 fake 模型被排除：它们需要本地基础设施或定制配置，
# 而非简单的「是否设置了 key」检查，因此不适合这种通用扫描。
PROVIDER_MODELS: dict[Provider, tuple[type[AllModelEnum], Callable[[], bool]]] = {
    Provider.OPENAI: (OpenAIModelName, lambda: bool(settings.OPENAI_API_KEY)),
    Provider.ANTHROPIC: (AnthropicModelName, lambda: bool(settings.ANTHROPIC_API_KEY)),
    Provider.GOOGLE: (GoogleModelName, lambda: bool(settings.GOOGLE_API_KEY)),
    Provider.GROQ: (GroqModelName, lambda: bool(settings.GROQ_API_KEY)),
    Provider.DEEPSEEK: (DeepseekModelName, lambda: bool(settings.DEEPSEEK_API_KEY)),
    Provider.OPENROUTER: (OpenRouterModelName, lambda: bool(settings.OPENROUTER_API_KEY)),
    Provider.AWS: (AWSModelName, lambda: settings.USE_AWS_BEDROCK),
    Provider.AZURE_OPENAI: (AzureOpenAIModelName, lambda: bool(settings.AZURE_OPENAI_API_KEY)),
    Provider.VERTEXAI: (VertexAIModelName, lambda: bool(settings.GOOGLE_APPLICATION_CREDENTIALS)),
}


async def check_model(model_name: AllModelEnum) -> tuple[bool, str]:
    try:
        model = get_model(model_name)
        result = await model.ainvoke(PROMPT)
        return True, str(result.content).strip()[:60]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:120]


async def main(provider_filter: set[str]) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for provider, (model_enum, has_credentials) in PROVIDER_MODELS.items():
        if provider_filter and provider.value not in provider_filter:
            continue
        if not has_credentials():
            rows.append((provider.value, "-", "SKIP", "no credentials configured"))
            continue
        for model_name in model_enum:
            ok, detail = await check_model(model_name)
            rows.append((provider.value, model_name.value, "PASS" if ok else "FAIL", detail))

    name_width = max((len(r[1]) for r in rows), default=10)
    for provider, model, status, detail in rows:
        print(f"{status:5} {provider:12} {model:<{name_width}} {detail}")

    if any(r[2] == "FAIL" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        nargs="*",
        default=[],
        metavar="PROVIDER",
        help="Limit to these provider values (e.g. anthropic google). Default: all configured.",
    )
    parser.add_argument(
        "--anthropic-api-key-env",
        default=None,
        metavar="VAR",
        help=(
            "Env var to read the Anthropic key from if ANTHROPIC_API_KEY itself isn't set. "
            "No fallback is checked unless this is passed. Applied before this script's own "
            "imports run, so the value here is informational -- set the env var itself before "
            "invoking the script."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(set(args.provider)))
