from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


type ModelT = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatVertexAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: 设置 streaming=True 的模型会在 token 生成时发送它们
    # 前提是调用 /stream 端点时传入 stream_tokens=True（默认值）
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        # GPT-5 生成基于推理，会拒绝 temperature（400）；省略它。
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        if model_name == AnthropicModelName.SONNET_5:
            # Claude Sonnet 5 会以 400 错误拒绝非默认采样参数（temperature、
            # top_p、top_k）——默认改为启用自适应思考。
            # 参见 https://docs.anthropic.com/en/docs/about-claude/models
            return ChatAnthropic(model_name=api_model_name, streaming=True)
        return ChatAnthropic(model_name=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in VertexAIModelName:
        return ChatVertexAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.GPT_OSS_SAFEGUARD_20B:
            return ChatGroq(model=api_model_name, temperature=0.0)  # type: ignore[call-arg]
        return ChatGroq(model=api_model_name, temperature=0.5)  # type: ignore[call-arg]
    if model_name in AWSModelName:
        if model_name == AWSModelName.BEDROCK_SONNET:
            # Sonnet 5 会拒绝非默认采样参数（400）；省略 temperature。
            return ChatBedrock(model=api_model_name)
        return ChatBedrock(model=api_model_name, temperature=0.5)
    if model_name in OllamaModelName:
        if not settings.OLLAMA_MODEL:
            raise ValueError("Ollama model must be configured")
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if model_name in OpenRouterModelName:
        # 没有显式 key 时，openai SDK 会回退到 OPENAI_API_KEY，
        # 这会把该 key 发送到 openrouter.ai。
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OpenRouter API key must be configured")

        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


@cache
def get_supervisor_model(model_name: AllModelEnum, /) -> ModelT:
    """为 supervisor 图返回一个已关闭 DeepSeek thinking 模式的模型。

    DeepSeek 的 thinking 模式要求请求历史里每条 content-only 的 assistant 消息
    都回传它生成时的 `reasoning_content`。`langgraph_supervisor` 恰恰把每个子 agent
    的最终答复拼成这样一条消息放进父图历史，而 `ChatOpenAI` 从不捕获该字段，
    于是每次转交都会以如下错误终止：

        400 - The reasoning_content in the thinking mode must be passed back to the API.

    关掉 thinking 就解除了这一要求，这是协议我方唯一可用的手段
    （见 docs/11-Agent能力完善方案.md 的路线 A）。

    两个有意为之的细节：

    - 用 `model_copy` 而不是重新构造实例，这样其他 agent 共享的
      那个缓存实例仍然保持 thinking 开启。
    - 仅对 DeepSeek 生效。其他 provider 会拒绝未知的请求字段，而该参数
      会被原样发到它们的 API。
    """
    model = get_model(model_name)
    if model_name in DeepseekModelName and isinstance(model, ChatOpenAI):
        return model.model_copy(update={"extra_body": {"thinking": {"type": "disabled"}}})
    return model
