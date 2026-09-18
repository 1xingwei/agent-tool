from enum import StrEnum
from json import loads
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    Provider,
    VertexAIModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """转换为 Python 日志级别常量。"""
        import logging

        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: SecretStr | None = None

    # 如果 DEFAULT_MODEL 为 None，将在 model_post_init 中设置
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # 设置 openai 兼容 api，主要用于概念验证
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    # Web 搜索（ddgs）配置
    WEB_SEARCH_PROXY: str = ""
    WEB_SEARCH_BACKENDS: str = "yahoo,duckduckgo"

    # MCP 配置
    GITHUB_PAT: SecretStr | None = None
    MCP_GITHUB_SERVER_URL: str = "https://api.githubcopilot.com/mcp/"

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # 数据库配置
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # 可选值：DatabaseType.SQLITE 或 DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "var/checkpoints.db"
    # 长期记忆 store；与上面的 checkpoints 数据库分开保存
    SQLITE_STORE_PATH: str = "var/memory_store.db"

    # Database_Search 使用的 RAG 向量数据库
    CHROMA_DIR: str = "./var/chroma_db"
    # 向量检索召回条数（原来是硬编码的 k=5）
    RAG_TOP_K: int = 5
    # 混合检索：向量 + FTS5（docs/15 P0-8）。FTS 侧车库与 Chroma 同目录。
    CHROMA_FTS_DB: str = "./var/chroma_fts.sqlite"
    # 混合召回各自取多少条做 RRF 融合（融合前每路的候选数）
    HYBRID_RECALL_K: int = 20

    # Embedding 配置：长期记忆 store 的语义检索与 RAG 向量化共用。
    # 默认本地 fastembed，不需要任何 API key；设为 openai 则需要 OPENAI_API_KEY。
    EMBEDDING_PROVIDER: str = "local"
    # 本地默认 bge-small-zh-v1.5（中文短文本，**512** 维，不是常被误记的 384）；
    # openai 时忽略

    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    # 显式向量维度。留空则首次使用时探测一次（见 core/embeddings.py）
    EMBEDDING_DIMS: int | None = None
    # fastembed 模型缓存目录；默认放项目内 var/（gitignore），
    # 避免与系统共享缓存相互污染（实测系统 Temp 缓存出现半下载的模型文件）
    EMBEDDING_CACHE_DIR: str = "var/fastembed_cache"

    # 会话蒸馏（长会话摘要压缩）。语义与不变量见 core/distill.py 的模块文档。
    # 默认关闭：开启后每个超阈值的请求会多一次摘要模型调用，是否划算取决于
    # 模型价格与常见会话长度，应由部署方而不是库作者决定。
    DISTILL_ENABLED: bool = False
    # 消息条数达到该值才考虑蒸馏。条数不是轮次——一条人类发言后面可能
    # 跟着工具消息，两者不成比例，因此这里按 `messages` 长度判定。
    DISTILL_TRIGGER_MESSAGES: int = 40
    # 至少要有多少条「新增」消息值得摘要。防止每次请求都重摘同一段。
    DISTILL_MIN_NEW_MESSAGES: int = 10
    # 送给模型的视图里始终保留最近多少条原文；无论轮次怎么算都不裁掉它们。
    DISTILL_KEEP_MESSAGES: int = 12
    # 按人类发言计，保留最近几轮不纳入摘要范围
    DISTILL_KEEP_TURNS: int = 3
    # 摘要输入的每条消息截断长度。工具结果动辄几 KB，全量喂进去
    # 会让摘要调用本身成为新的窗口压力。
    DISTILL_SNIPPET_CHARS: int = 800
    # 摘要用模型；留空则用 DEFAULT_MODEL
    DISTILL_MODEL: str | None = None
    DISTILL_SUMMARY_PROMPT: str = (
        "你负责把一段对话压缩成供后续对话使用的摘要。"
        "保留：用户的身份与目标、已达成的结论与决策、未解决的问题、"
        "具体的标识符（文件路径、仓库名、编号、URL）、以及任何被明确要求记住的约束。"
        "丢弃：寒暄、重复的确认、工具返回的原始噪声、以及已被后续结论推翻的中间猜测。"
        "用中文，用短段落或无序列表，不要写成对话记录。"
        "若用户提供了「已有摘要」，请把新内容合并进去，不要丢弃已有摘要里的信息。"
        "只输出摘要正文，不要加前言或说明。"
    )

    # 可选：设置后，checkpoint（短期）和 store（长期）都
    # 使用 Redis，从而在多个实例间共享状态。需要带有
    # RedisJSON + RediSearch 模块的 Redis 服务器（Redis 8.0+ 或 Redis Stack）。
    REDIS_URL: str | None = None

    # PostgreSQL 配置
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1

    # MongoDB 配置
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None
    MONGO_TLS: bool = False  # MongoDB 的可选 TLS；生产/Atlas 环境设为 True

    # Azure OpenAI 设置
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        # 即使存在真实 provider key，USE_FAKE_MODEL 也必须赢得默认值。
        if self.USE_FAKE_MODEL and self.DEFAULT_MODEL is None:
            self.DEFAULT_MODEL = FakeModelName.FAKE

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_V4_FLASH
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_45
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_38_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_38_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.GPT_OSS_20B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_38_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_5_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # 如果 Azure provider 可用，则校验 Azure OpenAI 设置
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # 如果 deployment map 是字符串则解析它
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # 校验所需的 deployment 是否存在
                    required_models = {"gpt-5", "gpt-5-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        """客户端应使用该 URL 访问此服务。

        `HOST` 是*绑定*地址。`0.0.0.0` 和 `::` 表示「所有接口」，
        这不是有效的连接目标——在 Windows 上连接 `0.0.0.0` 会
        直接以 WinError 10049 失败——因此这里将通配符映射为回环地址。

        任何发起连接的一方（示例客户端、UI）都应使用该 URL，而不是
        从 HOST 重新拼接 URL，通配符正是通过这种方式泄漏到调用方的。
        """
        loopback = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "[::1]"}
        host = loopback.get(self.HOST, self.HOST)
        return f"http://{host}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
