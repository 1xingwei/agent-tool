import asyncio

from dotenv import load_dotenv

from client import AgentClient
from core import settings
from schema import ChatMessage

# AgentClient 从进程环境读取 AUTH_SECRET（它是独立的客户端
# 库，刻意不导入 core.settings），因此必须在构造它之前
# 加载 .env。否则该示例在运行任何配置了 AUTH_SECRET 的服务时都会返回 401。
load_dotenv()


def _print_message(message: ChatMessage) -> None:
    """可读地打印一条消息（schema 层不再提供 UI 方法，归位到调用方）。"""
    base_title = message.type.title() + " Message"
    padded = " " + base_title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    title = f"{sep}{padded}{second_sep}"
    print(f"{title}\n\n{message.content}")  # noqa: T201


async def amain() -> None:
    #### 异步 ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    # 不显式指定模型：只有服务默认模型保证可用
    # （AVAILABLE_MODELS 取决于运维人员配置了哪些 provider key）。
    response = await client.ainvoke("Tell me a brief joke?")
    _print_message(response)

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            _print_message(message)
        else:
            print(f"ERROR: Unknown type - {type(message)}")


def main() -> None:
    #### 同步 ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    # 不显式指定模型：只有服务默认模型保证可用
    # （AVAILABLE_MODELS 取决于运维人员配置了哪些 provider key）。
    response = client.invoke("Tell me a brief joke?")
    _print_message(response)

    print("\nStream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            _print_message(message)
        else:
            print(f"ERROR: Unknown type - {type(message)}")


if __name__ == "__main__":
    print("Running in sync mode")
    main()
    print("\n\n\n\n\n")
    print("Running in async mode")
    asyncio.run(amain())
