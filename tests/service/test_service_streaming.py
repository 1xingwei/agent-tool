import pytest
from langchain_core.messages import AIMessage
from pydantic_core import ValidationError

from service.service import _create_ai_message


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1) 基本内容 + tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2) 未知键被忽略
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3) 额外的合法 AIMessage 参数（id、type）透传
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """
    _create_ai_message 应当：
      - 丢弃未知键（"foobar"）
      - 保留与 AIMessage 签名匹配的键
      - 对 parts 字典中的重复键使用最终值
    """
    msg: AIMessage = _create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_required_content_raises():
    """
    AIMessage 要求 'content'；若缺失，_create_ai_message 应当
    从构造函数抛出 pydantic ValidationError。

    LANGCHAIN V1 迁移说明：
    - 此前对缺失的必需参数抛出 TypeError
    - 在 langchain v1 与 Pydantic v2 中，验证错误现在是 pydantic_core.ValidationError
    - 这更明确地指出失败原因（验证）而非泛化的 TypeError
    """
    with pytest.raises(ValidationError):
        _create_ai_message({"tool_calls": []})


def test_create_ai_message_empty_dict_raises():
    """
    完全空的 parts 同样应无法构造 AIMessage，
    抛出 pydantic ValidationError。

    LANGCHAIN V1 迁移说明：
    - 异常类型从 TypeError 变为 pydantic_core.ValidationError
    - 反映了 langchain_core 转向使用 Pydantic v2 进行模型验证
    """
    with pytest.raises(ValidationError):
        _create_ai_message({})
