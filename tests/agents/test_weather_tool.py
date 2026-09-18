"""天气工具（wttr.in）的测试。全部离线，mock httpx。

替换背景：原 OpenWeatherMap 工具需要 API key、且背在已 sunset 的
langchain-community 包上（docs/10 P1-3）。改为 wttr.in 免 key。
"""

import json
from unittest.mock import patch

from agents.tools import weather, weather_func


def _fake_wttr_body() -> str:
    return json.dumps(
        {
            "current_condition": [
                {
                    "temp_C": "23",
                    "FeelsLikeC": "25",
                    "humidity": "60",
                    "windspeedKmph": "15",
                    "weatherDesc": [{"value": "Sunny"}],
                }
            ],
            "nearest_area": [
                {"areaName": [{"value": "Beijing"}], "region": [{"value": "Beijing"}]}
            ],
        }
    )


def _mock_get(text: str, status_code: int = 200):
    import httpx

    resp = httpx.Response(status_code, text=text, request=httpx.Request("GET", "http://x"))
    return patch("agents.tools.httpx.get", return_value=resp)


def test_weather_tool_is_registered_in_research_assistant() -> None:
    """天气工具必须进 research_assistant 的工具集（不再依赖 API key）。"""
    from agents.research_assistant import tools

    names = [t.name for t in tools]
    assert "Weather" in names


def test_weather_formatts_summary() -> None:
    with _mock_get(_fake_wttr_body()):
        out = weather_func("Beijing")
    assert "Beijing" in out
    assert "23" in out
    assert "Sunny" in out
    assert "humidity 60%" in out


def test_weather_validates_city_input() -> None:
    """城市名白名单：防路径注入。"""
    assert weather_func("../../etc/passwd").startswith("ERROR")
    assert weather_func("<script>").startswith("ERROR")
    assert weather_func("北京").startswith("ERROR")  # 仅英文/拼音


def test_weather_returns_error_on_http_failure() -> None:
    import httpx

    bad = httpx.Response(500, text="oops", request=httpx.Request("GET", "http://x"))
    with patch("agents.tools.httpx.get", return_value=bad):
        out = weather_func("Beijing")
    assert out.startswith("ERROR")


def test_weather_returns_error_on_parse_failure() -> None:
    with _mock_get("not json at all"):
        out = weather_func("Beijing")
    assert out.startswith("ERROR")


def test_weather_name_is_weather() -> None:
    assert weather.name == "Weather"
