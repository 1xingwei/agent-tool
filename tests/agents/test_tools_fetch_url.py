"""fetch_url 工具的测试（docs/11 §1.6）。全部离线，mock httpx。

铁律来自 docs/11 §1.3：**SSRF 防护是必须项，不是加分项。**
- 只允许 http/https
- 解析域名后校验全部 A/AAAA 记录，拒绝私网
- 拒绝 localhost / *.internal / *.local
- 响应体上限先按 Content-Length 预检，读取时硬截断
"""

from unittest.mock import patch

from agents.tools import (
    FETCH_URL_HARD_CAP,
    _hostname_blocked,
    _is_private_address,
    _validate_url_safe,
    fetch_url_func,
)

# --- SSRF 静态防护 ---


def test_url_scheme_ssrf_blocks() -> None:
    assert _validate_url_safe("file:///etc/passwd") is False
    assert _validate_url_safe("gopher://example.com/x") is False
    assert _validate_url_safe("data:text/plain,hi") is False
    assert _validate_url_safe("ftp://example.com/x") is False


def test_hostname_blocked() -> None:
    assert _hostname_blocked("localhost") is True
    assert _hostname_blocked("db.internal") is True
    assert _hostname_blocked("router.local") is True
    assert _hostname_blocked("example.com") is False


def test_private_ip_ranges() -> None:
    for ip in ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "::1"]:
        assert _is_private_address(ip), f"{ip} 应被拦"
    assert _is_private_address("8.8.8.8") is False


def test_validate_url_resolves_public_ip_ok() -> None:
    with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
        assert _validate_url_safe("http://example.com/") is True


def test_validate_url_blocks_private_resolution() -> None:
    """域名解析到私网 IP 必须被拦 —— 这是 SSRF 的实质防护。"""
    with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("127.0.0.1", 0))]):
        assert _validate_url_safe("http://evil.example.com/") is False


# --- fetch_url 行为 ---


def test_fetch_url_returns_error_for_blocked_url() -> None:
    result = fetch_url_func("http://127.0.0.1:8080/info")
    assert result.startswith("ERROR")


def test_fetch_url_fetches_and_extracts_text() -> None:
    html = "<html><head><style>a{}</style></head><body><h1>Hello</h1><p>World <b>foo</b></p></body></html>"
    fake_response = type(
        "R",
        (),
        {"raise_for_status": lambda self: None, "text": html, "is_redirect": False, "headers": {}},
    )()

    with patch("agents.tools.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_response
        out = fetch_url_func("http://example.com/page")

    assert "Hello" in out
    assert "World foo" in out
    assert "<style>" not in out
    assert "<b>" not in out


def test_fetch_url_returns_error_on_http_error() -> None:
    import httpx

    with patch("agents.tools.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.side_effect = httpx.ConnectError("refused")
        out = fetch_url_func("http://example.com/")
    assert out.startswith("ERROR")


def test_fetch_url_enforces_hard_cap() -> None:
    """max_chars 不能被顶点成任意大 —— 硬上限存在。"""
    big_html = "<p>" + "x" * 100000 + "</p>"
    fake_response = type(
        "R",
        (),
        {
            "raise_for_status": lambda self: None,
            "text": big_html,
            "is_redirect": False,
            "headers": {},
        },
    )()

    with patch("agents.tools.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_response
        out = fetch_url_func("http://example.com/", max_chars=999999)

    assert len(out) <= FETCH_URL_HARD_CAP
    assert len(out) < 200000, "硬上限未生效"


def test_fetch_url_pre_flights_content_length() -> None:
    """Content-Length 预检：过大页面直接拒绝，不读取。"""
    fake_response = type(
        "R",
        (),
        {
            "raise_for_status": lambda self: None,
            "text": "x",
            "is_redirect": False,
            "headers": {"content-length": "99999999"},
        },
    )()

    with patch("agents.tools.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_response
        out = fetch_url_func("http://example.com/")

    assert out.startswith("ERROR")


def test_fetch_url_blocks_redirect_to_internal() -> None:
    """docs/11 §1.6 第 6 条：公网 URL 302 到内网必须被拦，且不发出第二次请求。"""
    redirect = type(
        "R",
        (),
        {
            "raise_for_status": lambda self: None,
            "text": "",
            "is_redirect": True,
            "headers": {"location": "http://169.254.169.254/latest/meta-data/"},
        },
    )()

    with patch("agents.tools.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value = redirect
        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("93.184.216.34", 0))]):
            out = fetch_url_func("http://example.com/start")

    assert out.startswith("ERROR")
    assert mock_client.get.call_count == 1
