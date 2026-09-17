from pydantic import SecretStr


def test_no_auth_secret(mock_settings, mock_agent, test_client):
    """测试当未设置 AUTH_SECRET 时，所有请求都被允许"""
    mock_settings.AUTH_SECRET = None
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # 也应该在没有任何 auth 头的情况下工作
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent, test_client):
    """测试当设置了 AUTH_SECRET 时，携带正确 token 的请求被允许"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent, test_client):
    """测试当设置了 AUTH_SECRET 时，携带错误 token 的请求被拒绝"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # 也应该拒绝没有 auth 头的请求
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401
