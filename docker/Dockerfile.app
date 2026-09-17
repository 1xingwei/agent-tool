FROM python:3.13.14-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv==0.12.5

# 仅安装客户端应用所需的依赖
# --frozen：使用 lock 文件中的精确版本
# --only-group client：仅安装 pyproject.toml 中标记为 "client" 组的依赖
RUN uv sync --frozen --only-group client

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/voice/ ./voice/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
