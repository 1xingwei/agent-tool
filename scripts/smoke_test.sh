#!/usr/bin/env bash
# 针对基于 docker 的集成路径的按需冒烟测试，这些路径不在
# 快速单元测试套件或默认 CI 运行中：Postgres 和 MongoDB
# checkpointer、AG-UI 端点以及 LangFuse 追踪。让维护者（或
# agent）无需等待完整 CI 周期即可验证这些仍能正常工作。
#
# 用法：
#   ./scripts/smoke_test.sh                 # 默认目标：postgres、mongo、agui
#   ./scripts/smoke_test.sh mongo           # 运行单个目标
#   ./scripts/smoke_test.sh postgres agui   # 运行子集
#   ./scripts/smoke_test.sh langfuse        # 单独运行重量级的 langfuse 目标
#   ./scripts/smoke_test.sh all             # 全部，包括 langfuse
#
# 目标：postgres、mongo、agui、langfuse
#   langfuse 被排除在默认运行之外：它会启动 LangFuse 的完整
#   自托管栈（6 个服务，约 5GB 镜像）且耗时明显更长，因此
#   需显式运行或通过 `all` 运行。它还需要 cgr.dev 容器注册表
#   可访问（用于 minio 镜像）——在受限出网的云环境中，
#   请先将 cgr.dev 加入网络允许列表。
#
# 只有数据库在 Docker 中运行；服务本身通过 uv 在宿主机上运行，
# 指向 localhost。这是有意为之——构建服务镜像需要
# 从构建容器内部访问包注册表，这在某些沙箱化 agent 环境中
# 会被阻止。在宿主机上运行可绕过此问题，同时
# 仍能使用真实的数据库容器。
#
# 一次绿色的运行应当真正有意义。除了 pytest/API 检查之外，
# 每个目标都会独立验证确实使用了预期的依赖：
# DB 目标会向容器查询本次运行的 thread，langfuse 则查询
# 其 API 获取 trace。这能捕获静默回退（例如回退到 SQLite），
# 否则该回退会通过针对任何可用 checkpointer 的 API 级测试。
#
# 依赖：docker、docker compose、uv、node（AG-UI 客户端）、python3、curl
set -euo pipefail
cd "$(dirname "$0")/.."

# 每次运行唯一，以便下方的后端验证反映本次运行的数据，即使
# 数据库卷非空。导出以便 pytest 测试使用相同的
# thread id（见 tests/smoke/test_persistence.py）。
SMOKE_THREAD_ID="smoke-test-$(date +%s)-$$"
export SMOKE_THREAD_ID

# LangFuse 的自托管 compose 从此固定 tag 的上游获取，
# 而非 vendored 进仓库。升级此项即可迁移到更新的 LangFuse。
LANGFUSE_REF="v3.225.5"

SERVICE_PID=""
SERVICE_LOG=""
LANGFUSE_COMPOSE=""  # temp compose file, set while the langfuse target runs

start_service() {
  # 在宿主机上以给定的后端环境启动 agent 服务，然后等待
  # 直到其报告健康。Args: KEY=VALUE ... 连接设置。
  if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "  ✗ refusing to start: something is already listening on :8080"
    return 1
  fi
  SERVICE_LOG="$(mktemp)"
  env USE_FAKE_MODEL=true "$@" uv run python src/run_service.py > "$SERVICE_LOG" 2>&1 &
  SERVICE_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      echo "  ✗ service exited during startup; log:"
      cat "$SERVICE_LOG"
      return 1
    fi
    sleep 2
  done
  echo "  ✗ service did not become healthy within 60s; log:"
  cat "$SERVICE_LOG"
  return 1
}

stop_service() {
  if [[ -n "$SERVICE_PID" ]]; then
    kill "$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
    SERVICE_PID=""
  fi
  if [[ -n "$SERVICE_LOG" ]]; then
    rm -f "$SERVICE_LOG"
    SERVICE_LOG=""
  fi
}

wait_healthy() {
  # 等待容器报告健康。Args: 容器 id。
  local cid="$1" status=""
  for _ in $(seq 1 20); do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo missing)"
    [[ "$status" == "healthy" ]] && return 0
    echo "  waiting for database... ($status)"
    sleep 2
  done
  echo "  ✗ database never became healthy"
  return 1
}

assert_positive_count() {
  # Args: count-string、label。除非 count 是大于 0 的整数，否则显式报错退出。
  local n="$1" label="$2"
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )); then
    echo "  ✓ verified: $n $label"
  else
    echo "  ✗ FAIL: expected >0 $label, got '$n' — backend was NOT exercised as intended"
    return 1
  fi
}

cleanup() {
  echo "--- Tearing down ---"
  stop_service
  # down 会移除合并项目中的每个服务（postgres + mongo），因此这
  # 一次调用即可清理，无论之前运行的是哪个目标。
  docker compose -f compose.yaml -f docker/compose.mongo.yaml down -v >/dev/null 2>&1 || true
  if [[ -n "$LANGFUSE_COMPOSE" && -f "$LANGFUSE_COMPOSE" ]]; then
    docker compose -f "$LANGFUSE_COMPOSE" down -v >/dev/null 2>&1 || true
    rm -f "$LANGFUSE_COMPOSE"
  fi
}
trap cleanup EXIT

smoke_postgres() {
  echo "=== Postgres checkpointer (DATABASE_TYPE=postgres) ==="
  docker compose -f compose.yaml up -d postgres
  local cid
  cid="$(docker compose -f compose.yaml ps -q postgres)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=postgres POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec -e PGPASSWORD=postgres "$cid" psql -U postgres -d agent_service -tAc \
    "select count(*) from checkpoints where thread_id='$SMOKE_THREAD_ID'" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "postgres checkpoint rows for this run's thread"
  stop_service
  docker compose -f compose.yaml down -v
}

smoke_mongo() {
  echo "=== MongoDB checkpointer (DATABASE_TYPE=mongo) ==="
  local files=(-f compose.yaml -f docker/compose.mongo.yaml)
  docker compose "${files[@]}" up -d mongo
  local cid
  cid="$(docker compose "${files[@]}" ps -q mongo)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=mongo MONGO_HOST=localhost MONGO_PORT=27017 MONGO_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec "$cid" mongosh agent_service --quiet --eval \
    "db.checkpoints.countDocuments({thread_id:'$SMOKE_THREAD_ID'})" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "mongo checkpoint documents for this run's thread"
  stop_service
  docker compose "${files[@]}" down -v
}

smoke_agui() {
  echo "=== AG-UI endpoint ==="
  # AG-UI 与后端无关，因此这里默认的 SQLite checkpointer 即可，
  # 无需数据库容器。
  start_service
  local out
  out="$(cd scripts/agui-client && npm install --silent && \
    AGENT_URL=http://localhost:8080 node client.mjs "Tell me a joke!" chatbot)" || true
  echo "$out"
  # 绿色退出还不够：确认流确实已完成并返回
  # 假模型的响应，而非空或部分运行。
  if ! grep -q "RUN_FINISHED" <<<"$out"; then
    echo "  ✗ FAIL: AG-UI stream did not reach RUN_FINISHED"
    return 1
  fi
  if ! grep -q "This is a test response from the fake model." <<<"$out"; then
    echo "  ✗ FAIL: AG-UI did not return the expected assistant response"
    return 1
  fi
  echo "  ✓ verified: AG-UI streamed a complete run with the expected response"
  stop_service
}

smoke_langfuse() {
  echo "=== LangFuse tracing (self-hosted) ==="
  local pk="pk-lf-smoke-public" sk="sk-lf-smoke-secret"

  # 获取 LangFuse 官方自托管 compose（固定版本）而非 vendored 进仓库。
  # 使用裸 mktemp（无 --suffix）以兼容 macOS/BSD；`docker compose -f`
  # 不关心文件扩展名。
  LANGFUSE_COMPOSE="$(mktemp)"
  echo "  fetching LangFuse compose @ $LANGFUSE_REF"
  if ! curl -sSL "https://raw.githubusercontent.com/langfuse/langfuse/$LANGFUSE_REF/docker-compose.yml" \
    -o "$LANGFUSE_COMPOSE"; then
    echo "  ✗ FAIL: could not fetch LangFuse compose"
    return 1
  fi

  # LANGFUSE_INIT_* 在首次启动时初始化组织/项目/用户及已知 API 密钥，因此
  # 无需手动注册，且下方密钥是确定性的。
  echo "  starting LangFuse stack (this pulls ~5GB the first time)..."
  LANGFUSE_INIT_ORG_ID=smoke-org LANGFUSE_INIT_ORG_NAME=smoke \
  LANGFUSE_INIT_PROJECT_ID=smoke-project LANGFUSE_INIT_PROJECT_NAME=smoke \
  LANGFUSE_INIT_PROJECT_PUBLIC_KEY="$pk" LANGFUSE_INIT_PROJECT_SECRET_KEY="$sk" \
  LANGFUSE_INIT_USER_EMAIL=smoke@example.com LANGFUSE_INIT_USER_NAME=smoke \
  LANGFUSE_INIT_USER_PASSWORD=smokepassword123 \
    docker compose -f "$LANGFUSE_COMPOSE" up -d

  echo "  waiting for langfuse-web..."
  for _ in $(seq 1 40); do
    curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && break
    sleep 3
  done
  if ! curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1; then
    echo "  ✗ FAIL: langfuse-web did not become ready"
    return 1
  fi

  start_service LANGFUSE_TRACING=true LANGFUSE_HOST=http://localhost:3000 \
    LANGFUSE_PUBLIC_KEY="$pk" LANGFUSE_SECRET_KEY="$sk"

  # (1) 服务级：/health 对实例运行 langfuse.auth_check()。
  if curl -s http://localhost:8080/health | grep -q '"langfuse":"connected"'; then
    echo "  ✓ /health reports langfuse connected"
  else
    echo "  ✗ FAIL: /health did not report langfuse connected"
    return 1
  fi

  # (2) 决定性：一次被追踪的 invoke 必须实际在 LangFuse 中产生 trace。
  uv run python -c "
import sys; sys.path.insert(0, 'src')
from client import AgentClient
c = AgentClient('http://localhost:8080')
r = c.invoke('Trace me please', thread_id='$SMOKE_THREAD_ID', model='fake')
assert r.type == 'ai', r
print('  traced invoke ok')
"
  echo "  waiting for the trace to land in LangFuse (ingestion is async)..."
  local n=0
  for _ in $(seq 1 20); do
    n="$(curl -s -u "$pk:$sk" "http://localhost:3000/api/public/traces?limit=5" \
      | python3 -c 'import sys, json; print(len(json.load(sys.stdin).get("data", [])))' 2>/dev/null || echo 0)"
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )) && break
    sleep 3
  done
  assert_positive_count "$n" "LangFuse traces recorded for this run"

  stop_service
  docker compose -f "$LANGFUSE_COMPOSE" down -v
  rm -f "$LANGFUSE_COMPOSE"
  LANGFUSE_COMPOSE=""
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(postgres mongo agui)

# 将 "all" 展开为每个目标，包括重量级的 langfuse。
expanded=()
for t in "${targets[@]}"; do
  if [[ "$t" == "all" ]]; then
    expanded+=(postgres mongo agui langfuse)
  else
    expanded+=("$t")
  fi
done
targets=("${expanded[@]}")

for t in "${targets[@]}"; do
  case "$t" in
    postgres) smoke_postgres ;;
    mongo)    smoke_mongo ;;
    agui)     smoke_agui ;;
    langfuse) smoke_langfuse ;;
    *) echo "unknown target: $t (valid: postgres, mongo, agui, langfuse, all)"; exit 2 ;;
  esac
done

echo "--- All smoke tests passed ---"
