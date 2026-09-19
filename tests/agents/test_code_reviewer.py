import subprocess

import pytest
from langchain_core.messages import AIMessage

from agents.code import tools
from agents.code_reviewer import code_reviewer, remember_review
from agents.code_reviewer import tools as review_tools


def _make_repo(path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(path), check=True)
    for name, content in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        subprocess.run(["git", "add", name], cwd=str(path), check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"add {name}"], cwd=str(path), check=True)


def test_code_reviewer_compiles() -> None:
    nodes = set(code_reviewer.get_graph().nodes.keys())
    assert {"model", "tools", "remember_review"} <= nodes


def test_tools_are_read_only() -> None:
    names = [t.name for t in review_tools]
    assert "git_log" in names
    assert "git_diff" in names
    assert "file_search" in names
    assert "read_file" in names


def test_git_log_empty_repo(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert "any commits yet" in tools.git_log.func(repo_path=str(tmp_path))


def test_git_log_and_diff(tmp_path) -> None:
    _make_repo(tmp_path, {"a.py": "x = 1\n"})
    out = tools.git_log.func(repo_path=str(tmp_path))
    assert "add a.py" in out
    diff = tools.git_diff.func(repo_path=str(tmp_path), ref="HEAD")
    assert "add a.py" in diff and "+x = 1" in diff


def test_git_log_decodes_non_ascii_commits(tmp_path, monkeypatch) -> None:
    """Git 输出 UTF-8，但服务进程按 locale（Windows 上为 cp936）解码，
    导致 stdout 变空，并使此仓库上的每次 git_log 调用崩溃。

    仅断言文本还不够：pytest 从 shell 继承 PYTHONUTF8=1，且 CI 在 UTF-8 locale 下运行，
    因此即使没有修复，两者也能正确解码。所以改为显式固定编码。
    """
    _make_repo(tmp_path, {"a.py": "x = 1\n"})
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "修复：中文提交信息"],
        cwd=str(tmp_path),
        check=True,
    )

    calls: list[dict] = []
    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", spy)
    out = tools.git_log.func(repo_path=str(tmp_path))

    assert "修复：中文提交信息" in out
    assert calls, "expected _git to shell out to git"
    assert calls[0].get("encoding") == "utf-8"
    assert calls[0].get("errors") == "replace"


def test_file_search_and_read(tmp_path) -> None:
    _make_repo(tmp_path, {"mod/b.py": "def target(): pass\n"})
    hits = tools.file_search.func(repo_path=str(tmp_path), content_pattern="def target")
    assert "mod/b.py" in hits
    content = tools.read_file.func(repo_path=str(tmp_path), path="mod/b.py")
    assert "def target(): pass" in content


def test_read_file_blocks_traversal(tmp_path) -> None:
    _make_repo(tmp_path, {"a.py": "x = 1\n"})
    out = tools.read_file.func(repo_path=str(tmp_path), path="../outside.py")
    assert "stay inside" in out


def test_read_file_blocks_sensitive_paths(tmp_path) -> None:
    """read_file 必须拒绝 .env / privatecredentials 等敏感路径（docs/20 F3）。

    反例注入：把 denylist 从 read_file 删掉，本用例必须变红。
    """
    _make_repo(
        tmp_path,
        {
            ".env": "AUTH_SECRET=supersecret\nDEEPSEEK_API_KEY=sk-1234\n",
            "privatecredentials/.gitkeep": "",
            "config/settings.py": "AUTH_SECRET = get_env('AUTH_SECRET')",
        },
    )
    for path in (".env", "privatecredentials/.gitkeep"):
        out = tools.read_file.func(repo_path=str(tmp_path), path=path)
        assert "sensitive" in out, f"{path} 应被拒绝，实际返回了内容"
    # 正常文件不受影响
    out = tools.read_file.func(repo_path=str(tmp_path), path="config/settings.py")
    assert "get_env" in out


def test_read_file_blocks_env_variants_and_keys(tmp_path) -> None:
    """denylist 必须按模式而非精确名单拦截变体（docs/20 第五轮 V2）。

    `.env.local`、`.env.production`、`id_rsa`、`service-account.json`、`app.db`
    精确名单全部放行；反例注入：把 `_is_sensitive` 改回精确名集合，本用例必须变红。
    """
    _make_repo(
        tmp_path,
        {
            ".env.local": "DEEPSEEK_API_KEY=sk-5678\n",
            ".env.production": "AUTH_SECRET=prod-secret\n",
            "keys/id_rsa": "-----BEGIN PRIVATE KEY-----",
            "svc/service-account.json": '{"private_key": "x"}',
            "data/app.db": "\x00\x00\x01sqlite",
            "config/settings.py": "AUTH_SECRET = get_env('AUTH_SECRET')",
        },
    )
    for path in (
        ".env.local",
        ".env.production",
        "keys/id_rsa",
        "svc/service-account.json",
        "data/app.db",
    ):
        out = tools.read_file.func(repo_path=str(tmp_path), path=path)
        assert "sensitive" in out, f"{path} 应被拒绝，实际返回了内容"
    out = tools.read_file.func(repo_path=str(tmp_path), path="config/settings.py")
    assert "get_env" in out


def test_file_search_skips_sensitive_paths(tmp_path) -> None:
    """file_search 不得在敏感路径上命中内容（docs/20 F3）。

    反例注入：把 denylist 从 file_search 删掉，本用例必须变红。
    """
    _make_repo(
        tmp_path,
        {
            ".env": "AUTH_SECRET=supersecret\n",
            ".env.example": "AUTH_SECRET=\n",
            "privatecredentials/secret.txt": "AUTH_SECRET=hush\n",
            "config/settings.py": "AUTH_SECRET = get_env('AUTH_SECRET')",
        },
    )
    hits = tools.file_search.func(repo_path=str(tmp_path), content_pattern="AUTH_SECRET")
    lines = hits.splitlines()
    assert all((".env" not in line and "privatecredentials" not in line) for line in lines), lines
    # 正常路径仍可命中
    assert any("config/settings.py" in line for line in lines), lines


def test_file_search_skips_env_by_name(tmp_path) -> None:
    """即使不按内容过滤，.env 也不应出现在名称搜索结果里。"""
    _make_repo(
        tmp_path,
        {
            ".env": "AUTH_SECRET=x\n",
            "notes/env.md": "plain notes\n",
        },
    )
    hits = tools.file_search.func(repo_path=str(tmp_path), name_pattern=r"\.env")
    assert all(line != ".env" for line in hits.splitlines()), hits


@pytest.mark.asyncio
async def test_remember_review_writes_store(tmp_path) -> None:
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    state = {"messages": [AIMessage(content="review done")]}
    config = {"configurable": {"user_id": "u1"}}

    await remember_review(state, config, store)

    values = list(store.search(("code-reviewer", "u1")))
    assert len(values) == 1
    assert values[0].value["conclusion"] == "review done"


@pytest.mark.asyncio
async def test_remember_review_without_store_is_a_noop() -> None:
    """独立调用（`langgraph dev`、run_agent.py）传入 store=None。"""
    state = {"messages": [AIMessage(content="review done")]}
    assert await remember_review(state, {"configurable": {}}, None) == {"messages": []}


@pytest.mark.asyncio
async def test_remember_review_write_failure_is_nonfatal() -> None:
    """写路径失败（database is locked）不得拖垮已生成完的整次审查（docs/20 F7）。

    反例注入：把 remember_review 的 try/except 删掉，本用例必须变红。
    """

    class BoomStore:
        async def aput(self, *a, **k):
            raise RuntimeError("database is locked")

    state = {"messages": [AIMessage(content="review done")]}
    out = await remember_review(state, {"configurable": {"user_id": "u1"}}, BoomStore())
    assert out == {"messages": []}
