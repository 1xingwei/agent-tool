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
