import re
import subprocess
from functools import cache
from pathlib import Path

from langchain_core.tools import tool

DEFAULT_REPO = Path.cwd()

_SENSITIVE_PARTS = {
    ".venv",
    "__pycache__",
    "node_modules",
    "privatecredentials",
    ".git",
}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".sqlite", ".db")
# 精确名单拦不住 .env.local / .env.production / id_rsa / service-account.json
# 等常见变体（docs/20 第五轮 V2），改前缀与名字模式匹配。
_SENSITIVE_PART_PREFIXES = (".env",)
_SENSITIVE_FILE_NAMES = ("id_rsa", "id_ed25519", "id_dsa")
_SENSITIVE_NAME_SUBSTRINGS = ("service-account",)


def _is_sensitive(rel: str) -> bool:
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p in _SENSITIVE_PARTS for p in parts):
        return True
    if any(p.startswith(prefix) for p in parts for prefix in _SENSITIVE_PART_PREFIXES):
        return True
    if any(p in _SENSITIVE_FILE_NAMES for p in parts):
        return True
    if any(sub in p for p in parts for sub in _SENSITIVE_NAME_SUBSTRINGS):
        return True
    return any(rel.endswith(sfx) for sfx in _SENSITIVE_SUFFIXES)


def _git(repo_path: str, args: list[str]) -> str:
    cmd = ["git", "-c", "color.ui=false", "--no-pager", *args]
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        # 未显式指定编码时，文本模式会用 locale.getpreferredencoding() 解码——
        # 在 Windows 服务进程中即为 cp936。Git 输出 UTF-8，因此非 ASCII 的提交信息
        # 会在读取线程中抛出 UnicodeDecodeError，使 stdout 变成 None，
        # 调用方的 .strip() 于是抛出 AttributeError（见 docs/10-审核与修复总账.md）。
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return f"git error: {result.stderr.strip()}"
    return result.stdout.strip()


@cache
def _repo_root(repo_path: str) -> Path:
    """将 repo_path（文件或目录）解析为仓库根目录；若不是 git 仓库则抛出异常。"""
    path = Path(repo_path).resolve()
    if path.is_file():
        path = path.parent
    root = path
    while not (root / ".git").exists():
        if root == root.parent:
            raise ValueError(f"Not a git repository: {repo_path}")
        root = root.parent
    return root


@tool
def git_log(repo_path: str = str(DEFAULT_REPO), max_count: int = 20) -> str:
    """列出 git 仓库的近期提交历史，并附带每次提交的变更统计。

    Args:
        repo_path: git 仓库内文件或目录的路径。
        max_count: 返回的最大提交数。

    Returns:
        每次提交一行：hash、作者日期、主题以及变更文件数。
    """
    root = _repo_root(repo_path)
    out = _git(
        str(root),
        [
            "log",
            f"-{max(max_count, 1)}",
            "--date=short",
            "--pretty=format:%h|%ad|%an|%s",
            "--shortstat",
        ],
    )
    parts = re.split(r"\n\n", out)
    lines: list[str] = []
    for part in parts:
        head, _, stat = part.partition("\n")
        lines.append(f"{head}{stat}")
    return "\n".join(lines)


@tool
def git_diff(repo_path: str = str(DEFAULT_REPO), ref: str = "HEAD") -> str:
    """显示某次提交的完整 diff（或工作区变更相对于某个 ref 的 diff）。

    Args:
        repo_path: git 仓库内文件或目录的路径。
        ref: 要显示的提交 hash/ref。默认为 HEAD。

    Returns:
        完整的 diff 输出（遵循 git 的分页/长度限制）。
    """
    root = _repo_root(repo_path)
    return _git(str(root), ["show", "--format=%h %ad %s%n", "--date=short", ref])


@tool
def file_search(
    repo_path: str = str(DEFAULT_REPO),
    name_pattern: str = "",
    content_pattern: str = "",
    max_results: int = 20,
) -> str:
    """按名称和/或内容在仓库中搜索文件。

    Args:
        repo_path: git 仓库内文件或目录的路径。
        name_pattern: 与文件路径匹配的正则（为空则跳过）。
        content_pattern: 与文件内容匹配的正则（为空则跳过）。
        max_results: 返回的最大匹配数。

    Returns:
        匹配的路径，当设置了 content_pattern 时附加匹配的内容行。
    """
    root = _repo_root(repo_path)
    name_re = re.compile(name_pattern) if name_pattern else None
    content_re = re.compile(content_pattern) if content_pattern else None
    results: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if _is_sensitive(rel):
            continue
        if name_re and not name_re.search(rel):
            continue
        if content_re:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if content_re.search(line):
                    results.append(f"{rel}:{line.strip()[:120]}")
                    break
        else:
            results.append(rel)
        if len(results) >= max_results:
            break
    return "\n".join(results) or "No matches."


@tool
def read_file(repo_path: str = str(DEFAULT_REPO), path: str = "", max_chars: int = 8000) -> str:
    """从仓库中读取文件，并截断以避免上下文溢出。

    Args:
        repo_path: git 仓库内文件或目录的路径。
        path: 要读取的文件，相对于仓库根目录。
        max_chars: 返回的最大字符数。

    Returns:
        文件内容，若超过 max_chars 则截断并留下截断标记。
    """
    root = _repo_root(repo_path)
    if ".." in Path(path).parts:
        return "error: path must stay inside the repository"
    rel = Path(path).as_posix()
    if _is_sensitive(rel):
        return f"error: path is sensitive and cannot be read: {path}"
    full = (root / path).resolve()
    if not full.is_file() or not full.is_relative_to(root):
        return f"error: not a file inside the repository: {path}"
    content = full.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n...[truncated {len(content) - max_chars} chars]"
    return content
