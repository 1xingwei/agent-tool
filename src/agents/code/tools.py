import re
import subprocess
from functools import cache
from pathlib import Path

from langchain_core.tools import tool

DEFAULT_REPO = Path.cwd()


def _git(repo_path: str, args: list[str]) -> str:
    cmd = ["git", "-c", "color.ui=false", "--no-pager", *args]
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        # Without an explicit encoding, text mode decodes with
        # locale.getpreferredencoding() -- cp936 on a Windows service process. Git
        # emits UTF-8, so a non-ASCII commit message raises UnicodeDecodeError in
        # the reader thread, which leaves stdout as None and turns the caller's
        # .strip() into an AttributeError (see docs/notes/project_audit.md).
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
    """Resolve repo_path (file or dir) to the repo root; raise if not a git repo."""
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
    """List recent commit history of a git repository with per-commit change stats.

    Args:
        repo_path: Path to a file or directory inside the git repository.
        max_count: Maximum number of commits to return.

    Returns:
        One line per commit: hash, author date, subject, and number of files changed.
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
    """Show the full diff of a commit (or of working-tree changes vs a ref).

    Args:
        repo_path: Path to a file or directory inside the git repository.
        ref: Commit hash/ref to show. Defaults to HEAD.

    Returns:
        The complete diff output (respecting git's pager/length limits).
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
    """Search a repository for files by name and/or content.

    Args:
        repo_path: Path to a file or directory inside the git repository.
        name_pattern: Regex matched against file paths (empty to skip).
        content_pattern: Regex matched against file contents (empty to skip).
        max_results: Maximum number of matches to return.

    Returns:
        Matching paths, suffixed with the matching content line when content_pattern is set.
    """
    root = _repo_root(repo_path)
    name_re = re.compile(name_pattern) if name_pattern else None
    content_re = re.compile(content_pattern) if content_pattern else None
    results: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(p in path.parts for p in (".git", ".venv", "__pycache__", "node_modules")):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
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
    """Read a file from the repository, truncated to avoid overflowing context.

    Args:
        repo_path: Path to a file or directory inside the git repository.
        path: The file to read, relative to the repository root.
        max_chars: Maximum characters to return.

    Returns:
        The file content, truncated indicatically if longer than max_chars.
    """
    root = _repo_root(repo_path)
    if ".." in Path(path).parts:
        return "error: path must stay inside the repository"
    full = (root / path).resolve()
    if not full.is_file() or not full.is_relative_to(root):
        return f"error: not a file inside the repository: {path}"
    content = full.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n...[truncated {len(content) - max_chars} chars]"
    return content
