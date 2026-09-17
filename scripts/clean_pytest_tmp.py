"""删除 pytest 自身无法删除的 pytest 会话目录。

在 Windows 上，git 会将 ``.git/objects`` 下的松散对象标记为只读。
``shutil.rmtree``——以及 pytest 自身的 tmp_path 清理——会因
``PermissionError [WinError 5]`` 而中止。pytest 会吞掉该错误并继续，
导致每个会话留下一个 ``garbage-<uuid>`` 目录，系统
临时文件夹会逐渐被这些数兆字节的残留物填满。

此脚本先清除整棵树的只读位，然后删除。
它只触及 pytest 临时目录——绝不涉及项目数据。
不属于 pytest 测试套件；它是本地维护工具。

用法（从仓库根目录运行）：
    uv run python scripts/clean_pytest_tmp.py --dry-run
    uv run python scripts/clean_pytest_tmp.py
    uv run python scripts/clean_pytest_tmp.py .pytest_tmp_run
"""

import argparse
import getpass
import os
import shutil
import stat
import tempfile
from pathlib import Path


def _pytest_temp_root() -> Path:
    """pytest 存放其编号会话目录的父目录。"""
    return Path(tempfile.gettempdir()) / f"pytest-of-{getpass.getuser()}"


def _count_items(root: Path) -> int:
    return sum(1 for _ in root.rglob("*"))


def _clear_readonly(root: Path) -> int:
    """递归清除只读位，否则在 Windows 上 rmtree 会失败。"""
    cleared = 0
    for path in root.rglob("*"):
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            continue
        cleared += 1
    return cleared


def clean(root: Path, *, dry_run: bool) -> None:
    if not root.is_dir():
        print(f"skip   {root}  (not found)")
        return
    total = _count_items(root)
    if dry_run:
        print(f"would  {root}  ({total} items)")
        return
    cleared = _clear_readonly(root)
    try:
        shutil.rmtree(root)
    except OSError as exc:
        print(f"FAIL   {root}  ({total} items) - {exc}")
        return
    print(f"done   {root}  ({total} items removed, {cleared} read-only cleared)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="extra directories to clean, e.g. a --basetemp such as .pytest_tmp_run",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only, delete nothing")
    args = parser.parse_args()

    root = _pytest_temp_root()
    targets = [root] if root.is_dir() else []
    if not targets and not args.paths:
        print(f"nothing to do: {root} does not exist")
        return 0

    for target in [*targets, *args.paths]:
        clean(target, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
