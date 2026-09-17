"""Remove pytest session directories that pytest itself could not delete.

On Windows, git marks loose objects under ``.git/objects`` as read-only.
``shutil.rmtree`` -- and therefore pytest's own tmp_path cleanup -- aborts on
them with ``PermissionError [WinError 5]``. pytest swallows that error and moves
on, leaving one ``garbage-<uuid>`` directory behind per session, so the system
temp folder slowly fills up with these multi-megabyte leftovers.

This script clears the read-only bit across the tree first, then deletes it.
It only ever touches pytest scratch directories -- never project data.
Not part of the pytest suite; it is a local maintenance tool.

Usage (run from the repo root):
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
    """The parent directory pytest stores its numbered session dirs under."""
    return Path(tempfile.gettempdir()) / f"pytest-of-{getpass.getuser()}"


def _count_items(root: Path) -> int:
    return sum(1 for _ in root.rglob("*"))


def _clear_readonly(root: Path) -> int:
    """Recursively drop the read-only bit, without which rmtree fails on Windows."""
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
