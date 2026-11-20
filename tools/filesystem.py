import os
from pathlib import Path
from typing import Any, Optional, Union

from security.policy import Policy

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit by default


def _get_policy(policy: Optional[Policy]) -> Policy:
    """Helper to return passed policy or instantiate default Policy."""
    if policy is None:
        return Policy()
    return policy


def list_directory(path: Union[str, Path], policy: Optional[Policy] = None) -> list[dict[str, Any]]:
    """List directory contents strictly within permitted Policy locations.

    Args:
        path: Directory path to list.
        policy: Policy instance enforcing access control.

    Returns:
        List of entry dictionaries with name, path, is_dir, and is_file.

    Raises:
        PermissionError: If path access is denied by policy.
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If path is not a directory.
    """
    pol = _get_policy(policy)
    pol.check_access(path)

    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Directory not found: '{path}'")
    if not target_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{path}'")

    results = []
    for item in target_path.iterdir():
        # Verify individual items against policy to prevent symlink leaks
        if pol.is_allowed(item):
            try:
                resolved_item = item.resolve()
                if pol.is_allowed(resolved_item):
                    results.append({
                        "name": item.name,
                        "path": str(resolved_item),
                        "is_dir": resolved_item.is_dir(),
                        "is_file": resolved_item.is_file(),
                    })
            except (OSError, RuntimeError):
                continue

    return results


def search_files(
    directory: Union[str, Path],
    query: str,
    policy: Optional[Policy] = None
) -> list[dict[str, Any]]:
    """Recursively search for files matching query within permitted directory.

    Does not traverse outside permitted locations or follow escaping symlinks.

    Args:
        directory: Root directory to search within.
        query: Query string to match against filename.
        policy: Policy instance enforcing access control.

    Returns:
        List of dicts containing name and absolute path for each match.

    Raises:
        PermissionError: If directory access is denied by policy.
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If path is not a directory.
    """
    pol = _get_policy(policy)
    pol.check_access(directory)

    target_dir = Path(directory).expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: '{directory}'")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{directory}'")

    matches = []
    query_lower = query.lower()

    def _search_rec(curr_dir: Path):
        try:
            entries = list(curr_dir.iterdir())
        except OSError:
            return

        for item in entries:
            # Check item access
            if not pol.is_allowed(item):
                continue

            try:
                is_symlink = item.is_symlink()
                resolved_item = item.resolve()
            except (OSError, RuntimeError):
                continue

            # Ensure resolved target doesn't escape boundary
            if not pol.is_allowed(resolved_item):
                continue

            if resolved_item.is_dir():
                # Avoid following symlinks recursively to prevent loops & escapes
                if not is_symlink:
                    _search_rec(resolved_item)
            elif resolved_item.is_file():
                if query_lower in item.name.lower():
                    matches.append({
                        "name": item.name,
                        "path": str(resolved_item)
                    })

    _search_rec(target_dir)
    return matches


def read_file(
    path: Union[str, Path],
    policy: Optional[Policy] = None,
    max_bytes: int = DEFAULT_MAX_FILE_SIZE
) -> str:
    """Read file content safely within permitted locations.

    Only reads allowed text formats (.txt, .md, .json, .csv) and enforces size limits.

    Args:
        path: Path to the file.
        policy: Policy instance enforcing access control.
        max_bytes: Maximum allowed file size in bytes.

    Returns:
        Contents of the file as string.

    Raises:
        PermissionError: If file access is denied by policy.
        ValueError: If file extension is unsupported or file size exceeds max_bytes.
        FileNotFoundError: If file does not exist.
        IsADirectoryError: If path is a directory.
    """
    pol = _get_policy(policy)
    pol.check_access(path)

    target_file = Path(path).expanduser().resolve()

    if not target_file.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    if target_file.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: '{path}'")

    # Extension check
    ext = target_file.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Only {sorted(SUPPORTED_EXTENSIONS)} are allowed."
        )

    # Size check
    file_size = target_file.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"File size ({file_size} bytes) exceeds maximum permitted limit ({max_bytes} bytes)."
        )

    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
