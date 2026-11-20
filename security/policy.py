import json
from pathlib import Path
from typing import Union

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "permissions.json"


class Policy:
    """Security policy enforcer for SentinelAI filesystem access.

    Ensures that file operations are restricted strictly to explicitly permitted
    filesystem locations defined in the configuration.
    """

    def __init__(self, config_path: Union[str, Path, None] = None) -> None:
        """Initialize Policy with a config file path.

        If config_path is None, defaults to config/permissions.json relative to
        the project root.
        """
        if config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
        else:
            self.config_path = Path(config_path)

        self._allowed_locations: list[Path] = []
        self.load_policy()

    def load_policy(self) -> None:
        """Loads and resolves allowed locations from the permissions configuration file.

        Denies access by default if configuration is missing, malformed, or empty.
        """
        self._allowed_locations = []
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                locations = data.get("allowed_locations", [])
                if isinstance(locations, list):
                    for loc in locations:
                        if loc and isinstance(loc, str):
                            # Resolve path to canonical absolute form
                            resolved_loc = Path(loc).expanduser().resolve()
                            self._allowed_locations.append(resolved_loc)
        except (json.JSONDecodeError, OSError):
            self._allowed_locations = []

    def get_allowed_locations(self) -> list[str]:
        """Returns the list of configured allowed locations as canonical path strings."""
        return [str(loc) for loc in self._allowed_locations]

    def is_allowed(self, path: Union[str, Path]) -> bool:
        """Checks whether a target path is strictly within any allowed location.

        Resolves relative components (..) and symlinks to prevent path traversal
        and symlink escape vulnerabilities.
        """
        if not self._allowed_locations:
            return False

        if not path:
            return False

        try:
            target_path = Path(path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return False

        for allowed_dir in self._allowed_locations:
            try:
                if target_path.is_relative_to(allowed_dir):
                    return True
            except ValueError:
                continue

        return False

    def check_access(self, path: Union[str, Path]) -> None:
        """Enforces access control by checking the path.

        Raises:
            PermissionError: If access to the path is denied.
        """
        if not self.is_allowed(path):
            raise PermissionError(f"Access denied: Path '{path}' is outside permitted locations.")
