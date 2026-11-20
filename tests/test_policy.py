import json
import os
import tempfile
import unittest
from pathlib import Path

from security.policy import Policy


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        # Create test directory structure
        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()

        self.sibling_dir = self.base_path / "allowed_sibling"
        self.sibling_dir.mkdir()

        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        # Create config file pointing to allowed_dir
        self.config_file = self.base_path / "permissions.json"
        self.set_allowed_locations([str(self.allowed_dir)])

        self.policy = Policy(config_path=self.config_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def set_allowed_locations(self, locations: list[str]):
        config_data = {"allowed_locations": locations}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

    def test_allowed_file_inside_allowed_directory(self):
        file_path = self.allowed_dir / "test_file.txt"
        file_path.write_text("hello")

        self.assertTrue(self.policy.is_allowed(file_path))
        # Should not raise PermissionError
        self.policy.check_access(file_path)

    def test_allowed_nested_directory(self):
        nested_dir = self.allowed_dir / "nested" / "subfolder"
        nested_dir.mkdir(parents=True)
        file_path = nested_dir / "data.json"
        file_path.write_text("{}")

        self.assertTrue(self.policy.is_allowed(nested_dir))
        self.assertTrue(self.policy.is_allowed(file_path))
        self.policy.check_access(file_path)

    def test_denied_file_outside_allowed_directory(self):
        file_path = self.outside_dir / "secret.txt"
        file_path.write_text("secret")

        self.assertFalse(self.policy.is_allowed(file_path))
        with self.assertRaises(PermissionError):
            self.policy.check_access(file_path)

    def test_denied_sibling_directory_with_similar_name(self):
        # allowed_sibling has prefix "allowed", but is a distinct sibling directory
        sibling_file = self.sibling_dir / "file.txt"
        sibling_file.write_text("sibling")

        self.assertFalse(self.policy.is_allowed(self.sibling_dir))
        self.assertFalse(self.policy.is_allowed(sibling_file))
        with self.assertRaises(PermissionError):
            self.policy.check_access(sibling_file)

    def test_relative_path_traversal_attempt(self):
        traversal_path = self.allowed_dir / ".." / "outside" / "secret.txt"

        self.assertFalse(self.policy.is_allowed(traversal_path))
        with self.assertRaises(PermissionError):
            self.policy.check_access(traversal_path)

    def test_absolute_path_outside_allowed_directory(self):
        abs_outside = Path("/etc/passwd")

        self.assertFalse(self.policy.is_allowed(abs_outside))
        with self.assertRaises(PermissionError):
            self.policy.check_access(abs_outside)

    def test_empty_allowed_locations(self):
        self.set_allowed_locations([])
        empty_policy = Policy(config_path=self.config_file)

        file_path = self.allowed_dir / "test_file.txt"
        self.assertFalse(empty_policy.is_allowed(file_path))
        self.assertEqual(empty_policy.get_allowed_locations(), [])
        with self.assertRaises(PermissionError):
            empty_policy.check_access(file_path)

    def test_nonexistent_path_handling(self):
        # Non-existent path inside allowed dir should be allowed
        nonexistent_inside = self.allowed_dir / "new_file.txt"
        self.assertTrue(self.policy.is_allowed(nonexistent_inside))

        # Non-existent path outside allowed dir should be denied
        nonexistent_outside = self.outside_dir / "new_file.txt"
        self.assertFalse(self.policy.is_allowed(nonexistent_outside))

        # Traversal with non-existent components targetting outside
        nonexistent_traversal = self.allowed_dir / "foo" / ".." / ".." / "outside.txt"
        self.assertFalse(self.policy.is_allowed(nonexistent_traversal))

    def test_symlink_escape(self):
        symlink_path = self.allowed_dir / "escaped_link"
        try:
            os.symlink(self.outside_dir, symlink_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported on this platform/filesystem.")

        target_file_via_symlink = symlink_path / "secret.txt"
        (self.outside_dir / "secret.txt").write_text("secret content")

        self.assertFalse(self.policy.is_allowed(target_file_via_symlink))
        with self.assertRaises(PermissionError):
            self.policy.check_access(target_file_via_symlink)

    def test_get_allowed_locations(self):
        locations = self.policy.get_allowed_locations()
        self.assertEqual(locations, [str(self.allowed_dir)])


if __name__ == "__main__":
    unittest.main()
