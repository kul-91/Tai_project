import json
import os
import tempfile
import unittest
from pathlib import Path

from security.policy import Policy
from tools.filesystem import list_directory, read_file, search_files


class TestFilesystemTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        # Set up test directories
        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()

        self.sibling_dir = self.base_path / "allowed_sibling"
        self.sibling_dir.mkdir()

        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        # Config file pointing to allowed_dir
        self.config_file = self.base_path / "permissions.json"
        config_data = {"allowed_locations": [str(self.allowed_dir)]}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        self.policy = Policy(config_path=self.config_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_list_directory_allowed(self):
        (self.allowed_dir / "file1.txt").write_text("content 1")
        (self.allowed_dir / "file2.md").write_text("# Title")
        sub_dir = self.allowed_dir / "subdir"
        sub_dir.mkdir()

        result = list_directory(self.allowed_dir, policy=self.policy)
        names = {item["name"] for item in result}
        self.assertEqual(names, {"file1.txt", "file2.md", "subdir"})

        for item in result:
            if item["name"] == "subdir":
                self.assertTrue(item["is_dir"])
                self.assertFalse(item["is_file"])
            else:
                self.assertFalse(item["is_dir"])
                self.assertTrue(item["is_file"])

    def test_list_directory_denied(self):
        with self.assertRaises(PermissionError):
            list_directory(self.outside_dir, policy=self.policy)

    def test_search_files_allowed(self):
        (self.allowed_dir / "report_2026.txt").write_text("report data")
        sub_dir = self.allowed_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "annual_report.csv").write_text("a,b,c")
        (self.allowed_dir / "notes.md").write_text("some notes")

        matches = search_files(self.allowed_dir, query="report", policy=self.policy)
        matched_names = {m["name"] for m in matches}
        self.assertEqual(matched_names, {"report_2026.txt", "annual_report.csv"})

    def test_search_files_denied(self):
        (self.outside_dir / "secret_report.txt").write_text("secret")
        with self.assertRaises(PermissionError):
            search_files(self.outside_dir, query="report", policy=self.policy)

    def test_read_file_allowed(self):
        txt_file = self.allowed_dir / "sample.txt"
        txt_file.write_text("Hello World")

        json_file = self.allowed_dir / "data.json"
        json_file.write_text('{"key": "value"}')

        csv_file = self.allowed_dir / "data.csv"
        csv_file.write_text("id,name\n1,Alice")

        md_file = self.allowed_dir / "doc.md"
        md_file.write_text("# Header")

        self.assertEqual(read_file(txt_file, policy=self.policy), "Hello World")
        self.assertEqual(read_file(json_file, policy=self.policy), '{"key": "value"}')
        self.assertEqual(read_file(csv_file, policy=self.policy), "id,name\n1,Alice")
        self.assertEqual(read_file(md_file, policy=self.policy), "# Header")

    def test_read_file_denied(self):
        secret = self.outside_dir / "secret.txt"
        secret.write_text("top secret")
        with self.assertRaises(PermissionError):
            read_file(secret, policy=self.policy)

    def test_path_traversal_attempt(self):
        traversal = self.allowed_dir / ".." / "outside" / "secret.txt"
        with self.assertRaises(PermissionError):
            read_file(traversal, policy=self.policy)
        with self.assertRaises(PermissionError):
            list_directory(self.allowed_dir / ".." / "outside", policy=self.policy)
        with self.assertRaises(PermissionError):
            search_files(self.allowed_dir / ".." / "outside", query="secret", policy=self.policy)

    def test_sibling_directory_similar_name(self):
        sibling_file = self.sibling_dir / "file.txt"
        sibling_file.write_text("sibling content")

        with self.assertRaises(PermissionError):
            read_file(sibling_file, policy=self.policy)
        with self.assertRaises(PermissionError):
            list_directory(self.sibling_dir, policy=self.policy)
        with self.assertRaises(PermissionError):
            search_files(self.sibling_dir, query="file", policy=self.policy)

    def test_unsupported_file_extension(self):
        py_file = self.allowed_dir / "script.py"
        py_file.write_text("print('hello')")

        exe_file = self.allowed_dir / "app.exe"
        exe_file.write_text("binary content")

        with self.assertRaises(ValueError) as ctx:
            read_file(py_file, policy=self.policy)
        self.assertIn("Unsupported file extension", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            read_file(exe_file, policy=self.policy)
        self.assertIn("Unsupported file extension", str(ctx.exception))

    def test_oversized_file_rejection(self):
        large_file = self.allowed_dir / "large.txt"
        large_file.write_text("X" * 100)

        with self.assertRaises(ValueError) as ctx:
            read_file(large_file, policy=self.policy, max_bytes=50)
        self.assertIn("exceeds maximum permitted limit", str(ctx.exception))

    def test_symlink_escape(self):
        symlink_path = self.allowed_dir / "escaped_link"
        try:
            os.symlink(self.outside_dir, symlink_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported on this platform/filesystem.")

        (self.outside_dir / "secret.txt").write_text("secret content")

        # Access via symlink escape must raise PermissionError
        with self.assertRaises(PermissionError):
            read_file(symlink_path / "secret.txt", policy=self.policy)

        # Search should not follow symlinks outside allowed boundaries
        matches = search_files(self.allowed_dir, query="secret", policy=self.policy)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
