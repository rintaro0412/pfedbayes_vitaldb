"""Exercise export boundaries and exact file copying using tiny fixtures."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_public_release import MANIFEST_NAME, ReleaseError, prepare_release


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / "source"
        self.root.mkdir()
        self.output = base / "export"
        self.file_list = self.root / "publication_files.json"
        (self.root / "README.md").write_text("public code\n", encoding="utf-8")

    def select(self, files, excluded_context=None, source_overrides=None):
        self.file_list.write_text(
            json.dumps({"schema_version": 1, "files": files, "excluded_context": excluded_context or [], "source_overrides": source_overrides or {}}),
            encoding="utf-8",
        )

    def run_export(self, **kwargs):
        return prepare_release(self.root, self.file_list, self.output, **kwargs)

    def test_export_contains_only_selected_files_with_matching_hash(self):
        (self.root / "private.txt").write_text("not selected", encoding="utf-8")
        self.select(["README.md"])
        report = self.run_export()
        self.assertEqual(report["file_count"], 1)
        self.assertEqual((self.output / "README.md").read_bytes(), (self.root / "README.md").read_bytes())
        manifest = json.loads((self.output / MANIFEST_NAME).read_text())
        self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(b"public code\n").hexdigest())
        self.assertFalse((self.output / "private.txt").exists())
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["README.md", MANIFEST_NAME])

    def test_dry_run_does_not_create_output(self):
        self.select(["README.md"])
        self.assertFalse(self.run_export(dry_run=True)["manifest_written"])
        self.assertFalse(self.output.exists())

    def test_source_override_uses_public_copy_and_records_its_hash(self):
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "public_readme.md").write_text("publication instructions\n", encoding="utf-8")
        self.select(["README.md"], source_overrides={"README.md": "docs/public_readme.md"})
        self.run_export()
        self.assertEqual((self.output / "README.md").read_text(), "publication instructions\n")
        self.assertEqual((self.root / "README.md").read_text(), "public code\n")
        record = json.loads((self.output / MANIFEST_NAME).read_text())["files"][0]
        self.assertEqual(record["source_path"], "docs/public_readme.md")
        self.assertEqual(record["sha256"], hashlib.sha256(b"publication instructions\n").hexdigest())

    def test_override_cannot_select_private_data_or_nonselected_target(self):
        for target, source in [
            ("README.md", "paper_studies/private.md"),
            ("README.md", "tables/predictions.npz"),
            ("README.md", "../private.md"),
            ("other.md", "README.md"),
        ]:
            with self.subTest(target=target, source=source):
                self.select(["README.md"], source_overrides={target: source})
                with self.assertRaises(ReleaseError):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_override_source_respects_excluded_context_and_symlinks(self):
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "public_readme.md").write_text("public", encoding="utf-8")
        self.select(["README.md"], ["docs"], {"README.md": "docs/public_readme.md"})
        with self.assertRaises(ReleaseError):
            self.run_export()
        (self.root / "linked.md").symlink_to(docs / "public_readme.md")
        self.select(["README.md"], source_overrides={"README.md": "linked.md"})
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_nonempty_output_is_preserved(self):
        self.select(["README.md"])
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("untouched", encoding="utf-8")
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.assertEqual(marker.read_text(), "untouched")

    def test_rejects_traversal_absolute_and_glob_paths_before_writing(self):
        for path in ["../outside.txt", "/tmp/outside.txt", "./README.md", "README*.md"]:
            with self.subTest(path=path):
                self.select([path])
                with self.assertRaises(ReleaseError):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_rejects_duplicate_and_missing_files_before_writing(self):
        for files in [["README.md", "README.md"], ["README.md", "missing.py"]]:
            with self.subTest(files=files):
                self.select(files)
                with self.assertRaises(ReleaseError):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_rejects_source_symlink(self):
        (self.root / "linked.md").symlink_to(self.root / "README.md")
        self.select(["linked.md"])
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_rejects_output_symlink(self):
        target = self.output.parent / "other"
        target.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        self.select(["README.md"])
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.assertEqual(list(target.iterdir()), [])

    def test_rejects_data_checkpoint_private_context(self):
        for path in ["runs/run.json", "federated_data/a.csv", "tables/predictions.npz", "papers/paper.pdf"]:
            with self.subTest(path=path):
                self.select([path])
                with self.assertRaises(ReleaseError):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_excluded_context_prefix_is_enforced(self):
        self.select(["README.md"], ["README.md"])
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_in_repository_output_is_limited_to_staging_directory(self):
        self.select(["README.md"])
        self.output = self.root / "public"
        with self.assertRaises(ReleaseError):
            self.run_export()
        self.output = self.root / "runs" / "public_release" / "smoke"
        self.assertEqual(self.run_export()["mode"], "export")


if __name__ == "__main__":
    unittest.main()
