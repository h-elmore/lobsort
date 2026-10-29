"""Catch merged duplicate definitions and configuration keys before release."""
import ast
import json
from pathlib import Path
import unittest
import lobsort

ROOT = Path(lobsort.__file__).parent


class SourceIntegrityTests(unittest.TestCase):
    def test_no_duplicate_definitions_within_a_scope(self):
        for path in sorted(ROOT.rglob("*.py")):
            with self.subTest(module=str(path.relative_to(ROOT))):
                tree = ast.parse(path.read_text(), filename=str(path))
                for scope in ast.walk(tree):
                    if not isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    seen = set()
                    for node in scope.body:
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.assertNotIn(node.name, seen, f"{path.name}:{node.lineno}: repeated definition {node.name}")
                            seen.add(node.name)

    def test_defaults_have_unique_keys_and_keep_latest_settings(self):
        def unique_object(pairs):
            keys = [key for key, _ in pairs]
            self.assertEqual(len(keys), len(set(keys)), "Repeated settings key in defaults.json")
            return dict(pairs)

        settings = json.loads((ROOT / "defaults.json").read_text(), object_pairs_hook=unique_object)
        for key in ("artifact_thresholds_mV", "threshold_factor", "training_labels_by_channel",
                    "sorting_methods_by_channel", "detection_labels_by_channel", "channel_names"):
            self.assertIn(key, settings)
        self.assertIn("PD", settings["label_targets"])


if __name__ == "__main__":
    unittest.main()
