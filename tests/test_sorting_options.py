"""Detection-only sorting and per-channel training label choices."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets as W
from lobsort.curation import Review
from lobsort.demo import demo_channels
from lobsort.engine import PipelineSession, default_settings
from lobsort.gui import MainWindow


def settings_for_test():
    settings = default_settings()
    settings.update(threshold_factor=4.0, shadow_period_ms=2.0,
                    number_of_trees=20, number_of_waveform_pcs=2)
    return settings


class DetectionOnlyTests(unittest.TestCase):
    def test_recording_with_one_channel_and_one_saved_label(self):
        settings = settings_for_test()
        settings.update(label_targets={"PD": 25}, reference_channel_name="Red")
        with tempfile.TemporaryDirectory() as folder:
            session = PipelineSession(
                demo_channels(2)[:1], settings, folder, lambda _: None
            ).prepare()
            table = session.accept_candidates("PD")
            self.assertGreater(len(table), 0)
            self.assertTrue(table.accepted_as_spike.all())
            self.assertEqual(set(table.manual_label), {"PD"})

    def test_all_candidates_become_one_unit_and_remain_reviewable(self):
        with tempfile.TemporaryDirectory() as folder:
            session = PipelineSession(demo_channels(5), settings_for_test(), folder, lambda _: None).prepare()
            # Even a one-class label file and no active training labels are irrelevant.
            path = session.ns["training_labels_path"]
            pd.DataFrame(dict(sample_index=session.ns["candidate_indices"][:3], manual_label="PD")).to_csv(path, index=False)
            before = path.read_bytes()
            session.settings["training_labels_by_channel"] = {"Red": []}
            with patch.object(session, "train", side_effect=AssertionError("must not train")), patch.object(session, "features", side_effect=AssertionError("must not extract features")):
                table = session.accept_candidates("PD")
            np.testing.assert_array_equal(table.sample_index, session.ns["candidate_indices"])
            np.testing.assert_allclose(table.time_seconds, session.ns["candidate_times"])
            self.assertTrue(table.accepted_as_spike.all())
            self.assertEqual(set(table.manual_label), {"PD"})
            self.assertEqual(set(table.acceptance_reason), {"detection_only"})
            self.assertNotIn("prediction_probability", table)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(Path(folder).glob("*.joblib")))
            persisted = pd.read_csv(Path(folder) / "automatic_candidates.csv")
            self.assertEqual(len(persisted), len(table))
            review = Review(session.ns["processed_channels"], {"Red": table}, {"settings": session.serializable_settings()})
            self.assertTrue(review.tables["Red"].automatic_accepted.all())
            first = int(table.sample_index.iloc[0])
            review.change("Red", first, "remove")
            review.undo()
            self.assertTrue(review.tables["Red"].accepted_as_spike.all())
            review.save(Path(folder) / "review.lobsort")
            restored = Review.load(Path(folder) / "review.lobsort")
            restored.export(Path(folder) / "exports")
            spikes = pd.read_csv(Path(folder) / "exports/0_Red_curated_spikes.csv")
            np.testing.assert_array_equal(spikes.sample_index, table.sample_index)
            self.assertEqual(set(spikes.manual_label), {"PD"})
            self.assertEqual(restored.metadata["settings"]["sorting_methods_by_channel"]["Red"], "detection_only")

    def test_invalid_unit_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            session = PipelineSession(demo_channels(2), settings_for_test(), folder, lambda _: None).prepare()
            for label in ("Noise", "noise", "Uncertain", "Unlabeled", "", "Missing"):
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, "neuron label"):
                    session.accept_candidates(label)

    def test_selected_classes_used_by_both_classifiers_without_deleting_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = settings_for_test()
            library = copy.deepcopy(settings["label_targets"])
            settings["training_labels_by_channel"] = {"Red": ["PY", "Noise"]}
            session = PipelineSession(demo_channels(35), settings, folder, lambda _: None).prepare()
            indices = session.ns["candidate_indices"]
            examples = pd.DataFrame(dict(sample_index=indices, time_seconds=session.ns["candidate_times"], manual_label=np.array(["PY", "Noise", "PD"])[np.arange(len(indices)) % 3]))
            path = session.ns["training_labels_path"]
            examples.to_csv(path, index=False)
            for kind, classifier in (("context", "context_classifier"), ("waveform", "classifier")):
                with self.subTest(kind=kind):
                    session.train(kind)
                    self.assertEqual(set(session.ns[classifier].classes_), {"PY", "Noise"})
                    pd.testing.assert_frame_equal(pd.read_csv(path), examples)
                    self.assertEqual(session.settings["label_targets"], library)
            session.settings["training_labels_by_channel"]["Red"] = ["PD"]
            with self.assertRaisesRegex(ValueError, "Detection only"):
                session.train("context")
            session.settings["training_labels_by_channel"]["Red"] = []
            with self.assertRaisesRegex(ValueError, "at least one"):
                session.training_targets()
            session.settings["training_labels_by_channel"]["Red"] = ["Missing"]
            with self.assertRaisesRegex(ValueError, "Unknown training labels"):
                session.training_targets()


class SortingOptionsGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])

    def finish_worker(self, window):
        deadline = time.monotonic() + 20
        while window.worker and window.worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertFalse(window.worker and window.worker.isRunning())

    def make_window(self, folder):
        window = MainWindow()
        window.settings = settings_for_test()
        window.output.setText(folder)
        window.loaded((demo_channels(5), [], 1), "demo.synthetic", 0)
        window.reference.setCurrentText("Green")
        return window

    def close_window(self, window):
        self.finish_worker(window)
        window.dirty = False
        window.close()
        self.app.processEvents()

    def test_detection_only_action_and_per_channel_labeler_choices(self):
        with tempfile.TemporaryDirectory() as folder:
            window = self.make_window(folder)
            try:
                library = copy.deepcopy(window.settings["label_targets"])
                window.training_label_checks["PD"].setChecked(False)
                window.training_label_checks["LP"].setChecked(False)
                window.target.setCurrentText("Green")
                window.training_label_checks["PY"].setChecked(False)
                window.target.setCurrentText("Red")
                self.assertEqual(window.selected_training_labels(), ["PY", "Noise"])
                self.assertEqual(window.settings["label_targets"], library)
                window.prepare()
                self.finish_worker(window)
                session = window.session
                self.assertIsNotNone(session)
                original_folder = session.output_dir
                window.open_labeler()
                self.assertEqual(set(window.label_window.label_buttons), {"PY", "Noise"})
                window.close_labeler()
                window.training_label_checks["PD"].setChecked(True)
                self.assertIs(window.session, session)
                window.open_labeler()
                self.assertEqual(set(window.label_window.label_buttons), {"PY", "Noise", "PD"})
                window.close_labeler()
                window.kind.setCurrentIndex(2)
                window.detection_label.setCurrentText("PD")
                self.assertFalse(window.labels_button.isEnabled())
                self.assertFalse(window.train_button.text().startswith("3."))
                self.assertTrue(window.train_button.isEnabled())
                window.train_button.click()
                self.finish_worker(window)
                table = window.review.tables["Red"]
                self.assertEqual(len(table), len(session.ns["candidate_indices"]))
                self.assertTrue(table.accepted_as_spike.all())
                self.assertEqual(set(table.manual_label), {"PD"})
                self.assertIs(window.tabs.currentWidget(), window.viewer)
                # Label/method choices must not create a new candidate workspace.
                window.prepare()
                self.finish_worker(window)
                self.assertEqual(window.session.output_dir, original_folder)
                window.target.setCurrentText("Green")
                self.assertEqual(window.kind.currentIndex(), 0)
                self.assertNotIn("PY", window.selected_training_labels())
                window.target.setCurrentText("Red")
                self.assertEqual(window.kind.currentIndex(), 2)
                self.assertEqual(window.detection_label.currentText(), "PD")
            finally:
                self.close_window(window)

    def test_choices_survive_profiles_reviews_and_library_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            window = self.make_window(folder)
            other = MainWindow()
            try:
                window.training_label_checks["LP"].setChecked(False)
                window.kind.setCurrentIndex(2)
                window.detection_label.setCurrentText("PD")
                profile = str(Path(folder) / "profile.json")
                with patch.object(W.QFileDialog, "getSaveFileName", return_value=(profile, "")):
                    window.save_profile()
                stored = json.loads(Path(profile).read_text())
                self.assertNotIn("LP", stored["training_labels_by_channel"]["Red"])
                self.assertIn("LP", stored["label_targets"])
                with patch.object(W.QFileDialog, "getOpenFileName", return_value=(profile, "")), patch("lobsort.gui.SettingsDialog.exec", return_value=1):
                    other.load_profile()
                other.loaded((demo_channels(2), [], 1), "demo.synthetic", 0)
                self.assertEqual(other.kind.currentIndex(), 2)
                self.assertFalse(other.training_label_checks["LP"].isChecked())
                path = str(Path(folder) / "review.lobsort")
                window.review.save(path)
                window.dirty = False
                with patch.object(W.QFileDialog, "getOpenFileName", return_value=(path, "")):
                    window.open_project()
                self.assertFalse(window.training_label_checks["LP"].isChecked())
                self.assertEqual(window.kind.currentIndex(), 2)
                # Adding a library entry exposes it without retyping other labels.
                window.settings["label_targets"]["VD"] = 15
                window.refresh_channel_options()
                self.assertIn("VD", window.training_label_checks)
                self.assertEqual(window.detection_label.findText("VD") >= 0, True)
                del window.settings["label_targets"]["PY"]
                window.refresh_channel_options()
                self.assertNotIn("PY", window.training_label_checks)
                self.assertNotIn("PY", window.settings["training_labels_by_channel"]["Red"])
            finally:
                self.close_window(window)
                self.close_window(other)


if __name__ == "__main__":
    unittest.main()
