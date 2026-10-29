"""Animal-named exports and nerve aliases preserve stable signal identities."""
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
from PyQt6 import QtWidgets as W, QtCore
from lobsort.curation import Review
from lobsort.demo import demo_channels
from lobsort.gui import MainWindow, ChannelNamesDialog, SettingsDialog
from lobsort.recording_metadata import validate_channel_names, file_stem


def sample_review(animal="1001_41"):
    t = 12 + np.arange(100) / 1000
    data = {source: dict(time=t, raw_signal=np.sin(t), filtered_signal=np.sin(t), units="mV") for source in ("Red", "Green")}
    tables = {
        "Red": pd.DataFrame(dict(sample_index=[30, 10, 20], time_seconds=t[[30, 10, 20]], manual_label=["PD", "PD", "Noise"], accepted_as_spike=[True, True, False])),
        "Green": pd.DataFrame(dict(sample_index=[40], time_seconds=t[[40]], manual_label=["LP"], accepted_as_spike=[True])),
    }
    return Review(data, tables, dict(animal_id=animal, channel_names={"Red": "pdn", "Green": "lvn"}, recording="test.smr", segment=0))


class RecordingExportTests(unittest.TestCase):
    def test_animal_folder_has_one_accepted_csv_per_nerve_and_details(self):
        review = sample_review()
        before = review.tables["Red"].copy(deep=True)
        with tempfile.TemporaryDirectory() as folder:
            destination = review.export(folder)
            self.assertEqual(destination, Path(folder) / "1001_41")
            self.assertEqual({p.name for p in destination.glob("*.csv")}, {"pdn.csv", "lvn.csv"})
            pdn = pd.read_csv(destination / "pdn.csv")
            self.assertEqual(pdn.sample_index.tolist(), [10, 30])
            self.assertEqual(pdn.spike_number.tolist(), [1, 2])
            self.assertEqual(set(pdn.animal_id), {"1001_41"})
            self.assertEqual(set(pdn.channel), {"pdn"})
            self.assertEqual(set(pdn.source_channel), {"Red"})
            self.assertEqual(set(pdn.manual_label), {"PD"})
            np.testing.assert_allclose(pdn.time_seconds, review.processed["Red"]["time"][[10, 30]])
            details = destination / "review_details"
            self.assertEqual(len(pd.read_csv(details / "pdn_all_candidates.csv")), 3)
            self.assertTrue((details / "edit_history.json").is_file())
            saved = json.loads((details / "settings.json").read_text())
            self.assertEqual(saved["export_files"]["Red"], dict(name="pdn", file="pdn.csv"))
        pd.testing.assert_frame_equal(before, review.tables["Red"])

    def test_reexport_updates_spikes_and_can_add_another_sorted_nerve(self):
        review = sample_review()
        green = review.tables.pop("Green")
        with tempfile.TemporaryDirectory() as folder:
            destination = review.export(folder)
            review.change("Red", 30, "remove")
            review.set_table("Green", green)
            self.assertEqual(review.export(folder), destination)
            self.assertEqual(pd.read_csv(destination / "pdn.csv").sample_index.tolist(), [10])
            self.assertTrue((destination / "lvn.csv").is_file())
            before = (destination / "pdn.csv").read_bytes()
            review.metadata["recording"] = "other.smr"
            with self.assertRaisesRegex(ValueError, "different recording"):
                review.export(folder)
            self.assertEqual((destination / "pdn.csv").read_bytes(), before)

    def test_safe_unique_paths_and_empty_nerve_files(self):
        review = sample_review("../animal:41")
        review.metadata["channel_names"] = {"Red": "pd/n", "Green": "pd:n"}
        review.tables["Green"]["accepted_as_spike"] = False
        with tempfile.TemporaryDirectory() as folder:
            destination = review.export(folder)
            self.assertEqual(destination.parent, Path(folder))
            self.assertTrue((destination / "pd_n.csv").is_file())
            self.assertTrue(pd.read_csv(destination / "pd_n_2.csv").empty)
            self.assertEqual(file_stem("CON"), "_CON")
            self.assertNotIn("/", file_stem("../animal:41"))

    def test_names_and_original_keys_survive_review_save_and_undo(self):
        review = sample_review()
        review.change("Red", 10, "remove")
        review.undo()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.lobsort"
            review.save(path)
            restored = Review.load(path)
            self.assertEqual(restored.metadata["animal_id"], "1001_41")
            self.assertEqual(restored.channel_name("Red"), "pdn")
            self.assertEqual(set(restored.tables), {"Red", "Green"})
            self.assertEqual(restored.audit[0]["channel"], "Red")
            destination = restored.export(folder)
            self.assertEqual(len(pd.read_csv(destination / "pdn.csv")), 2)


class ChannelNamingGuiTests(unittest.TestCase):
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

    def close_window(self, window):
        self.finish_worker(window)
        window.dirty = False
        window.close()
        self.app.processEvents()

    def test_naming_dialog_validates_without_partially_applying(self):
        dialog = ChannelNamesDialog(["Red", "Green"], {})
        try:
            dialog.table.item(0, 1).setText("pdn")
            dialog.table.item(1, 1).setText("PDN")
            dialog.validate()
            self.assertEqual(dialog.result(), 0)
            self.assertIn("repeated", dialog.error.text())
            self.assertEqual(dialog.names, {})
            dialog.table.item(1, 1).setText("")
            dialog.validate()
            self.assertIn("needs a name", dialog.error.text())
            dialog.table.item(1, 1).setText("lvn")
            dialog.validate()
            self.assertEqual(dialog.names, {"Red": "pdn", "Green": "lvn"})
        finally:
            dialog.close()
        with self.assertRaises(ValueError):
            validate_channel_names(["Red", "Green"], {"Red": "Green"})

    def test_aliases_keep_candidates_calibration_labeler_and_export_on_correct_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow()
            try:
                window.animal_id.setText("1001_41")
                window.settings.update(threshold_factor=4, shadow_period_ms=2)
                window.output.setText(folder)
                window.loaded((demo_channels(5), [], 1), "demo.synthetic", 0)
                window.reference.setCurrentText("Green")
                window.prepare()
                self.finish_worker(window)
                session = window.session
                self.assertIsNotNone(session)
                output = session.output_dir
                masks = session.ns["processed_channels"]["Red"]["artifact_mask"].copy()
                thresholds = copy.deepcopy(window.settings["artifact_thresholds_mV"])
                window.preprocessing.controls["artifact"].setValue(90)
                revision = window.preprocessing.revision
                window.apply_channel_names({"Red": "pdn", "Green": "lvn"})
                self.assertIs(window.session, session)
                self.assertEqual(window.target.currentText(), "pdn")
                self.assertEqual(window.target.currentData(), "Red")
                self.assertEqual(window.reference.currentText(), "lvn")
                self.assertEqual(window.viewer.channel.currentText(), "pdn")
                self.assertEqual(window.preprocessing.channel.currentText(), "pdn")
                self.assertEqual(window.preprocessing.controls["artifact"].value(), 90)
                self.assertEqual(window.preprocessing.revision, revision)
                self.assertEqual(window.settings["artifact_thresholds_mV"], thresholds)
                np.testing.assert_array_equal(masks, session.ns["processed_channels"]["Red"]["artifact_mask"])
                window.open_labeler()
                self.assertIn("pdn", window.label_window.windowTitle())
                self.assertIn("lvn", window.label_window.reference_plot.plotItem.titleLabel.text)
                window.close_labeler()
                window.kind.setCurrentIndex(2)
                window.detection_label.setCurrentText("PD")
                window.train()
                self.finish_worker(window)
                self.assertTrue(window.review.tables["Red"].accepted_as_spike.all())
                self.assertEqual(window.viewer.channel.currentData(), "Red")
                self.assertEqual(window.viewer.channel.currentText(), "pdn")
                window.show()
                self.app.processEvents()
                widget = window.viewer
                widget.mode.setCurrentIndex(2)
                first = int(window.review.tables["Red"].sample_index.iloc[0])
                data = window.review.processed["Red"]
                point = widget.plots["Red"][0].getViewBox().mapViewToScene(
                    QtCore.QPointF(float(data["time"][first]), float(data["filtered_signal"][first]))
                )

                class Click:
                    def button(self):
                        return QtCore.Qt.MouseButton.LeftButton

                    def scenePos(self):
                        return point

                widget.clicked("Red", Click())
                self.assertFalse(window.review.tables["Red"].iloc[0].accepted_as_spike)
                widget.undo()
                self.assertTrue(window.review.tables["Red"].accepted_as_spike.all())
                with patch.object(W.QFileDialog, "getExistingDirectory", return_value=folder):
                    window.export()
                destination = Path(folder) / "1001_41"
                self.assertTrue((destination / "pdn.csv").is_file())
                self.assertTrue((destination / "review_details/review.lobsort").is_file())
                self.assertEqual(set(pd.read_csv(destination / "pdn.csv").manual_label), {"PD"})
                window.prepare()
                self.finish_worker(window)
                self.assertEqual(window.session.output_dir, output)
                self.assertEqual(window.session.settings["channel_name"], "Red")
            finally:
                self.close_window(window)

    def test_review_reopen_profile_and_threshold_table_preserve_names(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow()
            try:
                review = sample_review()
                review.metadata["settings"] = {**window.settings, "channel_names": review.metadata["channel_names"]}
                path = str(Path(folder) / "review.lobsort")
                review.save(path)
                with patch.object(W.QFileDialog, "getOpenFileName", return_value=(path, "")):
                    window.open_project()
                self.assertEqual(window.animal_id.text(), "1001_41")
                self.assertEqual(window.viewer.channel.currentText(), "pdn")
                self.assertEqual(window.viewer.channel.currentData(), "Red")
                self.assertTrue(window.name_channels.isEnabled())
                window.apply_channel_names({"Red": "pdn_left", "Green": "lvn"})
                self.assertEqual(window.viewer.channel.currentText(), "pdn_left")
                profile = str(Path(folder) / "profile.json")
                with patch.object(W.QFileDialog, "getSaveFileName", return_value=(profile, "")):
                    window.save_profile()
                self.assertEqual(json.loads(Path(profile).read_text())["channel_names"]["Red"], "pdn_left")
                # A nerve alias may equal another original channel name without
                # reassigning that source's physical artifact cutoff.
                window.settings["channel_names"] = {"Red": "Green", "Green": "lvn"}
                dialog = SettingsDialog(window.settings)
                before = copy.deepcopy(window.settings["artifact_thresholds_mV"])
                dialog.validate()
                self.assertEqual(dialog.result(), 1)
                self.assertEqual(dialog.settings["artifact_thresholds_mV"], before)
                dialog.close()
                older_profile = copy.deepcopy(window.settings)
                del older_profile["channel_names"]
                old_path = str(Path(folder) / "old_profile.json")
                Path(old_path).write_text(json.dumps(older_profile))
                with patch.object(W.QFileDialog, "getOpenFileName", return_value=(old_path, "")), patch("lobsort.gui.SettingsDialog.exec", return_value=1):
                    window.load_profile()
                self.assertEqual(window.settings["channel_names"], {"Red": "Green", "Green": "lvn"})
                self.assertEqual(window.review.channel_name("Red"), "Green")
            finally:
                self.close_window(window)


if __name__ == "__main__":
    unittest.main()
