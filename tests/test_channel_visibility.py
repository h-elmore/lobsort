"""Review visibility affects plotting and editing targets, not spike data."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets as W
from lobsort.curation import Review
from lobsort.viewer import ReviewWidget
from lobsort.gui import MainWindow


def review_with_channels():
    t = np.arange(5000) / 1000
    processed, tables = {}, {}
    for source, index in (("Red", 500), ("Green", 600), ("Blue", 700)):
        signal = np.sin(t * 100) * .1
        signal[index] = 4
        processed[source] = dict(time=t, raw_signal=signal, filtered_signal=signal, units="mV")
        tables[source] = pd.DataFrame(dict(sample_index=[index], time_seconds=[t[index]], manual_label=["PD"], accepted_as_spike=[True]))
    return Review(processed, tables, dict(animal_id="1001_41", recording="test.smr", segment=0, channel_names={"Red": "pdn", "Green": "lvn", "Blue": "mvn"}))


class ChannelVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])

    def setUp(self):
        self.review = review_with_channels()
        self.widget = ReviewWidget()
        self.widget.resize(1400, 1100)
        self.widget.set_review(self.review)
        self.widget.window.setValue(1)
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.timer.stop()
        self.widget.close()
        self.app.processEvents()

    def test_checkboxes_collapse_plots_skip_work_and_update_at_current_time(self):
        widget = self.widget
        initial_height = widget.plots["Red"][0].height()
        widget.channels_toggle.setChecked(True)
        self.assertFalse(widget.channel_panel.isHidden())
        widget.channel_checks["Green"].setChecked(False)
        widget.channel_checks["Blue"].setChecked(False)
        self.app.processEvents()
        self.assertEqual(widget.visible_channels(), ["Red"])
        self.assertTrue(widget.plots["Green"][0].isHidden())
        self.assertTrue(widget.plots["Blue"][0].isHidden())
        self.assertGreater(widget.plots["Red"][0].height(), initial_height)
        self.assertEqual(widget.channel_summary.text(), "1 of 3 channels shown")
        self.assertIn("1 accepted spikes visible", widget.status.text())
        hidden_trace = widget.plots["Green"][1]
        with patch.object(hidden_trace, "setData") as draw_hidden:
            widget.start.setValue(2)
            draw_hidden.assert_not_called()
        widget.channel_checks["Green"].setChecked(True)
        self.assertGreaterEqual(float(hidden_trace.xData[0]), 2)
        self.assertLessEqual(float(hidden_trace.xData[-1]), 3.001)
        widget.channels_toggle.setChecked(False)
        self.assertTrue(widget.channel_panel.isHidden())

    def test_hiding_edit_target_selects_visible_channel_and_hide_all_is_safe(self):
        widget = self.widget
        widget.channel_checks["Red"].setChecked(False)
        self.assertEqual(widget.channel.currentData(), "Green")
        self.assertEqual([widget.channel.itemText(i) for i in range(widget.channel.count())], ["lvn", "mvn"])
        widget.play.setChecked(True)
        widget.hide_all.click()
        self.assertEqual(widget.visible_channels(), [])
        self.assertEqual(widget.channel.count(), 0)
        self.assertFalse(widget.mode.isEnabled())
        self.assertFalse(widget.play.isChecked())
        self.assertFalse(widget.timer.isActive())
        self.assertFalse(widget.empty_view.isHidden())
        self.assertIn("0 accepted spikes visible", widget.status.text())
        widget.show_all.click()
        self.assertEqual(widget.visible_channels(), ["Red", "Green", "Blue"])
        self.assertTrue(widget.mode.isEnabled())
        self.assertTrue(widget.empty_view.isHidden())
        widget.channel_checks["Green"].setChecked(False)
        widget.select_channel("Green")
        self.assertTrue(widget.channel_checks["Green"].isChecked())
        self.assertEqual(widget.channel.currentData(), "Green")

    def test_choices_survive_view_rebuild_rename_save_and_new_channels(self):
        widget = self.widget
        widget.channel_checks["Red"].setChecked(False)
        self.review.metadata["channel_names"]["Red"] = "pdn_left"
        widget.update_channel_names()
        self.assertEqual(widget.channel_checks["Red"].text(), "pdn_left")
        self.assertFalse(widget.channel_checks["Red"].isChecked())
        widget.set_review(self.review)
        self.assertFalse(widget.channel_checks["Red"].isChecked())
        self.review.processed["Orange"] = self.review.processed["Blue"]
        widget.set_review(self.review)
        self.assertTrue(widget.channel_checks["Orange"].isChecked())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.lobsort"
            self.review.save(path)
            restored = Review.load(path)
            self.assertEqual(restored.metadata["hidden_channels"], ["Red"])
            widget.set_review(restored)
            self.assertFalse(widget.channel_checks["Red"].isChecked())
            self.assertTrue(widget.channel_checks["Orange"].isChecked())
            self.assertEqual(widget.channel_checks["Red"].text(), "pdn_left")

    def test_hiding_preserves_spike_decisions_and_all_channel_exports(self):
        before = copy.deepcopy(self.review.tables)
        with tempfile.TemporaryDirectory() as folder:
            destination = self.review.export(folder)
            contents = {path.name: path.read_bytes() for path in destination.glob("*.csv")}
            signals = []
            self.widget.edited.connect(lambda: signals.append(True))
            self.widget.hide_all.click()
            self.assertEqual(len(signals), 1)
            self.review.export(folder)
            self.assertEqual({path.name: path.read_bytes() for path in destination.glob("*.csv")}, contents)
            self.assertEqual(self.review.audit, [])
            for source, table in before.items():
                pd.testing.assert_frame_equal(table, self.review.tables[source])

    def test_old_reviews_show_all_channels_and_save_marks_view_change_dirty(self):
        self.assertNotIn("hidden_channels", self.review.metadata)
        self.assertEqual(len(self.widget.visible_channels()), 3)
        window = MainWindow()
        try:
            window.review = self.review
            window.viewer.set_review(self.review)
            window.viewer.channel_checks["Green"].setChecked(False)
            self.assertTrue(window.dirty)
            self.assertEqual(self.review.metadata["hidden_channels"], ["Green"])
        finally:
            window.dirty = False
            window.close()


if __name__ == "__main__":
    unittest.main()
