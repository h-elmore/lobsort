import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import unittest
import numpy as np
from PyQt6 import QtWidgets as W
from lobsort.engine import PipelineSession, default_settings
from lobsort.demo import demo_channels
from lobsort.preprocessing import preview_channel
from lobsort.gui import MainWindow


class PreviewTests(unittest.TestCase):
    def test_preview_matches_sorting_without_mutating_inputs(self):
        settings = default_settings()
        settings.update(threshold_factor=4, shadow_period_ms=2)
        original = copy.deepcopy(settings)
        channels = demo_channels(35)
        values = channels[0]["values"].copy()
        with tempfile.TemporaryDirectory() as folder:
            session = PipelineSession(channels, settings, folder, lambda _: None).prepare()
            p, d = preview_channel(channels[0], settings)
            expected = session.ns["processed_channels"]["Red"]
            for key in ("filtered_signal", "artifact_mask", "centered_signal"):
                np.testing.assert_allclose(p[key], expected[key], equal_nan=True)
            np.testing.assert_array_equal(d["spike_indices"], session.ns["candidate_indices"])
            np.testing.assert_allclose(d["lower_threshold_trace"], session.ns["candidate_result"]["lower_threshold_trace"])
        self.assertEqual(settings, original)
        np.testing.assert_array_equal(channels[0]["values"], values)

    def test_units_gaps_and_empty_detection(self):
        t = 12 + np.arange(30000) / 10000
        rng = np.random.default_rng(5)
        y = rng.normal(0, .1, len(t))
        y[8000:8005] = 10
        y[14000] = np.nan
        settings = default_settings()
        settings.update(default_artifact_threshold_mV=2, threshold_factor=1e8)
        results = []
        for units, scale in (("mV", 1), ("uV", 1000), ("V", .001)):
            p, d = preview_channel(dict(name="A", time=t, values=y * scale, units=units), settings)
            self.assertEqual(len(d["spike_indices"]), 0)
            self.assertTrue(p["artifact_mask"][8000])
            self.assertTrue(p["artifact_mask"][14000])
            self.assertTrue(np.isnan(p["filtered_signal"][14000]))
            results.append(p["artifact_mask"])
        for mask in results[1:]:
            np.testing.assert_array_equal(mask, results[0])

    def test_invalid_settings_and_all_masked(self):
        channel = demo_channels(2)[0]
        for update in (dict(filter_low_hz=2000, filter_high_hz=1000),
                       dict(filter_low_hz=1e6), dict(artifact_padding_ms=-1),
                       dict(threshold_factor=float("nan")), dict(filter_order=0)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                preview_channel(channel, {**default_settings(), **update})
        settings = default_settings()
        settings["artifact_thresholds_mV"]["Red"] = 1e-12
        with self.assertRaisesRegex(ValueError, "Too few"):
            preview_channel(channel, settings)


class PreviewGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])

    def await_preview(self, widget):
        deadline = time.monotonic() + 20
        while widget.result is None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNotNone(widget.result, widget.status.text())

    def test_draft_apply_reset_channel_and_stale_result(self):
        window = MainWindow()
        try:
            window.demo()
            widget = window.preprocessing
            self.await_preview(widget)
            old = window.settings["threshold_factor"]
            widget.controls["threshold_factor"].setValue(old + 2)
            self.assertFalse(widget.apply.isEnabled())
            self.assertEqual(window.settings["threshold_factor"], old)
            widget.received((widget.revision - 1, None, "obsolete failure"))
            self.assertNotIn("obsolete", widget.status.text())
            self.await_preview(widget)
            window.session = object()
            widget.apply.click()
            self.assertEqual(window.settings["threshold_factor"], old + 2)
            self.assertIsNone(window.session)
            widget.controls["artifact"].setValue(90)
            widget.channel.setCurrentText("Green")
            self.assertEqual(widget.controls["artifact"].value(), 1400)
            self.await_preview(widget)
            self.assertEqual(widget.result[0]["channel_index"], 0)
            np.testing.assert_array_equal(widget.result[0]["raw_signal"], next(c["values"] for c in window.channels if c["name"] == "Green"))
            widget.channel.setCurrentText("Red")
            self.assertEqual(widget.controls["artifact"].value(), 90)
            widget.reset.click()
            self.assertEqual(widget.controls["artifact"].value(), 120)
            self.await_preview(widget)
            # Invalid edits cannot apply; a corrected draft recovers inline.
            widget.controls["filter_low_hz"].setValue(2000)
            deadline = time.monotonic() + 20
            while "Preview unavailable" not in widget.status.text() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertIn("Preview unavailable", widget.status.text())
            self.assertFalse(widget.apply.isEnabled())
            widget.controls["filter_low_hz"].setValue(100)
            self.await_preview(widget)
            widget.time_slider.setValue(5000)
            self.assertGreater(widget.start.value(), 0)
            widget.start.setValue(10)
            widget.duration.setValue(.1)
            self.assertEqual(widget.result[0]["time"][0], 0)
        finally:
            window.dirty = False
            window.close()


if __name__ == "__main__":
    unittest.main()
