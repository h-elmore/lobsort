import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets as W, QtCore
from lobsort.gui import MainWindow, STYLE, SettingsDialog
from lobsort.curation import Review


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])
        cls.app.setStyleSheet(STYLE)

    def test_setup_settings_and_review(self):
        window = MainWindow()
        window.demo()
        window.show()
        self.app.processEvents()
        self.assertEqual(window.target.currentText(), "Red")
        self.assertTrue(window.prepare_button.isEnabled())
        self.assertFalse(window.train_button.isEnabled())
        dialog = SettingsDialog(window.settings)
        self.assertIn("artifact_thresholds_mV", dialog.widgets)
        t = np.arange(1000) / 1000
        y = np.zeros(1000)
        y[400] = 5
        review = Review(
            {"A": dict(time=t, raw_signal=y, filtered_signal=y, units="mV")}
        )
        window.viewer.set_review(review)
        window.tabs.setCurrentWidget(window.viewer)
        self.app.processEvents()
        widget = window.viewer
        widget.mode.setCurrentIndex(1)
        widget.label.setCurrentText("PY")
        widget.snap.setValue(2)
        plot = widget.plots["A"][0]
        point = plot.getViewBox().mapViewToScene(QtCore.QPointF(0.4, 5))

        class Click:
            def button(self):
                return QtCore.Qt.MouseButton.LeftButton

            def scenePos(self):
                return point

        widget.clicked("A", Click())
        self.assertEqual(review.tables["A"].sample_index.tolist(), [400])
        widget.mode.setCurrentIndex(2)
        widget.clicked("A", Click())
        self.assertFalse(review.tables["A"].iloc[0].accepted_as_spike)
        widget.undo()
        self.assertTrue(review.tables["A"].iloc[0].accepted_as_spike)
        widget.redo()
        self.assertFalse(review.tables["A"].iloc[0].accepted_as_spike)
        widget.slider.setValue(1000000)
        self.app.processEvents()
        widget.play.setChecked(True)
        widget.play.setChecked(False)
        window.dirty = False
        window.close()
        dialog.close()


if __name__ == "__main__":
    unittest.main()
