"""Interactive, draft-only preprocessing workspace."""
from __future__ import annotations
import copy
import numpy as np
from PyQt6 import QtCore, QtWidgets as W
import pyqtgraph as pg
from .preprocessing import preview_channel


class PreviewWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(object)

    def __init__(self, channel, settings, revision, parent):
        super().__init__(parent)
        self.channel, self.settings, self.revision = channel, settings, revision

    def run(self):
        try:
            result = preview_channel(self.channel, self.settings)
            self.result.emit((self.revision, result, None))
        except Exception as exc:
            self.result.emit((self.revision, None, str(exc)))


class PreprocessingWidget(W.QWidget):
    applied = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = []
        self.settings = {}
        self.revision = 0
        self.result = None
        self.worker = None
        self.updating = False
        layout = W.QVBoxLayout(self)
        description = W.QLabel(
            "Preview filtering before sorting · Drag the gold artifact cutoff lines or move the sliders. "
            "Red samples are excluded, including padding. Gold dots are candidate spikes, not classified neurons. "
            "Apply settings when satisfied; sorting still runs from Recording and sorting."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        bar = W.QHBoxLayout()
        self.channel = W.QComboBox()
        self.channel.currentIndexChanged.connect(self.select_channel)
        bar.addWidget(W.QLabel("Preview channel"))
        bar.addWidget(self.channel)
        self.start = W.QDoubleSpinBox()
        self.start.setDecimals(3)
        self.start.setSuffix(" s")
        self.start.valueChanged.connect(self.draw)
        bar.addWidget(W.QLabel("Start"))
        bar.addWidget(self.start)
        self.duration = W.QDoubleSpinBox()
        self.duration.setRange(.01, 120)
        self.duration.setValue(2)
        self.duration.setSuffix(" s")
        self.duration.valueChanged.connect(self.draw)
        bar.addWidget(W.QLabel("Window"))
        bar.addWidget(self.duration)
        layout.addLayout(bar)
        self.time_slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 10000)
        self.time_slider.valueChanged.connect(self.seek)
        layout.addWidget(self.time_slider)
        body = W.QHBoxLayout()
        form = W.QFormLayout()
        self.controls = {}
        self.sliders = {}
        for key, label, lo, hi, decimals in [
            ("artifact", "Artifact cutoff (mV, channel)", .000001, 1e9, 6),
            ("artifact_padding_ms", "Artifact padding (ms)", 0, 10000, 2),
            ("filter_low_hz", "Low cutoff (Hz)", .01, 1e7, 2),
            ("filter_high_hz", "High cutoff (Hz)", .01, 1e7, 2),
            ("filter_order", "Filter order", 1, 20, 0),
            ("threshold_factor", "Detection threshold (× noise)", .01, 1000, 2),
            ("shadow_period_ms", "Minimum spike spacing (ms)", .001, 10000, 3),
            ("adaptive_window_s", "Noise window (s)", .001, 100000, 3),
            ("adaptive_step_s", "Noise step (s)", .001, 100000, 3),
            ("noise_quantile", "Noise quantile", .001, .999, 3),
            ("minimum_valid_fraction", "Minimum valid fraction", .001, 1, 3),
        ]:
            box = W.QDoubleSpinBox()
            box.setDecimals(decimals)
            box.setRange(lo, hi)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self.changed)
            self.controls[key] = box
            row = W.QVBoxLayout()
            row.addWidget(box)
            if key in {"artifact", "filter_low_hz", "filter_high_hz", "threshold_factor"}:
                slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setRange(0, 1000)
                slider.valueChanged.connect(lambda v, k=key: self.slide(k, v))
                self.sliders[key] = slider
                row.addWidget(slider)
            form.addRow(label, row)
        self.polarity = W.QComboBox()
        self.polarity.addItems(["positive", "negative", "both"])
        self.polarity.currentTextChanged.connect(self.changed)
        form.addRow("Spike polarity", self.polarity)
        note = W.QLabel("Cutoffs, padding and detection settings apply globally.\nArtifact cutoffs are stored separately for each channel.\nOther pipeline settings remain available in Setup.")
        note.setWordWrap(True)
        form.addRow(note)
        self.apply = W.QPushButton("Apply preview settings to sorting")
        self.apply.clicked.connect(self.apply_settings)
        form.addRow(self.apply)
        self.reset = W.QPushButton("Reset draft to current sorting settings")
        self.reset.clicked.connect(lambda: self.set_recording(self.channels, self.baseline))
        form.addRow(self.reset)
        self.fit_cutoffs = W.QCheckBox("Fit artifact cutoff lines in raw plot")
        self.fit_cutoffs.toggled.connect(self.draw)
        form.addRow(self.fit_cutoffs)
        controls = W.QWidget()
        controls.setLayout(form)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(430)
        scroll.setMaximumWidth(470)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body.addWidget(scroll)
        plots = W.QVBoxLayout()
        self.raw = pg.PlotWidget(title="Raw signal centered at median · red = excluded")
        self.filtered = pg.PlotWidget(title="Filtered signal · adaptive thresholds and candidate spikes")
        self.filtered.setXLink(self.raw)
        for plot in (self.raw, self.filtered):
            plot.setLabel("bottom", "Recording time", units="s")
            plots.addWidget(plot)
        body.addLayout(plots, 1)
        layout.addLayout(body, 1)
        self.status = W.QLabel("Load a recording to preview its actual signal.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.calculate)
        self.apply.setEnabled(False)
        self.setEnabled(False)

    def set_recording(self, channels, settings):
        self.timer.stop()
        self.updating = True
        self.channels = channels
        self.baseline = copy.deepcopy(settings)
        self.settings = copy.deepcopy(settings)
        self.bounds_channel = None
        old = self.channel.currentText()
        self.channel.clear()
        self.channel.addItems([c["name"] for c in channels])
        if old in [c["name"] for c in channels]:
            self.channel.setCurrentText(old)
        self.updating = False
        self.setEnabled(bool(channels))
        self.select_channel()

    def select_channel(self, *_):
        if self.updating or not self.channels:
            return
        self.updating = True
        c = self.channels[self.channel.currentIndex()]
        for key, box in self.controls.items():
            value = (self.settings["artifact_thresholds_mV"].get(c["name"], self.settings["default_artifact_threshold_mV"])
                     if key == "artifact" else self.settings[key])
            box.setValue(value)
        self.polarity.setCurrentText(self.settings["polarity"])
        self.start.setRange(float(c["time"][0]), float(c["time"][-1]))
        self.updating = False
        self.changed()

    def seek(self, value):
        lo, hi = self.start.minimum(), self.start.maximum()
        self.start.setValue(lo + (hi - lo) * value / 10000)

    def slide(self, key, value):
        if not self.updating:
            lo, hi = self.slider_bounds[key]
            self.controls[key].setValue(lo * (hi / lo) ** (value / 1000))

    def changed(self, *_):
        if self.updating or not self.channels:
            return
        c = self.channels[self.channel.currentIndex()]
        for key, box in self.controls.items():
            if key == "artifact":
                self.settings["artifact_thresholds_mV"][c["name"]] = box.value()
            else:
                self.settings[key] = int(box.value()) if key == "filter_order" else box.value()
        self.settings["polarity"] = self.polarity.currentText()
        fs = 1 / np.median(np.diff(c["time"][:1000]))
        self.slider_bounds = {
            "artifact": (max(self.controls["artifact"].value() / 100, 1e-6), max(self.controls["artifact"].value() * 100, .001)),
            "filter_low_hz": (.01, max(fs * .45, .02)),
            "filter_high_hz": (.01, max(fs * .45, .02)),
            "threshold_factor": (.01, 1000),
        } if not hasattr(self, "slider_bounds") or getattr(self, "bounds_channel", None) != c["name"] else self.slider_bounds
        self.bounds_channel = c["name"]
        self.updating = True
        for key, slider in self.sliders.items():
            lo, hi = self.slider_bounds[key]
            slider.setValue(int(np.clip(np.log(max(self.controls[key].value(), lo) / lo) / np.log(hi / lo) * 1000, 0, 1000)))
        self.updating = False
        self.revision += 1
        self.result = None
        self.raw.clear()
        self.filtered.clear()
        self.apply.setEnabled(False)
        self.status.setText("Updating full-channel preview…")
        self.timer.start()

    def calculate(self):
        if self.worker and self.worker.isRunning():
            self.timer.start()
            return
        self.worker = PreviewWorker(self.channels[self.channel.currentIndex()], copy.deepcopy(self.settings), self.revision, self)
        self.worker.result.connect(self.received)
        self.worker.start()

    def received(self, payload):
        revision, result, error = payload
        if revision != self.revision:
            return
        if error:
            self.status.setText("Preview unavailable: " + error + " Adjust the settings and try again.")
            return
        self.result = result
        self.apply.setEnabled(True)
        p, d = result
        self.status.setText(
            f"Full channel: {100 * p['artifact_mask'].mean():.3f}% excluded · "
            f"{p['artifact_event_count']} artifact intervals · {len(d['spike_indices'])} candidate spikes · "
            f"Effective filter {p['filter_low_hz']:.2f}–{p['filter_high_hz']:.2f} Hz "
            f"(high cutoff capped at 45% of sampling rate). Draft settings; apply to use for sorting."
        )
        self.draw()

    def draw(self, *_):
        if self.result is None:
            return
        p, d = self.result
        t = p["time"]
        self.time_slider.blockSignals(True)
        span = self.start.maximum() - self.start.minimum()
        self.time_slider.setValue(int((self.start.value() - self.start.minimum()) / span * 10000) if span else 0)
        self.time_slider.blockSignals(False)
        a, b = np.searchsorted(t, [self.start.value(), self.start.value() + self.duration.value()])
        b = min(len(t), max(b, a + 2))
        x = t[a:b]
        self.raw.clear()
        self.filtered.clear()
        for plot in (self.raw, self.filtered):
            plot.setLabel("left", "Amplitude", units=str(p["units"]))
        def trace(plot, y, color):
            curve = plot.plot(x, y[a:b], pen=pg.mkPen(color, width=1), connect="finite")
            curve.setDownsampling(auto=True, method="peak")
        trace(self.raw, p["centered_signal"], "#aeb8c7")
        trace(self.raw, np.where(p["artifact_mask"], p["centered_signal"], np.nan), "#ff7676")
        for sign in (-1, 1):
            line = pg.InfiniteLine(sign * p["artifact_threshold_native"], angle=0, movable=True, pen="#ffd166")
            line.sigPositionChangeFinished.connect(lambda line, scale=p["artifact_threshold_native"] / p["artifact_threshold_mV"]: self.controls["artifact"].setValue(abs(line.value()) / scale))
            self.raw.addItem(line, ignoreBounds=not self.fit_cutoffs.isChecked())
        trace(self.filtered, d["centered_filtered_signal"], "#7ee1c5")
        for sign in (-1, 1):
            if self.settings["polarity"] == "both" or (sign == 1) == (self.settings["polarity"] == "positive"):
                trace(self.filtered, sign * d["lower_threshold_trace"], "#ffd166")
        idx = d["spike_indices"]
        idx = idx[(idx >= a) & (idx < b)]
        self.filtered.plot(t[idx], d["centered_filtered_signal"][idx], pen=None, symbol="o", symbolSize=6, symbolBrush="#ffd166")
        self.raw.enableAutoRange(axis="y")
        self.filtered.enableAutoRange(axis="y")
        if len(x):
            self.raw.setXRange(float(x[0]), float(x[-1]), padding=0)

    def apply_settings(self):
        if self.result is not None:
            self.baseline = copy.deepcopy(self.settings)
            self.applied.emit(copy.deepcopy(self.settings))
            self.status.setText("Preview settings applied. Return to Recording and sorting to filter and detect candidates.")

    def shutdown(self):
        self.timer.stop()
        self.revision += 1
        if self.worker:
            self.worker.wait()
