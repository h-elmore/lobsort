"""Linked multichannel recording browser with explicit edit modes."""

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets as W
import pyqtgraph as pg

COLORS = ["#43d9ad", "#ffcb6b", "#82aaff", "#f78c6c", "#c792ea", "#89ddff"]


def spin(low, high, value, step=0.1, decimals=3):
    box = W.QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class ScrollPlot(pg.PlotWidget):
    scrolled = QtCore.pyqtSignal(float)
    zoomed = QtCore.pyqtSignal(float)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.zoomed.emit(1.2 if event.angleDelta().y() < 0 else 1 / 1.2)
            event.accept()
        else:
            self.scrolled.emit(-event.angleDelta().y() / 120 * 0.15)
            event.accept()


class ReviewWidget(W.QWidget):
    edited = QtCore.pyqtSignal()
    message = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review = None
        self.plots = {}
        self.channel_checks = {}
        self.syncing = False
        outer = W.QVBoxLayout(self)
        header = W.QLabel("Review & refine")
        header.setObjectName("title")
        outer.addWidget(header)
        self.help = W.QLabel(
            "Scroll to move through time · Ctrl + wheel to zoom · Choose an edit tool, then click the trace."
        )
        outer.addWidget(self.help)
        tools = W.QHBoxLayout()
        outer.addLayout(tools)
        self.mode = W.QComboBox()
        self.mode.addItems(
            ["Browse", "Add / restore spike", "Deselect spike", "Relabel spike"]
        )
        self.channel = W.QComboBox()
        self.label = W.QComboBox()
        self.label.setEditable(True)
        self.label.addItems(["PY", "LP", "PD", "Spike"])
        self.snap = spin(0, 50, 2, 1)
        self.snap.setSuffix(" ms")
        self.signal = W.QComboBox()
        self.signal.addItems(["Filtered", "Raw"])
        self.rejected = W.QCheckBox("Show rejected")
        self.rejected.setChecked(True)
        self.yzoom = spin(0.1, 100, 1, 0.5, 2)
        for title, widget in [
            ("Tool", self.mode),
            ("Edit channel", self.channel),
            ("Label", self.label),
            ("Snap radius", self.snap),
            ("Trace", self.signal),
            ("Y zoom", self.yzoom),
        ]:
            tools.addWidget(W.QLabel(title))
            tools.addWidget(widget)
        tools.addWidget(self.rejected)
        visibility = W.QHBoxLayout()
        outer.addLayout(visibility)
        self.channels_toggle = W.QToolButton()
        self.channels_toggle.setText("Channels…")
        self.channels_toggle.setCheckable(True)
        self.channels_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.channels_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        visibility.addWidget(self.channels_toggle)
        self.channel_summary = W.QLabel("No channels loaded")
        visibility.addWidget(self.channel_summary)
        self.show_all = W.QPushButton("Show all")
        self.hide_all = W.QPushButton("Hide all")
        visibility.addWidget(self.show_all)
        visibility.addWidget(self.hide_all)
        visibility.addStretch()
        self.channel_panel = W.QScrollArea()
        self.channel_panel.setWidgetResizable(True)
        self.channel_panel.setMaximumHeight(125)
        self.channel_panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        choices = W.QWidget()
        self.channel_grid = W.QGridLayout(choices)
        self.channel_panel.setWidget(choices)
        self.channel_panel.setVisible(False)
        outer.addWidget(self.channel_panel)
        self.channels_toggle.toggled.connect(self.toggle_channel_panel)
        self.show_all.clicked.connect(lambda: self.set_all_channels(True))
        self.hide_all.clicked.connect(lambda: self.set_all_channels(False))
        for button in (self.channels_toggle, self.show_all, self.hide_all):
            button.setEnabled(False)
        self.scroll = W.QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self.plot_container = W.QWidget()
        self.plot_layout = W.QVBoxLayout(self.plot_container)
        self.scroll.setWidget(self.plot_container)
        nav = W.QHBoxLayout()
        outer.addLayout(nav)
        self.play = W.QPushButton("Play")
        self.play.setCheckable(True)
        self.start = spin(-1e9, 1e9, 0, 0.1)
        self.start.setSuffix(" s")
        self.window = spin(0.005, 1e8, 15, 1)
        self.window.setSuffix(" s")
        for title, fn in [
            ("←", lambda: self.shift(-0.75)),
            ("→", lambda: self.shift(0.75)),
            ("Previous spike", lambda: self.jump(-1)),
            ("Next spike", lambda: self.jump(1)),
            ("Undo", self.undo),
            ("Redo", self.redo),
        ]:
            b = W.QPushButton(title)
            b.clicked.connect(fn)
            nav.addWidget(b)
        nav.addWidget(self.play)
        nav.addWidget(W.QLabel("Start"))
        nav.addWidget(self.start)
        nav.addWidget(W.QLabel("Window"))
        nav.addWidget(self.window)
        self.slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000000)
        outer.addWidget(self.slider)
        self.status = W.QLabel("Open a recording or a saved review.")
        outer.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(lambda: self.shift(0.05))
        self.play.toggled.connect(
            lambda on: self.timer.start() if on else self.timer.stop()
        )
        self.start.valueChanged.connect(self.refresh)
        self.window.valueChanged.connect(self.refresh)
        self.slider.valueChanged.connect(self.slider_changed)
        for w in [self.signal, self.channel]:
            w.currentTextChanged.connect(self.refresh)
        self.rejected.toggled.connect(self.refresh)
        self.yzoom.valueChanged.connect(self.refresh)
        self.shortcuts = []
        for key, fn in [
            ("Left", lambda: self.shift(-0.25)),
            ("Right", lambda: self.shift(0.25)),
            ("Space", lambda: self.play.toggle()),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
        ]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(fn)
            self.shortcuts.append(shortcut)

    def set_review(self, review):
        self.timer.stop()
        self.play.setChecked(False)
        edit_source = self.channel.currentData()
        self.review = review
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.plots = {}
        while self.channel_grid.count():
            self.channel_grid.takeAt(0).widget().deleteLater()
        self.channel_checks = {}
        hidden = set(review.metadata.get("hidden_channels", []))
        self.first = min(float(d["time"][0]) for d in review.processed.values())
        self.last = max(float(d["time"][-1]) for d in review.processed.values())
        self.start.setRange(self.first, self.last)
        self.window.setMaximum(max(0.005, self.last - self.first))
        self.window.setValue(min(15, max(0.005, self.last - self.first)))
        self.start.setValue(self.first)
        for index, (name, data) in enumerate(review.processed.items()):
            check = W.QCheckBox(review.channel_name(name))
            check.setToolTip(f"{review.channel_name(name)} · recorded channel: {name}")
            check.setChecked(name not in hidden)
            check.toggled.connect(self.channel_visibility_changed)
            self.channel_grid.addWidget(check, index // 4, index % 4)
            self.channel_checks[name] = check
            plot = ScrollPlot()
            plot.setMinimumHeight(230)
            plot.setBackground("#141d2c")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("left", review.channel_name(name), units=data.get("units", ""))
            plot.setLabel("bottom", "Time", units="s")
            plot.setMouseEnabled(x=False, y=True)
            plot.scrolled.connect(self.shift)
            plot.zoomed.connect(
                lambda factor: self.window.setValue(self.window.value() * factor)
            )
            plot.scene().sigMouseClicked.connect(
                lambda event, n=name: self.clicked(n, event)
            )
            self.plot_layout.addWidget(plot)
            plot.setVisible(name not in hidden)
            self.plots[name] = (plot, plot.plot(pen=pg.mkPen("#a7b6cc", width=1)))
        self.empty_view = W.QLabel("No channels shown. Open Channels and check the nerves you want to view.")
        self.empty_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_view.setWordWrap(True)
        self.plot_layout.addWidget(self.empty_view)
        self.empty_view.setVisible(not self.visible_channels())
        labels = sorted(
            {
                str(label)
                for df in review.tables.values()
                for label in df.manual_label.unique()
                if str(label).lower() not in ["noise", "nan", "uncertain"]
            }
        )
        existing = {self.label.itemText(i) for i in range(self.label.count())}
        self.label.addItems([s for s in labels if s not in existing])
        self.apply_channel_visibility(preferred=edit_source)

    def toggle_channel_panel(self, expanded):
        self.channel_panel.setVisible(expanded)
        self.channels_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )

    def visible_channels(self):
        return [source for source, check in self.channel_checks.items() if check.isChecked()]

    def set_all_channels(self, visible):
        changed = False
        for check in self.channel_checks.values():
            changed |= check.isChecked() != visible
            check.blockSignals(True)
            check.setChecked(visible)
            check.blockSignals(False)
        if changed:
            self.channel_visibility_changed()

    def channel_visibility_changed(self, *_):
        if not self.review:
            return
        shown = set(self.visible_channels())
        self.review.metadata["hidden_channels"] = [source for source in self.review.processed if source not in shown]
        self.apply_channel_visibility()
        self.edited.emit()

    def apply_channel_visibility(self, preferred=None):
        visible = self.visible_channels()
        preferred = preferred or self.channel.currentData()
        self.channel.blockSignals(True)
        self.channel.clear()
        for source in visible:
            self.channel.addItem(self.review.channel_name(source), source)
        index = self.channel.findData(preferred)
        if index >= 0:
            self.channel.setCurrentIndex(index)
        self.channel.blockSignals(False)
        for source, (plot, _) in self.plots.items():
            plot.setVisible(source in visible)
        self.empty_view.setVisible(not visible)
        for widget in (self.channel, self.mode, self.label, self.snap, self.play):
            widget.setEnabled(bool(visible))
        if not visible:
            self.play.setChecked(False)
        self.channels_toggle.setEnabled(bool(self.channel_checks))
        self.show_all.setEnabled(len(visible) < len(self.channel_checks))
        self.hide_all.setEnabled(bool(visible))
        self.channel_summary.setText(f"{len(visible)} of {len(self.channel_checks)} channels shown")
        self.refresh()

    def update_channel_names(self):
        if not self.review:
            return
        self.channel.blockSignals(True)
        for index in range(self.channel.count()):
            source = self.channel.itemData(index)
            self.channel.setItemText(index, self.review.channel_name(source))
        self.channel.blockSignals(False)
        for source, (plot, _) in self.plots.items():
            self.channel_checks[source].setText(self.review.channel_name(source))
            self.channel_checks[source].setToolTip(f"{self.review.channel_name(source)} · recorded channel: {source}")
            plot.setLabel("left", self.review.channel_name(source), units=self.review.processed[source].get("units", ""))
        self.refresh()

    def select_channel(self, source):
        if source in self.channel_checks and not self.channel_checks[source].isChecked():
            self.channel_checks[source].setChecked(True)
        index = self.channel.findData(source)
        if index >= 0:
            self.channel.setCurrentIndex(index)

    def shift(self, fraction):
        if self.review:
            self.start.setValue(
                np.clip(
                    self.start.value() + fraction * self.window.value(),
                    self.first,
                    max(self.first, self.last - self.window.value()),
                )
            )

    def slider_changed(self, value):
        if self.review and not self.syncing:
            self.start.setValue(
                self.first
                + (max(self.first, self.last - self.window.value()) - self.first)
                * value
                / 1e6
            )

    def refresh(self, *_):
        if not self.review or not self.plots:
            return
        self.syncing = True
        begin = float(
            np.clip(
                self.start.value(),
                self.first,
                max(self.first, self.last - self.window.value()),
            )
        )
        end = min(self.last, begin + self.window.value())
        self.start.blockSignals(True)
        self.start.setValue(begin)
        self.start.blockSignals(False)
        self.slider.setValue(
            int(
                (begin - self.first)
                / max(1e-12, self.last - self.window.value() - self.first)
                * 1e6
            )
        )
        self.syncing = False
        count = 0
        all_labels = sorted(
            {
                str(label)
                for df in self.review.tables.values()
                for label in df.manual_label.unique()
            }
        )
        color_map = {
            label: COLORS[i % len(COLORS)] for i, label in enumerate(all_labels)
        }
        for name, (plot, trace) in self.plots.items():
            if not self.channel_checks[name].isChecked():
                continue
            data = self.review.processed[name]
            t = data["time"]
            values = data[
                (
                    "raw_signal"
                    if self.signal.currentText() == "Raw"
                    else "filtered_signal"
                )
            ]
            lo, hi = np.searchsorted(t, [begin, end])
            hi = min(len(t), hi + 1)
            # Pyqtgraph peak downsampling preserves narrow spikes in long windows.
            trace.setData(t[lo:hi], values[lo:hi], connect="finite")
            trace.setDownsampling(auto=True, method="peak")
            trace.setClipToView(True)
            for item in list(plot.items()):
                if isinstance(item, pg.ScatterPlotItem):
                    plot.removeItem(item)
            df = self.review.tables.get(name)
            if df is not None and len(df):
                visible = df[(df.time_seconds >= begin) & (df.time_seconds <= end)]
                for (label, accepted), group in visible.groupby(
                    ["manual_label", "accepted_as_spike"]
                ):
                    if not accepted and not self.rejected.isChecked():
                        continue
                    indices = group.sample_index.to_numpy(dtype=int)
                    indices = np.clip(indices, 0, len(t) - 1)
                    color = (
                        color_map.get(str(label), "#8899aa") if accepted else "#596578"
                    )
                    scatter = pg.ScatterPlotItem(
                        x=t[indices],
                        y=values[indices],
                        symbol="o" if accepted else "x",
                        size=10 if accepted else 7,
                        pen=pg.mkPen(color, width=2),
                        brush=pg.mkBrush(color) if accepted else None,
                        tip=lambda x, y, d: f"{d}",
                        data=[
                            f'{label} · {"accepted" if accepted else "rejected"}'
                            for _ in indices
                        ],
                    )
                    plot.addItem(scatter)
                    count += int(accepted) * len(group)
            plot.setXRange(begin, end, padding=0)
            finite = values[lo:hi]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                low, high = float(np.min(finite)), float(np.max(finite))
                center = (low + high) / 2
                half = max((high - low) / 2, 1e-9) / self.yzoom.value()
                plot.setYRange(center - half, center + half, padding=0.08)
            plot.setTitle(
                self.review.channel_name(name)
                + (" • editing" if name == self.channel.currentData() else ""),
                color="#43d9ad" if name == self.channel.currentData() else "#b5c3d9",
            )
        shown = set(self.visible_channels())
        visible_labels = sorted({
            str(label) for source, table in self.review.tables.items() if source in shown
            for label in table.manual_label.unique() if str(label).lower() != "noise"
        })
        legend = "  ·  ".join(visible_labels)
        self.status.setText(
            f"{begin:.3f}–{end:.3f} s  |  {count} accepted spikes visible  |  {legend}"
        )

    def clicked(self, name, event):
        if (
            not self.review
            or self.mode.currentIndex() == 0
            or event.button() != QtCore.Qt.MouseButton.LeftButton
        ):
            return
        if name != self.channel.currentData() or not self.channel_checks[name].isChecked():
            return
        plot = self.plots[name][0]
        if not plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = plot.getViewBox().mapSceneToView(event.scenePos())
        data = self.review.processed[name]
        t = data["time"]
        x = point.x()
        if x < t[0] or x > t[-1]:
            return
        action = ["browse", "add", "remove", "relabel"][self.mode.currentIndex()]
        index = int(
            np.argmin(
                abs(
                    t[
                        max(0, np.searchsorted(t, x) - 1) : min(
                            len(t), np.searchsorted(t, x) + 1
                        )
                    ]
                    - x
                )
            )
        ) + max(0, np.searchsorted(t, x) - 1)
        df = self.review.tables.get(name)
        # Existing marker hit tolerance follows screen pixels, independent of sampling rate.
        tolerance = self.window.value() * 12 / max(1, plot.width())
        if df is not None and len(df):
            eligible = df if action != "remove" else df[df.accepted_as_spike]
            if len(eligible):
                row = eligible.iloc[
                    int(np.argmin(abs(eligible.time_seconds.to_numpy() - x)))
                ]
                if abs(row.time_seconds - x) <= tolerance:
                    index = int(row.sample_index)
                elif action in ["remove", "relabel"]:
                    return
        elif action in ["remove", "relabel"]:
            return
        if (
            action == "add"
            and self.snap.value() > 0
            and (df is None or index not in set(df.sample_index))
        ):
            radius = self.snap.value() / 1000
            lo, hi = np.searchsorted(t, [x - radius, x + radius])
            hi = min(len(t), max(lo + 1, hi))
            values = data["filtered_signal"][lo:hi]
            if len(values) and np.any(np.isfinite(values)):
                polarity = self.review.metadata.get("settings", {}).get(
                    "polarity", "both"
                )
                score = (
                    values
                    if polarity == "positive"
                    else -values if polarity == "negative" else abs(values)
                )
                index = lo + int(np.nanargmax(score))
        label = self.label.currentText().strip()
        if action != "remove" and (
            not label or label.lower() in ["noise", "uncertain", "unlabeled"]
        ):
            self.message.emit(
                "Choose a neuron label before adding or relabeling a spike."
            )
            return
        try:
            if self.review.change(name, index, action, label):
                self.refresh()
                self.edited.emit()
        except ValueError as exc:
            self.message.emit(str(exc))

    def undo(self):
        if self.review and self.review.undo():
            self.refresh()
            self.edited.emit()

    def redo(self):
        if self.review and self.review.redo():
            self.refresh()
            self.edited.emit()

    def jump(self, direction):
        if not self.review:
            return
        df = self.review.tables.get(self.channel.currentData())
        if df is None:
            return
        times = np.sort(df.loc[df.accepted_as_spike, "time_seconds"].to_numpy())
        center = self.start.value() + self.window.value() / 2
        if not len(times):
            return
        i = np.searchsorted(
            times, center, side="right" if direction > 0 else "left"
        ) + (0 if direction > 0 else -1)
        self.start.setValue(
            np.clip(
                times[i % len(times)] - self.window.value() / 2,
                self.first,
                max(self.first, self.last - self.window.value()),
            )
        )

