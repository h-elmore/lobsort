"""Extracted from spike_sort_v3.ipynb, cell 22."""


class CandidateLabeler(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.order_position = 0
        self.current_candidate_position = None
        self.history = []
        self.setWindowTitle(f"{channel_name}: candidate labeling")
        self.resize(1450, 950)
        self.build_interface()
        self.install_shortcuts()
        self.advance_to_next_unlabeled()

    def build_interface(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.status_label)
        if hasattr(QtCore.Qt, "Orientation"):
            vertical_orientation = QtCore.Qt.Orientation.Vertical
        else:
            vertical_orientation = QtCore.Qt.Vertical
        self.plot_splitter = QtWidgets.QSplitter(vertical_orientation)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", f"{channel_name} filtered voltage")
        self.plot.setTitle(f"Candidate channel: {channel_name}")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot_splitter.addWidget(self.plot)
        if reference_channel_name is not None:
            self.reference_plot = pg.PlotWidget()
            self.reference_plot.showGrid(x=True, y=True, alpha=0.2)
            self.reference_plot.setLabel("bottom", "Time (seconds)")
            self.reference_plot.setLabel(
                "left", f"{reference_channel_name} {reference_signal_key}"
            )
            self.reference_plot.setTitle(f"Reference channel: {reference_channel_name}")
            self.reference_plot.setXLink(self.plot)
            self.reference_plot.getAxis("bottom").enableAutoSIPrefix(False)
            self.reference_plot.getAxis("left").enableAutoSIPrefix(False)
            self.plot.getAxis("bottom").setStyle(showValues=False)
            self.plot_splitter.addWidget(self.reference_plot)
            self.plot_splitter.setStretchFactor(0, 2)
            self.plot_splitter.setStretchFactor(1, 1)
        else:
            self.reference_plot = None
            self.plot.setLabel("bottom", "Time (seconds)")
        layout.addWidget(self.plot_splitter, stretch=1)
        button_layout = QtWidgets.QHBoxLayout()
        self.label_buttons = {}
        button_colors = ["#4daf4a", "#e41a1c", "#377eb8", "#984ea3"]
        for index, label in enumerate(label_targets):
            button = QtWidgets.QPushButton(f"{label} [{index + 1}]")
            button.setMinimumHeight(55)
            button.setStyleSheet(
                f"font-size: 16px; background-color: {button_colors[index % len(button_colors)]};"
            )
            button.clicked.connect(
                lambda checked=False, value=label: self.assign_label(value)
            )
            button_layout.addWidget(button)
            self.label_buttons[label] = button
        self.skip_button = QtWidgets.QPushButton("Skip [S]")
        self.skip_button.setMinimumHeight(55)
        self.skip_button.clicked.connect(self.skip_candidate)
        button_layout.addWidget(self.skip_button)
        self.undo_button = QtWidgets.QPushButton("Undo [U]")
        self.undo_button.setMinimumHeight(55)
        self.undo_button.clicked.connect(self.undo_last)
        button_layout.addWidget(self.undo_button)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setMinimumHeight(55)
        self.save_button.clicked.connect(self.save_labels)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def install_shortcuts(self):
        for index, label in enumerate(label_targets):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(str(index + 1)), self)
            shortcut.activated.connect(lambda value=label: self.assign_label(value))
            setattr(self, f"_shortcut_{index}", shortcut)
        self.skip_shortcut = QtGui.QShortcut(QtGui.QKeySequence("S"), self)
        self.skip_shortcut.activated.connect(self.skip_candidate)
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("U"), self)
        self.undo_shortcut.activated.connect(self.undo_last)

    def current_counts(self):
        counts = {}
        for label in label_targets:
            counts[label] = int(np.sum(manual_labels_df["manual_label"] == label))
        return counts

    def targets_complete(self):
        counts = self.current_counts()
        return all((counts[label] >= target for label, target in label_targets.items()))

    def update_buttons(self):
        counts = self.current_counts()
        for label, target in label_targets.items():
            completed = counts[label] >= target
            self.label_buttons[label].setEnabled(not completed)
            self.label_buttons[label].setText(f"{label}: {counts[label]}/{target}")

    def candidate_already_labeled(self, candidate_position):
        sample_index = int(candidate_indices[candidate_position])
        return bool(
            np.any(
                pd.to_numeric(manual_labels_df["sample_index"], errors="coerce")
                == sample_index
            )
        )

    def advance_to_next_unlabeled(self):
        if self.targets_complete():
            self.save_labels()
            self.status_label.setText(
                "Labeling targets complete. You can close this window and run the classifier block."
            )
            self.update_buttons()
            return
        while self.order_position < len(candidate_order):
            candidate_position = int(candidate_order[self.order_position])
            self.order_position += 1
            if not self.candidate_already_labeled(candidate_position):
                self.current_candidate_position = candidate_position
                self.show_current_candidate()
                return
        self.status_label.setText("No additional unlabeled candidates remain.")

    @staticmethod
    def set_robust_y_range(plot_widget, signal_values):
        """
        Set an independent Y-axis range using the visible
        signal values in one plot.
        """
        finite_values = np.asarray(signal_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) == 0:
            return
        low, high = np.percentile(finite_values, [0.5, 99.5])
        if high <= low:
            center = float(np.median(finite_values))
            half_range = max(abs(center) * 0.05, 1e-12)
            low = center - half_range
            high = center + half_range
        margin = max((high - low) * 0.15, 1e-12)
        plot_widget.setYRange(low - margin, high + margin, padding=0)

    def show_current_candidate(self):
        position = self.current_candidate_position
        spike_time = float(candidate_times[position])
        spike_index = int(candidate_indices[position])
        start_time = spike_time - context_before_seconds
        stop_time = spike_time + context_after_seconds
        left = np.searchsorted(candidate_time, start_time, side="left")
        right = np.searchsorted(candidate_time, stop_time, side="right")
        context_time = candidate_time[left:right]
        context_signal = candidate_signal[left:right]
        nearby_left = np.searchsorted(candidate_times, start_time, side="left")
        nearby_right = np.searchsorted(candidate_times, stop_time, side="right")
        nearby_times = candidate_times[nearby_left:nearby_right]
        nearby_indices = candidate_indices[nearby_left:nearby_right]
        nearby_values = candidate_signal[nearby_indices]
        self.plot.clear()
        trace = self.plot.plot(
            context_time, context_signal, pen=pg.mkPen(candidate_trace_color, width=1)
        )
        trace.setDownsampling(auto=True, method="peak")
        trace.setClipToView(True)
        self.plot.plot(
            nearby_times,
            nearby_values,
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=pg.mkPen("#777777"),
            symbolBrush=None,
        )
        current_value = candidate_signal[spike_index]
        self.plot.plot(
            [spike_time],
            [current_value],
            pen=None,
            symbol="o",
            symbolSize=14,
            symbolPen=pg.mkPen(candidate_marker_color, width=3),
            symbolBrush=None,
        )
        marker_line = pg.InfiniteLine(
            pos=spike_time, angle=90, pen=pg.mkPen(candidate_marker_color, width=2)
        )
        self.plot.addItem(marker_line)
        self.plot.setXRange(start_time, stop_time, padding=0)
        self.set_robust_y_range(self.plot, context_signal)
        if self.reference_plot is not None:
            reference_left = np.searchsorted(reference_time, start_time, side="left")
            reference_right = np.searchsorted(reference_time, stop_time, side="right")
            visible_reference_time = reference_time[reference_left:reference_right]
            visible_reference_signal = reference_signal[reference_left:reference_right]
            self.reference_plot.clear()
            if len(visible_reference_time):
                reference_trace = self.reference_plot.plot(
                    visible_reference_time,
                    visible_reference_signal,
                    pen=pg.mkPen(reference_trace_color, width=1),
                )
                reference_trace.setDownsampling(auto=True, method="peak")
                reference_trace.setClipToView(True)
                self.set_robust_y_range(self.reference_plot, visible_reference_signal)
            reference_marker_line = pg.InfiniteLine(
                pos=spike_time, angle=90, pen=pg.mkPen(candidate_marker_color, width=2)
            )
            self.reference_plot.addItem(reference_marker_line)
            self.reference_plot.setXRange(start_time, stop_time, padding=0)
        counts = self.current_counts()
        count_text = " | ".join(
            (
                f"{label}: {counts[label]}/{label_targets[label]}"
                for label in label_targets
            )
        )
        if reference_channel_name is None:
            reference_text = ""
        else:
            reference_text = f" | Reference: {reference_channel_name}"
        self.status_label.setText(
            f"Candidate at {spike_time:.6f} s | SNR={candidate_snr[position]:.2f}{reference_text} | {count_text}"
        )
        self.update_buttons()

    def assign_label(self, label):
        if self.current_candidate_position is None:
            return
        counts = self.current_counts()
        if counts[label] >= label_targets[label]:
            return
        position = self.current_candidate_position
        new_row = pd.DataFrame(
            [
                {
                    "channel": channel_name,
                    "sample_index": int(candidate_indices[position]),
                    "time_seconds": float(candidate_times[position]),
                    "manual_label": label,
                    "candidate_snr": float(candidate_snr[position]),
                }
            ]
        )
        global manual_labels_df
        manual_labels_df = pd.concat([manual_labels_df, new_row], ignore_index=True)
        self.history.append(int(candidate_indices[position]))
        self.save_labels()
        self.advance_to_next_unlabeled()

    def skip_candidate(self, *_):
        self.advance_to_next_unlabeled()

    def undo_last(self, *_):
        global manual_labels_df
        if not self.history:
            return
        sample_index = self.history.pop()
        matching_rows = manual_labels_df.index[
            pd.to_numeric(manual_labels_df["sample_index"], errors="coerce")
            == sample_index
        ]
        if len(matching_rows):
            manual_labels_df = manual_labels_df.drop(matching_rows[-1]).reset_index(
                drop=True
            )
        self.save_labels()
        self.show_current_candidate()

    def save_labels(self, *_):
        manual_labels_df.to_csv(training_labels_path, index=False)
