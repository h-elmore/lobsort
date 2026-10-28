"""Lobsort desktop application."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets as W
import pyqtgraph as pg
from .engine import PipelineSession, default_settings, load_recording
from .curation import Review
from .viewer import ReviewWidget, spin

STYLE = """
QWidget { background:#101827; color:#e4ecf7; font-size:13px; }
QLabel#title { font-size:25px; font-weight:600; color:#f1f6ff; padding:8px 0; }
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QTableWidget { background:#1b293c; border:1px solid #34445c; border-radius:5px; padding:6px; }
QPushButton { background:#263950; border:1px solid #3b536c; border-radius:6px; padding:9px 14px; }
QPushButton:hover { background:#355470; } QPushButton:disabled { color:#68768b; }
QPushButton#primary { background:#207b68; font-weight:600; } QGroupBox { border:1px solid #34445c; border-radius:8px; margin-top:16px; padding:12px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; color:#7ee1c5; }
QTabBar::tab { padding:12px 22px; background:#192539; } QTabBar::tab:selected { background:#2b425a; }
QScrollArea { border:none; } QHeaderView::section { background:#263950; padding:6px; }
"""


class Worker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    log = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


class SettingsDialog(W.QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline settings")
        self.resize(780, 760)
        self.settings = copy.deepcopy(settings)
        self.widgets = {}
        layout = W.QVBoxLayout(self)
        layout.addWidget(
            W.QLabel(
                "Notebook parameters · Saved with each review and trained context model"
            )
        )
        tabs = W.QTabWidget()
        layout.addWidget(tabs)
        groups = {
            "Filter & detection": [
                "artifact_thresholds_mV",
                "default_artifact_threshold_mV",
                "artifact_padding_ms",
                "filter_low_hz",
                "filter_high_hz",
                "filter_order",
                "threshold_factor",
                "shadow_period_ms",
                "polarity",
                "adaptive_window_s",
                "adaptive_step_s",
                "noise_quantile",
                "minimum_valid_fraction",
                "noise_scale_bounds",
            ],
            "Training": [
                "label_targets",
                "context_before_seconds",
                "context_after_seconds",
                "number_of_time_bins",
                "number_of_amplitude_bins",
                "random_seed",
                "waveform_before_ms",
                "waveform_after_ms",
                "minimum_classification_probability",
                "validation_block_seconds",
                "number_of_waveform_pcs",
                "number_of_validation_time_blocks",
                "number_of_trees",
                "minimum_samples_per_leaf",
                "classifier_random_seed",
            ],
            "Rhythm & acceptance": [
                "context_waveform_before_ms",
                "context_waveform_after_ms",
                "reference_burst_threshold_factor",
                "maximum_reference_burst_gap_s",
                "maximum_reliable_reference_burst_duration_s",
                "unreliable_reference_padding_s",
                "local_context_seconds",
                "reference_envelope_smoothing_ms",
                "minimum_reference_burst_duration_s",
                "isi_filtered_label",
                "high_confidence_probability",
                "supported_probability",
                "sequence_support_window_s",
                "minimum_same_label_neighbors",
                "minimum_preceding_spikes",
                "isi_standard_deviations",
                "minimum_isi_tolerance_ms",
                "apply_within_burst_isi_filter",
                "include_moderate_probability_spikes",
            ],
        }
        for title, keys in groups.items():
            scroll = W.QScrollArea()
            scroll.setWidgetResizable(True)
            page = W.QWidget()
            form = W.QFormLayout(page)
            scroll.setWidget(page)
            tabs.addTab(scroll, title)
            for key in keys:
                value = settings[key]
                if isinstance(value, dict):
                    widget = W.QTableWidget(len(value) + 3, 2)
                    widget.setHorizontalHeaderLabels(
                        ["Channel / label", "Threshold (mV) / example count"]
                    )
                    widget.horizontalHeader().setSectionResizeMode(
                        W.QHeaderView.ResizeMode.Stretch
                    )
                    widget.setMinimumHeight(220)
                    for row, (k, v) in enumerate(value.items()):
                        widget.setItem(row, 0, W.QTableWidgetItem(k))
                        widget.setItem(row, 1, W.QTableWidgetItem(str(v)))
                elif isinstance(value, bool):
                    widget = W.QCheckBox()
                    widget.setChecked(value)
                elif isinstance(value, int):
                    widget = W.QSpinBox()
                    widget.setRange(0, 100000)
                    widget.setValue(value)
                elif isinstance(value, float):
                    widget = spin(0, 1e8, value, 0.1, 5)
                elif key == "polarity":
                    widget = W.QComboBox()
                    widget.addItems(["positive", "negative", "both"])
                    widget.setCurrentText(value)
                else:
                    widget = W.QLineEdit(
                        json.dumps(value) if isinstance(value, list) else str(value)
                    )
                self.widgets[key] = widget
                form.addRow(
                    {
                        "isi_filtered_label": "Neuron for burst checks / transferred waveform model",
                        "include_moderate_probability_spikes": "Include sequence-supported moderate confidence spikes",
                        "artifact_thresholds_mV": "Artifact cutoffs by channel (mV)",
                        "label_targets": "Training labels and example counts",
                    }.get(
                        key, key.replace("_", " ").replace("mV", "(mV)").capitalize()
                    ),
                    widget,
                )
        buttons = W.QDialogButtonBox(
            W.QDialogButtonBox.StandardButton.Save
            | W.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def validate(self):
        try:
            for key, w in self.widgets.items():
                old = self.settings[key]
                if isinstance(old, dict):
                    value = {}
                    for row in range(w.rowCount()):
                        k = w.item(row, 0)
                        v = w.item(row, 1)
                        if k and k.text().strip():
                            value[k.text().strip()] = (
                                int(v.text())
                                if key == "label_targets"
                                else float(v.text())
                            )
                    if not value or any(x <= 0 for x in value.values()):
                        raise ValueError(
                            "Thresholds and label example counts must be positive."
                        )
                elif isinstance(old, bool):
                    value = w.isChecked()
                elif isinstance(old, (int, float)):
                    value = w.value()
                elif key == "polarity":
                    value = w.currentText()
                else:
                    value = (
                        json.loads(w.text())
                        if isinstance(old, list)
                        else w.text().strip()
                    )
                self.settings[key] = value
            s = self.settings
            if s["filter_low_hz"] >= s["filter_high_hz"]:
                raise ValueError(
                    "The lower filter cutoff must be below the upper cutoff."
                )
            for k in [
                "high_confidence_probability",
                "supported_probability",
                "minimum_classification_probability",
                "minimum_valid_fraction",
                "noise_quantile",
            ]:
                if not 0 < s[k] <= 1:
                    raise ValueError(f"{k} must be greater than 0 and at most 1.")
            if s["supported_probability"] > s["high_confidence_probability"]:
                raise ValueError(
                    "Supported probability must not exceed high confidence probability."
                )
            if (
                len(s["noise_scale_bounds"]) != 2
                or not 0 < s["noise_scale_bounds"][0] <= s["noise_scale_bounds"][1]
            ):
                raise ValueError(
                    "Noise scale bounds must be two increasing positive numbers, e.g. [0.25, 4]."
                )
            if not s["label_targets"]:
                raise ValueError("Define at least two training labels.")
            self.accept()
        except Exception as exc:
            W.QMessageBox.warning(self, "Check settings", str(exc))


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lobsort · Spike sorting & review")
        self.resize(1480, 1000)
        self.settings = default_settings()
        self.channels = []
        self.markers = []
        self.session = None
        self.review = None
        self.project_path = None
        self.dirty = False
        self.worker = None
        self.label_window = None
        self.tabs = W.QTabWidget()
        self.setCentralWidget(self.tabs)
        setup = W.QWidget()
        layout = W.QVBoxLayout(setup)
        layout.setContentsMargins(28, 22, 28, 22)
        self.tabs.addTab(setup, "1  Recording and sorting")
        title = W.QLabel("Lobsort")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(
            W.QLabel(
                "From recording to curated spikes. Load a file, label examples or reuse a classifier, then review every detection."
            )
        )
        recording = W.QGroupBox("Recording")
        f = W.QFormLayout(recording)
        layout.addWidget(recording)
        row = W.QHBoxLayout()
        self.path = W.QLineEdit()
        self.path.setPlaceholderText("Choose an SMR, SMRX, or ABF recording")
        self.browse = W.QPushButton("Choose file…")
        self.browse.clicked.connect(self.choose_file)
        row.addWidget(self.path, 1)
        row.addWidget(self.browse)
        f.addRow("File", row)
        row = W.QHBoxLayout()
        self.segment = W.QSpinBox()
        self.segment.setRange(0, 9999)
        self.load = W.QPushButton("Load recording")
        self.load.setObjectName("primary")
        self.load.clicked.connect(self.load_file)
        row.addWidget(W.QLabel("Segment (starts at 0)"))
        row.addWidget(self.segment)
        row.addWidget(self.load)
        row.addStretch()
        f.addRow(row)
        self.record_info = W.QLabel(
            "No recording loaded. You can also open a saved review from the File menu."
        )
        self.record_info.setWordWrap(True)
        f.addRow(self.record_info)
        sorting = W.QGroupBox("Sorting")
        f = W.QFormLayout(sorting)
        layout.addWidget(sorting)
        self.target = W.QComboBox()
        self.reference = W.QComboBox()
        f.addRow("Channel to sort", self.target)
        f.addRow("Rhythm reference channel", self.reference)
        self.output = W.QLineEdit(str(Path.cwd() / "lobsort_results"))
        outrow = W.QHBoxLayout()
        outrow.addWidget(self.output)
        b = W.QPushButton("Choose folder…")
        b.clicked.connect(self.choose_output)
        outrow.addWidget(b)
        f.addRow("Project folder", outrow)
        row = W.QHBoxLayout()
        self.configure = W.QPushButton("Adjust pipeline settings…")
        self.configure.clicked.connect(self.configure_settings)
        row.addWidget(self.configure)
        self.prepare_button = W.QPushButton("1. Filter and detect candidates")
        self.prepare_button.setObjectName("primary")
        self.prepare_button.clicked.connect(self.prepare)
        row.addWidget(self.prepare_button)
        f.addRow(row)
        self.target.currentTextChanged.connect(self.invalidate)
        self.reference.currentTextChanged.connect(self.invalidate)
        training = W.QGroupBox("Training or reuse")
        f = W.QFormLayout(training)
        layout.addWidget(training)
        row = W.QHBoxLayout()
        self.labels_button = W.QPushButton("2. Label training examples")
        self.labels_button.clicked.connect(self.open_labeler)
        row.addWidget(self.labels_button)
        self.kind = W.QComboBox()
        self.kind.addItems(["Context-aware classifier", "Waveform classifier"])
        row.addWidget(self.kind)
        self.train_button = W.QPushButton("3. Train and classify")
        self.train_button.setObjectName("primary")
        self.train_button.clicked.connect(self.train)
        row.addWidget(self.train_button)
        f.addRow(row)
        row = W.QHBoxLayout()
        self.import_labels_button = W.QPushButton("Import training labels…")
        self.import_labels_button.clicked.connect(self.import_labels)
        row.addWidget(self.import_labels_button)
        self.transfer_button = W.QPushButton("Reuse a saved classifier…")
        self.transfer_button.clicked.connect(self.transfer)
        row.addWidget(self.transfer_button)
        row.addStretch()
        f.addRow(row)
        f.addRow(
            W.QLabel(
                "Training labels save as you work. Review remains available for manual sorting even without a classifier."
            )
        )
        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText(
            "Pipeline progress and validation results appear here."
        )
        layout.addWidget(self.log, 1)
        self.progress = W.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.viewer = ReviewWidget()
        self.tabs.addTab(self.viewer, "2  Review and refine")
        self.viewer.edited.connect(self.changed)
        self.viewer.message.connect(
            lambda text: W.QMessageBox.information(self, "Review", text)
        )
        menu = self.menuBar().addMenu("File")
        for text, fn, key in [
            ("Open review…", self.open_project, "Ctrl+O"),
            ("Save review", self.save_project, "Ctrl+S"),
            ("Save review as…", lambda: self.save_project(True), "Ctrl+Shift+S"),
            ("Export curated spikes…", self.export, None),
            ("Import sorted spike CSV…", self.import_spikes, None),
            ("Export recording markers…", self.export_markers, None),
            ("Save settings profile…", self.save_profile, None),
            ("Load settings profile…", self.load_profile, None),
            ("Open demo recording", self.demo, None),
        ]:
            action = menu.addAction(text)
            action.triggered.connect(fn)
            if key:
                action.setShortcut(key)
        helpmenu = self.menuBar().addMenu("Help")
        helpmenu.addAction(
            "How to review spikes",
            lambda: W.QMessageBox.information(
                self,
                "Review guide",
                "Use the time slider, scroll wheel, arrow keys, or Play to browse. Choose Add / restore spike, Deselect spike, or Relabel spike. Select the edit channel and neuron label, then click its trace. Add snaps to the strongest local peak within the selected radius. Set the radius to 0 to place at the nearest sample. Rejected candidates remain visible as gray crosses. Undo/redo work within the current session. Save a review to preserve all decisions. Export writes accepted spike CSVs and all candidate CSVs with edit history.",
            ),
        )
        self.update_buttons()

    def append_log(self, text):
        if text.strip():
            self.log.appendPlainText(text.strip())

    def invalidate(self, *_):
        self.session = None
        self.update_buttons()

    def update_buttons(self):
        busy = bool(self.worker and self.worker.isRunning())
        for w in [
            self.load,
            self.browse,
            self.configure,
            self.target,
            self.reference,
            self.output,
            self.path,
            self.segment,
        ]:
            w.setEnabled(not busy)
        self.prepare_button.setEnabled(bool(self.channels) and not busy)
        for w in [
            self.labels_button,
            self.train_button,
            self.transfer_button,
            self.import_labels_button,
        ]:
            w.setEnabled(self.session is not None and not busy)
        self.viewer.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0 if busy else 100)

    def run(self, fn, done):
        if self.worker and self.worker.isRunning():
            return
        self.worker = Worker(fn)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(done)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.update_buttons)
        self.worker.start()
        self.update_buttons()

    def failed(self, error):
        self.append_log(error)
        W.QMessageBox.warning(
            self,
            "Could not complete this step",
            error.strip().splitlines()[-1] + "\n\nDetails are in the progress panel.",
        )

    def choose_file(self):
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Open recording", "", "Recordings (*.smr *.smrx *.abf)"
        )
        if path:
            self.path.setText(path)

    def choose_output(self):
        path = W.QFileDialog.getExistingDirectory(
            self, "Project folder", self.output.text()
        )
        if path:
            self.output.setText(path)
            self.invalidate()

    def load_file(self):
        if not self.can_leave():
            return
        self.close_labeler()
        path = self.path.text().strip()
        segment = self.segment.value()
        if not Path(path).is_file():
            W.QMessageBox.warning(
                self, "Recording", "Choose an existing recording file."
            )
            return
        self.run(
            lambda: load_recording(path, segment),
            lambda result: self.loaded(result, path, segment),
        )

    def loaded(self, result, path, segment):
        self.channels, self.markers, nsegments = result
        self.recording_path = str(Path(path).resolve())
        self.recording_segment = segment
        self.session = None
        self.project_path = None
        for combo in [self.target, self.reference]:
            combo.clear()
            combo.addItems([c["name"] for c in self.channels])
        for combo, key in [
            (self.target, "channel_name"),
            (self.reference, "reference_channel_name"),
        ]:
            if self.settings[key] in [c["name"] for c in self.channels]:
                combo.setCurrentText(self.settings[key])
        self.record_info.setText(
            f"{Path(path).name} · {len(self.channels)} channels · {nsegments} segments · {len(self.markers)} event markers"
        )
        processed = {
            c["name"]: dict(
                time=c["time"],
                raw_signal=c["values"],
                filtered_signal=c["values"],
                units=c["units"],
            )
            for c in self.channels
        }
        self.review = Review(
            processed,
            metadata=dict(
                recording=self.recording_path,
                segment=segment,
                markers=self.markers,
                settings=self.settings,
            ),
        )
        self.viewer.set_review(self.review)
        self.dirty = False
        self.append_log(
            "Recording loaded. Preview traces in Review & refine, or filter and detect candidates here."
        )
        self.update_buttons()

    def configure_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.settings
            self.invalidate()

    def prepare(self):
        self.close_labeler()
        self.settings.update(
            channel_name=self.target.currentText(),
            reference_channel_name=self.reference.currentText(),
        )
        settings = copy.deepcopy(self.settings)
        channels = self.channels
        root = Path(self.output.text()).expanduser().resolve()
        target = settings["channel_name"]
        identity = json.dumps(
            dict(
                path=self.recording_path,
                segment=self.recording_segment,
                settings=settings,
            ),
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        folder = root / digest
        self.append_log(f"Project workspace: {folder}")

        def work():
            session = PipelineSession(
                channels, settings, folder, lambda text: self.worker.log.emit(text)
            )
            return session.prepare()

        self.run(work, self.prepared)

    def prepared(self, session):
        self.session = session
        processed = session.ns["processed_channels"]
        if self.review is None:
            self.review = Review(processed)
        else:
            self.review.processed.update(processed)
        target = session.settings["channel_name"]
        if target not in self.review.tables:
            indices = session.ns["candidate_indices"]
            self.review.set_table(
                target,
                pd.DataFrame(
                    dict(
                        sample_index=indices,
                        time_seconds=session.ns["candidate_times"],
                        manual_label="Unlabeled",
                        accepted_as_spike=False,
                    )
                ),
            )
        self.review.metadata.update(
            settings=session.serializable_settings(),
            recording=self.recording_path,
            segment=self.recording_segment,
            markers=self.markers,
        )
        self.project_path = (
            Path(self.output.text()).expanduser().resolve() / "review.lobsort"
        )
        self.viewer.set_review(self.review)
        self.changed()
        self.append_log(
            "Candidates ready. Label examples, reuse a model, or review candidates manually."
        )
        self.update_buttons()

    def open_labeler(self):
        if not self.session:
            return
        try:
            self.label_window = self.session.labeler()
            self.label_window.show()
        except Exception:
            self.failed(traceback.format_exc())

    def close_labeler(self):
        if self.label_window:
            self.label_window.close()
            self.label_window = None

    def import_labels(self):
        if not self.session:
            return
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Import training labels", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            from .engine import canonical_label

            df = pd.read_csv(path)
            if not {"sample_index", "manual_label"}.issubset(df.columns):
                raise ValueError(
                    "Training CSV needs sample_index and manual_label columns."
                )
            df["sample_index"] = pd.to_numeric(df.sample_index, errors="raise").astype(
                int
            )
            allowed = set(self.session.ns["candidate_indices"])
            df = (
                df.loc[df.sample_index.isin(allowed)]
                .drop_duplicates("sample_index", keep="last")
                .copy()
            )
            if df.empty:
                raise ValueError(
                    "No labels match candidates in this recording. Check the recording and detection settings."
                )
            df["manual_label"] = df.manual_label.map(canonical_label)
            unknown = set(df.manual_label) - set(self.settings["label_targets"])
            if unknown:
                raise ValueError(
                    f"Add these labels to the training settings first: {sorted(unknown)}"
                )
            df["time_seconds"] = self.session.ns["candidate_time"][
                df.sample_index.to_numpy()
            ]
            target = self.session.ns["training_labels_path"]
            if target.exists():
                target.replace(target.with_suffix(".previous.csv"))
            df.to_csv(target, index=False)
            self.session.ns["manual_labels_df"] = df
            self.append_log(
                f"Imported {len(df)} training labels. Previous labels retained as training_labels.previous.csv."
            )
        except Exception:
            self.failed(traceback.format_exc())

    def import_spikes(self):
        if not self.review:
            return
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Import sorted spikes", "", "CSV (*.csv)"
        )
        if not path:
            return
        channel, ok = W.QInputDialog.getItem(
            self,
            "Spike channel",
            "Channel for this CSV",
            list(self.review.processed),
            0,
            False,
        )
        if not ok:
            return
        try:
            df = pd.read_csv(path)
            t = self.review.processed[channel]["time"]
            if "time_seconds" not in df:
                raise ValueError("Sorted CSV must contain time_seconds.")
            times = pd.to_numeric(df.time_seconds, errors="raise").to_numpy()
            if (
                not np.all(np.isfinite(times))
                or np.any(times < t[0])
                or np.any(times > t[-1])
            ):
                raise ValueError(
                    "CSV spike times must fall within this channel recording."
                )
            right = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
            left = np.maximum(0, right - 1)
            indices = np.where(
                abs(t[left] - times) <= abs(t[right] - times), left, right
            )
            df["sample_index"] = indices
            df["time_seconds"] = t[indices]
            if "manual_label" not in df:
                df["manual_label"] = (
                    df["predicted_label"] if "predicted_label" in df else "Spike"
                )
            if "accepted_as_spike" not in df:
                df["accepted_as_spike"] = (
                    ~df.manual_label.astype(str)
                    .str.lower()
                    .isin(["noise", "unlabeled", "uncertain"])
                )
            if channel in self.review.tables:
                df = pd.concat(
                    [self.review.tables[channel], df], ignore_index=True
                ).drop_duplicates("sample_index", keep="last")
            self.review.set_table(channel, df)
            self.viewer.set_review(self.review)
            self.viewer.channel.setCurrentText(channel)
            self.changed()
            self.tabs.setCurrentWidget(self.viewer)
        except Exception:
            self.failed(traceback.format_exc())

    def train(self):
        if self.session:
            self.close_labeler()
            kind = "context" if self.kind.currentIndex() == 0 else "waveform"
            self.run(lambda: self.session.train(kind), self.classified)

    def transfer(self):
        if not self.session:
            return
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Reuse a trusted classifier", "", "Classifier (*.joblib)"
        )
        if path:
            self.close_labeler()
            self.run(lambda: self.session.transfer(path), self.classified)

    def classified(self, table):
        target = self.session.settings["channel_name"]
        # Retain a pre-classification snapshot before replacing a manually reviewed channel.
        if (
            target in self.review.tables
            and self.review.tables[target].edit_source.eq("manual").any()
        ):
            self.review.save(
                self.session.output_dir / "before_reclassification.lobsort"
            )
        self.review.metadata.setdefault("channel_settings", {})[
            target
        ] = self.session.serializable_settings()
        self.review.set_table(target, table)
        self.viewer.set_review(self.review)
        self.viewer.channel.setCurrentText(target)
        self.changed()
        self.tabs.setCurrentWidget(self.viewer)
        self.append_log(
            "Classification finished. Review and refine spikes, then export curated results."
        )

    def changed(self):
        self.dirty = True
        self.setWindowTitle("Lobsort · Unsaved review changes")

    def save_project(self, save_as=False):
        if not self.review:
            return
        if save_as or not self.project_path:
            path, _ = W.QFileDialog.getSaveFileName(
                self,
                "Save review",
                str(self.project_path or Path(self.output.text()) / "review.lobsort"),
                "Lobsort review (*.lobsort)",
            )
            if not path:
                return
            self.project_path = Path(
                path if path.endswith(".lobsort") else path + ".lobsort"
            )
        try:
            self.review.save(self.project_path)
            self.dirty = False
            self.setWindowTitle(f"Lobsort · {self.project_path.name}")
            self.statusBar().showMessage(f"Saved {self.project_path}", 7000)
        except Exception:
            self.failed(traceback.format_exc())

    def open_project(self):
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Open review", "", "Lobsort review (*.lobsort)"
        )
        if not path:
            return
        if not self.can_leave():
            return
        try:
            self.close_labeler()
            self.review = Review.load(path)
            self.project_path = Path(path)
            self.session = None
            self.channels = []
            self.markers = self.review.metadata.get("markers", [])
            self.viewer.set_review(self.review)
            self.dirty = False
            self.tabs.setCurrentWidget(self.viewer)
            self.update_buttons()
        except Exception:
            self.failed(traceback.format_exc())

    def export(self):
        if not self.review:
            return
        folder = W.QFileDialog.getExistingDirectory(
            self, "Export curated spike CSVs", self.output.text()
        )
        if folder:
            try:
                self.review.export(folder)
                self.review.save(Path(folder) / "review.lobsort")
                self.statusBar().showMessage(
                    f"Exported curated spikes and review to {folder}", 10000
                )
            except Exception:
                self.failed(traceback.format_exc())

    def export_markers(self):
        if not self.markers:
            return
        path, _ = W.QFileDialog.getSaveFileName(
            self, "Export markers", "markers.csv", "CSV (*.csv)"
        )
        if path:
            pd.DataFrame(self.markers).to_csv(path, index=False)

    def save_profile(self):
        path, _ = W.QFileDialog.getSaveFileName(
            self, "Save settings", "lobsort_settings.json", "JSON (*.json)"
        )
        if path:
            Path(path).write_text(json.dumps(self.settings, indent=2))

    def load_profile(self):
        path, _ = W.QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)"
        )
        if path:
            try:
                settings = default_settings()
                loaded = json.loads(Path(path).read_text())
                settings.update({k: v for k, v in loaded.items() if k in settings})
                dialog = SettingsDialog(settings, self)
                if dialog.exec():
                    self.settings = dialog.settings
                    self.invalidate()
            except Exception:
                self.failed(traceback.format_exc())

    def demo(self):
        if not self.can_leave():
            return
        self.close_labeler()
        from .demo import demo_channels

        self.loaded((demo_channels(), [], 1), "demo.synthetic", 0)
        self.settings.update(
            channel_name="Red",
            reference_channel_name="Green",
            threshold_factor=4.0,
            shadow_period_ms=2.0,
        )
        self.target.setCurrentText("Red")
        self.reference.setCurrentText("Green")
        self.append_log("Synthetic demo loaded. No biological data are included.")

    def can_leave(self):
        if self.worker and self.worker.isRunning():
            W.QMessageBox.information(
                self,
                "Pipeline running",
                "Wait for the current step to finish before closing.",
            )
            return False
        if self.dirty:
            answer = W.QMessageBox.question(
                self,
                "Save review?",
                "Save your review changes before leaving?",
                W.QMessageBox.StandardButton.Save
                | W.QMessageBox.StandardButton.Discard
                | W.QMessageBox.StandardButton.Cancel,
            )
            if answer == W.QMessageBox.StandardButton.Cancel:
                return False
            if answer == W.QMessageBox.StandardButton.Save:
                self.save_project()
                return not self.dirty
        return True

    def closeEvent(self, event):
        if self.can_leave():
            self.close_labeler()
            event.accept()
        else:
            event.ignore()


def main():
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    app = W.QApplication.instance() or W.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if "--demo" in sys.argv:
        window.demo()
    sys.exit(app.exec())
