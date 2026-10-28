"""Explicit stages derived from the supplied notebook, with isolated per-channel state.

Stage sources are ordinary Python shipped with the package, never user-supplied code.
"""

from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import io
import contextlib
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter1d,
    binary_closing,
    binary_opening,
)
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
    GroupKFold,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

STAGES = Path(__file__).parent / "stages"
DEFAULTS = json.loads((Path(__file__).parent / "defaults.json").read_text())


def default_settings():
    return copy.deepcopy(DEFAULTS)


def canonical_label(label):
    return "Noise" if str(label).strip().lower() == "noise" else str(label).strip()


def load_recording(path, segment_index=0):
    """Read SMR/SMRX or ABF with Neo; return every segment's marker metadata."""
    from neo.io import Spike2IO, AxonIO

    path = Path(path)
    reader = (
        AxonIO(filename=str(path))
        if path.suffix.lower() == ".abf"
        else Spike2IO(filename=str(path))
    )
    block = reader.read_block(lazy=False)
    if not block.segments:
        raise ValueError("This recording has no segments.")
    if not 0 <= segment_index < len(block.segments):
        raise ValueError(f"Segment must be between 0 and {len(block.segments)-1}.")
    channels = []
    for gi, group in enumerate(block.segments[segment_index].analogsignals):
        data = np.asarray(group, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        times = np.asarray(group.times.rescale("s").magnitude, dtype=float)
        names = group.array_annotations.get("channel_names")
        for col in range(data.shape[1]):
            name = (
                str(names[col])
                if names is not None and col < len(names)
                else str(group.name or f"Signal {gi+1}")
                + (f" {col+1}" if data.shape[1] > 1 else "")
            )
            if any(c["name"] == name for c in channels):
                name = f"{name} [{gi}:{col}]"
            channels.append(
                dict(name=name, time=times, values=data[:, col], units=str(group.units))
            )
    if not channels:
        raise ValueError("No continuous analog channels found.")
    markers = []
    for si, segment in enumerate(block.segments):
        for event in segment.events:
            for time, label in zip(event.times.rescale("s").magnitude, event.labels):
                markers.append(
                    dict(
                        segment=si,
                        channel=str(event.name),
                        time_seconds=float(time),
                        label=(
                            label.decode("utf-8", errors="replace")
                            if isinstance(label, bytes)
                            else str(label)
                        ),
                    )
                )
    return channels, markers, len(block.segments)


class Overrides(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.values
        ):
            key = node.targets[0].id
            node.value = ast.Subscript(
                value=ast.Name(id="_settings", ctx=ast.Load()),
                slice=ast.Constant(key),
                ctx=ast.Load(),
            )
            return node
        return self.generic_visit(node)

    def visit_Call(self, node):
        # Detection kwargs were inline constants in the notebook.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "detect_spikes_processed_channel_adaptive"
        ):
            for kw in node.keywords:
                if kw.arg in self.values:
                    kw.value = ast.Subscript(
                        value=ast.Name(id="_settings", ctx=ast.Load()),
                        slice=ast.Constant(kw.arg),
                        ctx=ast.Load(),
                    )
        return self.generic_visit(node)


class PipelineSession:
    def __init__(self, channels, settings, output_dir, log=print):
        self.settings = copy.deepcopy(settings)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.ns = dict(globals())
        self.ns.update(copy.deepcopy(settings))
        self.ns.update(
            channels=channels,
            display=lambda x: print(x.to_string() if hasattr(x, "to_string") else x),
        )
        target = settings["channel_name"]
        self.ns.update(
            channel_name=target,
            reference_channel_name=settings["reference_channel_name"],
            classification_channel_name=target,
        )
        # Use paths inside a channel-specific project folder; never change process cwd.
        self.ns.update(
            training_labels_path=self.output_dir / "training_labels.csv",
            classification_output_path=self.output_dir / "waveform_candidates.csv",
            trained_model_path=self.output_dir / "waveform_classifier.joblib",
        )
        self.settings.update(
            training_labels_path=self.ns["training_labels_path"],
            classification_output_path=self.ns["classification_output_path"],
            trained_model_path=self.ns["trained_model_path"],
        )

    def run(self, stage, extra=None):
        values = {
            k: v
            for k, v in {**self.settings, **(extra or {})}.items()
            if k not in {"channel_name"}
        }
        self.ns.update(values)
        self.ns["_settings"] = values
        tree = ast.parse((STAGES / f"{stage}.py").read_text())
        tree = ast.fix_missing_locations(Overrides(values).visit(tree))
        capture = io.StringIO()
        self.log(f'Running {stage.replace("_", " ")}…')
        try:
            with contextlib.redirect_stdout(capture):
                exec(compile(tree, str(STAGES / f"{stage}.py"), "exec"), self.ns)
        finally:
            self.log(capture.getvalue())
            with (self.output_dir / "pipeline.log").open("a") as logfile:
                logfile.write(f"\n[{stage}]\n" + capture.getvalue())
            for num in plt.get_fignums():
                plt.figure(num).savefig(
                    self.output_dir / f"{stage}_diagnostic_{num}.png",
                    dpi=140,
                    bbox_inches="tight",
                )
            plt.close("all")

    def prepare(self):
        names = {c["name"] for c in self.ns["channels"]}
        target, ref = (
            self.settings["channel_name"],
            self.settings["reference_channel_name"],
        )
        if target not in names or ref not in names:
            raise ValueError("Select an available target and rhythm reference channel.")
        for key in [
            "threshold_factor",
            "shadow_period_ms",
            "filter_low_hz",
            "filter_high_hz",
            "adaptive_step_s",
            "adaptive_window_s",
        ]:
            if float(self.settings[key]) <= 0:
                raise ValueError(f"{key} must be positive.")
        self.run(
            "filter",
            {
                "channels_to_analyze": list(dict.fromkeys([target, ref])),
                "show_raw_preview": False,
                "show_final_plot": False,
            },
        )
        # The filter loop uses channel_name; restore the user's target.
        self.ns["channel_name"] = target
        self.run("detection")
        # An empty candidate set needs a helpful message before stratified sampling.
        result = self.ns["detect_spikes_processed_channel_adaptive"](
            self.ns["processed_channels"][target],
            **{
                k: self.settings[k]
                for k in [
                    "threshold_factor",
                    "shadow_period_ms",
                    "polarity",
                    "adaptive_window_s",
                    "adaptive_step_s",
                    "noise_quantile",
                    "minimum_valid_fraction",
                    "noise_scale_bounds",
                ]
            },
        )
        if len(result["spike_indices"]) == 0:
            raise ValueError(
                "No candidates detected. Lower the detection threshold or adjust polarity/artifact cutoffs."
            )
        self.run("candidates", {"candidate_result": result})
        labels_path = self.ns["training_labels_path"]
        columns = [
            "candidate_position",
            "sample_index",
            "time_seconds",
            "manual_label",
            "candidate_snr",
        ]
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df["manual_label"] = df["manual_label"].map(canonical_label)
        else:
            df = pd.DataFrame(columns=columns)
        self.ns["manual_labels_df"] = df
        df.to_csv(labels_path, index=False)
        self.ns["candidate_trace_color"] = "#aeb8c7"
        self.ns["reference_trace_color"] = "#76a5d5"
        self.ns["candidate_marker_color"] = "#ffd166"
        return self

    def labeler(self):
        from PyQt6 import QtCore, QtWidgets, QtGui
        import pyqtgraph as pg

        self.ns.update(QtCore=QtCore, QtWidgets=QtWidgets, QtGui=QtGui, pg=pg)
        self.run("labeler")
        return self.ns["CandidateLabeler"]()

    def features(self):
        self.run(
            "context_features",
            {
                "waveform_before_ms": self.settings["context_waveform_before_ms"],
                "waveform_after_ms": self.settings["context_waveform_after_ms"],
            },
        )

    def train(self, kind="context"):
        df = pd.read_csv(self.ns["training_labels_path"])
        df["manual_label"] = df["manual_label"].map(canonical_label)
        if df.manual_label.nunique() < 2 or df.manual_label.value_counts().min() < 3:
            raise ValueError(
                "Label at least three examples in each of at least two classes. More examples across the recording improve validation."
            )
        df.to_csv(self.ns["training_labels_path"], index=False)
        self.ns["manual_labels_df"] = df
        if kind == "waveform":
            self.run("waveform_train")
        else:
            self.features()
            self.run("context_train")
            package = dict(
                format="lobsort-context-v1",
                classifier=self.ns["context_classifier"],
                settings=self.serializable_settings(),
                sampling_rate=float(self.ns["candidate_result"]["sampling_rate"]),
            )
            joblib.dump(package, self.output_dir / "context_classifier.joblib")
            self.run("context_predict")
        return self.result(kind)

    def transfer(self, model_path):
        package = joblib.load(model_path)
        if package.get("format") == "lobsort-context-v1":
            model_settings = package["settings"]
            rate = float(self.ns["candidate_result"]["sampling_rate"])
            if not np.isclose(rate, package["sampling_rate"], rtol=1e-5):
                raise ValueError(
                    "This context model uses a different sampling rate. Train a model for this recording."
                )
            self.features_from_model(model_settings)
            self.ns["context_classifier"] = package["classifier"]
            self.run("context_predict")
            return self.result("context")
        self.run("waveform_features")
        self.run("waveform_predict", {"trained_model_path": Path(model_path)})
        for col in ["predicted_label", "manual_label"]:
            self.ns["classified_candidates_df"][col] = self.ns[
                "classified_candidates_df"
            ][col].map(canonical_label)
        self.features()
        self.run("transfer_merge")
        self.run("transfer_rules")
        return self.result("transfer")

    def features_from_model(self, saved):
        # Context model input depends on waveform geometry and reference envelope settings.
        keys = [
            "context_waveform_before_ms",
            "context_waveform_after_ms",
            "reference_signal_key",
            "reference_burst_threshold_factor",
            "maximum_reference_burst_gap_s",
            "maximum_reliable_reference_burst_duration_s",
            "unreliable_reference_padding_s",
            "local_context_seconds",
            "reference_envelope_smoothing_ms",
            "minimum_reference_burst_duration_s",
        ]
        overrides = {k: saved[k] for k in keys}
        overrides.update(
            waveform_before_ms=saved["context_waveform_before_ms"],
            waveform_after_ms=saved["context_waveform_after_ms"],
        )
        self.run("context_features", overrides)

    def result(self, kind):
        if kind == "waveform":
            df = self.ns["classified_candidates_df"].copy()
            df["accepted_as_spike"] = ~df.manual_label.isin(
                ["Noise", "Uncertain", "Unlabeled"]
            )
            df["acceptance_reason"] = np.where(
                df.accepted_as_spike,
                "high_classifier_probability",
                "insufficient_confidence",
            )
            df["prediction_probability"] = df["classification_probability"]
        else:
            df = self.ns["candidate_features_df"].copy()
            # Preserve the notebook's final high-confidence-only export policy.
            if not self.settings["include_moderate_probability_spikes"]:
                df["accepted_as_spike"] &= df.acceptance_reason.eq(
                    "high_classifier_probability"
                )
        df["channel"] = self.settings["channel_name"]
        df.to_csv(self.output_dir / "automatic_candidates.csv", index=False)
        return df

    def serializable_settings(self):
        return {
            k: (str(v) if isinstance(v, Path) else v) for k, v in self.settings.items()
        }
