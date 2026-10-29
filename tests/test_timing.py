"""Labeled-unit timing, frozen unsaved references, and confidence policy."""

import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets as W
from lobsort.curation import Review
from lobsort.engine import PipelineSession, default_settings
from lobsort.demo import demo_channels
from lobsort.gui import MainWindow
from lobsort.timing import (
    reference_features,
    reference_snapshot,
    same_label_neighbors,
    unit_context_features,
)


def review_with_reference():
    channels = demo_channels(5)
    processed = {
        c["name"]: dict(
            time=c["time"],
            raw_signal=c["values"],
            filtered_signal=c["values"],
            units=c["units"],
        )
        for c in channels
    }
    return Review(
        processed,
        {
            "Green": pd.DataFrame(
                dict(
                    sample_index=[2500, 3000, 7500, 8000],
                    time_seconds=[0.5, 0.6, 1.5, 1.6],
                    manual_label=["PD"] * 4,
                    accepted_as_spike=[True] * 4,
                )
            )
        },
    )


def synthetic_session(folder):
    settings = default_settings()
    settings.update(
        context_training_speed="custom",
        number_of_trees=50,
        number_of_validation_time_blocks=3,
        number_of_waveform_pcs=2,
        training_labels_by_channel={"Red": ["LP", "Noise"]},
    )
    fs = 2000
    clock = np.arange(160000) / fs
    ref = np.arange(0.5, 79.5, 0.5)
    rng = np.random.default_rng(12)
    positive = rng.integers(0, 2, len(ref)).astype(bool)
    spikes = ref + np.where(positive, 0.025, -0.025)
    indices = np.rint(spikes * fs).astype(int)
    signal = np.zeros(len(clock))
    signal[indices] = 3
    signal[indices - 1] = 1
    signal[indices + 1] = 1
    channels = [dict(name="Red", time=clock, values=signal, units="mV")]
    session = PipelineSession(channels, settings, folder, lambda _: None)
    session.ns["candidate_result"] = dict(
        time=clock,
        centered_filtered_signal=signal,
        spike_indices=indices,
        sampling_rate=fs,
    )
    pd.DataFrame(
        dict(
            sample_index=indices,
            time_seconds=spikes,
            manual_label=np.where(positive, "LP", "Noise"),
        )
    ).to_csv(session.ns["training_labels_path"], index=False)
    snapshot = dict(
        channel="Green", label="PD", times=ref.tolist(), recording_bounds=[0.0, 79.9995]
    )
    session.set_timing_reference(snapshot)
    return session, positive


class TimingMathTests(unittest.TestCase):
    def test_neighbor_counts_match_original_exactly(self):
        rng = np.random.default_rng(23)
        times = rng.integers(0, 100, 1000) / 10
        labels = rng.choice(["LP", "PY", "Noise"], len(times))
        window = 0.2
        expected = np.array(
            [
                sum((labels == label) & (times >= t - window) & (times <= t + window))
                - 1
                for t, label in zip(times, labels)
            ]
        )
        np.testing.assert_array_equal(
            same_label_neighbors(times, labels, window), expected
        )
        np.testing.assert_array_equal(
            same_label_neighbors([0, 0.2, 0.4], ["LP"] * 3, 0.2), [1, 2, 1]
        )

    def test_lags_counts_bounds_and_missing_reference(self):
        snap = dict(times=[1.0, 1.1, 2.0, 2.1], recording_bounds=[0.0, 3.0])
        df, onsets, offsets = reference_features(
            np.array([0.95, 1.0, 1.05, 1.15, 2.05, 4.0]), snap, 0.3, 0.15
        )
        self.assertAlmostEqual(df.unit_nearest_signed_lag_s.iloc[0], -0.05)
        self.assertAlmostEqual(df.unit_since_previous_s.iloc[2], 0.05)
        self.assertAlmostEqual(df.unit_until_next_s.iloc[2], 0.05)
        self.assertAlmostEqual(df.unit_interspike_phase.iloc[2], 0.5)
        self.assertEqual(df.unit_count_support.iloc[2], 2)
        self.assertEqual(df.unit_inside_burst.iloc[2], 1)
        self.assertEqual(df.unit_inside_burst.iloc[3], 0)
        self.assertEqual(df.unit_timing_available.iloc[-1], 0)
        self.assertTrue(np.isnan(df.unit_count_support.iloc[-1]))
        np.testing.assert_allclose(onsets, [1.0, 2.0])
        np.testing.assert_allclose(offsets, [1.1, 2.1])

    def test_unsaved_reference_snapshot_filters_labels_and_rejections(self):
        review = review_with_reference()
        snap = reference_snapshot(review, "Red", "Green", "PD")
        review.change("Green", 4000, "add", "LP")
        review.change("Green", 3000, "remove")
        updated = reference_snapshot(review, "Red", "Green", "PD")
        self.assertEqual(updated["times"], [0.5, 1.5, 1.6])
        self.assertNotEqual(updated["fingerprint"], snap["fingerprint"])
        self.assertEqual(snap["times"], [0.5, 0.6, 1.5, 1.6])
        with self.assertRaisesRegex(ValueError, "different channel"):
            reference_snapshot(review, "Green", "Green", "PD")
        review.metadata["timing_references"] = {
            "Green": {"channel": "Other"},
            "Other": {"channel": "Red"},
        }
        with self.assertRaisesRegex(ValueError, "circular"):
            reference_snapshot(review, "Red", "Green", "PD")

    def test_features_are_measured_without_target_labels_and_cache_invalidates(self):
        with tempfile.TemporaryDirectory() as folder:
            session, _ = synthetic_session(folder)
            session.features()
            a = session.ns["candidate_features_df"].copy()
            cache = session._unit_feature_cache
            session.ns["candidate_features_df"][
                "predicted_label"
            ] = "SHOULD_NOT_BE_CACHED"
            session.features()
            self.assertIs(session._unit_feature_cache, cache)
            pd.testing.assert_frame_equal(a, session.ns["candidate_features_df"])
            self.assertFalse(
                any(
                    "envelope" in x or "predicted" in x or "manual_label" in x
                    for x in session.ns["context_feature_columns"]
                )
            )
            labels = pd.read_csv(session.ns["training_labels_path"])
            labels.manual_label = "Noise"
            labels.to_csv(session.ns["training_labels_path"], index=False)
            session.features()
            pd.testing.assert_frame_equal(a, session.ns["candidate_features_df"])
            snap = copy.deepcopy(session.timing_reference)
            snap["times"] = np.array(snap["times"]) + 0.01
            session.set_timing_reference(snap)
            session.features()
            self.assertIsNot(session._unit_feature_cache, cache)
            self.assertFalse(
                np.allclose(
                    a.unit_nearest_signed_lag_s,
                    session.ns["candidate_features_df"].unit_nearest_signed_lag_s,
                )
            )

    def test_waveform_geometry_matches_notebook_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            session, _ = synthetic_session(folder)
            features = unit_context_features(
                session.ns["candidate_result"],
                session.timing_reference,
                session.settings,
            )
            df = features["candidate_features_df"]
            cols = features["waveform_feature_columns"]
            result = session.ns["candidate_result"]
            before = 10
            after = 10
            i = int(result["spike_indices"][0])
            wave = result["centered_filtered_signal"][i - before : i + after + 1].copy()
            wave -= np.median(wave[: before // 2])
            wave /= np.linalg.norm(wave)
            np.testing.assert_allclose(df.loc[0, cols].to_numpy(float), wave)


class TimingModelTests(unittest.TestCase):
    def test_learns_labeled_timing_and_model_reuse_requires_current_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            session, positive = synthetic_session(Path(folder) / "train")
            result = session.train("context")
            self.assertGreater(session.ns["validation_balanced_accuracy"], 0.95)
            self.assertGreater(
                np.mean((result.predicted_label == "LP").to_numpy() == positive), 0.98
            )
            model = Path(folder) / "train/context_classifier.joblib"
            package = joblib.load(model)
            self.assertEqual(package["format"], "lobsort-context-v2")
            self.assertNotIn("times", package["timing_reference_spec"])
            self.assertTrue(
                (Path(folder) / "train/timing_reference_snapshot.json").exists()
            )
            q, _ = synthetic_session(Path(folder) / "reuse")
            q.set_timing_reference(None)
            with self.assertRaisesRegex(ValueError, "labeled timing reference"):
                q.transfer(model)
            wrong = copy.deepcopy(session.timing_reference)
            wrong["label"] = "PY"
            q.set_timing_reference(wrong)
            with self.assertRaisesRegex(ValueError, "same reference unit label"):
                q.transfer(model)
            q.set_timing_reference(session.timing_reference)
            reused = q.transfer(model)
            np.testing.assert_array_equal(
                result.predicted_label, reused.predicted_label
            )
            np.testing.assert_allclose(
                result.prediction_probability, reused.prediction_probability
            )
            changed = copy.deepcopy(session.timing_reference)
            changed["times"] = (np.array(changed["times"]) + 0.05).tolist()
            q.set_timing_reference(changed)
            new = q.transfer(model)
            self.assertGreater(
                np.mean(new.predicted_label != result.predicted_label), 0.2
            )

    def test_fast_preset_and_custom_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            p, _ = synthetic_session(folder)
            p.settings["context_training_speed"] = "fast"
            self.assertEqual(
                p.context_training_options(),
                dict(number_of_trees=200, number_of_validation_time_blocks=3),
            )
            p.settings["context_training_speed"] = "custom"
            self.assertEqual(p.context_training_options()["number_of_trees"], 50)


class ConfidencePolicyTests(unittest.TestCase):
    def test_toggle_preserves_manual_edits_invalid_spikes_and_isi_rejections(self):
        r = review_with_reference()
        r.set_table(
            "Red",
            pd.DataFrame(
                dict(
                    sample_index=np.arange(7),
                    time_seconds=np.arange(7) / 10,
                    manual_label=["LP"] * 7,
                    accepted_as_spike=[True, False, False, False, False, False, True],
                    valid_waveform=[True, True, False, True, True, True, True],
                    same_label_neighbor_count=[0, 2, 2, 2, 2, 2, 0],
                    acceptance_reason=[
                        "high_classifier_probability",
                        "moderate_probability_with_sequence_support",
                        "moderate_probability_with_sequence_support",
                        "rejected_by_within_burst_maximum_isi",
                        "insufficient_confidence",
                        "high_classifier_probability",
                        "moderate_probability_with_sequence_support",
                    ],
                    edit_source=["automatic"] * 5 + ["manual", "manual"],
                )
            ),
        )
        r.apply_confidence_policy("Red", True)
        self.assertEqual(
            r.tables["Red"].accepted_as_spike.tolist(),
            [True, True, False, False, False, False, True],
        )
        r.apply_confidence_policy("Red", False)
        self.assertEqual(
            r.tables["Red"].accepted_as_spike.tolist(),
            [True, False, False, False, False, False, True],
        )
        r.undo()
        self.assertTrue(r.tables["Red"].accepted_as_spike.iloc[1])
        r.redo()
        self.assertFalse(r.tables["Red"].accepted_as_spike.iloc[1])


class TimingGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])

    def test_saved_unfinished_review_resumes_training_with_reference_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            review = review_with_reference()
            settings = default_settings()
            settings.update(threshold_factor=4.0, shadow_period_ms=2.0)
            settings["unit_timing_references_by_channel"] = {
                "Red": {"channel": "Green", "label": "PD"}
            }
            settings["include_moderate_probability_spikes"] = True
            review.metadata.update(
                settings=settings, recording="/missing/original.smr", segment=0
            )
            path = Path(folder) / "unfinished.lobsort"
            review.save(path)
            window = MainWindow()
            try:
                with patch.object(
                    W.QFileDialog, "getOpenFileName", return_value=(str(path), "")
                ):
                    window.open_project()
                self.assertTrue(window.prepare_button.isEnabled())
                self.assertEqual(len(window.channels), 2)
                self.assertTrue(window.timing_group.isChecked())
                self.assertEqual(window.timing_unit.currentText(), "PD")
                self.assertEqual(window.acceptance_choice.currentIndex(), 1)
                window.prepare()
                deadline = time.monotonic() + 20
                while (
                    window.worker
                    and window.worker.isRunning()
                    and time.monotonic() < deadline
                ):
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()
                self.assertIsNotNone(window.session)
                self.assertEqual(len(window.review.tables["Green"]), 4)
                window.attach_timing_reference()
                np.testing.assert_allclose(
                    window.session.timing_reference["times"],
                    [0.5, 0.6, 1.5, 1.6],
                    atol=1e-15,
                )
            finally:
                if window.worker:
                    window.worker.wait(20000)
                window.dirty = False
                window.close()

    def test_reference_choice_unsaved_edits_profile_and_session_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow()
            window.demo()
            window.review = review_with_reference()
            window.viewer.set_review(window.review)
            try:
                window.refresh_timing_sources()
                self.assertEqual(window.timing_channel.currentData(), "Green")
                window.timing_unit.setCurrentText("PD")
                window.timing_group.setChecked(True)
                window.settings["threshold_factor"] = 4.0
                window.settings["shadow_period_ms"] = 2.0
                window.session = PipelineSession(
                    demo_channels(5), window.settings, folder, lambda _: None
                ).prepare()
                window.attach_timing_reference()
                self.assertEqual(len(window.session.timing_reference["times"]), 4)
                window.review.change("Green", 3000, "remove")
                window.attach_timing_reference()
                self.assertEqual(len(window.session.timing_reference["times"]), 3)
                window.acceptance_choice.setCurrentIndex(1)
                self.assertTrue(
                    window.session.settings["include_moderate_probability_spikes"]
                )
                window.context_speed.setCurrentIndex(2)
                self.assertEqual(
                    window.session.settings["context_training_speed"], "custom"
                )
                # Confidence switching updates only automatic context decisions, without a worker/model fit.
                window.review.set_table(
                    "Red",
                    pd.DataFrame(
                        dict(
                            sample_index=[1000, 2000],
                            time_seconds=[0.2, 0.4],
                            manual_label=["LP", "LP"],
                            accepted_as_spike=[False, False],
                            valid_waveform=[True, True],
                            same_label_neighbor_count=[2, 0],
                            acceptance_reason=[
                                "moderate_probability_with_sequence_support",
                                "high_classifier_probability",
                            ],
                            edit_source=["automatic", "manual"],
                        )
                    ),
                )
                window.viewer.set_review(window.review)
                window.update_buttons()
                self.assertTrue(window.apply_acceptance_button.isEnabled())
                window.apply_acceptance_button.click()
                self.assertEqual(
                    window.review.tables["Red"].accepted_as_spike.tolist(),
                    [True, False],
                )
                self.assertIsNone(window.worker)
                window.acceptance_choice.setCurrentIndex(0)
                window.apply_acceptance_button.click()
                self.assertEqual(
                    window.review.tables["Red"].accepted_as_spike.tolist(),
                    [False, False],
                )
                window.acceptance_choice.setCurrentIndex(1)
                # Selectors and speed changes do not invalidate detector candidates.
                self.assertIsNotNone(window.session)
                profile = Path(folder) / "settings.json"
                with patch.object(
                    W.QFileDialog, "getSaveFileName", return_value=(str(profile), "")
                ):
                    window.save_profile()
                import json

                saved = json.loads(profile.read_text())
                self.assertEqual(
                    saved["unit_timing_references_by_channel"]["Red"],
                    dict(channel="Green", label="PD"),
                )
                self.assertTrue(saved["include_moderate_probability_spikes"])
                window.target.setCurrentText("Green")
                self.assertFalse(window.timing_group.isChecked())
                window.target.setCurrentText("Red")
                self.assertTrue(window.timing_group.isChecked())
                self.assertEqual(window.timing_unit.currentText(), "PD")
            finally:
                window.dirty = False
                window.close()


if __name__ == "__main__":
    unittest.main()
