import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from lobsort.engine import PipelineSession, default_settings
from lobsort.demo import demo_channels
from lobsort.curation import Review


class CurationTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(100) / 1000
        self.r = Review(
            {
                "A": dict(
                    time=t,
                    raw_signal=np.sin(t),
                    filtered_signal=np.sin(t),
                    artifact_mask=np.zeros(100, bool),
                    units="mV",
                )
            },
            {
                "A": pd.DataFrame(
                    dict(
                        sample_index=[10, 30],
                        time_seconds=[0.01, 0.03],
                        manual_label=["PY", "PY"],
                        accepted_as_spike=[True, True],
                    )
                )
            },
        )

    def test_manual_add_remove_undo_redo_export_reload(self):
        self.r.change("A", 20, "add", "LP")
        self.r.change("A", 10, "remove")
        self.assertEqual(sum(self.r.tables["A"].accepted_as_spike), 2)
        self.r.undo()
        self.assertEqual(sum(self.r.tables["A"].accepted_as_spike), 3)
        self.r.redo()
        self.r.change("A", 30, "relabel", "PD")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.lobsort"
            self.r.save(path)
            loaded = Review.load(path)
            pd.testing.assert_frame_equal(
                self.r.tables["A"], loaded.tables["A"], check_dtype=False
            )
            self.assertEqual(loaded.audit, self.r.audit)
            loaded.export(folder)
            exported = pd.read_csv(Path(folder) / "0_A_curated_spikes.csv")
            self.assertEqual(set(exported.sample_index), {20, 30})
            self.assertEqual(set(exported.manual_label), {"LP", "PD"})
            self.assertEqual(exported.spike_number.tolist(), [1, 2])

    def test_restore_has_no_duplicate(self):
        self.r.change("A", 10, "remove")
        self.r.change("A", 10, "add", "LP")
        self.assertEqual(len(self.r.tables["A"]), 2)
        self.assertEqual(self.r.tables["A"].iloc[0].manual_label, "LP")

    def test_artifact_and_bounds_rejected(self):
        self.r.processed["A"]["artifact_mask"][50] = True
        with self.assertRaises(ValueError):
            self.r.change("A", 50, "add")
        with self.assertRaises(ValueError):
            self.r.change("A", 100, "add")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.settings = default_settings()
        cls.settings.update(
            threshold_factor=4.0,
            shadow_period_ms=2.0,
            number_of_trees=20,
            number_of_waveform_pcs=2,
            isi_filtered_label="PY",
        )
        cls.channels = demo_channels(35)
        cls.p = PipelineSession(
            cls.channels, cls.settings, cls.root / "train", lambda x: None
        ).prepare()
        idx = cls.p.ns["candidate_indices"]
        times = cls.p.ns["candidate_times"]
        # Synthetic labels exercise both classes; they are not accuracy ground truth.
        pd.DataFrame(
            dict(
                sample_index=idx,
                manual_label=np.where(np.arange(len(idx)) % 2, "PY", "Noise"),
                time_seconds=times,
            )
        ).to_csv(cls.p.ns["training_labels_path"], index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_detector_matches_notebook(self):
        notebook = Path(__file__).parents[1] / "reference/spike_sort_v3.ipynb"
        n = json.loads(notebook.read_text())
        source = "".join(n["cells"][21]["source"])
        tree = ast.parse(source)
        functions = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
        original = dict(self.p.ns)
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "reference", "exec"),
            original,
        )
        kwargs = {
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
        }
        result = original["detect_spikes_processed_channel_adaptive"](
            self.p.ns["processed_channels"]["Red"], **kwargs
        )
        np.testing.assert_array_equal(
            result["spike_indices"], self.p.ns["candidate_indices"]
        )
        np.testing.assert_allclose(
            result["local_noise_trace"],
            self.p.ns["candidate_result"]["local_noise_trace"],
            equal_nan=True,
        )

    def test_train_transfer_both_classifier_paths(self):
        context = self.p.train("context")
        q = PipelineSession(
            self.channels, self.settings, self.root / "transfer", lambda x: None
        ).prepare()
        transferred = q.transfer(self.root / "train/context_classifier.joblib")
        np.testing.assert_array_equal(
            context.predicted_label, transferred.predicted_label
        )
        np.testing.assert_allclose(
            context.prediction_probability, transferred.prediction_probability
        )
        np.testing.assert_array_equal(
            context.accepted_as_spike, transferred.accepted_as_spike
        )
        waveform = self.p.train("waveform")
        transferred_waveform = q.transfer(
            self.root / "train/waveform_classifier.joblib"
        )
        a = self.p.ns["classified_candidates_df"]
        b = q.ns["classified_candidates_df"]
        np.testing.assert_array_equal(a.predicted_label, b.predicted_label)
        np.testing.assert_allclose(
            a.classification_probability, b.classification_probability
        )
        self.assertGreater(len(waveform), 0)
        self.assertGreater(len(transferred_waveform), 0)

    def test_empty_detection_has_useful_error(self):
        settings = {**self.settings, "threshold_factor": 1e8}
        with self.assertRaisesRegex(ValueError, "No candidates"):
            PipelineSession(
                self.channels, settings, self.root / "empty", lambda x: None
            ).prepare()


if __name__ == "__main__":
    unittest.main()
