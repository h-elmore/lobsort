# Notebook provenance and implementation

The supplied `spike_sort_v3.ipynb` is included under `reference/` with stored outputs and execution counts removed. It is reference material, not a runtime dependency. Notebook prose was treated as workflow documentation rather than instructions for the development agent.

The program runs ordinary Python stages in an isolated namespace per target channel. Configured parameters replace notebook assignments and inline detection arguments before stage execution; file paths are controlled by the project workspace. There is no runtime dependency on IPython or ipywidgets, and no global working-directory change.

| Notebook cells | Program |
| --- | --- |
| 7, 9, 11 | `engine.load_recording`: channels, segment selection, event markers; ABF additionally uses the already imported Axon reader |
| 14 | `viewer.ReviewWidget`: multichannel preview, navigation, playback, zoom |
| 16 | `stages/filter.py`: physical-unit artifact thresholds, padding, interpolation, zero-phase bandpass, masked output |
| 21, 22 | `stages/detection.py`, `candidates.py`, `labeler.py`: adaptive detection and original stratified manual labeling window |
| 24 | `waveform_train.py`: waveform/context features, random forest, validation, saved model and predictions |
| 27, 28 | `waveform_features.py`, `waveform_predict.py`: feature reconstruction and transfer; features are shared with cell 24 |
| 31, 33 | settings and `context_features.py`: waveform, temporal, reference envelope, bursts, reliability and cycle phase |
| 35, 37 | settings and `context_train.py`: PCA inside preprocessing, grouped time validation, context-aware random forest |
| 39, 41 | settings and `context_predict.py`: confidence, sequence support, optional within-burst ISI rejection |
| 44, 45 | `transfer_merge.py`, `transfer_rules.py`: transferred waveform predictions plus original transfer consistency rules |
| 42, 46 | `engine.result`, `curation.Review.export`: high-confidence policy plus curated/all-candidate exports |
| 49, 51 | sorted CSV import and `viewer.ReviewWidget`; adds manual spike editing |

## Intentional changes

- All former user-editable constants are interface settings; channel selection, recording path, output paths and segment are controls.
- The old `noise`/`Noise` inconsistency is normalized to `Noise`, including imported labels and transferred predictions. The default ISI/transfer neuron is `PY`, matching the initial Red/PY labeling setup, rather than the later notebook cell's independent `PD` example.
- Waveform models retain 2 ms before / 3 ms after; context waveform features retain their separate 5 ms before / 5 ms after defaults. These are distinct controls.
- Small cross-validation folds cap waveform PCA dimensionality to the available training rows. Missing finite raw samples are included in padded artifact masking instead of contaminating the full filtered signal.
- New context model packages can be saved and transferred. Their feature geometry/settings and sampling rate are recorded. Legacy notebook waveform models remain supported.
- Empty candidate sets produce a clear message before stratification tries to calculate minima/quantiles. Model training requires at least three labels per class in at least two classes.
- Notebook diagnostic figures are saved to the workspace instead of opening blocking Matplotlib windows. The progress log is also saved.
- Final human edits override automatic acceptance in exports. Original decisions, all candidates, audit events, and signal data remain available in saved review projects. No raw recordings or trained classifiers are shipped in the repository.
- The viewer uses peak-preserving display downsampling and full voltage extrema for initial Y scaling. Screen navigation never changes sample indices or detection arrays. OpenGL is disabled for portability.

## Known validation boundary

Automated tests use deterministic synthetic signals and check exact spike-index/noise-trace agreement with the supplied detector, model reuse consistency, and editing persistence. This demonstrates software behavior, not scientific classification accuracy. No available biological recording was included in the initial validation run.

## v0.2: learned labeled-unit timing and visible confidence choice

`timing.py` adds vectorized per-unit lag, phase, local activity and burst-timing
measurements from a frozen accepted-event snapshot of the open review. Selecting
this reference replaces the full raw-envelope context feature path with waveform
and spike-timing measurements. Only target manual labels passed to the training
fold enter the supervised fit; reference features do not contain target labels.
Known target/reference dependency cycles are rejected, while reference quality
and independence remain user responsibilities.

The same-label sequence-support rule is retained after prediction. Its counts
now use per-label sorted searches with the original closed-window boundary and
self-exclusion semantics. No target predictions are fed back into training.
A visible confidence selector controls high-only versus high-plus-supported-
moderate output. Existing context decisions can be switched without retraining,
with undo and retained manual overrides, noise/invalid rejection, and optional
ISI rejections. Numeric threshold edits still require classification again.

Model packages distinguish legacy envelope context (v1) and labeled-unit timing
(v2). V2 records exact expected feature columns and reference unit identity,
geometry, timing settings and sampling rate. Reuse requires a fresh reference
snapshot from the current recording. Snapshot provenance is kept in review
metadata and the workspace; the model contains a fingerprint/specification, not
training-recording timestamps to reuse on new data. Unit feature caches key on
reference spike content, waveform geometry, candidate set and timing settings.
