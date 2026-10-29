# Lobsort

A desktop application for the spike-sorting workflow in `spike_sort_v3.ipynb`, with a final manual curation step. No Jupyter session is required.

![Lobsort review screen, synthetic data](screenshot_review.png)

## Start the app

Requires Python 3.10 or newer. Python 3.11 or 3.12 is recommended for a fresh installation.

**macOS:** double-click **Launch Lobsort.command**. The first launch creates a private Python environment beside the app and installs its dependencies; it needs internet access. If Finder opens the file as text, run it from Terminal with `bash "Launch Lobsort.command"`.

**Windows:** double-click **Launch Lobsort.bat** with Python installed.

**Linux or manual installation:**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
lobsort
```

Use **File → Open demo recording** to try the interface with deterministic synthetic signals. You can also start with `lobsort --demo`.

## Sort a recording

1. Choose an `.smr` or `.abf` file and click **Load recording**. Choose a segment if the recording has multiple segments; numbering starts at zero. Neo must support the particular recording format/version. SMRX files require a separate CED reader and are not supported in this release.
2. Enter the **Animal ID** and use **Name channels…** to assign nerve names such as pdn and lvn. Choose the target channel, rhythm reference channel, and project folder. Preview all channels in **Review and refine**.
3. Open **Adjust pipeline settings**. Set each channel's artifact cutoff in **mV**, bandpass limits, detection threshold, polarity, and shadow period. Set neuron names and desired example counts under **Training**. Defaults are starting points from the notebook, not universal physiological settings.
4. Click **Filter and detect candidates**. Filtering runs for the target and reference channels. Masked artifacts appear as gaps in the filtered trace.
5. Click **Label training examples**. The original time/amplitude-stratified labeling window presents candidates alongside the reference trace. Use the labeled buttons or number keys, **S** to skip, and **U** to undo. Labels save immediately. You can import a compatible training-label CSV instead.
6. Choose **Context-aware classifier** or **Waveform classifier**, then **Train and classify**. Validation results appear in the progress panel; logs, models, candidate tables, and diagnostic images are saved in the channel/settings workspace. For later recordings, detect candidates and choose **Reuse a saved classifier** instead.
7. Review and correct the result, save the review, then export curated spikes. Repeat for other target channels; their tables accumulate in the review.

Label examples across the recording and include noise. At least three examples per class in two classes are needed to start training; the original larger per-class targets are preferable. Validation scores describe held-out manual labels, not guaranteed recording-wide accuracy.

**Transferred waveform models:** the notebook's transfer consistency rules accept the neuron named **Neuron for burst checks / transferred waveform model**. Set this to the neuron being sorted (default `PY`). Context classifiers can accept multiple neuron labels. Optional within-burst ISI rejection and sequence-supported moderate-confidence acceptance are exposed in settings. Moderate-confidence spikes are excluded from final automatic acceptance by default, matching the notebook's high-confidence export choice; rejected candidates remain available for manual restoration.

## Correct spikes by hand

- Browse with the time slider, mouse wheel, left/right keys, **Play**, or previous/next spike buttons. Use **Ctrl + wheel** or the window-length control to zoom in time. Switch between raw and filtered traces; use Y zoom for voltage scaling.
- Choose the **edit channel** and a **neuron label**.
- **Add / restore spike:** click a missed event or rejected marker. A new event snaps to the local peak according to detection polarity within the chosen snap radius. A radius of zero selects the nearest sample. Adding does not require an existing detected candidate.
- **Deselect spike:** click an accepted marker. It stays in the all-candidate table but is excluded from curated spike exports.
- **Relabel spike:** click a marker to assign the chosen neuron label.
- **Undo / Redo:** reverse or restore edits in the current session. The review keeps an audit history of edits. Undo stacks restart when reopening a project or replacing a channel table.

Zoom to individual events before editing densely packed spikes. Adds in filtered artifact gaps are blocked; revise filtering to recover events excluded by artifact settings. Automatic decisions are preserved in `automatic_label` and `automatic_accepted`; manual decisions use `manual_label`, `accepted_as_spike`, and `edit_source`.

## Save, resume, and export

**File → Save review** writes a `.lobsort` project containing raw/filtered signals, artifact masks, candidate decisions, metadata, channel settings, event markers, and edit history. The file is self-contained and can be reopened without the original recording. Saves replace the project atomically. Save periodically during final review; final edits are not automatically saved after each click. Training labels save separately as you label.

**File → Export curated spikes** asks for a parent folder. If an animal ID is provided, it creates an animal-named subfolder with one CSV per sorted nerve:

```text
1001_41/
  pdn.csv
  lvn.csv
  review_details/
    pdn_all_candidates.csv
    lvn_all_candidates.csv
    edit_history.json
    settings.json
    review.lobsort
```

The nerve CSVs contain accepted spikes, sorted by time, with `spike_number`, `time_seconds`, `sample_index`, `channel` (the nerve name), `source_channel` (the recorded identifier), `animal_id`, and `manual_label` (the neuron, such as PD). The details folder retains accepted and rejected candidates, available model predictions, manual provenance, settings, and a complete review snapshot. Only sorted/reviewed channels produce CSVs; merely displaying a rhythm reference does not sort it.

Without an animal ID, the previous flat export layout remains available: indexed `*_curated_spikes.csv`, `*_all_candidates.csv`, and the review files in the selected folder.

Exports honor manual overrides, including manually added spikes with no classifier confidence. Empty accepted tables still export column headers. Spike times are absolute recording seconds; indices are zero-based within the selected segment/channel.

You can import existing notebook spike CSVs through **File → Import sorted spike CSV**, select their channel, and curate them in the same viewer. CSVs need `time_seconds`; `manual_label` or `predicted_label` is used when present. Recording event markers can be exported separately. Settings profiles can also be saved and loaded.

Project folders contain a workspace identified by recording path, segment, and settings. Changing detection settings uses a different training-label workspace to avoid silently mixing incompatible candidates. Keep different recordings in distinct project folders. Before reclassifying a manually edited channel, the app saves `before_reclassification.lobsort` in its workspace.

## Models and resource use

Existing notebook waveform `.joblib` packages with `classifier`, `feature_columns`, and `feature_medians` are supported. New context models also store feature settings and sampling rate; reuse at a different sampling rate is rejected. Use matching preprocessing, channel meaning, voltage units, and recording conditions when transferring a model. Only open trusted joblib files, since this Python serialization format can execute code when loaded.

Recordings are loaded eagerly, as in the notebook. Long recordings and self-contained review projects may require substantial memory and storage. Filtering and training run in a background thread; project saving and exporting can take time for large recordings.

## Development and validation

```sh
python -m unittest discover -s tests -v
```

The tests exercise detection equivalence with the notebook, both training and model-transfer paths, empty detection, manual additions/removals/relabeling, undo/redo, artifact guards, review round-trips, curated exports, and GUI editing. CI runs the tests on Linux with an offscreen Qt interface.

Initial local validation used synthetic recordings. The notebook's hardcoded example SMR file was unavailable, so real-recording input and biological sorting accuracy remain unvalidated. Manual review supports correction but cannot guarantee 100% accuracy.

See [PORTING.md](PORTING.md) for notebook-stage mapping and intentional changes.


### Preview filtering before sorting

After loading an SMR/ABF recording (or the demo), open **Preview filtering on
signal…** or the **Preview filtering** tab. Choose any loaded channel, then browse
with **Start** and **Window**. The upper plot shows the median-centered raw signal;
red portions mark samples excluded by the artifact cutoff and padding. Drag either
gold horizontal line to change that channel's artifact cutoff, or use its slider
and numeric control. Thresholds are entered in mV and converted to the recording's
native voltage units.

Adjust the bandpass cutoffs, filter order, artifact padding, detection threshold,
spike spacing, polarity, and adaptive noise controls. The lower plot shows the
filtered signal, adaptive detection thresholds, and candidate peaks. Candidates
are previews, not classified spikes. Statistics report excluded time and candidate
counts across the complete channel. The effective high cutoff is capped at 45%
of the sampling rate, exactly as in sorting.

Edits are a draft until you click **Apply preview settings to sorting**. Artifact
cutoffs are channel-specific; bandpass, padding, and detection settings are shared.
You can preview the rhythm reference channel as well as the channel to sort.
**Reset draft** restores the most recently applied/current sorting settings.
Applying invalidates the previous candidate session; return to **Recording and
sorting** and run **Filter and detect candidates** to create candidates with the
new settings. Existing review decisions are retained by the existing review workflow.
Settings profiles and saved reviews continue to include the applied settings.

Preview calculations run in the background over the full channel, using the same
shipped filter and detector code as sorting. Display windows are sliced afterward
so moving around does not change adaptive noise estimates or introduce window-edge
filter artifacts. Large recordings can take time and additional memory; the latest
draft is displayed only when its calculation finishes. Invalid settings or excessive
artifact exclusion show an inline explanation and cannot be applied. No pipeline
output files are written by previewing.

### Single-unit channels: accept detections without a classifier

For a channel such as the pdn where all detected spikes belong to PD, choose
**Detection only (single unit)** in the sorting-method dropdown under
**Sorting method and training**. Set **Detected unit label** to **PD** (or another
neuron from your saved label library). After **Filter and detect candidates**,
click **Accept detections as spikes**. Every candidate produced by the current
filtering and detection settings becomes an accepted spike under that label.
No example labels, second class, waveform features, or classifier are required.
The detection thresholds and artifact exclusions still apply; no additional
classifier confidence or rhythm rules are applied.

The accepted spikes open in the normal review tab, where you can remove false
positives, add missed spikes, relabel, undo, save, and export. Detection-only
results record `acceptance_reason=detection_only` and retain detector amplitude
and noise measurements. They do not report classifier probabilities. Replacing
a channel with manual edits retains the existing pre-classification snapshot
behavior.

### Keep a label library and choose labels per channel

Keep all your neuron labels (for example **PD**, **LP**, **PY**, and **Noise**) and
example counts in **Adjust pipeline settings → Training → Saved labels and
example counts**. PD is included in the default library. In Setup, use the
**Training labels for this channel** checkboxes to choose the labels to collect
for the current channel. Only checked labels appear as buttons and keyboard
shortcuts in **Label training examples**, and both classifiers train using those
selected classes. Other previously saved examples remain on disk and become
available again if you re-enable their labels.

Classifiers still need examples from at least two selected classes. Single-unit
channels can use Detection only instead. Changing the active labels or sorting
method does not require rerunning detection, and these choices do not change
the candidate workspace path. Switching channels restores each channel's
checkboxes, sorting method, and detected-unit label. Settings profiles and saved
reviews preserve these choices alongside the complete label library.


### Animal IDs and nerve names

You can enter the animal ID before or after loading a recording. After loading,
click **Name channels…** beside **Load recording** to map recorded names such as
Red and Green to nerve names such as pdn and lvn. Names must be nonempty and unique
(ignoring capitalization). They appear in Setup, preprocessing, labeling, review,
and CSV import/export. A settings profile can retain the channel-name mapping;
older profiles without names preserve the currently assigned names. Animal IDs
are recording metadata, rather than filtering-profile parameters.

Renaming uses display aliases: the original signal keys remain stable. Existing
candidate sessions, voltage cutoffs, per-channel label choices, training examples,
review decisions, undo history, and raw recordings are retained. Naming a channel
pdn does not assign the neuron label PD automatically; choose the neuron label in
the usual sorting controls.

Animal IDs and nerve names are stored in saved reviews and restored on reopen.
**Name channels…** also works on a saved review. After reviewing spikes, export
them with **File → Export curated spikes…**. Select the parent destination;
Lobsort creates the animal-named folder itself and reports its location. Unsafe
filename characters are replaced, and filename collisions receive numeric
suffixes; the chosen names and original identifiers remain in the CSVs and the
export manifest inside `review_details/settings.json`.

Re-exporting the same recording updates its nerve CSVs and can add newly sorted
nerves. To preserve previous results, exporting another recording/segment under
the same animal ID, or changing an already exported nerve filename, requires a
different parent destination. Spike times stay in seconds on the original
recording timeline; naming does not shift times or combine separate recordings.
