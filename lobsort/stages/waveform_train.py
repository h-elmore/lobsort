"""Extracted from spike_sort_v3.ipynb, cell 24."""

waveform_before_ms = 2.0
waveform_after_ms = 3.0
minimum_classification_probability = 0.85
validation_block_seconds = 30.0
classification_output_path = Path(f"{channel_name}_supervised_candidates.csv")
manual_labels_df = pd.read_csv(training_labels_path)
manual_labels_df["sample_index"] = pd.to_numeric(
    manual_labels_df["sample_index"], errors="coerce"
)
manual_labels_df = manual_labels_df.dropna(subset=["sample_index"]).copy()
manual_labels_df["sample_index"] = manual_labels_df["sample_index"].astype(int)
print(manual_labels_df["manual_label"].value_counts())
sampling_rate = float(candidate_result["sampling_rate"])
samples_before = max(1, int(sampling_rate * waveform_before_ms / 1000))
samples_after = max(1, int(sampling_rate * waveform_after_ms / 1000))
waveform_length = samples_before + samples_after + 1
candidate_times_sorted = np.asarray(candidate_times, dtype=float)
preceding_isi = np.diff(candidate_times_sorted, prepend=np.nan)
following_isi = np.diff(candidate_times_sorted, append=np.nan)
preceding_isi = np.nan_to_num(preceding_isi, nan=10.0, posinf=10.0, neginf=10.0)
following_isi = np.nan_to_num(following_isi, nan=10.0, posinf=10.0, neginf=10.0)
preceding_isi = np.clip(preceding_isi, 0, 10)
following_isi = np.clip(following_isi, 0, 10)


def candidate_count_in_window(center_times, half_window):
    left = np.searchsorted(center_times, center_times - half_window, side="left")
    right = np.searchsorted(center_times, center_times + half_window, side="right")
    return right - left - 1


count_within_025 = candidate_count_in_window(candidate_times_sorted, 0.25)
count_within_100 = candidate_count_in_window(candidate_times_sorted, 1.0)
finite_signal = np.where(np.isfinite(candidate_signal), candidate_signal, 0.0)
valid_signal = np.isfinite(candidate_signal).astype(float)
absolute_cumsum = np.concatenate([[0.0], np.cumsum(np.abs(finite_signal))])
valid_cumsum = np.concatenate([[0.0], np.cumsum(valid_signal)])


def local_absolute_mean(sample_index, half_window_seconds):
    half_samples = int(sampling_rate * half_window_seconds)
    left = max(0, sample_index - half_samples)
    right = min(len(candidate_signal), sample_index + half_samples + 1)
    total = absolute_cumsum[right] - absolute_cumsum[left]
    count = valid_cumsum[right] - valid_cumsum[left]
    if count <= 0:
        return np.nan
    return total / count


feature_rows = []
normalized_waveforms = []
for candidate_position, sample_index in enumerate(candidate_indices):
    waveform_start = sample_index - samples_before
    waveform_stop = sample_index + samples_after + 1
    if waveform_start < 0 or waveform_stop > len(candidate_signal):
        continue
    waveform = candidate_signal[waveform_start:waveform_stop].copy()
    if len(waveform) != waveform_length or not np.all(np.isfinite(waveform)):
        continue
    baseline_count = max(1, samples_before // 2)
    baseline = np.median(waveform[:baseline_count])
    waveform = waveform - baseline
    absolute_peak = np.max(np.abs(waveform))
    if not np.isfinite(absolute_peak) or absolute_peak <= 0:
        continue
    normalized_waveform = waveform / absolute_peak
    peak_to_peak = np.ptp(waveform)
    waveform_energy = np.sum(waveform**2)
    positive_peak = np.max(waveform)
    negative_peak = np.min(waveform)
    local_noise = local_noise_trace[sample_index]
    snr = absolute_peak / local_noise if local_noise > 0 else np.nan
    feature_rows.append(
        {
            "candidate_position": candidate_position,
            "sample_index": int(sample_index),
            "time_seconds": float(candidate_times[candidate_position]),
            "absolute_peak": float(absolute_peak),
            "positive_peak": float(positive_peak),
            "negative_peak": float(negative_peak),
            "peak_to_peak": float(peak_to_peak),
            "waveform_energy": float(waveform_energy),
            "local_noise_sigma": float(local_noise),
            "snr": float(snr),
            "preceding_isi": float(preceding_isi[candidate_position]),
            "following_isi": float(following_isi[candidate_position]),
            "candidates_within_025s": int(count_within_025[candidate_position]),
            "candidates_within_1s": int(count_within_100[candidate_position]),
            "local_activity_025s": float(local_absolute_mean(int(sample_index), 0.25)),
            "local_activity_1s": float(local_absolute_mean(int(sample_index), 1.0)),
        }
    )
    normalized_waveforms.append(normalized_waveform)
feature_df = pd.DataFrame(feature_rows)
normalized_waveforms = np.asarray(normalized_waveforms, dtype=float)
for waveform_column in range(normalized_waveforms.shape[1]):
    feature_df[f"waveform_{waveform_column:03d}"] = normalized_waveforms[
        :, waveform_column
    ]
print(f"Created features for {len(feature_df):,} candidates.")
training_df = feature_df.merge(
    manual_labels_df[["sample_index", "manual_label"]], on="sample_index", how="inner"
)
training_df = training_df[training_df["manual_label"].isin(label_targets.keys())].copy()
print("\nTraining examples:")
print(training_df["manual_label"].value_counts())
non_feature_columns = {
    "candidate_position",
    "sample_index",
    "time_seconds",
    "manual_label",
}
feature_columns = [
    column for column in feature_df.columns if column not in non_feature_columns
]
X = training_df[feature_columns].to_numpy(dtype=float)
y = training_df["manual_label"].to_numpy()
feature_medians = np.nanmedian(X, axis=0)
invalid_rows, invalid_columns = np.where(~np.isfinite(X))
X[invalid_rows, invalid_columns] = feature_medians[invalid_columns]
validation_groups = np.floor(
    (training_df["time_seconds"].to_numpy() - candidate_times.min())
    / validation_block_seconds
).astype(int)
split_found = False
group_splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=12345)
for train_indices, test_indices in group_splitter.split(X, y, groups=validation_groups):
    if len(np.unique(y[train_indices])) == len(np.unique(y)) and len(
        np.unique(y[test_indices])
    ) == len(np.unique(y)):
        split_found = True
        break
if not split_found:
    print(
        "Could not form a time-blocked split containing all labels. Falling back to a stratified random split."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=12345
    )
else:
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
validation_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=12345,
)
validation_model.fit(X_train, y_train)
test_predictions = validation_model.predict(X_test)
print("\nHeld-out validation:")
print(classification_report(y_test, test_predictions, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, cmap="Blues")
plt.title("Held-out candidate classification")
plt.show()
classifier = RandomForestClassifier(
    n_estimators=750,
    class_weight="balanced",
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=12345,
)
classifier.fit(X, y)
trained_model_path = Path(f"{exp}_{signal}_classifier.joblib")
model_package = {
    "classifier": classifier,
    "feature_columns": feature_columns,
    "feature_medians": feature_medians,
}
joblib.dump(model_package, trained_model_path)
print("Saved trained classifier:", trained_model_path.resolve())
X_all = feature_df[feature_columns].to_numpy(dtype=float)
invalid_rows, invalid_columns = np.where(~np.isfinite(X_all))
X_all[invalid_rows, invalid_columns] = feature_medians[invalid_columns]
predicted_labels = classifier.predict(X_all)
predicted_probabilities = classifier.predict_proba(X_all)
maximum_probabilities = np.max(predicted_probabilities, axis=1)
final_labels = np.where(
    maximum_probabilities >= minimum_classification_probability,
    predicted_labels,
    "Uncertain",
)
classified_candidates_df = feature_df[
    ["candidate_position", "sample_index", "time_seconds", "absolute_peak", "snr"]
].copy()
classified_candidates_df["predicted_label"] = predicted_labels
classified_candidates_df["classification_probability"] = maximum_probabilities
classified_candidates_df["manual_label"] = final_labels
classified_candidates_df["channel"] = channel_name
for class_index, class_name in enumerate(classifier.classes_):
    classified_candidates_df[f"probability_{class_name}"] = predicted_probabilities[
        :, class_index
    ]
classified_candidates_df.to_csv(classification_output_path, index=False)
print("\nFinal classifications:")
print(classified_candidates_df["manual_label"].value_counts())
print("\nSaved:", classification_output_path.resolve())
