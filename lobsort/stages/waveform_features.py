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
