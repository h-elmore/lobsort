"""Extracted from spike_sort_v3.ipynb, cell 45."""

plot_label = isi_filtered_label
plot_start_seconds = float(candidate_time[0])
plot_duration_seconds = 30.0
show_rejected_candidates = True
acceptance_reason_colors = {
    "high_classifier_probability": "#0072B2",
    "moderate_probability_with_sequence_support": "#E69F00",
    "rejected_by_within_burst_maximum_isi": "#D55E00",
    "insufficient_confidence": "#999999",
    "predicted_noise": "#666666",
    "invalid_waveform": "#CC79A7",
    "missing_transferred_prediction": "#000000",
}
default_reason_color = "#CC79A7"
required_transfer_columns = {
    "predicted_label",
    "classification_probability",
    "sample_index",
    "time_seconds",
    "valid_waveform",
}
missing_transfer_columns = required_transfer_columns - set(
    candidate_features_df.columns
)
if missing_transfer_columns:
    raise KeyError(
        f"candidate_features_df is missing transferred-classifier columns: {missing_transfer_columns}"
    )
candidate_features_df["predicted_label"] = (
    candidate_features_df["predicted_label"].fillna("Unmatched").astype(str).str.strip()
)
candidate_features_df["prediction_probability"] = pd.to_numeric(
    candidate_features_df["classification_probability"], errors="coerce"
)
all_predicted_labels = candidate_features_df["predicted_label"].to_numpy(dtype=str)
missing_transferred_prediction = (
    candidate_features_df["predicted_label"].isin(["Unmatched", "nan", "None"])
    | candidate_features_df["prediction_probability"].isna()
)
candidate_features_df.loc[missing_transferred_prediction, "predicted_label"] = (
    "Unmatched"
)
all_predicted_labels = candidate_features_df["predicted_label"].to_numpy(dtype=str)
print("Transferred predicted labels:")
print(candidate_features_df["predicted_label"].value_counts(dropna=False))
print("\nTransferred probability summary:")
print(candidate_features_df["prediction_probability"].describe())
print(
    "\nCandidates without transferred predictions:",
    int(missing_transferred_prediction.sum()),
)
candidate_times_all = candidate_features_df["time_seconds"].to_numpy(dtype=float)
same_label_neighbor_count = same_label_neighbors(
    candidate_times_all, all_predicted_labels, sequence_support_window_s
)
candidate_features_df["same_label_neighbor_count"] = same_label_neighbor_count
valid_waveform_mask = candidate_features_df["valid_waveform"].fillna(False).astype(bool)
target_label_mask = candidate_features_df["predicted_label"] == isi_filtered_label
noise_label_mask = candidate_features_df["predicted_label"] == noise_label
high_confidence_mask = (
    target_label_mask
    & valid_waveform_mask
    & (candidate_features_df["prediction_probability"] >= high_confidence_probability)
)
moderate_confidence_mask = (
    target_label_mask
    & valid_waveform_mask
    & ~high_confidence_mask
    & (candidate_features_df["prediction_probability"] >= supported_probability)
    & (
        candidate_features_df["same_label_neighbor_count"]
        >= minimum_same_label_neighbors
    )
)
candidate_features_df["accepted_as_spike"] = (
    high_confidence_mask | moderate_confidence_mask
)
candidate_features_df["acceptance_reason"] = "insufficient_confidence"
candidate_features_df.loc[noise_label_mask, "acceptance_reason"] = "predicted_noise"
candidate_features_df.loc[moderate_confidence_mask, "acceptance_reason"] = (
    "moderate_probability_with_sequence_support"
)
candidate_features_df.loc[high_confidence_mask, "acceptance_reason"] = (
    "high_classifier_probability"
)
candidate_features_df.loc[~valid_waveform_mask, "acceptance_reason"] = (
    "invalid_waveform"
)
candidate_features_df.loc[missing_transferred_prediction, "acceptance_reason"] = (
    "missing_transferred_prediction"
)
print("\nTarget label:", isi_filtered_label)
print("Noise label:", noise_label)
print("High-confidence target candidates:", int(high_confidence_mask.sum()))
print("Moderate supported target candidates:", int(moderate_confidence_mask.sum()))
print(
    "Accepted before ISI filtering:",
    int(candidate_features_df["accepted_as_spike"].sum()),
)
print("\nAcceptance reasons before ISI filtering:")
print(candidate_features_df["acceptance_reason"].value_counts(dropna=False))
print("\nAccepted labels before ISI filtering:")
print(
    candidate_features_df.loc[
        candidate_features_df["accepted_as_spike"], "predicted_label"
    ].value_counts(dropna=False)
)
if "reference_burst_onsets" not in globals():
    raise NameError(
        "reference_burst_onsets was not found. Run waveform/temporal/rhythmic Block 1 first."
    )
if "reference_burst_offsets" not in globals():
    raise NameError(
        "reference_burst_offsets was not found. Run waveform/temporal/rhythmic Block 1 first."
    )
reference_burst_onsets_array = np.asarray(reference_burst_onsets, dtype=float)
reference_burst_offsets_array = np.asarray(reference_burst_offsets, dtype=float)
if len(reference_burst_onsets_array) != len(reference_burst_offsets_array):
    raise ValueError("Reference burst onset and offset arrays have different lengths.")
candidate_burst_id = np.full(len(candidate_features_df), -1, dtype=int)
if len(reference_burst_onsets_array) > 0 and len(reference_burst_offsets_array) > 0:
    possible_burst_ids = (
        np.searchsorted(reference_burst_onsets_array, candidate_times_all, side="right")
        - 1
    )
    valid_possible_burst = (possible_burst_ids >= 0) & (
        possible_burst_ids < len(reference_burst_offsets_array)
    )
    if "reference_reliable" in candidate_features_df.columns:
        valid_possible_burst = (
            valid_possible_burst
            & candidate_features_df["reference_reliable"]
            .fillna(0)
            .astype(bool)
            .to_numpy()
        )
    valid_positions = np.flatnonzero(valid_possible_burst)
    valid_ids = possible_burst_ids[valid_possible_burst]
    valid_times = candidate_times_all[valid_possible_burst]
    inside_burst = valid_times <= reference_burst_offsets_array[valid_ids]
    inside_positions = valid_positions[inside_burst]
    candidate_burst_id[inside_positions] = valid_ids[inside_burst]
candidate_features_df["reference_burst_id"] = candidate_burst_id
print(
    "\nCandidates inside reliable reference bursts:",
    int(np.sum(candidate_burst_id >= 0)),
)
print(
    "Candidates outside reliable reference bursts:", int(np.sum(candidate_burst_id < 0))
)
candidate_features_df["preceding_confirmed_spikes"] = 0
candidate_features_df["sequential_isi_seconds"] = np.nan
candidate_features_df["preceding_mean_isi_seconds"] = np.nan
candidate_features_df["preceding_isi_std_seconds"] = np.nan
candidate_features_df["maximum_allowed_isi_seconds"] = np.nan
candidate_features_df["within_burst_isi_passed"] = True
accepted_before_isi_filter = (
    candidate_features_df["accepted_as_spike"].fillna(False).astype(bool).copy()
)
predicted_as_isi_label = candidate_features_df["predicted_label"] == isi_filtered_label
eligible_for_isi_filter = (
    accepted_before_isi_filter
    & predicted_as_isi_label
    & (candidate_features_df["reference_burst_id"] >= 0)
)
eligible_burst_ids = np.unique(
    candidate_features_df.loc[eligible_for_isi_filter, "reference_burst_id"].to_numpy(
        dtype=int
    )
)
number_rejected_by_isi = 0
for burst_id in eligible_burst_ids:
    burst_rows = candidate_features_df.index[
        eligible_for_isi_filter
        & (candidate_features_df["reference_burst_id"] == burst_id)
    ]
    burst_rows = (
        candidate_features_df.loc[burst_rows]
        .sort_values("time_seconds")
        .index.to_numpy()
    )
    confirmed_times = []
    for row_index in burst_rows:
        current_time = float(candidate_features_df.at[row_index, "time_seconds"])
        preceding_spike_count = len(confirmed_times)
        candidate_features_df.at[row_index, "preceding_confirmed_spikes"] = (
            preceding_spike_count
        )
        if preceding_spike_count < minimum_preceding_spikes:
            confirmed_times.append(current_time)
            continue
        preceding_times_array = np.asarray(confirmed_times, dtype=float)
        preceding_isis = np.diff(preceding_times_array)
        preceding_isis = preceding_isis[
            np.isfinite(preceding_isis) & (preceding_isis > 0)
        ]
        if len(preceding_isis) == 0:
            confirmed_times.append(current_time)
            continue
        mean_preceding_isi = float(np.mean(preceding_isis))
        if len(preceding_isis) >= 2:
            preceding_isi_std = float(np.std(preceding_isis, ddof=1))
        else:
            preceding_isi_std = 0.0
        minimum_tolerance_seconds = minimum_isi_tolerance_ms / 1000.0
        effective_isi_std = max(preceding_isi_std, minimum_tolerance_seconds)
        maximum_allowed_isi = (
            mean_preceding_isi + isi_standard_deviations * effective_isi_std
        )
        current_isi = current_time - confirmed_times[-1]
        isi_passed = current_isi <= maximum_allowed_isi
        candidate_features_df.at[row_index, "sequential_isi_seconds"] = current_isi
        candidate_features_df.at[row_index, "preceding_mean_isi_seconds"] = (
            mean_preceding_isi
        )
        candidate_features_df.at[row_index, "preceding_isi_std_seconds"] = (
            preceding_isi_std
        )
        candidate_features_df.at[row_index, "maximum_allowed_isi_seconds"] = (
            maximum_allowed_isi
        )
        candidate_features_df.at[row_index, "within_burst_isi_passed"] = isi_passed
        if isi_passed:
            confirmed_times.append(current_time)
        else:
            number_rejected_by_isi += 1
            if apply_within_burst_isi_filter:
                candidate_features_df.at[row_index, "accepted_as_spike"] = False
                candidate_features_df.at[row_index, "acceptance_reason"] = (
                    "rejected_by_within_burst_maximum_isi"
                )
accepted_before_count = int(np.sum(accepted_before_isi_filter & predicted_as_isi_label))
accepted_after_count = int(
    np.sum(
        candidate_features_df["accepted_as_spike"].fillna(False).astype(bool)
        & predicted_as_isi_label
    )
)
print(
    f"\n{isi_filtered_label} accepted before within-burst ISI filtering: {accepted_before_count:,}"
)
print(
    f"{isi_filtered_label} rejected by the within-burst ISI rule: {number_rejected_by_isi:,}"
)
print(
    f"{isi_filtered_label} remaining after within-burst ISI filtering: {accepted_after_count:,}"
)
accepted_final_mask = (
    candidate_features_df["accepted_as_spike"].fillna(False).astype(bool)
)
final_spikes_df = (
    candidate_features_df.loc[accepted_final_mask]
    .copy()
    .sort_values("time_seconds")
    .reset_index(drop=True)
)
final_spikes_df["manual_label"] = final_spikes_df["predicted_label"]
if "spike_number" in final_spikes_df.columns:
    final_spikes_df = final_spikes_df.drop(columns="spike_number")
final_spikes_df.insert(0, "spike_number", np.arange(1, len(final_spikes_df) + 1))
print(f"\nPermissive candidates: {len(candidate_features_df):,}")
print(f"Final accepted spikes: {len(final_spikes_df):,}")
print("\nFinal spikes by predicted label:")
display(final_spikes_df["predicted_label"].value_counts().rename("accepted_spikes"))
