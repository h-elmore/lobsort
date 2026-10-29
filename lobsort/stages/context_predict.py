"""Extracted from spike_sort_v3.ipynb, cell 41."""

X_all_candidates = candidate_features_df[
    globals().get(
        "feature_columns_for_prediction",
        waveform_feature_columns + context_feature_columns,
    )
]
all_predicted_labels = context_classifier.predict(X_all_candidates)
all_prediction_probabilities = context_classifier.predict_proba(X_all_candidates)
classifier_classes = context_classifier.named_steps["classifier"].classes_
maximum_prediction_probability = np.max(all_prediction_probabilities, axis=1)
for class_position, class_name in enumerate(classifier_classes):
    safe_class_name = str(class_name).replace(" ", "_").replace("/", "_")
    candidate_features_df[f"probability_{safe_class_name}"] = (
        all_prediction_probabilities[:, class_position]
    )
candidate_features_df["predicted_label"] = all_predicted_labels
candidate_features_df["prediction_probability"] = maximum_prediction_probability
predicted_labels = np.asarray(all_predicted_labels).astype(str)
candidate_times_all = candidate_features_df["time_seconds"].to_numpy(dtype=float)
same_label_neighbor_count = same_label_neighbors(
    candidate_times_all, predicted_labels, sequence_support_window_s
)
candidate_features_df["same_label_neighbor_count"] = same_label_neighbor_count
predicted_as_neuron = (
    candidate_features_df["predicted_label"].astype(str) != noise_label
)
high_confidence = (
    candidate_features_df["prediction_probability"] >= high_confidence_probability
)
moderate_with_support = (
    candidate_features_df["prediction_probability"] >= supported_probability
) & (candidate_features_df["same_label_neighbor_count"] >= minimum_same_label_neighbors)
candidate_features_df["accepted_as_spike"] = (
    candidate_features_df["valid_waveform"]
    & predicted_as_neuron
    & (high_confidence | moderate_with_support)
)
candidate_features_df["acceptance_reason"] = np.select(
    [
        predicted_as_neuron & high_confidence,
        predicted_as_neuron & moderate_with_support,
        ~predicted_as_neuron,
    ],
    [
        "high_classifier_probability",
        "moderate_probability_with_sequence_support",
        "predicted_noise",
    ],
    default="insufficient_confidence",
)
if "reference_burst_onsets" not in globals():
    raise NameError(
        "reference_burst_onsets was not found. Run Block 1 before this block."
    )
if "reference_burst_offsets" not in globals():
    raise NameError(
        "reference_burst_offsets was not found. Run Block 1 before this block."
    )
reference_burst_onsets_array = np.asarray(reference_burst_onsets, dtype=float)
reference_burst_offsets_array = np.asarray(reference_burst_offsets, dtype=float)
candidate_burst_id = np.full(len(candidate_features_df), -1, dtype=int)
if len(reference_burst_onsets_array) > 0 and len(reference_burst_offsets_array) > 0:
    possible_burst_ids = (
        np.searchsorted(reference_burst_onsets_array, candidate_times_all, side="right")
        - 1
    )
    valid_possible_burst = (possible_burst_ids >= 0) & (
        possible_burst_ids < len(reference_burst_offsets_array)
    )
    valid_positions = np.flatnonzero(valid_possible_burst)
    valid_ids = possible_burst_ids[valid_possible_burst]
    valid_times = candidate_times_all[valid_possible_burst]
    inside_burst = valid_times <= reference_burst_offsets_array[valid_ids]
    inside_positions = valid_positions[inside_burst]
    candidate_burst_id[inside_positions] = valid_ids[inside_burst]
candidate_features_df["reference_burst_id"] = candidate_burst_id
candidate_features_df["preceding_confirmed_spikes"] = 0
candidate_features_df["sequential_isi_seconds"] = np.nan
candidate_features_df["preceding_mean_isi_seconds"] = np.nan
candidate_features_df["preceding_isi_std_seconds"] = np.nan
candidate_features_df["maximum_allowed_isi_seconds"] = np.nan
candidate_features_df["within_burst_isi_passed"] = True
accepted_before_isi_filter = candidate_features_df["accepted_as_spike"].astype(bool)
predicted_as_isi_label = (
    candidate_features_df["predicted_label"].astype(str) == isi_filtered_label
)
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
        candidate_features_df["accepted_as_spike"].astype(bool) & predicted_as_isi_label
    )
)
print(
    f"{isi_filtered_label} accepted before within-burst ISI filtering: {accepted_before_count:,}"
)
print(
    f"{isi_filtered_label} rejected by the within-burst ISI rule: {number_rejected_by_isi:,}"
)
print(
    f"{isi_filtered_label} remaining after within-burst ISI filtering: {accepted_after_count:,}"
)
final_spikes_df = (
    candidate_features_df[candidate_features_df["accepted_as_spike"]]
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
display(
    final_spikes_df[
        [
            "spike_number",
            "time_seconds",
            "sample_index",
            "predicted_label",
            "prediction_probability",
            "reference_burst_id",
            "preceding_confirmed_spikes",
            "sequential_isi_seconds",
            "maximum_allowed_isi_seconds",
            "within_burst_isi_passed",
            "acceptance_reason",
        ]
    ].head(20)
)
