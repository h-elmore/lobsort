"""Extracted from spike_sort_v3.ipynb, cell 22."""

number_of_time_bins = 20
number_of_amplitude_bins = 4
random_seed = 12345
reference_signal_key = "filtered_signal"
candidate_trace_color = "#8f8f8f"
reference_trace_color = "#76a5d5"
candidate_marker_color = "#ffff00"
training_labels_path = Path(f"{channel_name}_manual_training_labels.csv")
if reference_channel_name is not None:
    if reference_channel_name not in processed_channels:
        raise KeyError(
            f"Reference channel {reference_channel_name!r} was not found.\nAvailable channels: {list(processed_channels.keys())}"
        )
    reference_result = processed_channels[reference_channel_name]
    if reference_signal_key not in reference_result:
        raise KeyError(
            f"{reference_signal_key!r} was not found in processed_channels[{reference_channel_name!r}].\nAvailable keys: {list(reference_result.keys())}"
        )
    reference_time = np.asarray(reference_result["time"], dtype=float)
    reference_signal = np.asarray(reference_result[reference_signal_key], dtype=float)
    if len(reference_time) != len(reference_signal):
        raise ValueError(
            "The reference-channel time and signal arrays do not have the same length."
        )
else:
    reference_time = None
    reference_signal = None
candidate_result = detect_spikes_processed_channel_adaptive(
    result=processed_channels[channel_name],
    threshold_factor=8,
    upper_threshold_factor=None,
    prominence_factor=None,
    shadow_period_ms=6,
    polarity="positive",
    adaptive_window_s=30.0,
    adaptive_step_s=15.0,
    noise_quantile=0.25,
    minimum_valid_fraction=0.25,
    noise_scale_bounds=(0.25, 4.0),
)
candidate_time = np.asarray(candidate_result["time"], dtype=float)
candidate_signal = np.asarray(candidate_result["centered_filtered_signal"], dtype=float)
candidate_indices = np.asarray(candidate_result["spike_indices"], dtype=int)
candidate_times = candidate_time[candidate_indices]
local_noise_trace = np.asarray(candidate_result["local_noise_trace"], dtype=float)
candidate_amplitudes = np.abs(candidate_signal[candidate_indices])
candidate_snr = candidate_amplitudes / local_noise_trace[candidate_indices]
print(f"Generated {len(candidate_indices):,} permissive candidates for {channel_name}.")
rng = np.random.default_rng(random_seed)
time_edges = np.linspace(
    candidate_times.min(), candidate_times.max(), number_of_time_bins + 1
)
time_bins = np.clip(
    np.digitize(candidate_times, time_edges[1:-1]), 0, number_of_time_bins - 1
)
amplitude_edges = np.unique(
    np.quantile(candidate_snr, np.linspace(0, 1, number_of_amplitude_bins + 1))
)
if len(amplitude_edges) > 2:
    amplitude_bins = np.clip(
        np.digitize(candidate_snr, amplitude_edges[1:-1]), 0, len(amplitude_edges) - 2
    )
else:
    amplitude_bins = np.zeros(len(candidate_snr), dtype=int)
candidate_groups = {}
for position in range(len(candidate_indices)):
    group_key = (int(time_bins[position]), int(amplitude_bins[position]))
    candidate_groups.setdefault(group_key, []).append(position)
for group_positions in candidate_groups.values():
    rng.shuffle(group_positions)
candidate_order = []
group_keys = list(candidate_groups.keys())
rng.shuffle(group_keys)
groups_remaining = True
while groups_remaining:
    groups_remaining = False
    for group_key in group_keys:
        group_positions = candidate_groups[group_key]
        if group_positions:
            candidate_order.append(group_positions.pop())
            groups_remaining = True
candidate_order = np.asarray(candidate_order, dtype=int)
if training_labels_path.exists():
    manual_labels_df = pd.read_csv(training_labels_path)
    print(f"Resuming from {training_labels_path}: {len(manual_labels_df)} labels.")
else:
    manual_labels_df = pd.DataFrame(
        columns=[
            "channel",
            "sample_index",
            "time_seconds",
            "manual_label",
            "candidate_snr",
        ]
    )
