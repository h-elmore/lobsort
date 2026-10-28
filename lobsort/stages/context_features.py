"""Extracted from spike_sort_v3.ipynb, cell 33."""

local_context_seconds = 0.25
reference_envelope_smoothing_ms = 40.0
minimum_reference_burst_duration_s = 0.1
reference_diagnostic_seconds = 25.0
reference_diagnostic_start_s = float(candidate_times[0])
candidate_time = np.asarray(candidate_result["time"], dtype=float)
candidate_signal = np.asarray(candidate_result["centered_filtered_signal"], dtype=float)
candidate_indices = np.asarray(candidate_result["spike_indices"], dtype=int)
candidate_times = candidate_time[candidate_indices]
if len(candidate_time) < 2:
    raise ValueError("Not enough target-channel samples.")
target_sampling_rate = 1.0 / np.median(np.diff(candidate_time))
samples_before = max(1, int(round(waveform_before_ms * target_sampling_rate / 1000.0)))
samples_after = max(1, int(round(waveform_after_ms * target_sampling_rate / 1000.0)))
waveform_sample_count = samples_before + samples_after + 1
print(f"Target sampling rate: {target_sampling_rate:.2f} Hz")
print(f"Waveform contains {waveform_sample_count} samples.")
waveforms = np.full(
    (len(candidate_indices), waveform_sample_count), np.nan, dtype=float
)
valid_waveform = np.zeros(len(candidate_indices), dtype=bool)
for position, sample_index in enumerate(candidate_indices):
    start_index = sample_index - samples_before
    stop_index = sample_index + samples_after + 1
    if start_index < 0:
        continue
    if stop_index > len(candidate_signal):
        continue
    waveform = candidate_signal[start_index:stop_index].copy()
    if len(waveform) != waveform_sample_count:
        continue
    if not np.all(np.isfinite(waveform)):
        continue
    baseline_sample_count = max(1, samples_before // 2)
    baseline = np.median(waveform[:baseline_sample_count])
    waveform = waveform - baseline
    waveforms[position] = waveform
    valid_waveform[position] = True
normalized_waveforms = np.full_like(waveforms, np.nan)
waveform_norms = np.linalg.norm(np.nan_to_num(waveforms), axis=1)
normalizable = valid_waveform & np.isfinite(waveform_norms) & (waveform_norms > 0)
normalized_waveforms[normalizable] = (
    waveforms[normalizable] / waveform_norms[normalizable, np.newaxis]
)
peak_positive = np.nanmax(waveforms, axis=1)
peak_negative = np.nanmin(waveforms, axis=1)
absolute_peak = np.nanmax(np.abs(waveforms), axis=1)
peak_to_peak = peak_positive - peak_negative
waveform_energy = np.nansum(waveforms**2, axis=1)
waveform_area = np.nansum(np.abs(waveforms), axis=1)
peak_sample_within_waveform = np.nanargmax(np.abs(np.nan_to_num(waveforms)), axis=1)
peak_offset_ms = (
    (peak_sample_within_waveform - samples_before) / target_sampling_rate * 1000.0
)
candidate_sort_order = np.argsort(candidate_times)
sorted_candidate_times = candidate_times[candidate_sort_order]
previous_isi_s_sorted = np.full(len(candidate_times), np.nan)
next_isi_s_sorted = np.full(len(candidate_times), np.nan)
if len(candidate_times) > 1:
    candidate_intervals = np.diff(sorted_candidate_times)
    previous_isi_s_sorted[1:] = candidate_intervals
    next_isi_s_sorted[:-1] = candidate_intervals
previous_isi_s = np.full(len(candidate_times), np.nan)
next_isi_s = np.full(len(candidate_times), np.nan)
previous_isi_s[candidate_sort_order] = previous_isi_s_sorted
next_isi_s[candidate_sort_order] = next_isi_s_sorted
local_candidate_count_sorted = np.zeros(len(candidate_times), dtype=int)
for sorted_position, event_time in enumerate(sorted_candidate_times):
    left_position = np.searchsorted(
        sorted_candidate_times, event_time - local_context_seconds, side="left"
    )
    right_position = np.searchsorted(
        sorted_candidate_times, event_time + local_context_seconds, side="right"
    )
    local_candidate_count_sorted[sorted_position] = max(
        0, right_position - left_position - 1
    )
local_candidate_count = np.zeros(len(candidate_times), dtype=int)
local_candidate_count[candidate_sort_order] = local_candidate_count_sorted
local_candidate_rate_hz = local_candidate_count / (2.0 * local_context_seconds)
if reference_channel_name not in processed_channels:
    raise KeyError(
        f"{reference_channel_name!r} was not found. Available channels: {list(processed_channels.keys())}"
    )
reference_result = processed_channels[reference_channel_name]
reference_time = np.asarray(reference_result["time"], dtype=float)
reference_signal = np.asarray(reference_result[reference_signal_key], dtype=float)
if len(reference_time) != len(reference_signal):
    raise ValueError("Reference time and signal lengths differ.")
if len(reference_time) < 2:
    raise ValueError("Not enough reference-channel samples.")
reference_sampling_rate = 1.0 / np.median(np.diff(reference_time))
finite_reference = np.isfinite(reference_signal)
if np.sum(finite_reference) < 10:
    raise ValueError("Too few finite reference-channel samples.")
reference_center = np.nanmedian(reference_signal)
reference_filled = reference_signal.copy()
reference_filled[~finite_reference] = reference_center
reference_centered = reference_filled - reference_center
smoothing_sigma_samples = max(
    1, int(round(reference_envelope_smoothing_ms * reference_sampling_rate / 1000.0))
)
reference_envelope = gaussian_filter1d(
    np.abs(reference_centered), sigma=smoothing_sigma_samples, mode="nearest"
)
reference_envelope_center = np.median(reference_envelope)
reference_envelope_sigma = (
    np.median(np.abs(reference_envelope - reference_envelope_center)) / 0.6745
)
if not np.isfinite(reference_envelope_sigma) or reference_envelope_sigma <= 0:
    reference_envelope_sigma = np.std(reference_envelope)
reference_burst_threshold = (
    reference_envelope_center
    + reference_burst_threshold_factor * reference_envelope_sigma
)
reference_active = reference_envelope >= reference_burst_threshold
minimum_burst_samples = max(
    1, int(round(minimum_reference_burst_duration_s * reference_sampling_rate))
)
maximum_gap_samples = max(
    1, int(round(maximum_reference_burst_gap_s * reference_sampling_rate))
)
reference_active = binary_closing(
    reference_active, structure=np.ones(maximum_gap_samples, dtype=bool)
)
reference_active = binary_opening(
    reference_active, structure=np.ones(minimum_burst_samples, dtype=bool)
)
reference_transitions = np.diff(reference_active.astype(int), prepend=0, append=0)
reference_onset_indices = np.flatnonzero(reference_transitions == 1)
reference_offset_indices = np.flatnonzero(reference_transitions == -1)
reference_burst_onsets = reference_time[
    np.clip(reference_onset_indices, 0, len(reference_time) - 1)
]
reference_burst_offsets = reference_time[
    np.clip(reference_offset_indices - 1, 0, len(reference_time) - 1)
]
reference_burst_durations = reference_burst_offsets - reference_burst_onsets
reference_burst_is_reliable = np.isfinite(reference_burst_durations) & (
    reference_burst_durations <= maximum_reliable_reference_burst_duration_s
)
reference_burst_is_unreliable = ~reference_burst_is_reliable
unreliable_reference_mask = np.zeros(len(reference_time), dtype=bool)
unreliable_padding_samples = max(
    0, int(round(unreliable_reference_padding_s * reference_sampling_rate))
)
for onset_index, offset_index, is_unreliable in zip(
    reference_onset_indices, reference_offset_indices, reference_burst_is_unreliable
):
    if not is_unreliable:
        continue
    padded_start = max(0, int(onset_index) - unreliable_padding_samples)
    padded_stop = min(
        len(reference_time), int(offset_index) + unreliable_padding_samples
    )
    unreliable_reference_mask[padded_start:padded_stop] = True
reference_reliable_mask = ~unreliable_reference_mask
reliable_reference_burst_onsets = reference_burst_onsets[reference_burst_is_reliable]
reliable_reference_burst_offsets = reference_burst_offsets[reference_burst_is_reliable]
reliable_reference_burst_durations = reference_burst_durations[
    reference_burst_is_reliable
]
print("Reliable reference bursts:", len(reliable_reference_burst_onsets))
print(
    "Continuously active/unreliable reference periods:",
    int(np.sum(reference_burst_is_unreliable)),
)
print(f"Reference channel: {reference_channel_name}")
print(f"Reference sampling rate: {reference_sampling_rate:.2f} Hz")
print(f"Reference bursts detected: {len(reference_burst_onsets):,}")
candidate_reference_envelope = np.interp(
    candidate_times, reference_time, reference_envelope, left=np.nan, right=np.nan
)
candidate_reference_z = (
    candidate_reference_envelope - reference_envelope_center
) / reference_envelope_sigma
candidate_reference_active = (
    np.interp(
        candidate_times, reference_time, reference_active.astype(float), left=0, right=0
    )
    >= 0.5
)
candidate_reference_reliable = (
    np.interp(
        candidate_times,
        reference_time,
        reference_reliable_mask.astype(float),
        left=0,
        right=0,
    )
    >= 0.5
)
candidate_reference_unreliable = ~candidate_reference_reliable
candidate_reference_envelope[candidate_reference_unreliable] = np.nan
candidate_reference_z[candidate_reference_unreliable] = np.nan
candidate_reference_active_feature = candidate_reference_active.astype(float)
candidate_reference_active_feature[candidate_reference_unreliable] = np.nan
cycle_phase = np.full(len(candidate_times), np.nan)
time_from_previous_reference_onset = np.full(len(candidate_times), np.nan)
time_until_next_reference_onset = np.full(len(candidate_times), np.nan)
if len(reliable_reference_burst_onsets) >= 2:
    previous_onset_positions = (
        np.searchsorted(reliable_reference_burst_onsets, candidate_times, side="right")
        - 1
    )
    next_onset_positions = previous_onset_positions + 1
    valid_cycle = (previous_onset_positions >= 0) & (
        next_onset_positions < len(reliable_reference_burst_onsets)
    )
    previous_onsets = reliable_reference_burst_onsets[
        previous_onset_positions[valid_cycle]
    ]
    next_onsets = reliable_reference_burst_onsets[next_onset_positions[valid_cycle]]
    cycle_durations = next_onsets - previous_onsets
    interval_is_reliable = np.ones(len(cycle_durations), dtype=bool)
    for interval_position, (previous_onset, next_onset) in enumerate(
        zip(previous_onsets, next_onsets)
    ):
        interval_left = np.searchsorted(reference_time, previous_onset, side="left")
        interval_right = np.searchsorted(reference_time, next_onset, side="right")
        if np.any(unreliable_reference_mask[interval_left:interval_right]):
            interval_is_reliable[interval_position] = False
    positive_cycle = (
        (cycle_durations > 0)
        & interval_is_reliable
        & candidate_reference_reliable[np.flatnonzero(valid_cycle)]
    )
    valid_candidate_positions = np.flatnonzero(valid_cycle)
    valid_candidate_positions = valid_candidate_positions[positive_cycle]
    previous_onsets = previous_onsets[positive_cycle]
    next_onsets = next_onsets[positive_cycle]
    cycle_durations = cycle_durations[positive_cycle]
    cycle_phase[valid_candidate_positions] = (
        candidate_times[valid_candidate_positions] - previous_onsets
    ) / cycle_durations
    time_from_previous_reference_onset[valid_candidate_positions] = (
        candidate_times[valid_candidate_positions] - previous_onsets
    )
    time_until_next_reference_onset[valid_candidate_positions] = (
        next_onsets - candidate_times[valid_candidate_positions]
    )
cycle_phase_sin = np.sin(2.0 * np.pi * cycle_phase)
cycle_phase_cos = np.cos(2.0 * np.pi * cycle_phase)
candidate_features_df = pd.DataFrame(
    {
        "candidate_position": np.arange(len(candidate_indices)),
        "sample_index": candidate_indices,
        "time_seconds": candidate_times,
        "positive_peak": peak_positive,
        "negative_peak": peak_negative,
        "absolute_peak": absolute_peak,
        "peak_to_peak": peak_to_peak,
        "waveform_energy": waveform_energy,
        "waveform_area": waveform_area,
        "peak_offset_ms": peak_offset_ms,
        "previous_isi_ms": previous_isi_s * 1000.0,
        "next_isi_ms": next_isi_s * 1000.0,
        "local_candidate_count": local_candidate_count,
        "local_candidate_rate_hz": local_candidate_rate_hz,
        "reference_envelope": candidate_reference_envelope,
        "reference_envelope_z": candidate_reference_z,
        "reference_active": candidate_reference_active_feature,
        "reference_reliable": candidate_reference_reliable.astype(int),
        "time_from_reference_onset_s": time_from_previous_reference_onset,
        "time_until_reference_onset_s": time_until_next_reference_onset,
        "cycle_phase": cycle_phase,
        "cycle_phase_sin": cycle_phase_sin,
        "cycle_phase_cos": cycle_phase_cos,
        "valid_waveform": valid_waveform,
    }
)
waveform_feature_columns = []
for waveform_sample in range(waveform_sample_count):
    column_name = f"waveform_{waveform_sample:03d}"
    candidate_features_df[column_name] = normalized_waveforms[:, waveform_sample]
    waveform_feature_columns.append(column_name)
context_feature_columns = [
    "positive_peak",
    "negative_peak",
    "absolute_peak",
    "peak_to_peak",
    "waveform_energy",
    "waveform_area",
    "peak_offset_ms",
    "previous_isi_ms",
    "next_isi_ms",
    "local_candidate_count",
    "local_candidate_rate_hz",
    "reference_envelope_z",
    "reference_active",
    "reference_reliable",
    "time_from_reference_onset_s",
    "time_until_reference_onset_s",
    "cycle_phase_sin",
    "cycle_phase_cos",
]
print(f"Created features for {len(candidate_features_df):,} candidates.")
