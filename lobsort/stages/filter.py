"""Extracted from spike_sort_v3.ipynb, cell 16."""

channels_to_analyze = [
    "Red",
    "Lavender",
    "Yellow",
    "Orange",
    "Green",
    "Blue",
    "Purple",
    "Black",
]
artifact_thresholds_mV = {
    "Red": 120.0,
    "Lavender": 50.0,
    "Yellow": 20.0,
    "Orange": 30.0,
    "Green": 1400.0,
    "Blue": 45.0,
    "Purple": 28.0,
    "Black": 35.0,
}
default_artifact_threshold_mV = 28.0
artifact_padding_ms = 10.0
filter_low_hz = 100.0
filter_high_hz = 1500.0
filter_order = 3
preview_start_s = None
preview_duration_s = 120.0
final_plot_start_s = None
final_plot_duration_s = 30.0
show_raw_preview = True
show_final_plot = True


def find_channel(channel_identifier, all_channels):
    """
    Find a channel by integer index or channel name.
    """
    if isinstance(channel_identifier, int):
        if not 0 <= channel_identifier < len(all_channels):
            raise IndexError(
                f"Channel index {channel_identifier} is outside 0–{len(all_channels) - 1}."
            )
        return (channel_identifier, all_channels[channel_identifier])
    requested_name = str(channel_identifier).strip().lower()
    exact_matches = [
        (index, channel)
        for index, channel in enumerate(all_channels)
        if str(channel["name"]).strip().lower() == requested_name
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    partial_matches = [
        (index, channel)
        for index, channel in enumerate(all_channels)
        if requested_name in str(channel["name"]).strip().lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        matching_names = [channel["name"] for _, channel in partial_matches]
        raise ValueError(
            f"{channel_identifier!r} matched multiple channels: {matching_names}"
        )
    available_names = [channel["name"] for channel in all_channels]
    raise ValueError(
        f"Could not find channel {channel_identifier!r}. Available channels: {available_names}"
    )


def threshold_mV_to_native_units(threshold_mV, units):
    """
    Convert an artifact threshold in mV to the native units
    used in the channel array.
    """
    units_text = str(units).lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    if "mv" in units_text:
        return threshold_mV
    if "uv" in units_text:
        return threshold_mV * 1000.0
    if "v" in units_text:
        return threshold_mV / 1000.0
    raise ValueError(
        f"Could not interpret channel units {units!r}. Expected V, mV, or uV."
    )


def robust_noise_estimate(signal):
    """
    Estimate background scale using median absolute deviation.
    """
    signal = np.asarray(signal, dtype=float)
    center = np.nanmedian(signal)
    mad_sigma = np.nanmedian(np.abs(signal - center)) / 0.6745
    if not np.isfinite(mad_sigma) or mad_sigma <= 0:
        raise ValueError(
            "Could not calculate a positive robust noise estimate for this channel."
        )
    return (center, mad_sigma)


def make_artifact_mask(time, raw_signal, threshold_native, padding_ms):
    """
    Identify samples whose absolute deviation from the median
    exceeds the physical amplitude threshold.
    """
    time = np.asarray(time, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)
    sampling_rate = 1 / np.median(np.diff(time))
    center, noise_sigma = robust_noise_estimate(raw_signal)
    centered_signal = raw_signal - center
    artifact_z_threshold = threshold_native / noise_sigma
    robust_amplitude = np.abs(centered_signal) / noise_sigma
    initial_mask = np.abs(centered_signal) >= threshold_native
    padding_samples = max(1, int(sampling_rate * padding_ms / 1000))
    padded_mask = binary_dilation(initial_mask, iterations=padding_samples)
    return {
        "center": center,
        "noise_sigma": noise_sigma,
        "artifact_z_threshold": artifact_z_threshold,
        "robust_amplitude": robust_amplitude,
        "initial_mask": initial_mask,
        "artifact_mask": padded_mask,
        "sampling_rate": sampling_rate,
        "padding_samples": padding_samples,
    }


def interpolate_masked_samples(signal, mask):
    """
    Linearly interpolate across artifact intervals.

    This interpolated signal is used only as input to the
    filter. The final masked filtered signal still contains
    NaN gaps at excluded locations.
    """
    signal = np.asarray(signal, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    sample_indices = np.arange(len(signal))
    good_samples = ~mask & np.isfinite(signal)
    if np.sum(good_samples) < 2:
        raise ValueError("Too few unmasked samples remain for interpolation.")
    interpolated = signal.copy()
    interpolated[mask] = np.interp(
        sample_indices[mask], sample_indices[good_samples], signal[good_samples]
    )
    return interpolated


def count_masked_events(mask):
    """
    Count contiguous masked intervals.
    """
    changes = np.diff(
        np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(int)
    )
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (len(starts), starts, stops)


selected_channels = []
for requested_channel in channels_to_analyze:
    channel_index, channel = find_channel(requested_channel, channels)
    selected_channels.append({"index": channel_index, "channel": channel})
print("Selected channels:")
for item in selected_channels:
    print(
        f"  index={item['index']}, name={item['channel']['name']!r}, units={item['channel']['units']}"
    )
processed_channels = {}
for item in selected_channels:
    channel_index = item["index"]
    channel = item["channel"]
    channel_name = str(channel["name"])
    time = np.asarray(channel["time"], dtype=float)
    raw_signal = np.asarray(channel["values"], dtype=float)
    threshold_mV = artifact_thresholds_mV.get(
        channel_name, default_artifact_threshold_mV
    )
    threshold_native = threshold_mV_to_native_units(
        threshold_mV=threshold_mV, units=channel["units"]
    )
    mask_result = make_artifact_mask(
        time=time,
        raw_signal=raw_signal,
        threshold_native=threshold_native,
        padding_ms=artifact_padding_ms,
    )
    artifact_mask = mask_result["artifact_mask"] | binary_dilation(
        ~np.isfinite(raw_signal), iterations=mask_result["padding_samples"]
    )
    sampling_rate = mask_result["sampling_rate"]
    centered_signal = raw_signal - mask_result["center"]
    interpolation_signal = interpolate_masked_samples(
        signal=centered_signal, mask=artifact_mask
    )
    actual_high_hz = min(filter_high_hz, sampling_rate * 0.45)
    if filter_low_hz >= actual_high_hz:
        raise ValueError(
            f"Channel {channel_name!r} has a sampling rate of {sampling_rate:.2f} Hz. The requested filter {filter_low_hz}–{actual_high_hz:.2f} Hz is invalid. Reduce filter_low_hz."
        )
    sos = butter(
        N=filter_order,
        Wn=[filter_low_hz, actual_high_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    filtered_interpolated = sosfiltfilt(sos, interpolation_signal)
    filtered_masked = filtered_interpolated.copy()
    filtered_masked[artifact_mask] = np.nan
    event_count, event_starts, event_stops = count_masked_events(artifact_mask)
    processed_channels[channel_name] = {
        "channel_index": channel_index,
        "units": channel["units"],
        "time": time,
        "raw_signal": raw_signal,
        "centered_signal": centered_signal,
        "interpolated_signal": interpolation_signal,
        "filtered_interpolated": filtered_interpolated,
        "filtered_signal": filtered_masked,
        "artifact_mask": artifact_mask,
        "initial_artifact_mask": mask_result["initial_mask"],
        "robust_amplitude": mask_result["robust_amplitude"],
        "robust_noise": mask_result["noise_sigma"],
        "artifact_threshold_mV": threshold_mV,
        "artifact_threshold_native": threshold_native,
        "artifact_z_threshold": mask_result["artifact_z_threshold"],
        "sampling_rate": sampling_rate,
        "filter_low_hz": filter_low_hz,
        "filter_high_hz": actual_high_hz,
        "artifact_event_count": event_count,
        "artifact_event_starts": event_starts,
        "artifact_event_stops": event_stops,
    }
    print()
    print(f"Channel: {channel_name}")
    print(f"  Sampling rate: {sampling_rate:.2f} Hz")
    print(f"  Robust noise: {mask_result['noise_sigma']:.8f} {channel['units']}")
    print(f"  Physical artifact cutoff: {threshold_mV:.3f} mV")
    print(f"  Equivalent robust-Z cutoff: {mask_result['artifact_z_threshold']:.2f}")
    print(f"  Artifact intervals: {event_count}")
    print(f"  Recording excluded: {artifact_mask.mean() * 100:.5f}%")
    print(f"  Filter: {filter_low_hz:.1f}–{actual_high_hz:.1f} Hz")
