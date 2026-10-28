"""Extracted from spike_sort_v3.ipynb, cell 21."""


def estimate_background_noise(values, quantile=0.25):
    """
    Estimate the center and background-noise scale of a signal.

    Using the lower portion of the amplitude distribution makes
    the estimate less sensitive to genuine spikes and bursts.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return (np.nan, np.nan)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")
    center = np.median(values)
    absolute_deviation = np.abs(values - center)
    scale_constant = norm.ppf((1 + quantile) / 2)
    noise_sigma = np.quantile(absolute_deviation, quantile) / scale_constant
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = np.median(absolute_deviation) / 0.6745
    return (center, noise_sigma)


def detect_spikes_processed_channel_adaptive(
    result,
    threshold_factor=2,
    upper_threshold_factor=None,
    shadow_period_ms=1.0,
    prominence_factor=None,
    polarity="both",
    adaptive_window_s=30.0,
    adaptive_step_s=15.0,
    noise_quantile=0.25,
    minimum_valid_fraction=0.25,
    noise_scale_bounds=(0.25, 4.0),
):
    """
    Detect extracellular spikes using locally adaptive noise,
    threshold, and prominence estimates.

    Parameters
    ----------
    result : dict
        One channel from processed_channels.

    threshold_factor : float
        Lower detection threshold in local noise units.

    upper_threshold_factor : float or None
        Optional upper threshold in local noise units.

    shadow_period_ms : float
        Minimum time between detected events.

    prominence_factor : float or None
        Optional minimum prominence in local noise units.

    polarity : str
        Which spike direction to detect:

            "positive"
            "negative"
            "both"

    adaptive_window_s : float
        Duration of each local noise-estimation window.

    adaptive_step_s : float
        Time between consecutive window centers.

        For example, a 30-second window with a 15-second step
        gives 50% overlap.

    noise_quantile : float
        Portion of the absolute-amplitude distribution used to
        estimate background noise.

    minimum_valid_fraction : float
        Minimum valid fraction required in each adaptive window.

    noise_scale_bounds : tuple or None
        Lower and upper bounds for local noise relative to the
        global noise estimate.

        For example, (0.25, 4.0) prevents the local threshold
        from falling below one-quarter or rising above four
        times the global estimate.

    Returns
    -------
    dict
        Detected spikes, adaptive thresholds, noise estimates,
        diagnostic information, and a spike DataFrame.
    """
    polarity = str(polarity).strip().lower()
    allowed_polarities = {"positive", "negative", "both"}
    if polarity not in allowed_polarities:
        raise ValueError("polarity must be 'positive', 'negative', or 'both'.")
    if threshold_factor <= 0:
        raise ValueError("threshold_factor must be positive.")
    if upper_threshold_factor is not None:
        if upper_threshold_factor <= threshold_factor:
            raise ValueError(
                "upper_threshold_factor must be greater than threshold_factor."
            )
    if shadow_period_ms <= 0:
        raise ValueError("shadow_period_ms must be positive.")
    if adaptive_window_s <= 0:
        raise ValueError("adaptive_window_s must be positive.")
    if adaptive_step_s <= 0:
        raise ValueError("adaptive_step_s must be positive.")
    time = np.asarray(result["time"], dtype=float)
    filtered_signal = np.asarray(result["filtered_signal"], dtype=float)
    artifact_mask = np.asarray(result["artifact_mask"], dtype=bool)
    sampling_rate = float(result["sampling_rate"])
    if not len(time) == len(filtered_signal) == len(artifact_mask):
        raise ValueError(
            "Time, filtered signal, and artifact mask must have equal lengths."
        )
    if len(time) < 10:
        raise ValueError("The recording contains too few samples.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("The time array must be strictly increasing.")
    valid_samples = np.isfinite(filtered_signal) & ~artifact_mask
    clean_values = filtered_signal[valid_samples]
    if len(clean_values) < 10:
        raise ValueError("Too few valid samples remain for detection.")
    global_center, global_noise_sigma = estimate_background_noise(
        clean_values, quantile=noise_quantile
    )
    if not np.isfinite(global_noise_sigma) or global_noise_sigma <= 0:
        raise ValueError("Could not estimate the global noise level.")
    recording_start = time[0]
    recording_stop = time[-1]
    window_starts = np.arange(recording_start, recording_stop, adaptive_step_s)
    if len(window_starts) == 0:
        window_starts = np.array([recording_start])
    window_centers = []
    local_centers = []
    local_noise_values = []
    valid_counts = []
    expected_samples_per_window = max(1, int(adaptive_window_s * sampling_rate))
    minimum_valid_samples = max(
        10, int(expected_samples_per_window * minimum_valid_fraction)
    )
    for window_start in window_starts:
        window_stop = min(window_start + adaptive_window_s, recording_stop)
        left_index = np.searchsorted(time, window_start, side="left")
        right_index = np.searchsorted(time, window_stop, side="right")
        window_valid = valid_samples[left_index:right_index]
        window_values = filtered_signal[left_index:right_index][window_valid]
        window_center_time = (window_start + window_stop) / 2
        window_centers.append(window_center_time)
        valid_counts.append(len(window_values))
        if len(window_values) < minimum_valid_samples:
            local_centers.append(np.nan)
            local_noise_values.append(np.nan)
            continue
        local_center, local_sigma = estimate_background_noise(
            window_values, quantile=noise_quantile
        )
        local_centers.append(local_center)
        local_noise_values.append(local_sigma)
    window_centers = np.asarray(window_centers, dtype=float)
    local_centers = np.asarray(local_centers, dtype=float)
    local_noise_values = np.asarray(local_noise_values, dtype=float)
    valid_counts = np.asarray(valid_counts, dtype=int)
    valid_noise_windows = np.isfinite(local_noise_values) & (local_noise_values > 0)
    if not np.any(valid_noise_windows):
        local_noise_values[:] = global_noise_sigma
    else:
        local_noise_values = np.interp(
            window_centers,
            window_centers[valid_noise_windows],
            local_noise_values[valid_noise_windows],
        )
    valid_center_windows = np.isfinite(local_centers)
    if not np.any(valid_center_windows):
        local_centers[:] = global_center
    else:
        local_centers = np.interp(
            window_centers,
            window_centers[valid_center_windows],
            local_centers[valid_center_windows],
        )
    raw_local_noise_values = local_noise_values.copy()
    if noise_scale_bounds is not None:
        minimum_noise_scale, maximum_noise_scale = noise_scale_bounds
        if minimum_noise_scale <= 0 or maximum_noise_scale <= minimum_noise_scale:
            raise ValueError(
                "noise_scale_bounds must contain positive increasing values."
            )
        local_noise_values = np.clip(
            local_noise_values,
            global_noise_sigma * minimum_noise_scale,
            global_noise_sigma * maximum_noise_scale,
        )
    local_noise_trace = np.interp(time, window_centers, local_noise_values)
    local_center_trace = np.interp(time, window_centers, local_centers)
    centered_filtered = filtered_signal - local_center_trace
    lower_threshold_trace = threshold_factor * local_noise_trace
    if upper_threshold_factor is None:
        upper_threshold_trace = None
    else:
        upper_threshold_trace = upper_threshold_factor * local_noise_trace
    if prominence_factor is None:
        prominence_trace = None
    else:
        prominence_trace = prominence_factor * local_noise_trace
    if polarity == "positive":
        detection_signal = centered_filtered.copy()
    elif polarity == "negative":
        detection_signal = -centered_filtered
    else:
        detection_signal = np.abs(centered_filtered)
    detection_signal[~valid_samples] = -np.inf
    minimum_distance_samples = max(1, int(sampling_rate * shadow_period_ms / 1000))
    height_range = (lower_threshold_trace, upper_threshold_trace)
    if prominence_trace is None:
        prominence_range = None
    else:
        prominence_range = (prominence_trace, None)
    spike_indices, properties = find_peaks(
        detection_signal,
        height=height_range,
        distance=minimum_distance_samples,
        prominence=prominence_range,
    )
    spike_times = time[spike_indices]
    signed_amplitudes = centered_filtered[spike_indices]
    absolute_amplitudes = np.abs(signed_amplitudes)
    local_noise_at_spikes = local_noise_trace[spike_indices]
    amplitudes_in_noise_units = absolute_amplitudes / local_noise_at_spikes
    lower_threshold_at_spikes = lower_threshold_trace[spike_indices]
    if upper_threshold_trace is None:
        upper_threshold_at_spikes = np.full(len(spike_indices), np.nan)
    else:
        upper_threshold_at_spikes = upper_threshold_trace[spike_indices]
    spike_polarities = np.where(signed_amplitudes >= 0, "positive", "negative")
    spikes_df = pd.DataFrame(
        {
            "spike_number": np.arange(1, len(spike_indices) + 1),
            "sample_index": spike_indices,
            "time_seconds": spike_times,
            "signed_amplitude": signed_amplitudes,
            "absolute_amplitude": absolute_amplitudes,
            "local_noise_sigma": local_noise_at_spikes,
            "lower_threshold": lower_threshold_at_spikes,
            "upper_threshold": upper_threshold_at_spikes,
            "amplitude_noise_units": amplitudes_in_noise_units,
            "polarity": spike_polarities,
            "polarity_mode": polarity,
        }
    )
    noise_windows_df = pd.DataFrame(
        {
            "window_center_seconds": window_centers,
            "valid_samples": valid_counts,
            "local_center": local_centers,
            "raw_local_noise_sigma": raw_local_noise_values,
            "used_local_noise_sigma": local_noise_values,
            "relative_to_global": local_noise_values / global_noise_sigma,
        }
    )
    return {
        "time": time,
        "filtered_signal": filtered_signal,
        "centered_filtered_signal": centered_filtered,
        "artifact_mask": artifact_mask,
        "valid_samples": valid_samples,
        "sampling_rate": sampling_rate,
        "polarity_mode": polarity,
        "noise_sigma": global_noise_sigma,
        "global_noise_sigma": global_noise_sigma,
        "global_center": global_center,
        "local_noise_trace": local_noise_trace,
        "local_center_trace": local_center_trace,
        "noise_windows_df": noise_windows_df,
        "adaptive_window_s": adaptive_window_s,
        "adaptive_step_s": adaptive_step_s,
        "threshold_factor": threshold_factor,
        "upper_threshold_factor": upper_threshold_factor,
        "prominence_factor": prominence_factor,
        "lower_threshold_trace": lower_threshold_trace,
        "upper_threshold_trace": upper_threshold_trace,
        "prominence_trace": prominence_trace,
        "threshold": np.median(lower_threshold_trace[valid_samples]),
        "lower_threshold": np.median(lower_threshold_trace[valid_samples]),
        "upper_threshold": (
            None
            if upper_threshold_trace is None
            else np.median(upper_threshold_trace[valid_samples])
        ),
        "shadow_period_ms": shadow_period_ms,
        "spike_indices": spike_indices,
        "spike_times": spike_times,
        "signed_amplitudes": signed_amplitudes,
        "absolute_amplitudes": absolute_amplitudes,
        "amplitudes_in_noise_units": amplitudes_in_noise_units,
        "spike_polarities": spike_polarities,
        "properties": properties,
        "spikes_df": spikes_df,
    }
