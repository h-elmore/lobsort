"""Fast, label-independent timing features from an immutable review snapshot.

No target labels or classifier predictions enter these features. Relationships
between these measurements and target labels are learned inside the classifier.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

EXCLUDED = {"noise", "uncertain", "unlabeled", "nan", ""}


def same_label_neighbors(times, labels, half_window):
    """Count other same-label candidates in the closed +/- window in O(N log N)."""
    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if not np.isfinite(half_window) or half_window < 0:
        raise ValueError("Sequence support window must be finite and nonnegative.")
    counts = np.zeros(len(times), dtype=int)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        group = np.sort(times[positions])
        counts[positions] = (
            np.searchsorted(group, times[positions] + half_window, side="right")
            - np.searchsorted(group, times[positions] - half_window, side="left")
            - 1
        )
    return counts


def reference_snapshot(review, target, channel, label):
    """Read accepted events directly from an open Review, without saving CSV."""
    if channel == target:
        raise ValueError(
            "Choose a timing reference on a different channel from the target."
        )
    if not label or label.strip().lower() in EXCLUDED:
        raise ValueError("Choose a labeled neuron as the timing reference.")
    if channel not in review.processed or channel not in review.tables:
        raise ValueError(
            "The timing reference channel has no labeled spikes in this review."
        )
    # Prevent known direct/indirect dependence on the target being classified.
    dependencies = review.metadata.get("timing_references", {})
    visited = set()
    ancestor = channel
    while ancestor in dependencies and ancestor not in visited:
        visited.add(ancestor)
        ancestor = dependencies[ancestor].get("channel")
        if ancestor == target:
            raise ValueError(
                "This timing reference was itself classified using the target channel. Choose an independent reference to avoid circular support."
            )
    df = review.tables[channel]
    selected = df.loc[df.accepted_as_spike & df.manual_label.eq(label)]
    times = np.unique(
        pd.to_numeric(selected.time_seconds, errors="coerce").to_numpy(dtype=float)
    )
    times = times[np.isfinite(times)]
    data = review.processed[channel]
    bounds = [float(data["time"][0]), float(data["time"][-1])]
    times = times[(times >= bounds[0]) & (times <= bounds[1])]
    if len(times) < 3:
        raise ValueError(
            "The reference unit needs at least three accepted spikes. Sort or label that channel first."
        )
    snapshot = dict(
        channel=channel,
        label=label,
        times=times.tolist(),
        recording_bounds=bounds,
        accepted_count=len(times),
        manual_count=int(
            selected.get("edit_source", pd.Series(dtype=str)).eq("manual").sum()
        ),
        review_audit_length=len(review.audit),
    )
    snapshot["fingerprint"] = reference_fingerprint(snapshot)
    return snapshot


def reference_fingerprint(snapshot):
    payload = {
        key: snapshot[key] for key in ("channel", "label", "times", "recording_bounds")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _lags(queries, events, max_lag):
    """Previous/next lags, with no extrapolation farther than max_lag."""
    right = np.searchsorted(events, queries, side="right")
    left = np.searchsorted(events, queries, side="left")
    previous = np.full(len(queries), np.nan)
    following = previous.copy()
    valid = right > 0
    previous[valid] = queries[valid] - events[right[valid] - 1]
    valid = left < len(events)
    following[valid] = events[left[valid]] - queries[valid]
    previous[previous > max_lag] = np.nan
    following[following > max_lag] = np.nan
    return previous, following


def reference_features(candidate_times, snapshot, window=1.0, burst_gap=0.15):
    """Vectorized spike, local-activity and burst timing for one labeled unit.

    Future events are allowed: this is offline sorting. Outside recording overlap,
    or when there is no accepted reference event within the window, timing is
    missing rather than a fabricated phase. Incomplete reference labels remain
    provisional measurements, never assumed biological ground truth.
    """
    if (
        not np.isfinite(window)
        or window <= 0
        or not np.isfinite(burst_gap)
        or burst_gap <= 0
    ):
        raise ValueError(
            "Reference timing window and burst gap must be positive and finite."
        )
    t = np.asarray(candidate_times, dtype=float)
    ref = np.unique(np.asarray(snapshot["times"], dtype=float))
    if len(ref) < 3 or not np.all(np.isfinite(ref)):
        raise ValueError(
            "The reference unit needs at least three finite accepted spike times."
        )
    bounds = snapshot["recording_bounds"]
    covered = (t >= bounds[0]) & (t <= bounds[1])
    previous, following = _lags(t, ref, window)
    nearest = np.where(
        np.isnan(previous) | (following < previous), -following, previous
    )
    intervals = previous + following
    between = np.isfinite(intervals) & (intervals > 0)
    phase = np.full(len(t), np.nan)
    phase[between] = previous[between] / intervals[between]
    features = dict(
        unit_since_previous_s=previous,
        unit_until_next_s=following,
        unit_nearest_signed_lag_s=nearest,
        unit_local_isi_s=intervals,
        unit_interspike_phase=phase,
    )
    for title, radius in [
        ("near", min(0.05, window)),
        ("support", min(0.2, window)),
        ("wide", window),
    ]:
        features[f"unit_count_{title}"] = (
            np.searchsorted(ref, t + radius, side="right")
            - np.searchsorted(ref, t - radius, side="left")
        ).astype(float)
    starts = np.r_[0, np.flatnonzero(np.diff(ref) > burst_gap) + 1]
    stops = np.r_[starts[1:] - 1, len(ref) - 1]
    # Singletons are not bursts. Burst boundaries use accepted spikes only.
    reliable = stops > starts
    onsets = ref[starts[reliable]]
    offsets = ref[stops[reliable]]
    since_onset, until_onset = _lags(t, onsets, window)
    since_offset, until_offset = _lags(t, offsets, window)
    features.update(
        unit_since_burst_onset_s=since_onset,
        unit_until_burst_onset_s=until_onset,
        unit_since_burst_offset_s=since_offset,
        unit_until_burst_offset_s=until_offset,
    )
    burst_id = np.searchsorted(onsets, t, side="right") - 1
    inside = np.zeros(len(t), bool)
    if len(offsets):
        possible = burst_id >= 0
        inside[possible] = t[possible] <= offsets[burst_id[possible]]
    features["unit_inside_burst"] = inside.astype(float)
    available = covered & (np.isfinite(previous) | np.isfinite(following))
    for value in features.values():
        value[~available] = np.nan
    features["unit_timing_available"] = available.astype(float)
    return pd.DataFrame(features), onsets, offsets


def unit_context_features(result, snapshot, settings):
    """Waveform geometry plus target timing and labeled cross-channel timing.

    Waveforms use bounded chunks. Reference features only search sorted event
    times and never smooth or iterate through the full reference signal.
    """
    t = np.asarray(result["time"], dtype=float)
    signal = np.asarray(result["centered_filtered_signal"], dtype=float)
    indices = np.asarray(result["spike_indices"], dtype=int)
    fs = float(result["sampling_rate"])
    times = t[indices]
    before = max(1, round(settings["context_waveform_before_ms"] * fs / 1000))
    after = max(1, round(settings["context_waveform_after_ms"] * fs / 1000))
    width = before + after + 1
    waves = np.full((len(indices), width), np.nan)
    valid = np.zeros(len(indices), bool)
    shifts = np.arange(-before, after + 1)
    for begin in range(0, len(indices), 4096):
        end = min(len(indices), begin + 4096)
        locations = indices[begin:end, None] + shifts
        block = signal[np.clip(locations, 0, len(signal) - 1)]
        okay = (
            (locations[:, 0] >= 0)
            & (locations[:, -1] < len(signal))
            & np.all(np.isfinite(block), axis=1)
        )
        block = block - np.median(block[:, : max(1, before // 2)], axis=1)[:, None]
        block[~okay] = np.nan
        waves[begin:end] = block
        valid[begin:end] = okay
    filled = np.nan_to_num(waves)
    norm = np.linalg.norm(filled, axis=1)
    normalized = np.full_like(waves, np.nan)
    good = valid & (norm > 0)
    normalized[good] = waves[good] / norm[good, None]
    positive = np.max(filled, axis=1)
    negative = np.min(filled, axis=1)
    abs_peak = np.max(abs(filled), axis=1)
    order = np.argsort(times)
    prev = np.full(len(times), np.nan)
    nxt = prev.copy()
    prev[order[1:]] = np.diff(times[order])
    nxt[order[:-1]] = np.diff(times[order])
    sorted_times = times[order]
    radius = settings["sequence_support_window_s"]
    density = (
        np.searchsorted(sorted_times, times + radius, side="right")
        - np.searchsorted(sorted_times, times - radius, side="left")
        - 1
    )
    frame = pd.DataFrame(
        dict(
            candidate_position=np.arange(len(indices)),
            sample_index=indices,
            time_seconds=times,
            positive_peak=positive,
            negative_peak=negative,
            absolute_peak=abs_peak,
            peak_to_peak=positive - negative,
            waveform_energy=np.sum(filled**2, axis=1),
            waveform_area=np.sum(abs(filled), axis=1),
            peak_offset_ms=(np.argmax(abs(filled), axis=1) - before) / fs * 1000,
            previous_isi_ms=prev * 1000,
            next_isi_ms=nxt * 1000,
            local_candidate_count=density,
            valid_waveform=valid,
        )
    )
    waveform_columns = [f"waveform_{i:03d}" for i in range(width)]
    frame = pd.concat(
        [frame, pd.DataFrame(normalized, columns=waveform_columns)], axis=1
    )
    timing, onsets, offsets = reference_features(
        times,
        snapshot,
        settings["unit_timing_window_s"],
        settings["unit_timing_burst_gap_s"],
    )
    frame = pd.concat([frame, timing], axis=1)
    context_columns = [
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
    ] + list(timing)
    return dict(
        candidate_features_df=frame,
        waveform_feature_columns=waveform_columns,
        context_feature_columns=context_columns,
        candidate_time=t,
        candidate_signal=signal,
        candidate_indices=indices,
        candidate_times=times,
        reference_burst_onsets=onsets,
        reference_burst_offsets=offsets,
    )
