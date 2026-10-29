"""Side-effect-free previews using the sorting pipeline's shipped stage sources."""
from __future__ import annotations

import ast
import copy
import numpy as np
from . import engine

DETECTION_KEYS = (
    "threshold_factor", "shadow_period_ms", "polarity", "adaptive_window_s",
    "adaptive_step_s", "noise_quantile", "minimum_valid_fraction", "noise_scale_bounds",
)


def preview_channel(channel, settings):
    """Process a complete channel so adaptive noise and filter edges match sorting.

    No project files, candidate labels, or pipeline state are created or changed.
    Display windows are sliced only after this calculation.
    """
    settings = copy.deepcopy(settings)
    # The filter loop assigns channel_name for each channel; never override it.
    settings.pop("channel_name", None)
    t = np.asarray(channel["time"], dtype=float)
    y = np.asarray(channel["values"], dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 10:
        raise ValueError("Preview requires at least ten matching time and signal samples.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Recording times must be finite and strictly increasing.")
    threshold = settings["artifact_thresholds_mV"].get(
        channel["name"], settings["default_artifact_threshold_mV"]
    )
    for key, value in {
        "artifact threshold": threshold, "filter_low_hz": settings["filter_low_hz"],
        "filter_high_hz": settings["filter_high_hz"],
        **{k: settings[k] for k in DETECTION_KEYS if k not in {"polarity", "noise_scale_bounds"}},
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not np.isfinite(settings["artifact_padding_ms"]) or settings["artifact_padding_ms"] < 0:
        raise ValueError("Artifact padding must be finite and nonnegative.")
    if not 1 <= settings["filter_order"] <= 20:
        raise ValueError("Filter order must be between 1 and 20.")
    if not settings["filter_low_hz"] < settings["filter_high_hz"]:
        raise ValueError("Low cutoff must be below high cutoff.")
    ns = dict(vars(engine))
    settings.update(channels_to_analyze=[channel["name"]], show_raw_preview=False, show_final_plot=False)
    ns.update(channels=[channel], _settings=settings, print=lambda *a, **kw: None)
    tree = ast.parse((engine.STAGES / "filter.py").read_text())
    tree = ast.fix_missing_locations(engine.Overrides(settings).visit(tree))
    exec(compile(tree, "filter preview", "exec"), ns)
    processed = ns["processed_channels"][channel["name"]]
    tree = ast.parse((engine.STAGES / "detection.py").read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, "detection preview", "exec"), ns)
    detection = ns["detect_spikes_processed_channel_adaptive"](
        processed, **{k: settings[k] for k in DETECTION_KEYS}
    )
    return processed, detection
