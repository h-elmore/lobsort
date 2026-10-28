"""Deterministic synthetic recording for trying the interface without lab data."""

import numpy as np


def demo_channels(duration=90, fs=5000):
    rng = np.random.default_rng(2026)
    t = np.arange(int(duration * fs)) / fs
    target = rng.normal(0, 0.15, len(t))
    reference = rng.normal(0, 0.12, len(t))
    for time in np.arange(0.5, duration - 0.1, 0.14):
        index = int(time * fs)
        width = np.arange(-15, 16) / fs
        target[index - 15 : index + 16] += 3 * np.exp(
            -((width / 0.0007) ** 2)
        ) - 0.7 * np.exp(-(((width - 0.0014) / 0.0008) ** 2))
    for time in np.arange(0.7, duration - 0.1, 0.23):
        index = int(time * fs)
        width = np.arange(-15, 16) / fs
        target[index - 15 : index + 16] += 1.5 * np.exp(-((width / 0.001) ** 2))
    active = (t % 2) < 0.7
    reference += active * 0.8 * np.sin(2 * np.pi * 180 * t)
    return [
        dict(name="Red", time=t, values=target, units="1.0 mV"),
        dict(name="Green", time=t, values=reference, units="1.0 mV"),
    ]
