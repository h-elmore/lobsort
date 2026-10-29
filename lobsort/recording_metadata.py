"""Recording identities and user-facing nerve names, separate from signal keys."""
from __future__ import annotations
import re


def validate_channel_names(sources, names):
    """Validate the full displayed channel list before applying any aliases."""
    result = {}
    seen = set()
    for source in sources:
        name = str(names.get(source, source)).strip()
        if not name:
            raise ValueError("Every channel needs a name. Keep the recorded name if unsure.")
        if any(ord(char) < 32 for char in name):
            raise ValueError("Channel names cannot contain control characters.")
        if name.casefold() in seen:
            raise ValueError(f"Channel name {name!r} is repeated. Give each channel a unique name.")
        seen.add(name.casefold())
        if name != source:
            result[source] = name
    return result


def file_stem(value, fallback="spikes"):
    """A safe, recognizable filename component on macOS, Windows, and Linux."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    value = value[:120].rstrip(" .") or fallback
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        value = "_" + value
    return value

