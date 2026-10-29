"""Lossless review sessions, manual decisions, and atomic project saves."""

from __future__ import annotations
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .recording_metadata import file_stem


def now():
    return datetime.now(timezone.utc).isoformat()


class Review:
    def __init__(self, processed, tables=None, metadata=None):
        self.processed = processed
        self.tables = {}
        self.metadata = metadata or {}
        self.audit = []
        self.undo_stack = []
        self.redo_stack = []
        for channel, table in (tables or {}).items():
            self.set_table(channel, table)

    def channel_name(self, source):
        return self.metadata.get("channel_names", {}).get(source, source)

    def set_table(self, channel, table):
        if channel in self.tables:
            self.undo_stack.clear()
            self.redo_stack.clear()
        df = table.copy().reset_index(drop=True)
        if "manual_label" not in df:
            df["manual_label"] = df.get("predicted_label", "Spike")
        if "accepted_as_spike" not in df:
            df["accepted_as_spike"] = True
        for col in ["automatic_label", "automatic_accepted", "edit_source"]:
            if col not in df:
                df[col] = (
                    df.manual_label
                    if col == "automatic_label"
                    else (
                        df.accepted_as_spike
                        if col == "automatic_accepted"
                        else "automatic"
                    )
                )
        df["channel"] = channel
        df["accepted_as_spike"] = df.accepted_as_spike.astype(bool)
        self.tables[channel] = df

    def apply_confidence_policy(self, channel, include_moderate):
        """Reapply existing context decisions without fitting a model or changing edits."""
        if channel not in self.tables:
            raise ValueError("This channel has no classifier results to update.")
        df = self.tables[channel]
        required = {"acceptance_reason", "same_label_neighbor_count", "valid_waveform"}
        if not required.issubset(df):
            raise ValueError(
                "This channel has no context-classifier decisions. Classify it first."
            )
        before = df.copy(deep=True)
        reasons = ["high_classifier_probability"]
        if include_moderate:
            reasons.append("moderate_probability_with_sequence_support")
        auto = ~df.edit_source.eq("manual")
        accepted = df.acceptance_reason.isin(reasons) & df.valid_waveform.fillna(
            False
        ).astype(bool)
        # Preserve noise, invalid waveforms, optional ISI rejections, and every manual decision.
        df.loc[auto, "accepted_as_spike"] = accepted.loc[auto]
        self.metadata.setdefault("confidence_policies", {})[channel] = (
            "high_and_supported_moderate" if include_moderate else "high_only"
        )
        if df.accepted_as_spike.equals(before.accepted_as_spike):
            return False
        self.undo_stack.append((channel, before, df.copy(deep=True)))
        self.redo_stack.clear()
        self.audit.append(
            dict(
                time=now(),
                action="confidence_policy",
                channel=channel,
                include_moderate=bool(include_moderate),
                accepted_count=int(df.accepted_as_spike.sum()),
            )
        )
        return True

    def change(self, channel, sample_index, action, label="Spike"):
        data = self.processed[channel]
        if not 0 <= sample_index < len(data["time"]):
            raise ValueError("Sample is outside the recording.")
        if action in ["add", "relabel"] and (
            not np.isfinite(data["filtered_signal"][sample_index])
            or data.get("artifact_mask", np.zeros(len(data["time"]), bool))[
                sample_index
            ]
        ):
            raise ValueError(
                "This sample is in an excluded artifact gap. Adjust filtering before adding it."
            )
        if channel not in self.tables:
            self.set_table(
                channel,
                pd.DataFrame(
                    columns=[
                        "sample_index",
                        "time_seconds",
                        "manual_label",
                        "accepted_as_spike",
                    ]
                ),
            )
        df = self.tables[channel]
        matching = df.index[df.sample_index == sample_index]
        if action == "remove" and not len(matching):
            return False
        before = df.copy(deep=True)
        if len(matching):
            row = matching[0]
            if action == "remove":
                df.loc[row, "accepted_as_spike"] = False
            else:
                df.loc[row, ["manual_label", "accepted_as_spike"]] = [label, True]
            df.loc[row, "edit_source"] = "manual"
        else:
            entry = dict(
                channel=channel,
                sample_index=int(sample_index),
                time_seconds=float(data["time"][sample_index]),
                manual_label=label,
                accepted_as_spike=True,
                automatic_label="",
                automatic_accepted=False,
                edit_source="manual",
                acceptance_reason="manually_added",
            )
            df = (
                pd.DataFrame([entry])
                if df.empty
                else pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            )
        self.tables[channel] = df.sort_values("time_seconds").reset_index(drop=True)
        self.undo_stack.append((channel, before, self.tables[channel].copy(deep=True)))
        self.redo_stack.clear()
        self.audit.append(
            dict(
                time=now(),
                action=action,
                channel=channel,
                sample_index=int(sample_index),
                label=label,
            )
        )
        return True

    def undo(self):
        if self.undo_stack:
            item = self.undo_stack.pop()
            self.redo_stack.append(item)
            self.tables[item[0]] = item[1].copy(deep=True)
            self.audit.append(dict(time=now(), action="undo", channel=item[0]))
            return True
        return False

    def redo(self):
        if self.redo_stack:
            item = self.redo_stack.pop()
            self.undo_stack.append(item)
            self.tables[item[0]] = item[2].copy(deep=True)
            self.audit.append(dict(time=now(), action="redo", channel=item[0]))
            return True
        return False

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        manifest = dict(
            version=1, metadata=self.metadata, audit=self.audit, channels=[]
        )
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for i, (name, data) in enumerate(self.processed.items()):
                    record = dict(name=name, units=data.get("units", ""), arrays=[])
                    for key in [
                        "time",
                        "raw_signal",
                        "filtered_signal",
                        "artifact_mask",
                    ]:
                        if key in data:
                            buf = io.BytesIO()
                            np.save(buf, data[key], allow_pickle=False)
                            z.writestr(f"{i}/{key}.npy", buf.getvalue())
                            record["arrays"].append(key)
                    if name in self.tables:
                        z.writestr(
                            f"{i}/spikes.json",
                            self.tables[name].to_json(
                                orient="table", index=False, double_precision=15
                            ),
                        )
                        record["table"] = "json"
                    manifest["channels"].append(record)
                z.writestr("manifest.json", json.dumps(manifest, indent=2))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        processed = {}
        tables = {}
        with zipfile.ZipFile(path) as z:
            m = json.loads(z.read("manifest.json"))
            if m["version"] != 1:
                raise ValueError("Unsupported review project version.")
            for i, r in enumerate(m["channels"]):
                processed[r["name"]] = {
                    "units": r["units"],
                    **{
                        key: np.load(
                            io.BytesIO(z.read(f"{i}/{key}.npy")), allow_pickle=False
                        )
                        for key in r["arrays"]
                    },
                }
                if r.get("table") == "json":
                    tables[r["name"]] = pd.read_json(
                        io.StringIO(z.read(f"{i}/spikes.json").decode()), orient="table"
                    )
                elif r.get("table"):
                    tables[r["name"]] = pd.read_csv(
                        io.BytesIO(z.read(f"{i}/spikes.csv"))
                    )
        obj = cls(processed, tables, m["metadata"])
        obj.audit = m["audit"]
        return obj

    def export(self, folder):
        folder = Path(folder)
        animal = str(self.metadata.get("animal_id", "")).strip()
        if animal:
            folder = folder / file_stem(animal)
        details = folder / "review_details" if animal else folder
        plans = []
        used = set()
        for i, (source, table) in enumerate(self.tables.items()):
            nerve = self.channel_name(source)
            if animal:
                base = file_stem(nerve, "channel")
                safe = base
                suffix = 2
                while safe.casefold() in used:
                    safe = f"{base}_{suffix}"
                    suffix += 1
                used.add(safe.casefold())
                filename = f"{safe}.csv"
            else:
                safe = (
                    str(i)
                    + "_"
                    + "".join(c if c.isalnum() or c in "-_" else "_" for c in nerve)
                )
                filename = f"{safe}_curated_spikes.csv"
            plans.append((source, nerve, table, safe, filename))

        manifest = {
            source: {"name": nerve, "file": filename}
            for source, nerve, _, _, filename in plans
        }
        # Re-exporting this recording updates its CSVs; a different recording or
        # changed nerve-file layout needs a separate destination to keep both exports.
        prior = details / "settings.json"
        if animal and prior.is_file():
            existing = json.loads(prior.read_text())
            if (
                existing.get("animal_id") != animal
                or existing.get("recording") != self.metadata.get("recording")
                or existing.get("segment") != self.metadata.get("segment")
                or any(
                    manifest.get(source) != info
                    for source, info in existing.get("export_files", {}).items()
                )
            ):
                raise ValueError(
                    "This animal folder contains a different recording or nerve-file layout. Choose a different parent folder to keep both exports."
                )
        folder.mkdir(parents=True, exist_ok=True)
        details.mkdir(parents=True, exist_ok=True)
        for source, nerve, table, safe, filename in plans:
            df = table.copy()
            if animal or nerve != source:
                df["source_channel"] = source
                df["channel"] = nerve
            if animal:
                df["animal_id"] = animal
            df.to_csv(details / f"{safe}_all_candidates.csv", index=False)
            accepted = df.loc[df.accepted_as_spike].sort_values("time_seconds").copy()
            accepted = accepted.drop(columns=["spike_number"], errors="ignore")
            accepted.insert(0, "spike_number", np.arange(1, len(accepted) + 1))
            accepted.to_csv(folder / filename, index=False)
        (details / "edit_history.json").write_text(json.dumps(self.audit, indent=2))
        metadata = (
            {**self.metadata, "export_files": manifest} if animal else self.metadata
        )
        (details / "settings.json").write_text(json.dumps(metadata, indent=2))
        return folder
