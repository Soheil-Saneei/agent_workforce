"""Append-only per-run storage under ``runs/<run-id>/``.

    manifest.json      resolved config, config hash, code version, grid
    items/<bench>.jsonl  exact items used (so a run never depends on a dataset changing)
    calls.jsonl        every completed model call: request, response, usage, latency, seed
    errors.jsonl       failed calls (not cached; retried on resume)
    trials.jsonl       one record per completed (benchmark, condition, item, replication)
    events.jsonl       run lifecycle (start, stop reason, spend)

``calls.jsonl`` doubles as the call cache: a resumed run replays the protocol and every call
whose key is already logged is served from the log instead of being paid for again.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workforce.design import ExperimentConfig, config_from_dict
from workforce.tasks.base import Item


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records, skipping a torn final line left by an interrupted write."""
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class RunStore:
    def __init__(self, run_dir: Path, manifest: dict[str, Any]):
        self.dir = run_dir
        self.manifest = manifest
        self.calls_path = run_dir / "calls.jsonl"
        self.trials_path = run_dir / "trials.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        self.events_path = run_dir / "events.jsonl"
        self._calls = {r["key"]: r for r in read_jsonl(self.calls_path)}
        self._trials = {r["trial_key"]: r for r in read_jsonl(self.trials_path)}

    @classmethod
    def create(
        cls, run_dir: Path, config: ExperimentConfig, config_text: str | None = None
    ) -> RunStore:
        if (run_dir / "manifest.json").exists():
            raise FileExistsError(f"{run_dir} already holds a run; use 'resume'")
        run_dir.mkdir(parents=True, exist_ok=True)
        from workforce import __version__

        manifest = {
            "run_id": run_dir.name,
            "created_at": now_iso(),
            "config_hash": config.hash,
            "config": config.raw,
            "workforce_version": __version__,
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "conditions": [c.to_dict() for c in config.conditions()],
            "skipped_degenerate_conditions": [c.id for c in config.skipped_conditions()],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if config_text is not None:
            (run_dir / "config.toml").write_text(config_text)
        return cls(run_dir, manifest)

    @classmethod
    def open(cls, run_dir: Path) -> RunStore:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no run at {run_dir} (missing manifest.json)")
        return cls(run_dir, json.loads(manifest_path.read_text()))

    def config(self) -> ExperimentConfig:
        config = config_from_dict(self.manifest["config"])
        if config.hash != self.manifest["config_hash"]:
            raise RuntimeError("stored config does not match its recorded hash")
        return config

    # Items -------------------------------------------------------------------------------
    def items(self, benchmark: str) -> list[Item] | None:
        path = self.dir / "items" / f"{benchmark}.jsonl"
        if not path.exists():
            return None
        return [Item.from_dict(r) for r in read_jsonl(path)]

    def save_items(self, benchmark: str, items: list[Item]) -> None:
        path = self.dir / "items" / f"{benchmark}.jsonl"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(i.to_dict(), ensure_ascii=False) + "\n" for i in items))
        tmp.replace(path)

    # Calls ---------------------------------------------------------------------------------
    def cached_call(self, key: str) -> dict[str, Any] | None:
        return self._calls.get(key)

    def append_call(self, record: dict[str, Any]) -> None:
        _append(self.calls_path, record)
        self._calls[record["key"]] = record

    def calls(self) -> list[dict[str, Any]]:
        return list(self._calls.values())

    @property
    def spent_usd(self) -> float:
        return sum(r.get("cost_usd") or 0.0 for r in self._calls.values())

    # Trials --------------------------------------------------------------------------------
    def has_trial(self, trial_key: str) -> bool:
        return trial_key in self._trials

    def append_trial(self, record: dict[str, Any]) -> None:
        _append(self.trials_path, record)
        self._trials[record["trial_key"]] = record

    def trials(self) -> list[dict[str, Any]]:
        return list(self._trials.values())

    # Diagnostics ---------------------------------------------------------------------------
    def append_error(self, record: dict[str, Any]) -> None:
        _append(self.errors_path, {"ts": now_iso(), **record})

    def append_event(self, event: str, **fields: Any) -> None:
        _append(self.events_path, {"ts": now_iso(), "event": event, **fields})
