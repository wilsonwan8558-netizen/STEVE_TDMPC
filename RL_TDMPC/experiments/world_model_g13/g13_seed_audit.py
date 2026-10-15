"""Audit and freeze G13 checkpoint-selection development seeds before rollout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/world_model_force_aware_planning_g13"
SEEDS = set(range(350000, 350030))
ROOTS = [RL / "logs", Path("/home/bizon/Desktop/stEVE_data/experiments")]
SEED_PATH = re.compile(r"(?:^|[/_-])seed[_-]?(\d{5,6})(?:$|[/_.-])")
SEED_KEYS = {"seed", "seeds", "seed_start", "seed_end", "seed_range",
             "base_seed", "episode_seed", "eval_seed", "environment_seed",
             "training_seed", "ordered_seeds", "development_seeds",
             "confirmation_seeds"}


def seed_key(key: str) -> bool:
    low = key.lower()
    return low in SEED_KEYS or low.endswith(("_seed", "_seeds"))


def maybe_seed(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in SEEDS else None
    if isinstance(value, str) and value.isdecimal():
        number = int(value)
        return number if number in SEEDS else None
    return None


def inspect_json(value, location: str, artifact: str, findings: list[dict]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{location}.{key}"
            if seed_key(key):
                values = item if isinstance(item, list) else [item]
                for entry in values:
                    found = maybe_seed(entry)
                    if found is not None:
                        findings.append({"seed": found, "artifact": artifact, "field": here})
            inspect_json(item, here, artifact, findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                inspect_json(item, f"{location}[{index}]", artifact, findings)


def main() -> None:
    audit_path = OUT / "checkpoint_selection_seed_audit.json"
    manifest_path = OUT / "checkpoint_selection_seed_manifest.json"
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError("G13 checkpoint-selection seed audit already frozen")
    findings: list[dict] = []
    counts = {"json": 0, "csv_with_seed_columns": 0, "seed_named_paths": 0}
    for root in ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            path_match = SEED_PATH.search(str(path))
            if path_match:
                counts["seed_named_paths"] += 1
                found = maybe_seed(path_match.group(1))
                if found is not None:
                    findings.append({"seed": found, "artifact": str(path), "field": "filename"})
            if path.suffix == ".json":
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                counts["json"] += 1
                inspect_json(document, "$", str(path), findings)
            elif path.suffix == ".csv":
                try:
                    with path.open(newline="", encoding="utf-8") as stream:
                        reader = csv.DictReader(stream)
                        columns = [key for key in (reader.fieldnames or [])
                                   if key and seed_key(key)]
                        if columns:
                            counts["csv_with_seed_columns"] += 1
                            for line, row in enumerate(reader, start=2):
                                for column in columns:
                                    found = maybe_seed(row[column])
                                    if found is not None:
                                        findings.append({"seed": found, "artifact": str(path),
                                                         "field": f"CSV:{column}:{line}"})
                except (UnicodeDecodeError, csv.Error):
                    continue
    report = {
        "phase": "G13",
        "purpose": "fresh development cohort for navigation checkpoint selection only",
        "proposed_seed_range": [min(SEEDS), max(SEEDS)],
        "scope": "structured JSON seed fields, CSV seed columns, seed-named paths in branch logs and archived experiments",
        "inspected_roots": [str(path) for path in ROOTS],
        "counts": counts,
        "overlaps": findings,
        "status": "PASS_UNUSED_IN_SCANNED_ARTIFACTS" if not findings else "STOP_OVERLAP",
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if findings:
        raise RuntimeError(f"G13 development seed overlap: {findings[:10]}")
    manifest = {
        "status": "FROZEN_BEFORE_G13_CHECKPOINT_EVALUATION",
        "ordered_seeds": sorted(SEEDS),
        "purpose": "task-only navigation checkpoint selection from 240k, 260k, 280k, 300k",
        "one_reused_scene_per_checkpoint": True,
        "seed_reset_order": "ascending",
        "force_planning_outcomes_excluded_from_selection": True,
        "audit": audit_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"], counts, flush=True)


if __name__ == "__main__":
    main()
