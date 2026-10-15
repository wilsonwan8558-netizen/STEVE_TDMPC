"""Check a proposed fresh G12.1 development cohort before any rollout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from g121_planner import OUT

SEEDS = list(range(112000, 112030))
ROOTS = [
    OUT.parents[1],  # branch RL_TDMPC/logs
    Path("/home/bizon/Desktop/stEVE_data/experiments"),
]
SEED_NAME = re.compile(r"(?:^|[_/])seed[_-]?(\d{5,6})(?:$|[_/.])")


def candidate_number(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in SEEDS else None
    if isinstance(value, str) and value.isdecimal():
        number = int(value)
        return number if number in SEEDS else None
    return None


def inspect_json(value, path: str, artifact: str, findings: list[dict]):
    if isinstance(value, dict):
        for key, item in value.items():
            low = key.lower()
            key_is_seed = low in {"seed", "seeds", "seed_start", "seed_end", "seed_range"} or low.endswith("_seed") or low.endswith("_seeds")
            if key_is_seed:
                values = item if isinstance(item, list) else [item]
                for entry in values:
                    number = candidate_number(entry)
                    if number is not None:
                        findings.append({"seed": number, "artifact": artifact,
                                         "field": f"{path}.{key}"})
            inspect_json(item, f"{path}.{key}", artifact, findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                inspect_json(item, f"{path}[{index}]", artifact, findings)


def main() -> None:
    if (OUT / "development_seed_manifest.json").exists() or (OUT / "development_seed_audit.json").exists():
        raise FileExistsError("G12.1 development seed audit already exists")
    findings = []
    counts = {"json": 0, "csv_with_seed_column": 0, "seed_named_paths": 0}
    inspected_roots = []
    for root in ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(root)
        inspected_roots.append(str(root.resolve()))
        for path in root.rglob("*"):
            if not path.is_file() or OUT in path.parents:
                continue
            match = SEED_NAME.search(str(path))
            if match:
                counts["seed_named_paths"] += 1
                number = candidate_number(match.group(1))
                if number is not None:
                    findings.append({"seed": number, "artifact": str(path), "field": "filename"})
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
                        headers = reader.fieldnames or []
                        columns = [key for key in headers if key and (key.lower() in {"seed", "episode_seed", "eval_seed", "environment_seed", "training_seed"} or key.lower().endswith("_seed"))]
                        if columns:
                            counts["csv_with_seed_column"] += 1
                            for row_number, row in enumerate(reader, start=2):
                                for column in columns:
                                    number = candidate_number(row[column])
                                    if number is not None:
                                        findings.append({"seed": number, "artifact": str(path),
                                                         "field": f"CSV:{column}:{row_number}"})
                except (UnicodeDecodeError, csv.Error):
                    continue
    report = {
        "phase": "G12.1",
        "purpose": "pre-rollout freshness check for proposed development episode seeds",
        "proposed_seed_range": [SEEDS[0], SEEDS[-1]],
        "scope": "structured JSON seed fields, CSV seed columns, and seed-named paths in branch RL_TDMPC/logs and prior stEVE_data/experiments; training-step counters are not episode seeds",
        "inspected_roots": inspected_roots,
        "counts": counts,
        "overlaps": findings,
        "status": "PASS_UNUSED_IN_SCANNED_ARTIFACTS" if not findings else "STOP_OVERLAP",
    }
    with (OUT / "development_seed_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if findings:
        raise RuntimeError(f"G12.1 development seed overlap: {findings[:10]}")
    with (OUT / "development_seed_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"status":"FROZEN_BEFORE_G121_CLOSED_LOOP_DEVELOPMENT",
                   "ordered_seeds":SEEDS,"seed_count":len(SEEDS),
                   "arms":"B0 lambda=0 and B1 at each of the three fixed nonzero lambda values",
                   "scene_lifecycle":"one reused Generic scene per arm, ordered resets",
                   "audit":"development_seed_audit.json"},stream,indent=2,sort_keys=True)
        stream.write("\n")
    print("G121_DEVELOPMENT_SEEDS_FROZEN",len(SEEDS),counts,flush=True)


if __name__ == "__main__":
    main()
