#!/usr/bin/env python3
"""Prepare an immutable ABC diagnostic matrix or run one allocated cell."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kinesync.external_abc import diagnostic_cases, require_gpu_allocation

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--cell", type=int, choices=range(4))
    args = parser.parse_args()
    destination = args.root.resolve()
    if args.prepare:
        destination.mkdir(parents=True, exist_ok=False)
        snapshot = destination / "source_snapshot"
        snapshot.mkdir()
        paths = [Path(__file__).resolve(), ROOT / "scripts/run_abc_external_eval.py",
                 ROOT / "src/kinesync/external_abc.py", ROOT / "scripts/run_abc_diagnostic.sbatch"]
        sources = []
        for path in paths:
            shutil.copy2(path, snapshot / path.name)
            sources.append({"path": str(path), "sha256": digest(path)})
        payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "evidence_stage": "backend_horizon_development_not_method_benefit",
                   "policy_rng_scope": "one_independent_episode_per_process",
                   "sources": sources, "cases": diagnostic_cases()}
        (destination / "matrix.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"matrix": str(destination), "episodes": 12}))
        return

    job_id = require_gpu_allocation()
    matrix = json.loads((destination / "matrix.json").read_text())
    for source in matrix["sources"]:
        if digest(Path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"Source changed since matrix declaration: {source['path']}")
    selected = [case for case in matrix["cases"] if case["cell"] == args.cell]
    if len(selected) != 3:
        raise ValueError("The declared diagnostic cell must contain three episodes")
    receipt = {"slurm_job_id": job_id, "cell": args.cell, "runs": []}
    for case in selected:
        command = [sys.executable, str(ROOT / "scripts/run_abc_external_eval.py"),
                   "--run-root", str(destination), "--run-id", case["id"],
                   "--worlds", "1", "--seed", str(case["seed"]),
                   "--policy-seed", str(case["policy_seed"]),
                   "--chunks", str(case["chunks"]), "--physics", case["physics"]]
        print("RUN " + json.dumps(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        receipt["runs"].append({"case": case, "argv": command, "returncode": result.returncode})
        (destination / f"cell_{args.cell}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f"Cell {args.cell} completed all three episodes", flush=True)


if __name__ == "__main__":
    main()
