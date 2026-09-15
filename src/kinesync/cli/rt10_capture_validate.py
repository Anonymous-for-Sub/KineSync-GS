"""Validate local RT10 capture evidence without any hardware interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from kinesync.capture.contracts import (
    load_capture_schedule,
    materialize_formal_matrix_receipt,
    validate_camera_calibration,
    validate_capture_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local RT10 observation capture session")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", help="Explicit destination for the formal matrix receipt")
    args = parser.parse_args(argv)
    schedule = load_capture_schedule(args.schedule)
    calibration = validate_camera_calibration(args.calibration)
    manifest = validate_capture_manifest(args.manifest, schedule, calibration)
    receipt = materialize_formal_matrix_receipt(schedule, calibration, manifest)
    if args.receipt:
        target = Path(args.receipt)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
