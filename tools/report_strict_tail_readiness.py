#!/usr/bin/env python3
"""Report strict_zero_tail readiness for bundled turbine resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _check_file(path: Path) -> tuple[bool, str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return False, "invalid_yaml"
    if "V" not in data or "cf" not in data:
        return False, "missing_V_or_cf"
    try:
        u = np.array(list(map(float, data["V"])), dtype=np.float64)
        p = np.array(list(map(float, data["cf"])), dtype=np.float64)
    except Exception:
        return False, "non_numeric_values"
    if u.ndim != 1 or p.ndim != 1 or u.size != p.size or u.size < 2:
        return False, "invalid_shapes"
    if np.any(np.diff(u) <= 0):
        return False, "non_monotone_wind_speed"
    if np.any(p < 0) or np.any(p > 1):
        return False, "cf_out_of_bounds"
    if not np.isclose(float(p[-1]), 0.0, atol=1e-12):
        return False, "tail_not_zero"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resources-dir", default="cleo/resources", help="Path to resources dir")
    ap.add_argument("--json-out", default="", help="Optional JSON output path")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when at least one turbine is not strict-ready",
    )
    args = ap.parse_args()

    resources_dir = Path(args.resources_dir)
    files = sorted(resources_dir.glob("*.yml"))
    files = [p for p in files if p.stem not in {"clc_codes", "cost_assumptions"}]

    results = []
    for path in files:
        ok, reason = _check_file(path)
        results.append({"turbine_id": path.stem, "strict_ready": ok, "reason": reason})

    strict_ready = [r["turbine_id"] for r in results if r["strict_ready"]]
    not_ready = [r for r in results if not r["strict_ready"]]

    payload = {
        "resources_dir": str(resources_dir),
        "total": len(results),
        "strict_ready_count": len(strict_ready),
        "not_ready_count": len(not_ready),
        "strict_ready": strict_ready,
        "not_ready": not_ready,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and not_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
