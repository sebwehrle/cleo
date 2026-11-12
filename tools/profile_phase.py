"""Run one CLEO capacity-factor phase, for memory and runtime measurement.

Maintainer tooling. Run each phase in its own process under ``/usr/bin/time -v``
to get that phase's peak RSS ("Maximum resident set size") and wall time::

    /usr/bin/time -v python tools/profile_phase.py WORKDIR PHASE [--backend threads --workers 4]

Phases:

- ``build``: ``Atlas.build()``; run on a workdir with GWA rasters but no stores.
- ``build_noop``: ``Atlas.build()`` on an already-built workdir (idempotent rebuild).
- ``compute``: compute capacity factors and evaluate them under the Atlas backend.
- ``materialize``: compute capacity factors and ``.materialize()`` them.
- ``to_raster``: the reporter's path, ``.data.rio.to_raster()`` with no Dask
  configuration, i.e. Dask's default threaded scheduler.

The compute phases skip ``build()`` (which loads all GWA rasters) by marking the
existing stores ready, so the peak reflects the phase alone.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from cleo import Atlas
from cleo.dask_utils import compute

COUNTRY = "AUT"
CRS = "EPSG:3035"
TURBINE = "Vestas.V162.7200"
PHASES = ("build", "build_noop", "compute", "materialize", "to_raster")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--backend", default="serial")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    atlas = Atlas(args.workdir, country=COUNTRY, crs=CRS, compute_backend=args.backend, compute_workers=args.workers)
    atlas.configure_turbines([TURBINE])

    start = time.perf_counter()
    if args.phase in ("build", "build_noop"):
        atlas.build()
    else:
        if not (atlas.wind_store_path.exists() and atlas.landscape_store_path.exists()):
            sys.exit("Stores missing: run the 'build' phase first.")
        atlas._canonical_ready = True  # skip build(): it loads all GWA rasters before its identity check
        result = atlas.wind.compute(
            "capacity_factors", turbines=[TURBINE], method="rotor_node_average", air_density=True
        )
        if args.phase == "compute":
            compute(result.data, backend=atlas.compute_backend, num_workers=atlas.compute_workers)
        elif args.phase == "materialize":
            result.materialize()
        else:
            with tempfile.TemporaryDirectory() as tmp:
                result.data.squeeze().rio.to_raster(Path(tmp) / "cf.tif")
    print(f"phase={args.phase} backend={args.backend} workers={args.workers} seconds={time.perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
