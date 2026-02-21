"""Turbine and cost ingestion helpers extracted from ``cleo.unify``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from cleo.unification.fingerprint import fingerprint_file


def _load_turbine_yaml(yaml_path: Path) -> dict:
    """Load turbine YAML file.

    Args:
        yaml_path: Path to turbine YAML file.

    Returns:
        Dict with turbine data.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Known non-turbine resource file stems to exclude from default turbine discovery
_NON_TURBINE_RESOURCE_STEMS = {"clc_codes", "cost_assumptions"}


def _default_turbines_from_resources(resources_dir: Path) -> list[str]:
    """Discover default turbines from resources directory.

    Globs for *.yml and *.yaml files, excludes known non-turbine resources,
    and returns sorted list of turbine IDs (file stems).

    Args:
        resources_dir: Path to atlas resources directory.

    Returns:
        Sorted list of turbine IDs (e.g. ["Enercon.E40.500", "Vestas.V90.2000"]).
    """
    if not resources_dir.exists():
        return []

    turbine_ids = []
    for pattern in ("*.yml", "*.yaml"):
        for yaml_path in resources_dir.glob(pattern):
            stem = yaml_path.stem
            if stem not in _NON_TURBINE_RESOURCE_STEMS:
                turbine_ids.append(stem)

    # Remove duplicates (in case both .yml and .yaml exist) and sort
    turbine_ids = sorted(set(turbine_ids))
    return turbine_ids


def _ingest_turbines_and_costs(
    atlas,
    fingerprint_method: str = "path_mtime_size",
) -> tuple[xr.Dataset, list[dict], list[dict]]:
    """Ingest turbine power curves and cost assumptions.

    Args:
        atlas: Atlas instance.
        fingerprint_method: Method for fingerprinting source files.

    Returns:
        Tuple of (dataset, sources, variables) where:
        - dataset: xr.Dataset with power_curve and turbine metadata
        - sources: List of source dicts for manifest
        - variables: List of variable dicts for manifest
    """
    resources_dir = Path(atlas.path) / "resources"

    # Determine turbines: use configured list or discover from resources
    turbines = atlas.turbines_configured
    if turbines is not None:
        turbine_names = list(turbines)
        turbines_mode = "configured"
        missing_configured = [
            str(resources_dir / f"{turbine_id}.yml")
            for turbine_id in turbine_names
            if not (resources_dir / f"{turbine_id}.yml").exists()
        ]
        if missing_configured:
            raise FileNotFoundError(
                "Configured turbine YAML files not found:\n" + "\n".join(missing_configured)
            )
    else:
        turbine_names = _default_turbines_from_resources(resources_dir)
        turbines_mode = "default"
        if not turbine_names:
            raise RuntimeError(
                "No turbines configured and none found in <atlas>/resources. "
                "Either call atlas.configure_turbines([...]) or add turbine YAMLs to resources/."
            )

    sources = []
    variables = []

    # Canonical wind_speed grid
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    power_curves = []
    meta_manufacturer = []
    meta_model = []
    meta_capacity = []
    meta_hub_height = []
    meta_rotor_diameter = []
    meta_commissioning_year = []
    meta_model_key = []
    turbine_ids = []

    for turbine_id in turbine_names:
        yaml_path = resources_dir / f"{turbine_id}.yml"

        data = _load_turbine_yaml(yaml_path)

        # Register source
        sources.append({
            "source_id": f"turbine:{turbine_id}",
            "name": turbine_id,
            "kind": "yaml",
            "path": str(yaml_path),
            "params_json": json.dumps({"turbine_id": turbine_id}),
            "fingerprint": fingerprint_file(yaml_path, fingerprint_method),
        })

        # Extract data
        manufacturer = str(data["manufacturer"])
        model = str(data["model"])
        capacity = float(data["capacity"])
        hub_height = float(data["hub_height"])
        rotor_diameter = float(data["rotor_diameter"])
        commissioning_year = int(data["commissioning_year"])
        model_key = f"{manufacturer}.{model}.{capacity}"

        # Resample power curve to canonical wind_speed grid
        old_u = np.array(list(map(float, data["V"])))
        old_p = np.array(list(map(float, data["cf"])))
        new_p = np.interp(wind_speed, old_u, old_p, left=0.0, right=0.0)

        power_curves.append(new_p)
        meta_manufacturer.append(manufacturer)
        meta_model.append(model)
        meta_capacity.append(capacity)
        meta_hub_height.append(hub_height)
        meta_rotor_diameter.append(rotor_diameter)
        meta_commissioning_year.append(commissioning_year)
        meta_model_key.append(model_key)
        turbine_ids.append(turbine_id)

    # Build dataset if we have turbines
    if turbine_ids:
        # Create turbines bundle source
        turbine_source_ids = [f"turbine:{tid}" for tid in turbine_ids]
        bundle_fingerprint = hashlib.sha256(
            json.dumps(turbine_source_ids, sort_keys=True).encode()
        ).hexdigest()[:16]

        sources.append({
            "source_id": "turbines:bundle",
            "name": "turbines_bundle",
            "kind": "bundle",
            "path": "",
            "params_json": json.dumps({"source_ids": turbine_source_ids}),
            "fingerprint": bundle_fingerprint,
        })

        # Power curve array: (turbine, wind_speed)
        # Use integer indices for turbine dim; store string metadata in attrs JSON
        n_turbines = len(turbine_ids)
        turbine_indices = np.arange(n_turbines, dtype=np.int64)

        pc_data = np.array(power_curves)
        power_curve_da = xr.DataArray(
            pc_data,
            dims=["turbine", "wind_speed"],
            coords={"turbine": turbine_indices, "wind_speed": wind_speed},
            name="power_curve",
        )

        # Build turbine metadata JSON (stored in attrs, NOT as arrays)
        # This avoids Zarr v3 string dtype warnings
        turbines_meta = []
        for i, tid in enumerate(turbine_ids):
            turbines_meta.append({
                "id": tid,
                "manufacturer": meta_manufacturer[i],
                "model": meta_model[i],
                "model_key": meta_model_key[i],
            })

        # Only numeric arrays stored in dataset (no string dtype)
        ds = xr.Dataset({
            "power_curve": power_curve_da,
            "turbine_capacity": ("turbine", np.array(meta_capacity, dtype=np.float64)),
            "turbine_hub_height": ("turbine", np.array(meta_hub_height, dtype=np.float64)),
            "turbine_rotor_diameter": ("turbine", np.array(meta_rotor_diameter, dtype=np.float64)),
            "turbine_commissioning_year": ("turbine", np.array(meta_commissioning_year, dtype=np.int64)),
        })
        ds = ds.assign_coords(turbine=turbine_indices)

        # Store turbine metadata as JSON in attrs (avoids string arrays)
        ds.attrs["cleo_turbines_json"] = json.dumps(
            turbines_meta, separators=(",", ":"), ensure_ascii=False
        )

        # Register variables (only the numeric ones we're storing)
        for var_name in ds.data_vars:
            variables.append({
                "variable_name": var_name,
                "source_id": "turbines:bundle",
                "resampling_method": "none",
                "nodata_policy": "none",
                "dtype": str(ds[var_name].dtype),
            })
    else:
        ds = xr.Dataset()
        ds = ds.assign_coords(wind_speed=wind_speed)

    # Load cost assumptions if available
    cost_path = resources_dir / "cost_assumptions.yml"
    if cost_path.exists():
        sources.append({
            "source_id": "costs:default",
            "name": "cost_assumptions",
            "kind": "yaml",
            "path": str(cost_path),
            "params_json": "{}",
            "fingerprint": fingerprint_file(cost_path, fingerprint_method),
        })

    return ds, sources, variables
