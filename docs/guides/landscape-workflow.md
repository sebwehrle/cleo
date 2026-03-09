# Landscape Guide

## Goal

Compute/add/rasterize landscape variables through the atlas and materialize into `atlas.landscape.data`.

## Canonical pattern

```python
atlas.landscape.compute(metric="distance", source="settlements", name="distance_settlements", if_exists="replace").materialize()

atlas.landscape.add(name="custom_layer", source_path="/path/to/layer.tif").materialize()

atlas.landscape.rasterize(
    shape="/path/to/areas.geojson",
    name="areas_mask",
    column=None,
    all_touched=False,
    if_exists="error",
).materialize()

land_ds = atlas.landscape.data
```

## Knobs you will use most

### Distance compute knobs

- `source` (required): one variable name or list/tuple of names from active landscape store
- `name` (optional): output name(s), default `distance_<source>`
- `if_exists`: `error`, `replace`, `noop`

### Raster add knobs

- `name` (required)
- `source_path` (required)
- `if_exists`: `error`, `replace`, `noop`

### Vector rasterize knobs

- `shape` (required): path-like source or GeoDataFrame
- `name` (required)
- `column` (optional)
- `all_touched` (optional)
- `if_exists`: `error`, `replace`, `noop`

## Methods (explicit)

### 1. `compute(metric="distance", ...)`

How it is computed:

1. Validate source variable exists in active landscape store (`atlas.landscape.data`).
2. Build target mask from source rule: finite and strictly positive cells are target cells.
3. Enforce exact y/x coordinate alignment with `valid_mask`.
4. Enforce projected CRS with meter units (distance in meters only).
5. Run Euclidean distance transform on grid spacing derived from x/y coordinates.
6. Return distance raster (`units=m`) with deterministic method attrs.

What this means operationally:

- Distances are geometric grid distances, not network distance or cost distance.
- Output is masked/defined on the canonical valid grid domain.

### 2. `add(name=..., source_path=...)`

How it is computed:

1. Register raster source in unification source registry for the named variable.
2. Normalize/reproject/align raster source to atlas canonical grid and CRS via unification pipeline.
3. Stage resulting aligned variable as overlay.
4. `materialize()` writes staged variable into active landscape store.

What this means operationally:

- The stored variable is always atlas-grid aligned, even if source raster was not.

### 3. `rasterize(shape=..., ...)`

How it is computed:

1. Register vector source + rasterization parameters (`column`, `all_touched`) in unification source registry.
2. Rasterize onto atlas canonical grid and CRS.
3. Stage rasterized variable.
4. `materialize()` writes staged variable into active landscape store.

What this means operationally:

- `all_touched=False` keeps stricter pixel inclusion; `True` increases inclusion on coarse grids/thin geometries.

### 4. `add_clc_category(...)`

How it is computed:

1. Ensure CLC source cache is prepared/aligned (`build_clc`).
2. Resolve category selection (`all`, one code, or list of codes).
3. Derive/validate output variable name.
4. Route through `add(...)` with CLC-specific params and stage/materialize pipeline.

What this means operationally:

- CLC additions are still atlas-aligned landscape variables in the same workflow model.

## Common errors and fixes

- Error: distance `source` not in active landscape store.
  Fix: ensure source variable exists in `atlas.landscape.data` first.
- Error: overwrite conflict.
  Fix: set `if_exists="replace"` (overwrite) or `if_exists="noop"` (keep existing).
- Error: expecting staging to persist automatically.
  Fix: call `.materialize()`.
- Error: CRS not projected meters for distance.
  Fix: ensure active store CRS is projected with metric units.

## Reference

For full parameter tables and behavior notes:

- `Reference -> Knobs and Methods`
