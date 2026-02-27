# Start Here

## Goal

Run a complete atlas-centric workflow in one session and keep outputs in atlas domain data.

## 1. Install

```bash
python -m pip install -e ".[dev,docs]"
```

## 2. Build base stores

```python
from cleo import Atlas

atlas = Atlas("/path/to/workdir", country="AUT", crs="EPSG:3035")
atlas.build()
```

What this does:

- Materializes canonical base stores (`wind.zarr`, `landscape.zarr`).
- Initializes the atlas workspace for domain computations.

## 3. Compute and materialize wind output

```python
atlas.wind.select(turbines=["Enercon.E40.500"])
atlas.wind.compute("capacity_factors", mode="direct_cf_quadrature", air_density=False, rews_n=12, loss_factor=1.0).materialize()
```

Result location:

- `atlas.wind.data["capacity_factors"]`

## 4. Compute and materialize landscape output

```python
atlas.landscape.compute(metric="distance", source="settlements", name="distance_settlements", if_exists="replace").materialize()
```

Result location:

- `atlas.landscape.data["distance_settlements"]`

## 5. Keep working from atlas domain data

```python
wind_ds = atlas.wind.data
landscape_ds = atlas.landscape.data
```

This is the canonical continuation point for analysis/export.

## 6. Strict docs check (maintainer)

```bash
python -m mkdocs build --strict
```

## Next pages

- `Canonical Workflow` for region selection/materialization semantics.
- `Domain Guides` for wind/landscape-specific knobs.
- `Reference -> Knobs and Methods` for complete parameter and method tables.
