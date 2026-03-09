# Data Model

## The central model

CLEO is atlas-centric: one mutable `Atlas` object is the working state container.

- Domain data lives in `atlas.wind.data` and `atlas.landscape.data`.
- `compute(...)` / `add(...)` / `rasterize(...)` stage results in domain state.
- `materialize(...)` writes staged results to active stores and keeps atlas data authoritative.

## Store layers

- Base stores: country-level canonical stores from `Atlas.build()`
- Area stores: area-scoped stores selected via `atlas.select(..., inplace=True)`
- Results stores: optional run artifacts under `results/<run_id>/...`

## Operational distinction

- `materialize(...)` is the canonical way to continue working inside atlas domain data.
- `persist(...)` is for result-management/export pipelines outside the active domain stores.

## Supported but secondary patterns

- Clone-style selection via `select(..., inplace=False)`
- Detached wrapper objects (`DomainResult`, landscape result wrappers)

These exist for flexibility, but are not the primary workflow model.
