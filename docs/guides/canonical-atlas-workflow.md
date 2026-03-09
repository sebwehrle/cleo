# Canonical Atlas Workflow

## Goal

Operate CLEO as one mutable `Atlas` object and keep computed state in atlas domain stores.

## Preconditions

- Atlas constructed with valid `path`, `country`, `crs`.
- `atlas.build()` completed at least once.

## Workflow

### 1. Configure atlas-level assumptions

```python
atlas.configure_turbines(["Enercon.E40.500", "Vestas.V112.3075"])
atlas.configure_timebase(hours_per_year=8766.0)
atlas.configure_economics(discount_rate=0.05, lifetime_a=25)
```

### 2. Select area routing (in place)

```python
atlas.select(area="Niederösterreich", inplace=True)
atlas.build()  # ensures area stores for current selection
```

Use `inplace=True` as default. Clone-style selection with `inplace=False` is supported but secondary.

### 3. Compute -> materialize -> continue

```python
atlas.wind.compute("wind_speed", method="height_weibull_mean", height=100).materialize()
atlas.landscape.compute(metric="distance", source="settlements").materialize()

wind_ds = atlas.wind.data
land_ds = atlas.landscape.data
```

### 4. Persist only when you need run artifacts

```python
run_path = atlas.wind.compute("wind_speed", method="height_weibull_mean", height=100).persist()
```

- `materialize(...)`: writes to active domain store and updates atlas-facing data.
- `persist(...)`: writes under `results_root` for run/result-management workflows.

### 5. Cleanup

```python
atlas.clean_results(metric_name="mean_wind_speed")
atlas.clean_areas(include_incomplete=True)
```

## Common mistakes

- Passing materialize-only args to `compute(...)` instead of `.materialize(...)`.
- Passing `hours_per_year` in wind `compute(...)` instead of `configure_timebase(...)`.
- Treating result wrappers as the primary long-lived data container.

## Where to look next

- Wind-specific knobs and methods: `Domain Guides -> Wind`
- Landscape-specific knobs and methods: `Domain Guides -> Landscape`
- Full parameter/method tables: `Reference -> Knobs and Methods`
