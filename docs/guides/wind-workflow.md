# Wind Guide

## Goal

Compute wind metrics through the atlas and materialize results back into `atlas.wind.data`.

## Canonical pattern

```python
atlas.wind.select(turbines=["Enercon.E40.500"])
atlas.wind.compute(
    "capacity_factors",
    method="rotor_node_average",
    interpolation="auto",
    air_density=False,
    rews_n=12,
    loss_factor=1.0,
).materialize()

wind_ds = atlas.wind.data
```

## Knobs you will use most

### Selection knobs

- `atlas.wind.select(turbines=[...])`
- `atlas.wind.select(turbine_indices=[...])`
- `atlas.wind.clear_selection()`

### Metric knobs

- `wind_speed`: `method`, plus:
  - `height` for `method="height_weibull_mean"`
  - `air_density`, `rews_n`, `interpolation` for `method="rotor_equivalent"`
- `capacity_factors`: `method`, `interpolation`, `air_density`, `rews_n`, `loss_factor`, plus turbine selection
- LCOE-family (`lcoe`, `min_lcoe_turbine`, `optimal_power`, `optimal_energy`): grouped `cf={...}` and `economics={...}`

### LCOE-family grouped spec shape

```python
atlas.wind.compute(
    "lcoe",
    cf={
        "method": "rotor_node_average",
        "interpolation": "auto",
        "air_density": False,
        "rews_n": 12,
        "loss_factor": 1.0,
    },
    economics={
        "discount_rate": 0.05,
        "lifetime_a": 25,
        "om_fixed_eur_per_kw_a": 20.0,
        "om_variable_eur_per_kwh": 0.008,
        # optional: bos_cost_share, grid_connect_cost_eur_per_kw
    },
).materialize()
```

## Methods by metric (explicit)

### 1. `wind_speed(method="height_weibull_mean")`

How it is computed:

1. Select Weibull parameters `A` and `k` at requested `height` from the wind store.
2. Compute expected wind speed from Weibull moments:
   `mean_wind_speed = A * Gamma(1 + 1/k)`.
3. Apply `valid_mask` from landscape (if available), so invalid cells remain masked.

What this means operationally:

- No turbine dimension is involved.
- This is a direct statistical expectation from Weibull parameters, not a power-curve integration.

### 2. `capacity_factors`

How it is computed (common pipeline):

1. Resolve selected turbines and extract turbine metadata from store attrs and turbine variables:
   hub heights, power curves, rotor diameters.
2. Read Weibull parameter stacks (`A(height,y,x)`, `k(height,y,x)`) and wind-speed grid.
3. Run turbine-by-turbine CF numerics (`capacity_factors_v1`) with the selected method.
4. Apply `loss_factor` during CF integration.
5. Apply `valid_mask` so output is defined only on valid cells.

Method-specific details:

- `rotor_node_average` (default):
  1. Construct rotor sample nodes along hub +/- rotor_radius using Gauss-Legendre nodes.
  2. Apply rotor-chord weighting so node weights represent swept-area contribution.
  3. At each node height, evaluate vertical Weibull parameters using vertical policy.
  4. For each node, integrate expected normalized power against Weibull PDF on wind-speed grid.
  5. Area-weight the node CF values and sum to final CF.

- `rotor_moment_matched_weibull`:
  1. Sample rotor nodes as above.
  2. Compute rotor-averaged first and third wind-speed moments (`m1`, `m3`).
  3. Solve an equivalent Weibull shape `k_rot` from ratio `m3 / m1^3`.
  4. Recover equivalent scale `A_rot` from `m1` and `k_rot`.
  5. Integrate CF once using this equivalent Weibull.

- `hub_height_weibull`:
  1. Interpolate Weibull parameters to hub height only.
  2. Integrate expected normalized power once at hub-height distribution.

- `hub_height_weibull_rews_scaled`:
  1. Interpolate Weibull at hub height.
  2. Compute REWS moment factor from rotor/hub cubic moments.
  3. Scale hub Weibull `A` by this factor.
  4. Integrate expected normalized power once with adjusted distribution.

Air-density behavior (`air_density=True`):

- Uses `rho` field from wind store.
- Applies speed-equivalent scaling based on `(rho / rho0)^(1/3)` within vertical evaluation path.

### 3. `wind_speed(method="rotor_equivalent")`

How it is computed:

1. Resolve turbines and rotor geometry (hub height, rotor diameter).
2. Reuse the rotor quadrature core used by the rotor-aware CF methods, but in wind-speed-only mode.
3. Compute rotor-equivalent wind speed from rotor-averaged cubic moment:
   `REWS = (E_rotor[U^3])^(1/3)`.
4. Apply `valid_mask`.

What this means operationally:

- Output is physical wind speed (`m/s`) per turbine and cell, materialized as `rotor_equivalent_wind_speed`.
- No power-curve expectation is returned here (CF part is skipped internally).
- `interpolation="auto"` resolves to `mu_cv_loglog`; explicit `ak_logz` is
  also available when you want strict no-extrapolation rotor-height queries.

### 4. `lcoe`

How it is computed:

1. Resolve grouped specs:
   - `cf={...}` controls CF method (same pipeline as `capacity_factors`).
   - `economics={...}` plus atlas baseline economics provide cost assumptions.
2. Reuse precomputed matching `capacity_factors` from active store if spec/turbines match; otherwise compute CF.
3. Extract turbine rated power and overnight cost (from metadata or fallback model).
4. Compute discounted cost and discounted energy over lifetime.
5. Return `LCOE = NPV(costs) / NPV(energy)` in `EUR/MWh`.

Key economics structure used:

- Lifetime discount factor for annual streams.
- Cost stream includes fixed/variable O&M and one-time CAPEX components.
- Energy stream is `CF * power * hours_per_year`, discounted over lifetime.

### 5. `min_lcoe_turbine`

How it is computed:

1. Compute `lcoe` cube (or use reused computation path).
2. Take argmin over turbine dimension per cell.
3. Encode all-NaN cells with nodata index `-1`.

What this means operationally:

- Output is an integer turbine index map, not turbine ID strings.
- Turbine ID mapping is stored in attrs JSON for deterministic decoding.

### 6. `optimal_power`

How it is computed:

1. Compute `lcoe` cube.
2. Find turbine index with minimum LCOE per cell.
3. Select rated power (`kW`) of that turbine per cell.

What this means operationally:

- This is "power of economically optimal turbine", not maximum technical power.

### 7. `optimal_energy`

How it is computed:

1. Compute/reuse CF and compute LCOE cube.
2. Find minimum-LCOE turbine index per cell.
3. Select CF and rated power for that turbine.
4. Compute annual energy:
   `optimal_energy = CF_selected * power_selected_kw * hours_per_year / 1e6` (GWh/a).

What this means operationally:

- Energy is for the economically selected turbine per cell, not for each turbine independently.

## Common errors and fixes

- Error: flat CF/economics kwargs for LCOE-family metrics.
  Fix: pass grouped `cf={...}` / `economics={...}`.
- Error: `hours_per_year` passed to `compute(...)`.
  Fix: set `atlas.configure_timebase(hours_per_year=...)`.
- Error: expecting compute to write stores.
  Fix: call `.materialize()`.
- Error: missing turbine metadata (`rotor_diameter`, costs).
  Fix: ensure turbine registry/store attrs are complete for selected turbines.

## Reference

For complete defaults, required fields, and cross-metric tables:

- `Reference -> Knobs and Methods`
