# CLEO Documentation

CLEO is an Atlas-first analysis package. The intended usage is one working `Atlas` object whose wind and landscape datasets are continuously updated through compute/materialize operations.

## Read This In Order

1. `Start Here` for the 10-minute, end-to-end canonical flow.
2. `Canonical Workflow` for area routing, materialization, and cleanup behavior.
3. Domain guide for the task at hand (`Wind` or `Landscape`).
4. `Reference -> Knobs and Methods` when selecting parameters and algorithm variants.
5. API reference pages for exact signatures and docstring-level details.

## Core Rule

- Compute on `atlas.wind` or `atlas.landscape`.
- Materialize back into atlas active stores.
- Continue from `atlas.wind.data` / `atlas.landscape.data`.

Detached result wrappers are supported operation handles, but they are not the primary long-lived working model.

## Normative Contract

Behavioral and architectural invariants are defined in:

- [Unified Atlas Contract](CONTRACT_UNIFIED_ATLAS.md)
