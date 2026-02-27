# PV Domain (Planned)

This repository is expected to add a PV analysis domain. This page defines documentation expectations before implementation.

## Planned documentation contract

When PV is introduced, docs should be updated in the same change set for:

- Public API surface (`atlas.pv`, compute/materialize/persist behavior)
- Parameter contracts (units, defaults, accepted ranges)
- Data model impact (new store variables, attrs, provenance)
- At least one end-to-end workflow guide
- API reference entries generated from PV docstrings

## Naming and shape guidance

- Reuse existing domain patterns (`compute(...) -> result wrapper -> materialize/persist`)
- Avoid transitory names (`v1`, `phase2`, `new_*`, `old_*`) in canonical symbols
- Keep dependency boundaries aligned with existing layer rules

## Placeholder status

This page intentionally contains no PV API details until the domain exists.
