# Result Wrapper API

`DomainResult` is an operation wrapper returned by wind compute calls.

Primary intent:

- inspect `.data`
- call `.materialize(...)` to write into active atlas store
- optionally `.persist(...)` to create run artifacts under `results_root`

::: cleo.results.DomainResult
