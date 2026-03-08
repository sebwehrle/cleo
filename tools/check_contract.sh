#!/usr/bin/env bash
# Contract enforcement for artifact and compute invariants.
# Portable: works on macOS, Linux, and Windows Git Bash.

set -euo pipefail

# Explicit code roots to search (no repo-root/docs scanning)
# tools/** is intentionally excluded from architecture-layer checks below.
CODE_ROOTS=(cleo tools)

# Common glob excludes (portable, no pipe filtering)
RG_EXCLUDES=(
    --glob '!tests/**'
    --glob '!docs/**'
    --glob '!data/**'
    --glob '!.git/**'
    --glob '!.venv/**'
    --glob '!venv/**'
    --glob '!build/**'
    --glob '!dist/**'
    --glob '!site/**'
    --glob '!.mypy_cache/**'
    --glob '!.pytest_cache/**'
    --glob '!__pycache__/**'
    --glob '!*.egg-info/**'
    --glob '!*.md'
    --glob '!*.sh'
)

FAIL_COUNT=0

# Helper: fail if pattern is found
# Usage: fail_if_found "description" "pattern" [extra_globs...]
fail_if_found() {
    local description="$1"
    local pattern="$2"
    shift 2
    local extra_globs=("$@")

    local matches
    matches="$(rg -n --no-heading --hidden --follow -S \
        "${RG_EXCLUDES[@]}" "${extra_globs[@]}" \
        -e "$pattern" "${CODE_ROOTS[@]}" 2>/dev/null || true)"

    if [[ -n "$matches" ]]; then
        echo "FAIL: ${description}"
        echo "$matches"
        echo ""
        echo "Hint: move I/O into dedicated I/O modules"
        echo "      (cleo/unification/**, cleo/results.py, cleo/exports.py)."
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Helper: fail if pattern is found in a specific file
# Usage: fail_if_found_in_file "description" "pattern" "filepath"
fail_if_found_in_file() {
    local description="$1"
    local pattern="$2"
    local filepath="$3"

    if [[ ! -f "$filepath" ]]; then
        return
    fi

    local matches
    matches="$(rg -n --no-heading -S -e "$pattern" "$filepath" 2>/dev/null || true)"

    if [[ -n "$matches" ]]; then
        echo "FAIL: ${description}"
        echo "$matches"
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

echo "=== Unified Atlas Contract Check ==="
echo ""

# -----------------------------------------------------------------------------
# A) No consolidated metadata dependency (Zarr v3 future-proof)
# -----------------------------------------------------------------------------
echo "Checking: No consolidated metadata usage..."

# 1) Forbid consolidated=True (explicit opt-in to consolidated metadata)
fail_if_found \
  "xr.open_zarr(consolidated=True) is forbidden (Zarr v3: no consolidated metadata dependency)" \
  'xr\.open_zarr\([^\)]*consolidated\s*=\s*True' \
  --glob '!cleo/unification/**' --glob '!cleo/results.py'

# 2) Forbid metadata consolidation writes (creates consolidated metadata)
fail_if_found \
  "zarr.consolidate_metadata() is forbidden (Zarr v3: no consolidated metadata writes)" \
  'zarr\.consolidate_metadata\('

fail_if_found \
  ".consolidate_metadata() is forbidden (Zarr v3: no consolidated metadata writes)" \
  '\.consolidate_metadata\('

# Raw I/O patterns forbidden in pure-compute modules.
PURE_COMPUTE_IO_PATTERNS=(
    'rxr\.open_rasterio\('
    'rioxarray\.open_rasterio\('
    'rasterio\.open\('
    'gpd\.read_file\('
    'geopandas\.read_file\('
    'yaml\.safe_load\('
    'yaml\.load\('
    'xr\.open_dataset\('
    'xr\.open_mfdataset\('
    'xr\.open_dataarray\('
    'xr\.open_zarr\('
    '\.to_zarr\('
    'to_netcdf\('
)

# These patterns are always forbidden (force Dask computation)
STRICT_EAGER_PATTERNS=(
  '\.\s*compute\s*\('
  '\.\s*load\s*\('
  '\.\s*item\s*\('
)

check_pure_compute_module() {
    local module_path="$1"
    local module_label="$2"

    if [[ ! -f "$module_path" ]]; then
        return
    fi

    echo "Checking: ${module_path} is pure compute (no I/O)..."
    for pattern in "${PURE_COMPUTE_IO_PATTERNS[@]}"; do
        fail_if_found_in_file \
            "${module_path} must be pure compute, found I/O: $pattern" \
            "$pattern" \
            "$module_path"
    done

    echo "Checking: ${module_path} must not eager-evaluate..."
    for pattern in "${STRICT_EAGER_PATTERNS[@]}"; do
      matches="$(rg -n --no-heading -e "$pattern" "$module_path" 2>/dev/null | rg -v '^\s*#' || true)"
      if [[ -n "$matches" ]]; then
        echo "FAIL: ${module_path} must stay lazy, found pattern: $pattern"
        echo "$matches"
        echo ""
        echo "Hint: ${module_label} must stay lazy; move eager steps to dedicated persistence/export layers or make evaluation an explicit user action."
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
    done

    # .values check: allow on coords (small 1D arrays) and 1D curves
    # Filter: .coords[...].values, _coord.values, and lines with "# coord" or "# 1D" markers
    values_matches="$(rg -n --no-heading -e '\.\s*values\b' "$module_path" 2>/dev/null \
      | rg -v '^\s*#' \
      | rg -v '\.coords\[' \
      | rg -v '_coord\.values' \
      | rg -v '# coord' \
      | rg -v '# 1D' \
      || true)"
    if [[ -n "$values_matches" ]]; then
      echo "FAIL: ${module_path} must stay lazy, found .values on non-coord array"
      echo "$values_matches"
      echo ""
      echo "Hint: ${module_label} must stay lazy; move eager steps to dedicated persistence/export layers or make evaluation an explicit user action."
      echo ""
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# -----------------------------------------------------------------------------
# D) Compute purity/laziness (extra strict)
# -----------------------------------------------------------------------------
check_pure_compute_module "cleo/assess.py" "assess.py"
check_pure_compute_module "cleo/economics.py" "economics.py"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "FAILED: $FAIL_COUNT contract violation(s) found."
    exit 1
else
    echo "OK: contract-check passed"
fi
