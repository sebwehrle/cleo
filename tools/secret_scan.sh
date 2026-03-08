#!/usr/bin/env bash

# Shared detect-secrets entrypoint for local maintenance and CI.
set -euo pipefail

BASELINE_PATH=".secrets.baseline"
JSON_OUT=""
MODE="all"
HOOK_BASELINE_PATH="$BASELINE_PATH"
TEMP_BASELINE_PATH=""

usage() {
  cat <<'EOF'
Usage: tools/secret_scan.sh [--all | --staged | --update-baseline] [--json-out PATH]

Modes:
  --all              Scan repository files against .secrets.baseline (default).
  --staged           Scan staged files against .secrets.baseline.
  --update-baseline  Regenerate/upgrade .secrets.baseline with current exclusions.

Options:
  --json-out PATH    Write detect-secrets JSON output to PATH for check modes.
  -h, --help         Show this help text.
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

resolve_python_bin() {
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi
  return 1
}

resolve_detect_secrets_scan_cmd() {
  if command -v detect-secrets >/dev/null 2>&1; then
    DETECT_SECRETS_SCAN_CMD=("detect-secrets")
    return 0
  fi

  local pybin
  pybin="$(resolve_python_bin)" || return 1
  DETECT_SECRETS_SCAN_CMD=("$pybin" "-m" "detect_secrets.main")
}

resolve_detect_secrets_hook_cmd() {
  if command -v detect-secrets-hook >/dev/null 2>&1; then
    DETECT_SECRETS_HOOK_CMD=("detect-secrets-hook")
    return 0
  fi

  local pybin
  pybin="$(resolve_python_bin)" || return 1
  DETECT_SECRETS_HOOK_CMD=("$pybin" "-m" "detect_secrets.pre_commit_hook")
}

collect_all_files() {
  local path
  while IFS= read -r -d '' path; do
    FILES+=("$path")
  done < <(git ls-files -z --cached --others --exclude-standard --)
}

collect_staged_files() {
  local path
  while IFS= read -r -d '' path; do
    FILES+=("$path")
  done < <(git diff --cached --name-only --diff-filter=ACMR -z --)
}

run_secret_check() {
  local -a cmd=()
  resolve_detect_secrets_hook_cmd || {
    echo "Missing detect-secrets runtime. Install detect-secrets==1.5.0 or run via pre-commit." >&2
    exit 2
  }
  cmd=(
    "${DETECT_SECRETS_HOOK_CMD[@]}"
    --baseline
    "$HOOK_BASELINE_PATH"
    --exclude-files
    '^requirements-lock\.txt$'
    --exclude-files
    '(^|/).*\.zarr(/|$)'
    --exclude-files
    '.*\.tif$'
    --exclude-files
    '.*\.png$'
    --exclude-files
    '^\.secrets\.baseline$'
    --exclude-secrets
    '^[A-Fa-f0-9]{64}$'
  )

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files selected for secret scanning."
    if [[ -n "$JSON_OUT" ]]; then
      printf '{\n  "results": {}\n}\n' > "$JSON_OUT"
    fi
    return 0
  fi

  if [[ -n "$JSON_OUT" ]]; then
    if "${cmd[@]}" --json "${FILES[@]}" > "$JSON_OUT"; then
      echo "OK: No new secrets detected"
      return 0
    fi
    echo "WARNING: New potential secrets detected. Review $JSON_OUT" >&2
    echo "If these are false positives, update $BASELINE_PATH with tools/secret_scan.sh --update-baseline" >&2
    return 1
  fi

  "${cmd[@]}" "${FILES[@]}"
}

update_baseline() {
  resolve_detect_secrets_scan_cmd || {
    echo "Missing detect-secrets runtime. Install detect-secrets==1.5.0 or run via pre-commit." >&2
    exit 2
  }
  "${DETECT_SECRETS_SCAN_CMD[@]}" scan \
    --baseline "$BASELINE_PATH" \
    --exclude-files '^requirements-lock\.txt$' \
    --exclude-files '(^|/).*\.zarr(/|$)' \
    --exclude-files '.*\.tif$' \
    --exclude-files '.*\.png$' \
    --exclude-files '^\.secrets\.baseline$' \
    --exclude-secrets '^[A-Fa-f0-9]{64}$'
  echo "Updated $BASELINE_PATH"
}

prepare_hook_baseline() {
  if ! git diff --quiet -- "$BASELINE_PATH"; then
    TEMP_BASELINE_PATH="$(mktemp "${TMPDIR:-/tmp}/cleo-secrets-baseline.XXXXXX")"
    cp "$BASELINE_PATH" "$TEMP_BASELINE_PATH"
    HOOK_BASELINE_PATH="$TEMP_BASELINE_PATH"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODE="all"
      shift
      ;;
    --staged)
      MODE="staged"
      shift
      ;;
    --update-baseline)
      MODE="update-baseline"
      shift
      ;;
    --json-out)
      JSON_OUT="${2:-}"
      if [[ -z "$JSON_OUT" ]]; then
        echo "--json-out requires a path argument" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command git

if [[ "$MODE" == "update-baseline" ]]; then
  update_baseline
  exit 0
fi

if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "Missing baseline: $BASELINE_PATH" >&2
  echo "Create or refresh it with tools/secret_scan.sh --update-baseline" >&2
  exit 2
fi

trap '[[ -n "$TEMP_BASELINE_PATH" ]] && rm -f "$TEMP_BASELINE_PATH"' EXIT

declare -a FILES=()
if [[ "$MODE" == "staged" ]]; then
  collect_staged_files
else
  collect_all_files
fi

prepare_hook_baseline
run_secret_check
