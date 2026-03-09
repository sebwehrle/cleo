#!/usr/bin/env bash
# Complexity gate for tracked package code.
# Fails if any function has cyclomatic complexity > 20.

set -e

# Check for Grade D functions (CC > 20)
GRADE_D=$(python -m radon cc cleo/ -s -n D 2>/dev/null | grep -v "^$" || true)

if [ -n "$GRADE_D" ]; then
    echo "ERROR: Grade D functions found (CC > 20):"
    echo "$GRADE_D"
    echo ""
    echo "Refactor these functions before committing."
    exit 1
fi

echo "OK: No grade D functions found"
