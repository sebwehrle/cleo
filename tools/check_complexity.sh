#!/usr/bin/env bash
# Pre-commit hook to check for Grade E complexity functions
# Fails if any function has cyclomatic complexity > 30

set -e

# Check for Grade E functions
GRADE_E=$(python -m radon cc cleo/ -s -n E 2>/dev/null | grep -v "^$" || true)

if [ -n "$GRADE_E" ]; then
    echo "ERROR: Grade E functions found (CC > 30):"
    echo "$GRADE_E"
    echo ""
    echo "Refactor these functions before committing."
    exit 1
fi

echo "OK: No grade E functions found"
