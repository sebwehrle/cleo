#!/usr/bin/env bash
# Complexity report for CLEO codebase.
# Run before and after refactoring to track progress.
# See AGENTS.md for refactoring requirements.

set -euo pipefail

# Colors for output (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    GREEN='\033[0;32m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    YELLOW=''
    GREEN=''
    BOLD=''
    NC=''
fi

echo -e "${BOLD}=== CLEO Complexity Report ===${NC}"
echo ""
echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Check radon is installed
if ! command -v radon &> /dev/null; then
    echo -e "${RED}ERROR: radon not found. Install with: pip install radon${NC}"
    exit 1
fi

# Grade E functions (CC > 30) - CRITICAL
echo -e "${BOLD}Grade E functions (CC > 30) - CRITICAL:${NC}"
GRADE_E=$(radon cc cleo --min E -s 2>/dev/null || true)
if [[ -n "$GRADE_E" ]]; then
    echo -e "${RED}$GRADE_E${NC}"
    GRADE_E_COUNT=$(echo "$GRADE_E" | grep -c "^cleo" || echo "0")
else
    echo -e "${GREEN}  None found.${NC}"
    GRADE_E_COUNT=0
fi
echo ""

# Grade D functions (CC 21-30) - BLOCKING
echo -e "${BOLD}Grade D functions (CC 21-30) - BLOCKING:${NC}"
GRADE_D=$(radon cc cleo --min D --max D -s 2>/dev/null || true)
if [[ -n "$GRADE_D" ]]; then
    echo -e "${YELLOW}$GRADE_D${NC}"
    GRADE_D_COUNT=$(echo "$GRADE_D" | grep -c "^cleo" || echo "0")
else
    echo -e "${GREEN}  None found.${NC}"
    GRADE_D_COUNT=0
fi
echo ""

# Grade C functions (CC 11-20) - MONITOR
echo -e "${BOLD}Grade C functions (CC 11-20) - MONITOR:${NC}"
GRADE_C=$(radon cc cleo --min C --max C -s 2>/dev/null || true)
if [[ -n "$GRADE_C" ]]; then
    echo "$GRADE_C"
    GRADE_C_COUNT=$(echo "$GRADE_C" | grep -c "^cleo" || echo "0")
else
    echo -e "${GREEN}  None found.${NC}"
    GRADE_C_COUNT=0
fi
echo ""

# Average complexity
echo -e "${BOLD}Average Complexity:${NC}"
radon cc cleo -a 2>/dev/null | tail -1
echo ""

# Maintainability Index
echo -e "${BOLD}Maintainability Concerns (MI < 20, grade C or below):${NC}"
LOW_MI=$(radon mi cleo -s 2>/dev/null | grep -E "^cleo.*- [CF]" || true)
if [[ -n "$LOW_MI" ]]; then
    echo -e "${YELLOW}$LOW_MI${NC}"
else
    echo -e "${GREEN}  All modules have acceptable maintainability (grade A or B).${NC}"
fi
echo ""

# Dead code check (if vulture is available)
if command -v vulture &> /dev/null; then
    echo -e "${BOLD}Potential Dead Code (90%+ confidence):${NC}"
    DEAD_CODE=$(vulture cleo --min-confidence 90 2>/dev/null || true)
    if [[ -n "$DEAD_CODE" ]]; then
        echo -e "${YELLOW}$DEAD_CODE${NC}"
    else
        echo -e "${GREEN}  No high-confidence dead code detected.${NC}"
    fi
    echo ""
fi

# Summary
echo -e "${BOLD}=== Summary ===${NC}"
echo "  Grade E (CC > 30):  $GRADE_E_COUNT functions - Critical"
echo "  Grade D (CC 21-30): $GRADE_D_COUNT functions - Blocks merge"
echo "  Grade C (CC 11-20): $GRADE_C_COUNT functions - Monitor"
echo ""

# Exit status based on grade D-or-worse presence
if [[ "$GRADE_D_COUNT" -gt 0 || "$GRADE_E_COUNT" -gt 0 ]]; then
    echo -e "${RED}FAIL: Grade D or worse functions must be refactored.${NC}"
    exit 1
else
    echo -e "${GREEN}OK: No grade D or worse functions.${NC}"
    exit 0
fi
