#!/bin/bash
# Check progress of multi-seed training runs
#
# Usage:
#   ./check_progress.sh           # Check seeds 0-299 (default)
#   ./check_progress.sh 0 99      # Check seeds 0-99

START_SEED=${1:-0}
END_SEED=${2:-299}

completed=0
missing=()

for seed in $(seq $START_SEED $END_SEED); do
    if ls logs/seed${seed}_*_eval_losses.pt 1> /dev/null 2>&1; then
        ((completed++))
    else
        missing+=($seed)
    fi
done

total=$((END_SEED - START_SEED + 1))
pct=$((completed * 100 / total))

echo "=============================================="
echo "Progress: $completed / $total seeds completed ($pct%)"
echo "=============================================="

if [ ${#missing[@]} -gt 0 ]; then
    if [ ${#missing[@]} -le 20 ]; then
        echo "Missing seeds: ${missing[*]}"
    else
        echo "Missing seeds: ${missing[*]:0:10} ... (${#missing[@]} total)"
    fi
fi

# Show disk usage
if [ -d logs ]; then
    echo ""
    echo "Disk usage:"
    du -sh logs/
    echo "Files: $(ls logs/*.pt 2>/dev/null | wc -l) .pt files, $(ls logs/*.txt 2>/dev/null | wc -l) .txt files"
fi

