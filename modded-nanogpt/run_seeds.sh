#!/bin/bash
# Run modded-nanogpt training across multiple seeds on 8 GPUs
#
# Usage:
#   ./run_seeds.sh                    # Run seeds 0-299 (default)
#   ./run_seeds.sh 0 49               # Run seeds 0-49
#   ./run_seeds.sh 100 199            # Run seeds 100-199
#
# Features:
#   - Automatically skips already-completed seeds (resume-friendly)
#   - Organized output in logs/ directory
#   - Progress tracking

set -e

START_SEED=${1:-0}
END_SEED=${2:-299}
TOTAL_SEEDS=$((END_SEED - START_SEED + 1))

echo "=============================================="
echo "Multi-seed training: seeds $START_SEED to $END_SEED ($TOTAL_SEEDS seeds)"
echo "Started at $(date)"
echo "=============================================="

completed=0
skipped=0

for seed in $(seq $START_SEED $END_SEED); do
    # Check if this seed already completed (look for eval_losses file)
    if ls logs/seed${seed}_*_eval_losses.pt 1> /dev/null 2>&1; then
        echo "[Seed $seed] Already completed, skipping..."
        ((++skipped))
        continue
    fi
    
    progress=$((completed + skipped + 1))
    remaining=$((TOTAL_SEEDS - progress + 1))
    
    echo ""
    echo "=============================================="
    echo "[Seed $seed] Starting ($progress/$TOTAL_SEEDS, $remaining remaining)"
    echo "[Seed $seed] $(date)"
    echo "=============================================="
    
    torchrun --standalone --nproc_per_node=8 train_gpt.py --seed=$seed
    
    ((++completed))
    echo "[Seed $seed] Completed at $(date)"
done

echo ""
echo "=============================================="
echo "Finished at $(date)"
echo "Completed: $completed | Skipped (already done): $skipped | Total: $TOTAL_SEEDS"
echo "Results saved in logs/ directory"
echo "=============================================="
