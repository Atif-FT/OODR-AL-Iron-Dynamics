#!/bin/bash
# train_ensemble.sh
# Trains an ensemble of 4 MACE models using MACE's native single-stage fine-tuning.
# Each member is trained directly from the foundation model with a different random seed,
# producing 4 independently-initialised SWA models for uncertainty quantification (UQ).
#
# Usage:
#   bash train_ensemble.sh <foundation_model.model> <train.xyz> <valid.xyz> [device]
#
# Outputs (saved by mace_run_train in the current directory under checkpoints/):
#   checkpoints/Fe_Ensemble_Member_0_run-42_stagetwo.model
#   checkpoints/Fe_Ensemble_Member_1_run-123_stagetwo.model
#   checkpoints/Fe_Ensemble_Member_2_run-999_stagetwo.model
#   checkpoints/Fe_Ensemble_Member_3_run-777_stagetwo.model
#
# After training, models are ALSO copied to:
#   ./Fe_Ensemble_Member_X_stagetwo.model   (for easy access)

set -e   # Exit immediately if any command fails

# Always use the location of this script as the directory reference,
# so that python finetune_mace.py can be found from wherever this script is called.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: bash train_ensemble.sh <foundation_model.model> <train.xyz> <valid.xyz> [device]"
    exit 1
fi

FOUNDATION_MODEL=$1
TRAIN_FILE=$2
VALID_FILE=$3
DEVICE=${4:-cuda}   # Default to cuda; pass 'cpu' if no GPU available

echo "=== STARTING MACE ENSEMBLE TRAINING (N=4 MEMBERS) ==="
echo "  Foundation model : $FOUNDATION_MODEL"
echo "  Train file       : $TRAIN_FILE"
echo "  Valid file       : $VALID_FILE"
echo "  Device           : $DEVICE"
echo ""

# 4 ensemble members with different random seeds for diverse UQ coverage
SEEDS=(42 123 999 777)

for i in {0..3}
do
    SEED=${SEEDS[$i]}
    MEMBER_NAME="Fe_Ensemble_Member_$i"

    # --- Auto-resume: check for the actual file MACE saves ---
    # MACE saves Stage 2 (SWA) model as:
    #   checkpoints/{name}_run-{seed}_stagetwo.model
    CHECKPOINT_FILE="checkpoints/${MEMBER_NAME}_run-${SEED}_stagetwo.model"
    FINAL_FILE="${MEMBER_NAME}_stagetwo.model"

    if [ -f "$FINAL_FILE" ] || [ -f "$CHECKPOINT_FILE" ]; then
        echo "=============================================="
        echo " Member $i already trained — skipping."
        echo " Found: $(ls -lh $CHECKPOINT_FILE 2>/dev/null || ls -lh $FINAL_FILE 2>/dev/null | tail -1)"
        echo "=============================================="
        # Ensure final copy exists even if we're resuming
        if [ -f "$CHECKPOINT_FILE" ] && [ ! -f "$FINAL_FILE" ]; then
            cp "$CHECKPOINT_FILE" "$FINAL_FILE"
            echo "  Copied to $FINAL_FILE"
        fi
        continue
    fi

    echo ""
    echo "=============================================="
    echo " Training Ensemble Member $i / 4 (Seed: $SEED)"
    echo "=============================================="

    python "$SCRIPT_DIR/finetune_mace.py" \
        "$FOUNDATION_MODEL" \
        "$TRAIN_FILE" \
        "$VALID_FILE" \
        "$MEMBER_NAME" \
        "$SEED" \
        "$DEVICE"

    # --- Copy SWA model to a clean, predictable filename ---
    # MACE naming convention: checkpoints/{name}_run-{seed}_stagetwo.model
    if [ -f "$CHECKPOINT_FILE" ]; then
        cp "$CHECKPOINT_FILE" "$FINAL_FILE"
        echo "  Saved final model → $FINAL_FILE"
    else
        echo "  WARNING: Expected checkpoint not found: $CHECKPOINT_FILE"
        echo "  Contents of checkpoints/:"
        ls -lh checkpoints/ 2>/dev/null || echo "  (checkpoints dir not found)"
    fi

    echo "Member $i complete → $FINAL_FILE"
done

echo ""
echo "=== MACE ENSEMBLE TRAINING COMPLETE ==="
echo "Ensemble SWA (Stage 2) models:"
ls -lh Fe_Ensemble_Member_*_stagetwo.model 2>/dev/null \
    || ls -lh checkpoints/Fe_Ensemble_Member_*_stagetwo.model 2>/dev/null \
    || echo "  (No stagetwo models found in current dir or checkpoints/)"
exit 0
