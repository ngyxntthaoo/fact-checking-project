#!/bin/bash
# =============================================================================
# run_train.sh — Launch full Concept-HGN training on vast.ai
#
# Usage:
#   ./run_train.sh              # full training
#   ./run_train.sh --smoke      # smoke test (200 samples)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCEPT_HGN="$SCRIPT_DIR/HeterFC/concept-hgn"

# ── GPU training config (overrides defaults in train_fever.py) ────────────────
export SMOKE_TEST=0                              # full dataset
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints" # save outside concept-hgn dir

# Optional: point to custom data location (defaults auto-detected from repo root)
# export TRAIN_PATH=/workspace/data/all_train.json
# export DEV_PATH=/workspace/data/all_dev.json
# export CONCEPT_CACHE=/workspace/concept_cache.pkl

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$CONCEPT_HGN/preprocessed"

echo "Starting Concept-HGN training..."
echo "  CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "  SMOKE_TEST: $SMOKE_TEST"
echo ""

cd "$CONCEPT_HGN"
python train_fever.py "$@"
