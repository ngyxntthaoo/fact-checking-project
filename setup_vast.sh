#!/bin/bash
# =============================================================================
# setup_vast.sh — One-shot setup script for vast.ai
#
# Usage (on the vast.ai instance, after cloning the repo):
#   chmod +x setup_vast.sh
#   ./setup_vast.sh
#
# Then to train:
#   ./run_train.sh
# =============================================================================

set -e  # exit on first error

echo "============================================"
echo "  Concept-HGN — vast.ai Setup Script"
echo "============================================"

# ── 1. Detect workspace root ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
CONCEPT_HGN="$REPO_ROOT/HeterFC/concept-hgn"

echo "[1/6] Repo root: $REPO_ROOT"
echo "      concept-hgn: $CONCEPT_HGN"

# ── 2. Install Python deps ────────────────────────────────────────────────────
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r "$CONCEPT_HGN/requirements.txt"

echo "[3/6] Downloading spaCy model..."
python -m spacy download en_core_web_lg

# ── 3. Verify CUDA ────────────────────────────────────────────────────────────
echo "[4/6] CUDA check..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ── 4. Check data ─────────────────────────────────────────────────────────────
echo "[5/6] Checking data files..."
DATA_DIR="$REPO_ROOT/KernelGAT/data/KernelGAT/data"
for f in all_train.json all_dev.json; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f found"
    else
        echo "  ✗ MISSING: $DATA_DIR/$f"
        echo "    → Upload it via: scp or vast.ai cloud storage"
    fi
done

# Check concept cache
if [ -f "$CONCEPT_HGN/concept_cache.pkl" ]; then
    echo "  ✓ concept_cache.pkl found"
else
    echo "  ✗ concept_cache.pkl not found — CES will be disabled during training"
fi

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo "[6/6] Setup complete!"
echo ""
echo "To start full training:"
echo "  cd $CONCEPT_HGN"
echo "  SMOKE_TEST=0 python train_fever.py --no-smoke"
echo ""
echo "Or use the run_train.sh script:"
echo "  ./run_train.sh"
