#!/bin/bash
# WSL2 MAPPO Environment Setup Script
# Run this inside WSL2 Ubuntu after cloning/copying the repository
#
# Usage: bash scripts/setup_wsl2.sh

set -e

echo "============================================"
echo " JSBSim MARL — WSL2 Environment Setup"
echo "============================================"

# ── System dependencies ──────────────────────────────────────────
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev \
    build-essential git curl unzip \
    libopenblas-dev liblapack-dev 2>/dev/null

# ── Python virtual environment ───────────────────────────────────
echo ""
echo "[2/5] Creating Python virtual environment..."
python3.10 -m venv jsbsim_rl --prompt jsbsim_rl
source jsbsim_rl/bin/activate

# ── Python packages ──────────────────────────────────────────────
echo ""
echo "[3/5] Installing Python dependencies..."

# Core ML
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy gymnasium stable-baselines3

# RLlib MAPPO
pip install "ray[rllib]==2.40.0"

# Utilities
pip install tensorboard matplotlib pyyaml

# JSBSim Python bindings
pip install jsbsim==1.3.1

echo ""
echo "[4/5] Verifying installation..."

python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import ray; print(f'Ray: {ray.__version__}')
import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')
import jsbsim; print(f'JSBSim: OK')
print('All core packages: OK')
"

# ── JSBSim aircraft data ─────────────────────────────────────────
echo ""
echo "[5/5] Checking JSBSim data files..."
if [ -d "data/jsbsim" ]; then
    echo "  JSBSim aircraft data: found"
else
    echo "  WARNING: data/jsbsim/ not found. JSBSim needs F-16 config files."
    echo "  Copy them from your Windows installation or install jsbsim package data."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Activate environment: source jsbsim_rl/bin/activate"
echo " Verify: python scripts/verify_installation.py"
echo "============================================"
