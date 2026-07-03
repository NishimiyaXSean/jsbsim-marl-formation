#!/bin/bash
# WSL2 Environment Setup — Pure PyTorch MAPPO (no RLlib)
# IMPORTANT: If you copied this file from Windows, run first:
#   sudo apt install dos2unix -y && dos2unix scripts/setup_wsl2.sh
# Then: bash scripts/setup_wsl2.sh

set -e

echo "============================================"
echo " JSBSim MARL — WSL2 Environment Setup"
echo " (Pure PyTorch MAPPO — no RLlib)"
echo "============================================"

# ── System dependencies ──────────────────────────────────────────
echo ""
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev \
    build-essential git curl unzip dos2unix \
    libopenblas-dev liblapack-dev 2>/dev/null

# ── Python virtual environment ───────────────────────────────────
echo ""
echo "[2/4] Creating Python virtual environment..."
python3.10 -m venv jsbsim_rl --prompt jsbsim_rl
source jsbsim_rl/bin/activate

# ── Python packages ──────────────────────────────────────────────
echo ""
echo "[3/4] Installing Python dependencies..."

pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy gymnasium stable-baselines3 tensorboard matplotlib pyyaml
pip install jsbsim==1.3.1

echo ""
echo "[4/4] Verifying installation..."

python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')
import jsbsim; print(f'JSBSim: OK')
from torch.utils.tensorboard import SummaryWriter; print('TensorBoard: OK')
print('All core packages: OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Activate: source jsbsim_rl/bin/activate"
echo " Verify:  python scripts/verify_installation.py"
echo ""
echo " Run training (dual-actor cooperative):"
echo "   python scripts/train_dual_actor.py --mode 2v1 --steps 500000 \\"
echo "       --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \\"
echo "       --cooperative --warmup 100000"
echo "============================================"
