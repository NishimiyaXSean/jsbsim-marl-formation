# jsbsim-marl-formation

Multi-agent reinforcement learning for formation cooperative route decision and planning, using JSBSim F-16 flight dynamics and MAPPO.

## Install

```bash
git clone https://github.com/NishimiyaXSean/jsbsim-marl-formation.git
cd jsbsim-marl-formation
pip install -e .
```

### JSBSim aircraft data

Download the official JSBSim aircraft/engines/systems data into `data/jsbsim/`:

```bash
git clone https://github.com/JSBSim-Team/jsbsim.git /tmp/jsbsim
cp -r /tmp/jsbsim/aircraft data/jsbsim/
cp -r /tmp/jsbsim/engines  data/jsbsim/
cp -r /tmp/jsbsim/systems  data/jsbsim/
```

Or set `JSBSIM_DATA_DIR` environment variable to point to an existing JSBSim installation.

### Verify installation

```bash
python scripts/verify_installation.py
```

## Quick Start

```bash
# Single-agent tracking task
python scripts/run_single_agent.py

# 1v1 air combat training
python scripts/run_1v1_training.py
```

## Project Structure

```
src/
├── dynamics/         # JSBSim F-16 wrapper + autopilot
├── environment/      # Gymnasium multi-agent env
├── models/           # MAPPO neural networks
├── training/         # RLlib training pipeline
├── visualization/    # 3D rendering (Tacview export)
├── logging/          # TensorBoard + experiment tracking
└── utils/            # Geometry, kinematics, units

configs/              # Hydra YAML configs
scripts/              # Entry-point scripts
tests/                # Unit tests
```
