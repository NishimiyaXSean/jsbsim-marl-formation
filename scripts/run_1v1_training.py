"""Launch 1v1 MAPPO training with JSBSim F-16 dynamics.

Usage:
    python scripts/run_1v1_training.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_mappo import train

if __name__ == "__main__":
    train(
        train_iterations=500,
        eval_interval=10,
        test_episodes=50,
        target_success_rate=0.70,
        checkpoint_freq=20,
        resume_checkpoint=None,
    )
