"""Phase 4: staged 2v1 cooperative pursuit training.

Stage 1 (0–50K):  Critic warm-up.  Actor weights frozen (copied from
    Phase 3.6 single-pursuer model).  Formation spacing reward = 0.
    Pursuers learn to coexist without colliding.

Stage 2 (50K–150K): Actor unfrozen with low LR.  Formation spacing
    reward gradually introduced.  Pursuers learn cooperative geometry.

Stage 3 (150K+): Full training.  Target difficulty increases.

Usage:
    python scripts/train_formation_2v1.py
    python scripts/train_formation_2v1.py --steps 200000 --tiled-model data/expert/tiled_2v1_phase36.zip
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque, defaultdict

from src.environment.formation_env import FormationEnv

TOTAL_TIMESTEPS = 200_000
STAGE1_STEPS = 50_000    # Critic-only warm-up
STAGE2_STEPS = 150_000   # Actor unfrozen, formation ramp
ACTOR_LR_STAGE1 = 0.0    # frozen
ACTOR_LR_STAGE2 = 1e-5   # very low — gentle fine-tune
CRITIC_LR = 5e-4         # aggressive — needs to catch up


FORMATION_RAMP_START = 50_000   # start when Actor unfreezes
FORMATION_RAMP_END = 150_000    # full weight by end of Stage 2

class Staged2v1Callback(BaseCallback):
    """Staged training: freeze/train Actor, ramp formation reward."""

    def __init__(self, model: PPO, total_steps: int, train_env=None, verbose: int = 0):
        super().__init__(verbose)
        self._model = model
        self._total_steps = total_steps
        self._train_env = train_env
        self._term_counts = defaultdict(int)
        self._term_total = 0
        self._formation_weight = 0.0

    def _on_step(self) -> bool:
        # ── Formation weight annealing ───────────────────────────────
        if self.num_timesteps < FORMATION_RAMP_START:
            new_fw = 0.0
        elif self.num_timesteps < FORMATION_RAMP_END:
            frac = (self.num_timesteps - FORMATION_RAMP_START) / (
                FORMATION_RAMP_END - FORMATION_RAMP_START)
            new_fw = float(np.clip(frac, 0.0, 1.0))
        else:
            new_fw = 1.0
        if abs(new_fw - self._formation_weight) > 1e-6:
            self._formation_weight = new_fw
            if self._train_env is not None:
                try:
                    self._train_env.set_formation_weight(new_fw)
                except AttributeError:
                    pass

        # ── Stage switching ──────────────────────────────────────────
        if self.num_timesteps == STAGE1_STEPS:
            self._unfreeze_actor()
            print(f"\n[2v1] === Stage 2: Actor unfrozen, formation_w={new_fw:.2f} ===")
        if self.num_timesteps == STAGE2_STEPS:
            print(f"\n[2v1] === Stage 3: Full training, formation_w={new_fw:.2f} ===")

        # ── Termination logging ──────────────────────────────────────
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    self._term_total += 1
                    reason = infos[i].get("termination_reason", "unknown")
                    self._term_counts[reason] += 1

        if self.n_calls % 10_000 == 0:
            pct = 100.0 * self.num_timesteps / self._total_steps
            parts = [f"{k}={v}" for k, v in sorted(self._term_counts.items())]
            stage = 1 if self.num_timesteps < STAGE1_STEPS else (
                2 if self.num_timesteps < STAGE2_STEPS else 3)
            print(f"[2v1 S{stage}] step={self.num_timesteps:>7d}  "
                  f"progress={pct:.0f}%  terms:[{' '.join(parts)}]")
            self._term_counts.clear()
            self._term_total = 0

        return True

    def _unfreeze_actor(self):
        """Unfreeze Actor layers with very low LR."""
        for name, param in self._model.policy.named_parameters():
            if 'value' not in name:
                param.requires_grad = True
        # Update optimizer param groups
        self._model.policy.optimizer.param_groups[0]['lr'] = ACTOR_LR_STAGE2
        # Re-create optimizer with separate groups
        policy_params = [p for n, p in self._model.policy.named_parameters()
                        if 'value' not in n]
        value_params = [p for n, p in self._model.policy.named_parameters()
                       if 'value' in n]
        self._model.policy.optimizer = torch.optim.Adam([
            {'params': policy_params, 'lr': ACTOR_LR_STAGE2},
            {'params': value_params, 'lr': CRITIC_LR},
        ])


def build_env(difficulty: float = 0.0):
    return FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)


def train(seed: int = 42, total_steps: int = TOTAL_TIMESTEPS, difficulty: float = 0.0,
          tiled_model: str | None = None):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"formation_2v1_{ts}_s{seed}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[2v1] Formation training — 2 pursuers + 1 target")
    print(f"  Steps: {total_steps:,}  Difficulty: {difficulty:.2f}")
    print(f"  Stage 1: Critic warm-up (0–{STAGE1_STEPS//1000}K, Actor frozen)")
    print(f"  Stage 2: Actor unfrozen ({STAGE1_STEPS//1000}K–{STAGE2_STEPS//1000}K, LR={ACTOR_LR_STAGE2})")
    print(f"  Log: {log_dir}")

    env = build_env(difficulty=difficulty)
    env = Monitor(env)

    # Build model
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                         activation_fn=torch.nn.Tanh)
    model = PPO("MlpPolicy", env, learning_rate=CRITIC_LR, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
                verbose=1, seed=seed, device="auto")

    # ── Load checkpoint weights (policy only, skip optimizer) ─────────
    if tiled_model:
        print(f"  Loading checkpoint: {tiled_model}")
        import zipfile, io
        with zipfile.ZipFile(tiled_model, 'r') as zf:
            with zf.open('policy.pth') as f:
                policy_state = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                          weights_only=True)
        model.policy.load_state_dict(policy_state)
        print("  Policy weights loaded (optimizer fresh).")

    # ── Stage 1: Freeze Actor ───────────────────────────────────────
    for name, param in model.policy.named_parameters():
        if 'value' not in name:
            param.requires_grad = False
    # Rebuild optimizer with frozen policy params
    policy_params = [p for n, p in model.policy.named_parameters() if 'value' not in n]
    value_params = [p for n, p in model.policy.named_parameters() if 'value' in n]
    model.policy.optimizer = torch.optim.Adam([
        {'params': policy_params, 'lr': 0.0},
        {'params': value_params, 'lr': CRITIC_LR},
    ])
    print(f"  Stage 1: Actor frozen ({sum(p.numel() for p in policy_params):,} params)")
    print(f"           Critic active ({sum(p.numel() for p in value_params):,} params, LR={CRITIC_LR})")

    # ── Train ───────────────────────────────────────────────────────
    callback = Staged2v1Callback(model, total_steps, train_env=env, verbose=1)
    model.learn(total_timesteps=total_steps, callback=callback,
                tb_log_name="formation_2v1", progress_bar=False)

    model.save(os.path.join(log_dir, "formation_2v1_final"))
    print(f"[2v1] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--tiled-model", type=str,
                        default="data/expert/tiled_2v1_phase36.zip")
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(seed=args.seed, total_steps=args.steps, difficulty=args.difficulty,
          tiled_model=args.tiled_model)
