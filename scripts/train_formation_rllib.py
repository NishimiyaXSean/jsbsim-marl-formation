"""RLlib MAPPO 2v1 Cooperative Formation Training (Parameter-Shared CTDE).

Rebuilt training pipeline using Ray RLlib for scalable multi-agent training.
Features:
  - Parameter-Shared MAPPO (shared policy across p0/p1 agents)
  - Self-Attention CTDE model (AttentionFormationActor + AttentionCritic)
  - BC weight hot-loading (inject pretrained weights before PPO)
  - Two-phase cooperative training (OR-gate → AND-gate)
  - Phase 5 pincer rewards + dynamic role assignment + distance asymmetry penalty
  - 5 Hz decision rate (was 2 Hz)

Usage:
    conda activate marl_env
    python scripts/train_formation_rllib.py
    python scripts/train_formation_rllib.py --iterations 500 --difficulty 0.0
    python scripts/train_formation_rllib.py --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth
    python scripts/train_formation_rllib.py --cooperative --warmup 200000
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext

from src.environment.formation_rllib_env import FormationRLlibEnv, COOP_PHASE_OR, COOP_PHASE_AND
from src.models.formation_rllib_model import RLlibAttentionActor

# ── AND-gate curriculum stages ──────────────────────────────────────────────
# Performance-based scheduler: progression gated by eval sync-entry rate.
# Each stage defines: (and_dist, and_angle, bearing_min, bearing_max)
CURRICULUM_STAGES = {
    1: {"and_dist": 1200.0, "and_angle": 35.0, "bearing_min": -30.0, "bearing_max": 30.0,
        "target_dist_min": 1600.0, "target_dist_max": 2200.0, "sustain_steps": 2},
    2: {"and_dist": 1000.0, "and_angle": 30.0, "bearing_min": -45.0, "bearing_max": 45.0,
        "target_dist_min": 1200.0, "target_dist_max": 2000.0, "sustain_steps": 4},
    3: {"and_dist": 800.0, "and_angle": 20.0, "bearing_min": -180.0, "bearing_max": 180.0,
        "target_dist_min": 900.0, "target_dist_max": 1800.0, "sustain_steps": 6},
}
CURRICULUM_WINDOW = 3        # number of eval rounds for moving-average sync rate
CURRICULUM_MIN_WINDOW = 3    # minimum evals before checking gate
CURRICULUM_STAGE1_GATE = 0.60  # stage 1→2: >60% sync rate
CURRICULUM_STAGE2_GATE = 0.50  # stage 2→3: >50% sync rate


TRANSITION_ITERS = 15  # iterations over which to smoothly interpolate stage params

def _apply_curriculum_stage(algo, stage: int) -> None:
    """Hot-update all env runners to use a new curriculum stage (immediate)."""
    params = CURRICULUM_STAGES[stage]

    def _set_stage(env):
        if hasattr(env, "set_curriculum_stage_full"):
            env.set_curriculum_stage_full(
                stage, params["and_dist"], params["and_angle"],
                params["bearing_min"], params["bearing_max"],
                params.get("target_dist_min", 900.0),
                params.get("target_dist_max", 1300.0),
                params.get("sustain_steps", 6))

    try:
        algo.env_runner_group.foreach_env(_set_stage)
    except Exception as e:
        print(f"  [WARN] Could not apply curriculum stage {stage}: {e}")


def _interpolate_params(old_stage: int, new_stage: int, progress: float) -> dict:
    """Linearly interpolate curriculum params between two stages."""
    old = CURRICULUM_STAGES[old_stage]
    new = CURRICULUM_STAGES[new_stage]
    t = max(0.0, min(1.0, progress))
    return {
        "stage": new_stage,
        "and_dist": old["and_dist"] * (1 - t) + new["and_dist"] * t,
        "and_angle": old["and_angle"] * (1 - t) + new["and_angle"] * t,
        "bearing_min": old["bearing_min"] * (1 - t) + new["bearing_min"] * t,
        "bearing_max": old["bearing_max"] * (1 - t) + new["bearing_max"] * t,
        "target_dist_min": old.get("target_dist_min", 900) * (1 - t) + new.get("target_dist_min", 900) * t,
        "target_dist_max": old.get("target_dist_max", 1300) * (1 - t) + new.get("target_dist_max", 1300) * t,
        "sustain_steps": round(old.get("sustain_steps", 6) * (1 - t) + new.get("sustain_steps", 6) * t),
    }


def _apply_smooth_transition(algo, old_stage: int, new_stage: int, progress: float) -> None:
    """Apply interpolated curriculum params during transition."""
    p = _interpolate_params(old_stage, new_stage, progress)

    def _set_transition(env):
        if hasattr(env, "set_curriculum_stage_full"):
            env.set_curriculum_stage_full(
                p["stage"], p["and_dist"], p["and_angle"],
                p["bearing_min"], p["bearing_max"],
                p["target_dist_min"], p["target_dist_max"],
                p.get("sustain_steps", 6))

    try:
        algo.env_runner_group.foreach_env(_set_transition)
    except Exception as e:
        print(f"  [WARN] Transition update failed: {e}")

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Registration
# ═══════════════════════════════════════════════════════════════════════════════

def env_creator(config: EnvContext):
    return FormationRLlibEnv(config)


# ═══════════════════════════════════════════════════════════════════════════════
#  BC Weight Hot-Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_bc_weights(algo, bc_ckpt_path: str, policy_ids: list[str]) -> bool:
    """Inject BC-pretrained Actor weights into RLlib policies.

    Supports two checkpoint formats:
      1. Continuous BC:
         {"actor_state": AttentionFormationActor.state_dict(), "val_loss": ..., "epoch": ...}
      2. Discrete BC:
         {"actor_state": AttentionFormationActor.state_dict(),   ← backbone
          "turn_head.weight": ..., "turn_head.bias": ...,         ← discrete heads (top-level)
          "speed_head.weight": ..., "speed_head.bias": ...,
          "val_loss": ..., "epoch": ...}

    Backbone keys are prefixed with "actor." (RLlibAttentionActor.actor).
    Discrete head keys map directly (RLlibAttentionActor.turn_head / .speed_head).

    Args:
        algo: Built RLlib PPO algorithm instance.
        bc_ckpt_path: Path to BC checkpoint .pth file.
        policy_ids: List of policy IDs to load weights into (e.g., ["p0_policy", "p1_policy"]).

    Returns:
        True if successful, False if checkpoint couldn't be loaded.
    """
    try:
        bc_ckpt = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[BC Load] ERROR: Cannot load checkpoint: {e}")
        return False

    # ── Determine discrete vs continuous ───────────────────────────────────
    DISCRETE_HEAD_KEYS = {
        "turn_head.weight", "turn_head.bias",
        "speed_head.weight", "speed_head.bias",
    }
    has_discrete_heads = any(k in bc_ckpt for k in DISCRETE_HEAD_KEYS)

    # Determine the backbone state dict key
    if "actor_state" in bc_ckpt:
        bc_state = bc_ckpt["actor_state"]
        key_name = "actor_state"
    elif "actor" in bc_ckpt:
        bc_state = bc_ckpt["actor"]
        key_name = "actor"
    else:
        print(f"[BC Load] ERROR: Unknown checkpoint format. Keys: {list(bc_ckpt.keys())}")
        return False

    bc_type = "discrete" if has_discrete_heads else "continuous"
    print(f"[BC Load] Loading {bc_type} BC weights from '{key_name}' "
          f"({len(bc_state)} backbone params, val_loss={bc_ckpt.get('val_loss', 'N/A')})")
    if has_discrete_heads:
        tw = bc_ckpt.get('turn_head.weight', None)
        sw = bc_ckpt.get('speed_head.weight', None)
        print(f"[BC Load]   + discrete heads: turn_head ({tw.shape if tw is not None else 'N/A'}), "
              f"speed_head ({sw.shape if sw is not None else 'N/A'})")

    for policy_id in policy_ids:
        policy = algo.get_policy(policy_id)
        rllib_model = policy.model  # RLlibAttentionActor instance

        rllib_state = rllib_model.state_dict()
        mapped_state = {}
        skipped = 0
        head_loaded = 0

        # ── Phase 0: Surgical padding for obs-dim expansion ──────────
        # BC was trained on 33-dim obs (self=13, mate=6). New model uses
        # 39-dim obs (self=15, mate=10). Pad projection layers with zeros.
        surgical_pads = [
            ("self_proj.weight", "actor.self_proj.weight", 13),  # +2 onehot
            ("mate_proj.weight", "actor.mate_proj.weight", 6),   # +4 broadcast
        ]
        for bc_key, rllib_key, old_cols in surgical_pads:
            if bc_key in bc_state and rllib_key in rllib_state:
                bc_w = bc_state[bc_key]
                target_shape = rllib_state[rllib_key].shape
                if bc_w.shape != target_shape:
                    new_w = torch.zeros(target_shape, dtype=bc_w.dtype, device=bc_w.device)
                    new_w[:, :old_cols] = bc_w[:, :old_cols]  # preserve BC weights
                    bc_state[bc_key] = new_w
                    print(f"[BC Load]   + surgically padded {bc_key}: "
                          f"{list(bc_w.shape)} → {list(new_w.shape)}")

        # ── Phase 1: Backbone keys (actor_state → "actor.*") ───────────
        for bc_key, bc_val in bc_state.items():
            rllib_key = f"actor.{bc_key}"
            if rllib_key in rllib_state:
                if rllib_state[rllib_key].shape == bc_val.shape:
                    mapped_state[rllib_key] = bc_val
                else:
                    print(f"  [WARN] Shape mismatch for {rllib_key}: "
                          f"RLlib={rllib_state[rllib_key].shape}, BC={bc_val.shape}")
                    skipped += 1
            else:
                skipped += 1

        # ── Phase 2: Discrete head keys (top-level → direct mapping) ───
        if has_discrete_heads:
            for bc_key in DISCRETE_HEAD_KEYS:
                if bc_key not in bc_ckpt:
                    continue
                bc_val = bc_ckpt[bc_key]
                # Direct mapping: turn_head.weight → turn_head.weight (no prefix)
                if bc_key in rllib_state:
                    if rllib_state[bc_key].shape == bc_val.shape:
                        mapped_state[bc_key] = bc_val
                        head_loaded += 1
                    else:
                        print(f"  [WARN] Shape mismatch for discrete head {bc_key}: "
                              f"RLlib={rllib_state[bc_key].shape}, BC={bc_val.shape}")
                        skipped += 1
                else:
                    print(f"  [WARN] Discrete head key {bc_key} not found in RLlib model")
                    skipped += 1

        if not mapped_state:
            print(f"[BC Load] WARNING: No keys matched for {policy_id}. "
                  f"Available RLlib keys: {list(rllib_state.keys())[:5]}...")
            continue

        # Load with strict=False: Critic keys (not in BC) stay random-initialized
        missing, unexpected = rllib_model.load_state_dict(mapped_state, strict=False)
        print(f"[BC Load] {policy_id}: loaded {len(mapped_state)} keys "
              f"(backbone={len(mapped_state) - head_loaded}, heads={head_loaded}), "
              f"skipped={skipped}, missing_critic={len(missing)}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    iterations: int = 500,
    difficulty: float = 0.0,
    cooperative: bool = True,
    warmup_steps: int = 0,
    load_bc: str | None = None,
    load_discrete_bc: str | None = None,
    resume_from: str | None = None,
    checkpoint_freq: int = 50,
    eval_interval: int = 25,
    eval_episodes: int = 20,
    lr: float | None = None,
    entropy_coeff: float = 0.03,
    seed: int = 42,
):
    # ── Setup ────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"rllib_formation_{timestamp}_s{seed}"
    project_root = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(f"{project_root}/checkpoints", exist_ok=True)

    ray.init(ignore_reinit_error=True, num_cpus=4)

    env_name = "formation_2v1_rllib"
    tune.register_env(env_name, env_creator)
    ModelCatalog.register_custom_model("attention_formation", RLlibAttentionActor)

    # Probe env for spaces
    temp_env = env_creator({"difficulty_level": difficulty})
    obs_space_p0 = temp_env.observation_space["p0"]
    act_space_p0 = temp_env.action_space["p0"]
    obs_space_p1 = temp_env.observation_space["p1"]
    act_space_p1 = temp_env.action_space["p1"]
    temp_env.close()

    print(f"Observation space (p0): {obs_space_p0}")
    print(f"Action space (p0):      {act_space_p0}")

    # ── BC path resolution + LR auto-adjust (must be before config) ──────
    bc_path = load_discrete_bc or load_bc
    if lr is None:
        lr = 2e-4 if bc_path else 3e-4
    if resume_from:
        print(f"[LR] lr={lr:.1e} (resume — explicit or default)")
    else:
        print(f"[LR] lr={lr:.1e} ({'BC hotstart — 0.67× cold-start' if bc_path else 'cold-start — standard'})")

    # ── PPOConfig with Parameter-Shared MAPPO ──────────────────────────
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,       # Old API for TorchModelV2
            enable_env_runner_and_connector_v2=False,  # Prevent Rollout Worker crash
        )
        .environment(env_name, env_config={
            "difficulty_level": difficulty,
            "lock_altitude": True,
            "record_tacview": False,
            "cooperative_mode": cooperative,
        })
        .framework("torch")
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(num_env_runners=2)
        .multi_agent(
            policies={
                "policy_0": (
                    None, obs_space_p0, act_space_p0,
                    {"model": {"custom_model": "attention_formation"}}
                ),
                "policy_1": (
                    None, obs_space_p1, act_space_p1,
                    {"model": {"custom_model": "attention_formation"}}
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id.split('_')[-1]}",
            policies_to_train=["policy_0", "policy_1"],
        )
        .training(
            lr=lr,
            train_batch_size=8192,
            minibatch_size=1024,
            num_epochs=10,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=entropy_coeff,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            model={"vf_share_layers": False},
        )
        .debugging(seed=seed)
    )

    algo = config.build()
    print(f"[RLlib MAPPO] Algorithm built: {type(algo).__name__}")

    # ── BC weight hot-loading (skip when resuming — checkpoint has weights) ─
    if bc_path and not resume_from:
        success = load_bc_weights(algo, bc_path, ["policy_0", "policy_1"])
        if not success:
            print("[BC Load] Continuing with random initialization...")
        else:
            bc_type = "discrete" if load_discrete_bc else "continuous"
            print(f"[BC Load] Successfully loaded {bc_type} BC weights into independent policies")

    # ── Training loop ─────────────────────────────────────────────────────
    current_phase = COOP_PHASE_OR
    coop_warmup_done = False
    best_avg_reward = -float("inf")
    curriculum_stage = 0     # 0=pre-AND, 1/2/3=curriculum stages
    sync_history = []        # rolling window of sync rates for stage gating
    # Smooth transition state
    transition_active = False
    transition_start_iter = 0
    target_stage = 0

    # If --resume-from is set, restore checkpoint and skip warmup
    and_start_iter = 0
    if resume_from:
        print(f"\n[Resume] Restoring from: {resume_from}")
        algo.restore(resume_from)
        if cooperative and warmup_steps > 0:
            coop_warmup_done = True
            current_phase = COOP_PHASE_AND
            curriculum_stage = 1
            and_start_iter = 0
            print(f"[Resume] Warmup skipped — starting directly in AND-gate Stage 1")
            p = CURRICULUM_STAGES[1]
            print(f"[Resume]    AND: {p}")
            def _resume_stage1(env):
                if hasattr(env, 'set_coop_phase'):
                    env.set_coop_phase(COOP_PHASE_AND)
                if hasattr(env, 'set_curriculum_stage_full'):
                    env.set_curriculum_stage_full(
                        1, p["and_dist"], p["and_angle"],
                        p["bearing_min"], p["bearing_max"],
                        p.get("target_dist_min", 900.0),
                        p.get("target_dist_max", 1300.0),
                        p.get("sustain_steps", 2))
            try:
                algo.env_runner_group.foreach_env(_resume_stage1)
            except Exception as e:
                print(f"  [WARN] Could not set stage 1 on workers: {e}")

    print(f"\n{'='*60}")
    print(f"RLlib MAPPO 2v1 Cooperative Formation Training")
    print(f"Run: {run_name}")
    print(f"Mode: {'Cooperative (OR→AND)' if cooperative else 'Non-cooperative'}")
    print(f"Iterations: {iterations}  |  Difficulty: {difficulty:.2f}  |  Seed: {seed}")
    print(f"BC Pretrain: {load_bc or 'None'}")
    print(f"Architecture: Independent-Actor MAPPO (policy_0 / policy_1, no parameter sharing)")
    print(f"Action Space: MultiDiscrete([5 turn, 3 speed]) = 15 primitives")
    print(f"Decision Rate: 5 Hz (DECISION_DT=0.2s)")
    print(f"{'='*60}\n")

    try:
        for i in range(iterations):
            result = algo.train()

            # Extract metrics
            env_stats = result.get("env_runners", result)
            ep_rew = env_stats.get("episode_reward_mean", 0.0)
            ep_len = env_stats.get("episode_len_mean", 0.0)

            # Independent policy metrics (avg across P0/P1)
            policy_rewards = env_stats.get("policy_reward_mean", {})
            p0_r = policy_rewards.get("policy_0", 0.0)
            p1_r = policy_rewards.get("policy_1", 0.0)
            avg_policy_r = (p0_r + p1_r) / 2.0

            # Entropy / KL (use policy_0)
            info = result.get("info", {})
            learner_info = info.get("learner", {})
            p0_learner = learner_info.get("policy_0", {})
            learner_stats = p0_learner.get("learner_stats", p0_learner)
            entropy = learner_stats.get("entropy", 0.0)
            kl = learner_stats.get("kl", 0.0)

            if i % 10 == 0:
                print(f"[{i:4d}] ep_rew={ep_rew:8.1f}  "
                      f"policy_r={avg_policy_r:8.1f}  "
                      f"ep_len={ep_len:6.1f}  "
                      f"ent={entropy:.4f}  kl={kl:.4f}")

            # ── Cooperative phase switching ──────────────────────────────
            if cooperative and not coop_warmup_done and warmup_steps > 0:
                # Use iteration counter * train_batch_size as step estimate
                total_env_steps = (i + 1) * 8192
                if total_env_steps >= warmup_steps:
                    current_phase = COOP_PHASE_AND
                    coop_warmup_done = True
                    curriculum_stage = 1
                    and_start_iter = i + 1
                    print(f"\n>>> AND-gate Activated: Curriculum Stage 1 (Greenhouse) "
                          f"at iter {and_start_iter} (~{total_env_steps} steps)")
                    # Apply Stage 1 on all workers
                    def _start_curriculum(env):
                        if hasattr(env, 'set_coop_phase'):
                            env.set_coop_phase(COOP_PHASE_AND)
                        if hasattr(env, 'set_curriculum_stage_full'):
                            p = CURRICULUM_STAGES[1]
                            env.set_curriculum_stage_full(
                                1, p["and_dist"], p["and_angle"],
                                p["bearing_min"], p["bearing_max"],
                                p.get("target_dist_min", 900.0),
                                p.get("target_dist_max", 1300.0),
                                p.get("sustain_steps", 2))
                    try:
                        algo.env_runner_group.foreach_env(_start_curriculum)
                        print(f"    AND: {CURRICULUM_STAGES[1]}")
                    except Exception as e:
                        print(f"  [WARN] Could not start curriculum: {e}")
                    print()

            # ── Dynamic AND-distance annealing (legacy, skipped in curriculum mode) ──
            if coop_warmup_done and cooperative and curriculum_stage == 0:
                # Legacy annealing: only used if warmup was set but curriculum
                # was somehow not activated (backward compat)
                from src.environment.formation_rllib_env import (
                    COOP_PHASE2_AND_DIST_INIT, COOP_PHASE2_AND_DIST)
                decay_end_iter = 200
                if i < decay_end_iter:
                    decay_progress = (i - and_start_iter) / max(decay_end_iter - and_start_iter, 1)
                    decay_progress = max(0.0, min(1.0, decay_progress))
                    current_and_dist = (COOP_PHASE2_AND_DIST_INIT -
                                       decay_progress *
                                       (COOP_PHASE2_AND_DIST_INIT - COOP_PHASE2_AND_DIST))
                else:
                    current_and_dist = COOP_PHASE2_AND_DIST

                # Broadcast to workers every 5 iters (or on significant change)
                if i % 5 == 0 or i == and_start_iter:
                    dist_val = current_and_dist
                    def set_dist(env):
                        if hasattr(env, 'set_and_distance'):
                            env.set_and_distance(dist_val)
                    try:
                        algo.env_runner_group.foreach_env(set_dist)
                    except Exception:
                        pass

                # Also set on the eval env (run_evaluation uses a standalone env)
                if i % 10 == 0:
                    print(f"    [AND-curriculum] iter={i+1:3d}  "
                          f"AND_dist={current_and_dist:.0f}m")

            # ── Smooth curriculum transition (every iter, not just eval) ───
            if coop_warmup_done and transition_active:
                transition_progress = (i - transition_start_iter) / TRANSITION_ITERS
                if transition_progress >= 1.0:
                    transition_active = False
                    curriculum_stage = target_stage
                    print(f"    Transition complete → Stage {target_stage}\n")
                else:
                    _apply_smooth_transition(algo, curriculum_stage, target_stage,
                                             transition_progress)

            # ── Evaluation ───────────────────────────────────────────────
            if eval_interval > 0 and (i + 1) % eval_interval == 0:
                # Determine AND distance and stage params for eval env
                if curriculum_stage >= 1:
                    eval_stage_params = dict(CURRICULUM_STAGES[curriculum_stage])
                    eval_stage_params["stage"] = curriculum_stage
                else:
                    eval_stage_params = None

                eval_rewards, sync_rate = run_evaluation(
                    algo, eval_episodes, difficulty, current_phase, cooperative,
                    stage_params=eval_stage_params)
                avg_eval = np.mean(eval_rewards) if eval_rewards else 0.0
                print(f"  [EVAL] iter={i+1:4d}  avg_rew={avg_eval:8.1f}  "
                      f"sync={sync_rate:.0%}  n={len(eval_rewards)}")

                if avg_eval > best_avg_reward:
                    best_avg_reward = avg_eval
                    best_path = os.path.join(
                        project_root, "checkpoints",
                        f"best_iter_{i+1:04d}_rew_{avg_eval:.0f}")
                    algo.save(best_path)
                    print(f"  [SAVE] New best: {best_path}")

                # ── AND-gate curriculum: performance-based stage advancement ──
                if coop_warmup_done and curriculum_stage < 3 \
                        and not transition_active:
                        sync_history.append(sync_rate)
                        if len(sync_history) > CURRICULUM_WINDOW:
                            sync_history = sync_history[-CURRICULUM_WINDOW:]
                        recent_sync = np.mean(sync_history)

                        if curriculum_stage == 1 and len(sync_history) >= CURRICULUM_MIN_WINDOW \
                                and recent_sync > CURRICULUM_STAGE1_GATE:
                            old_stage = curriculum_stage
                            target_stage = 2
                            transition_active = True
                            transition_start_iter = i + 1
                            sync_history = []
                            print(f"\n>>> Curriculum Stage 1→2: smooth transition "
                                  f"({TRANSITION_ITERS} iters) "
                                  f"(sync={recent_sync:.0%} > {CURRICULUM_STAGE1_GATE:.0%})\n")

                        elif curriculum_stage == 2 and len(sync_history) >= CURRICULUM_MIN_WINDOW \
                                and recent_sync > CURRICULUM_STAGE2_GATE:
                            old_stage = curriculum_stage
                            target_stage = 3
                            transition_active = True
                            transition_start_iter = i + 1
                            sync_history = []
                            print(f"\n>>> Curriculum Stage 2→3: smooth transition "
                                  f"({TRANSITION_ITERS} iters) "
                                  f"(sync={recent_sync:.0%} > {CURRICULUM_STAGE2_GATE:.0%})\n")

            # ── Periodic checkpoint ──────────────────────────────────────
            if checkpoint_freq > 0 and i > 0 and (i + 1) % checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    project_root, "checkpoints", f"checkpoint_{i+1:06d}")
                algo.save(ckpt_path)
                print(f"  [SAVE] Checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving final checkpoint...")
        algo.save(os.path.join(project_root, "checkpoints", "checkpoint_final"))

    finally:
        # Final save
        final_path = os.path.join(project_root, "checkpoints", "checkpoint_final")
        algo.save(final_path)
        print(f"[Final] Model saved: {final_path}")
        ray.shutdown()
        print("Training complete.")


def run_evaluation(algo, n_episodes: int, difficulty: float,
                   coop_phase: int, cooperative_mode: bool = True,
                   and_distance: float | None = None,
                   stage_params: dict | None = None) -> tuple[list[float], float]:
    """Evaluate the current policy in a separate env instance.

    Returns:
        (episode_rewards, sync_entry_rate) — sync rate is fraction of episodes
        that ended with cooperative_success (AND-gate) termination.
    """
    env = FormationRLlibEnv({
        "difficulty_level": difficulty,
        "lock_altitude": True,
        "record_tacview": False,
        "cooperative_mode": cooperative_mode,
    })
    if cooperative_mode:
        env.set_coop_phase(coop_phase)
    if stage_params is not None:
        if hasattr(env, 'set_curriculum_stage_full'):
            env.set_curriculum_stage_full(
                stage_params.get("stage", 1),
                stage_params.get("and_dist", 2000.0),
                stage_params.get("and_angle", 40.0),
                stage_params.get("bearing_min", -20.0),
                stage_params.get("bearing_max", 20.0),
                stage_params.get("target_dist_min", 900.0),
                stage_params.get("target_dist_max", 1300.0),
                stage_params.get("sustain_steps", 6))
    elif and_distance is not None:
        env.set_and_distance(and_distance)
    env._difficulty = difficulty

    episode_rewards = []
    coop_success_count = 0

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        total_r = 0.0

        while not done:
            actions = {}
            for aid in env._agent_ids:
                if aid in obs_dict:
                    actions[aid] = algo.compute_single_action(
                        obs_dict[aid],
                        policy_id=f"policy_{aid.split('_')[-1]}",
                        explore=False,
                    )

            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            total_r += sum(rewards.values())

            # Track cooperative success in AND-gate phase
            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
            if done:
                # Check if termination was cooperative_success via env's internal state
                reason = getattr(env, "_last_termination_reason", "timeout")
                if reason == "cooperative_success":
                    coop_success_count += 1

        episode_rewards.append(total_r)

    sync_rate = coop_success_count / n_episodes if n_episodes > 0 else 0.0
    env.close()
    return episode_rewards, sync_rate


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLlib MAPPO 2v1 Cooperative Formation Training")

    # Training
    parser.add_argument("--iterations", type=int, default=500,
                       help="Number of training iterations")
    parser.add_argument("--difficulty", type=float, default=0.0,
                       help="Initial difficulty level [0, 1]")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Cooperative
    parser.add_argument("--cooperative", action="store_true", default=True,
                       help="Enable cooperative Phase 5 (pincer+AND-gate)")
    parser.add_argument("--no-cooperative", action="store_false", dest="cooperative",
                       help="Disable cooperative mode")
    parser.add_argument("--warmup", type=int, default=0,
                       help="Env steps before switching to AND-gate (0 = OR only)")

    # BC
    parser.add_argument("--load-bc", type=str, default=None,
                       help="Path to continuous BC checkpoint (legacy)")
    parser.add_argument("--load-discrete-bc", type=str, default=None,
                       help="Path to discrete BC checkpoint (MultiDiscrete)")
    parser.add_argument("--no-bc", action="store_true", default=False,
                       help="Skip BC loading")

    # Checkpointing
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: 2e-4 with BC, 3e-4 cold-start)")
    parser.add_argument("--entropy-coeff", type=float, default=0.03,
                       help="Entropy coefficient for PPO (higher = more exploration)")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--eval-interval", type=int, default=25,
                       help="Run evaluation every N iterations")
    parser.add_argument("--eval-episodes", type=int, default=20,
                       help="Number of evaluation episodes")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from RLlib checkpoint directory")

    args = parser.parse_args()

    load_bc_path = None
    load_disc_bc_path = None
    if not args.no_bc:
        if args.load_discrete_bc:
            load_disc_bc_path = args.load_discrete_bc
        elif args.load_bc:
            load_bc_path = args.load_bc
        else:
            # Default: prefer discrete BC if available
            default_disc = "data/expert/discrete_attention_bc.pth"
            if os.path.exists(default_disc):
                load_disc_bc_path = default_disc
            elif os.path.exists("data/expert/attention_bc_2v1_filtered_pretrained.pth"):
                load_bc_path = "data/expert/attention_bc_2v1_filtered_pretrained.pth"

    train(
        iterations=args.iterations,
        difficulty=args.difficulty,
        cooperative=args.cooperative,
        warmup_steps=args.warmup,
        load_bc=load_bc_path,
        load_discrete_bc=load_disc_bc_path,
        resume_from=args.resume_from,
        checkpoint_freq=args.checkpoint_freq,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
    )
