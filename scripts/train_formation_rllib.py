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

    Maps AttentionFormationActor state_dict keys from the BC checkpoint
    into RLlib's TorchModelV2 namespace (prefix: "actor.").

    BC checkpoint format:
      {"actor_state": AttentionFormationActor.state_dict(), "val_loss": ..., "epoch": ...}
    or:
      {"actor": AttentionFormationActor.state_dict(), ...}

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

    # Determine the correct key for actor state dict
    if "actor_state" in bc_ckpt:
        bc_state = bc_ckpt["actor_state"]
        key_name = "actor_state"
    elif "actor" in bc_ckpt:
        bc_state = bc_ckpt["actor"]
        key_name = "actor"
    else:
        print(f"[BC Load] ERROR: Unknown checkpoint format. Keys: {list(bc_ckpt.keys())}")
        return False

    print(f"[BC Load] Loading BC weights from '{key_name}' "
          f"({len(bc_state)} params, val_loss={bc_ckpt.get('val_loss', 'N/A')})")

    for policy_id in policy_ids:
        policy = algo.get_policy(policy_id)
        rllib_model = policy.model  # RLlibAttentionActor instance

        # Map BC keys to RLlib namespace: bc_key → "actor.{bc_key}"
        rllib_state = rllib_model.state_dict()
        mapped_state = {}
        skipped = 0

        for bc_key, bc_val in bc_state.items():
            rllib_key = f"actor.{bc_key}"
            if rllib_key in rllib_state:
                # Check shape match
                if rllib_state[rllib_key].shape == bc_val.shape:
                    mapped_state[rllib_key] = bc_val
                else:
                    print(f"  [WARN] Shape mismatch for {rllib_key}: "
                          f"RLlib={rllib_state[rllib_key].shape}, BC={bc_val.shape}")
                    skipped += 1
            else:
                skipped += 1

        if not mapped_state:
            print(f"[BC Load] WARNING: No keys matched for {policy_id}. "
                  f"Available RLlib keys: {list(rllib_state.keys())[:5]}...")
            continue

        # Load with strict=False: Critic keys (not in BC) stay random-initialized
        missing, unexpected = rllib_model.load_state_dict(mapped_state, strict=False)
        print(f"[BC Load] {policy_id}: loaded {len(mapped_state)} keys, "
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
    checkpoint_freq: int = 50,
    eval_interval: int = 25,
    eval_episodes: int = 20,
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
                "shared_policy": (
                    None, obs_space_p0, act_space_p0,
                    {"model": {"custom_model": "attention_formation"}}
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .training(
            lr=3e-4,
            train_batch_size=8192,
            minibatch_size=1024,
            num_epochs=10,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            model={"vf_share_layers": False},
        )
        .debugging(seed=seed)
    )

    algo = config.build()
    print(f"[RLlib MAPPO] Algorithm built: {type(algo).__name__}")

    # ── BC weight hot-loading ─────────────────────────────────────────────
    if load_bc:
        success = load_bc_weights(algo, load_bc, ["shared_policy"])
        if not success:
            print("[BC Load] Continuing with random initialization...")
        else:
            print("[BC Load] Successfully loaded BC weights into shared policy")

    # ── Training loop ─────────────────────────────────────────────────────
    current_phase = COOP_PHASE_OR
    coop_warmup_done = False
    best_avg_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"RLlib MAPPO 2v1 Cooperative Formation Training")
    print(f"Run: {run_name}")
    print(f"Mode: {'Cooperative (OR→AND)' if cooperative else 'Non-cooperative'}")
    print(f"Iterations: {iterations}  |  Difficulty: {difficulty:.2f}  |  Seed: {seed}")
    print(f"BC Pretrain: {load_bc or 'None'}")
    print(f"Architecture: Parameter-Shared MAPPO (shared_policy for p0/p1)")
    print(f"Decision Rate: 5 Hz (DECISION_DT=0.2s)")
    print(f"{'='*60}\n")

    try:
        for i in range(iterations):
            result = algo.train()

            # Extract metrics
            env_stats = result.get("env_runners", result)
            ep_rew = env_stats.get("episode_reward_mean", 0.0)
            ep_len = env_stats.get("episode_len_mean", 0.0)

            # Shared policy metrics
            policy_rewards = env_stats.get("policy_reward_mean", {})
            shared_r = policy_rewards.get("shared_policy", 0.0)

            # Entropy / KL
            info = result.get("info", {})
            learner_info = info.get("learner", {})
            shared_learner = learner_info.get("shared_policy", {})
            learner_stats = shared_learner.get("learner_stats", shared_learner)
            entropy = learner_stats.get("entropy", 0.0)
            kl = learner_stats.get("kl", 0.0)

            if i % 10 == 0:
                print(f"[{i:4d}] ep_rew={ep_rew:8.1f}  "
                      f"policy_r={shared_r:8.1f}  "
                      f"ep_len={ep_len:6.1f}  "
                      f"ent={entropy:.4f}  kl={kl:.4f}")

            # ── Cooperative phase switching ──────────────────────────────
            if cooperative and not coop_warmup_done and warmup_steps > 0:
                total_env_steps = result.get("env_runners", {}).get(
                    "num_env_steps_sampled", 0)
                if total_env_steps >= warmup_steps:
                    print(f"\n>>> Phase 2: Switching to AND-gate "
                          f"(800m/30°) at {total_env_steps} steps\n")
                    current_phase = COOP_PHASE_AND
                    coop_warmup_done = True
                    # Set phase on all env runners
                    def set_and_phase(env):
                        if hasattr(env, 'set_coop_phase'):
                            env.set_coop_phase(COOP_PHASE_AND)
                    try:
                        algo.env_runner_group.foreach_env(set_and_phase)
                    except Exception as e:
                        print(f"  [WARN] Could not set coop phase on workers: {e}")

            # ── Evaluation ───────────────────────────────────────────────
            if eval_interval > 0 and (i + 1) % eval_interval == 0:
                eval_rewards = run_evaluation(
                    algo, eval_episodes, difficulty, current_phase, cooperative)
                avg_eval = np.mean(eval_rewards) if eval_rewards else 0.0
                print(f"  [EVAL] iter={i+1:4d}  avg_rew={avg_eval:8.1f}  "
                      f"n={len(eval_rewards)}")

                if avg_eval > best_avg_reward:
                    best_avg_reward = avg_eval
                    best_path = os.path.join(
                        project_root, "checkpoints",
                        f"best_iter_{i+1:04d}_rew_{avg_eval:.0f}")
                    algo.save(best_path)
                    print(f"  [SAVE] New best: {best_path}")

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
                   coop_phase: int, cooperative_mode: bool = True) -> list[float]:
    """Evaluate the current policy in a separate env instance.

    Returns per-episode total reward.
    """
    env = FormationRLlibEnv({
        "difficulty_level": difficulty,
        "lock_altitude": True,
        "record_tacview": False,
        "cooperative_mode": cooperative_mode,
    })
    if cooperative_mode:
        env.set_coop_phase(coop_phase)
    env._difficulty = difficulty

    episode_rewards = []

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
                        policy_id="shared_policy",
                        explore=False,
                    )

            obs_dict, rewards, terminateds, truncateds, _ = env.step(actions)
            total_r += sum(rewards.values())

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        episode_rewards.append(total_r)

    env.close()
    return episode_rewards


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
                       help="Path to BC pretrained checkpoint")
    parser.add_argument("--no-bc", action="store_true", default=False,
                       help="Skip BC loading even if checkpoint available")

    # Checkpointing
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--eval-interval", type=int, default=25,
                       help="Run evaluation every N iterations")
    parser.add_argument("--eval-episodes", type=int, default=20,
                       help="Number of evaluation episodes")

    args = parser.parse_args()

    load_bc_path = None
    if args.load_bc and not args.no_bc:
        load_bc_path = args.load_bc
    elif not args.no_bc:
        # Default BC checkpoint
        default_bc = "data/expert/attention_bc_2v1_filtered_pretrained.pth"
        if os.path.exists(default_bc):
            load_bc_path = default_bc

    train(
        iterations=args.iterations,
        difficulty=args.difficulty,
        cooperative=args.cooperative,
        warmup_steps=args.warmup,
        load_bc=load_bc_path,
        checkpoint_freq=args.checkpoint_freq,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
