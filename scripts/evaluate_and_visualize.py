"""Evaluate a trained agent and visualize the result.

Produces:
  1. Tacview ACMI file  (open with Tacview)
  2. 3D trajectory plot as PNG
  3. Console summary of the engagement

Usage:
  conda activate jsbsim_rl
  python scripts/evaluate_and_visualize.py
  python scripts/evaluate_and_visualize.py --model sb3_continuous  # default
  python scripts/evaluate_and_visualize.py --model sb3_bfm
  python scripts/evaluate_and_visualize.py --model mappo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.air_combat_env import AirCombatEnv
from src.environment.rewards import RewardConfig

# ── Matplotlib ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_sb3_model(run_dir: str, action_mode: str):
    from stable_baselines3 import PPO
    model_path = os.path.join(run_dir, "attacker_policy.zip")
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    print(f"  Loading SB3 PPO model: {model_path}")
    return PPO.load(model_path)


def load_mappo_model(run_dir: str):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from ray.rllib.models import ModelCatalog
    from src.models.mappo_model import MAPPOModel

    ray.init(ignore_reinit_error=True)
    register_env("air_combat_1v1", lambda cfg: AirCombatEnv(gui=False, record_tacview=False))
    ModelCatalog.register_custom_model("mappo_ctde_model", MAPPOModel)

    temp_env = AirCombatEnv(gui=False, record_tacview=False)
    algo = (
        PPOConfig()
        .environment("air_combat_1v1")
        .framework("torch")
        .resources(num_gpus=0)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={
                "policy_attacker": (None, temp_env.observation_spaces["attacker_0"], temp_env.action_spaces["attacker_0"], {}),
                "policy_evader": (None, temp_env.observation_spaces["evader_0"], temp_env.action_spaces["evader_0"], {}),
            },
            policy_mapping_fn=lambda agent_id, *a, **kw: "policy_attacker" if agent_id == "attacker_0" else "policy_evader",
        )
        .training(model={"custom_model": "mappo_ctde_model"})
    ).build()

    # Find the checkpoint
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    candidates = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_best")])
    if not candidates:
        candidates = sorted(os.listdir(ckpt_dir))
    assert candidates, f"No checkpoint found in {ckpt_dir}"
    ckpt_path = os.path.join(ckpt_dir, candidates[-1])
    print(f"  Loading MAPPO checkpoint: {ckpt_path}")
    algo.restore(ckpt_path)
    return algo


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluate(model, action_mode: str, use_mappo: bool, n_episodes: int = 5):
    """Run evaluation episodes, return trajectory data and Tacview frames."""
    env = AirCombatEnv(gui=False, record_tacview=True, action_mode=action_mode)
    env.set_curriculum_stage(1)  # easiest stage for demo

    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}

        ep_traj = {
            "attacker_ned": [],
            "evader_ned": [],
            "attacker_rpy": [],
            "evader_rpy": [],
            "times": [],
            "reasons": {},
            "total_reward": 0.0,
        }

        while not (terminated["__all__"] or truncated["__all__"]):
            t = env.step_counter / env.CTRL_FREQ

            # Record trajectory
            ep_traj["attacker_ned"].append(env.attacker.position_ned.copy())
            ep_traj["evader_ned"].append(env.evader.position_ned.copy())
            ep_traj["attacker_rpy"].append(env.attacker.rpy_rad.copy())
            ep_traj["evader_rpy"].append(env.evader.rpy_rad.copy())
            ep_traj["times"].append(t)

            # Get actions
            if use_mappo:
                action_A = model.compute_single_action(
                    obs["attacker_0"], policy_id="policy_attacker", explore=False
                )
                actions = {"attacker_0": action_A}
                if "evader_0" in obs:
                    action_E = model.compute_single_action(
                        obs["evader_0"], policy_id="policy_evader", explore=False
                    )
                    actions["evader_0"] = action_E
            else:
                # SB3: single-agent wrapper — attacker = model, evader = fixed
                action_A, _ = model.predict(obs["attacker_0"]["obs"], deterministic=True)
                actions = {"attacker_0": action_A}

                if action_mode == "bfm":
                    actions["evader_0"] = 0  # level flight
                else:
                    actions["evader_0"] = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

            obs, rewards, terminated, truncated, infos = env.step(actions)

            if "attacker_0" in rewards:
                ep_traj["total_reward"] += rewards["attacker_0"]
            if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                ep_traj["reasons"]["attacker_0"] = infos["attacker_0"]["reason"]

        reason = ep_traj["reasons"].get("attacker_0", "timeout")
        ep_traj["n_steps"] = env.step_counter
        ep_traj["duration_s"] = env.step_counter / env.CTRL_FREQ

        # Convert to arrays
        ep_traj["attacker_ned"] = np.array(ep_traj["attacker_ned"])
        ep_traj["evader_ned"] = np.array(ep_traj["evader_ned"])
        ep_traj["times"] = np.array(ep_traj["times"])

        all_episodes.append(ep_traj)

        print(f"  Episode {ep+1}: {reason:15s}  "
              f"reward={ep_traj['total_reward']:+7.1f}  "
              f"steps={ep_traj['n_steps']}  "
              f"duration={ep_traj['duration_s']:.0f}s")

    return all_episodes, env  # env holds Tacview frames


def run_pursuit_evaluate(model, n_episodes: int = 5):
    """Evaluate a SinglePursuitEnv-trained model."""
    from src.environment.single_pursuit_env import SinglePursuitEnv
    env = SinglePursuitEnv(curriculum_stage=1, record_tacview=True)

    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        ep_traj = {
            "pursuer_ned": [],
            "target_ned": [],
            "times": [],
        }
        reason = "timeout"
        min_dist = 8000.0

        while not done:
            t = env._step_counter / 60.0
            ep_traj["pursuer_ned"].append(env.pursuer.position_ned.copy())
            ep_traj["target_ned"].append(env.target_ac.position_ned.copy())
            ep_traj["times"].append(t)

            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += rew
            if "reason" in info:
                reason = info["reason"]
            if "min_dist" in info:
                min_dist = min(min_dist, info["min_dist"])

        ep_traj["pursuer_ned"] = np.array(ep_traj["pursuer_ned"])
        ep_traj["target_ned"] = np.array(ep_traj["target_ned"])
        ep_traj["times"] = np.array(ep_traj["times"])
        ep_traj["total_reward"] = total_r
        ep_traj["reasons"] = {"reason": reason, "min_dist": min_dist}
        all_episodes.append(ep_traj)

        print(f"  Episode {ep+1}: {reason:15s}  "
              f"reward={total_r:+7.1f}  min_dist={min_dist:.0f}m")

    return all_episodes, env


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory(episode: dict, output_path: str):
    """3D trajectory plot + 2D top-down view."""
    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7),
                                      subplot_kw={"projection": "3d"})

    # ── 3D view ──────────────────────────────────────────────────────────
    a_traj = episode["attacker_ned"]
    e_traj = episode["evader_ned"]

    ax3d.plot(a_traj[:, 0], a_traj[:, 1], -a_traj[:, 2],
              "r-", linewidth=1.5, alpha=0.8, label="Attacker")
    ax3d.plot(e_traj[:, 0], e_traj[:, 1], -e_traj[:, 2],
              "b-", linewidth=1.5, alpha=0.8, label="Evader")

    # Start/end markers
    ax3d.scatter(a_traj[0, 0],  a_traj[0, 1],  -a_traj[0, 2],
                 color="darkred",  s=80, marker="o", label="A start")
    ax3d.scatter(a_traj[-1, 0], a_traj[-1, 1], -a_traj[-1, 2],
                 color="red",     s=80, marker="x", label="A end")
    ax3d.scatter(e_traj[0, 0],  e_traj[0, 1],  -e_traj[0, 2],
                 color="darkblue", s=80, marker="o", label="E start")
    ax3d.scatter(e_traj[-1, 0], e_traj[-1, 1], -e_traj[-1, 2],
                 color="blue",    s=80, marker="x", label="E end")

    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(loc="best", fontsize=9)

    # ── 2D top-down ──────────────────────────────────────────────────────
    ax2d.plot(a_traj[:, 0], a_traj[:, 1], "r-", linewidth=1.5, alpha=0.8, label="Attacker")
    ax2d.plot(e_traj[:, 0], e_traj[:, 1], "b-", linewidth=1.5, alpha=0.8, label="Evader")
    ax2d.scatter(a_traj[0, 0],  a_traj[0, 1],  color="darkred",  s=80, marker="o")
    ax2d.scatter(a_traj[-1, 0], a_traj[-1, 1], color="red",      s=80, marker="x")
    ax2d.scatter(e_traj[0, 0],  e_traj[0, 1],  color="darkblue", s=80, marker="o")
    ax2d.scatter(e_traj[-1, 0], e_traj[-1, 1], color="blue",     s=80, marker="x")

    # Annotate start/end distances
    d0 = np.linalg.norm(a_traj[0] - e_traj[0])
    d1 = np.linalg.norm(a_traj[-1] - e_traj[-1])
    ax2d.annotate(f"Start dist: {d0:.0f}m",
                  xy=(a_traj[0, 0], a_traj[0, 1]),
                  fontsize=9, color="gray")
    ax2d.annotate(f"End dist: {d1:.0f}m",
                  xy=(a_traj[-1, 0], a_traj[-1, 1]),
                  fontsize=9, color="gray")

    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title("Top-Down View")
    ax2d.legend(loc="best", fontsize=9)
    ax2d.axis("equal")

    # ── Altitude profile ─────────────────────────────────────────────────
    fig2, ax_alt = plt.subplots(figsize=(10, 4))
    times = episode["times"]
    ax_alt.plot(times, a_traj[:, 2], "r-", linewidth=1.5, label="Attacker")
    ax_alt.plot(times, e_traj[:, 2], "b-", linewidth=1.5, label="Evader")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude Profile")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)

    # Save
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    alt_path = output_path.replace(".png", "_altitude.png")
    fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Trajectory plot saved: {output_path}")
    print(f"  Altitude plot saved:   {alt_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "single_agent": {
        "path": "marl_runs/sb3_run_0611_1610",
        "action_mode": "continuous",
        "mappo": False,
    },
    "single_pursuit": {
        "path": None,   # path is set dynamically from latest marl_runs/single_pursuit_*
        "action_mode": "pursuit",
        "mappo": False,
    },
    "sb3_continuous": {
        "path": "marl_runs/sb3_continuous_0611_1810",
        "action_mode": "continuous",
        "mappo": False,
    },
    "sb3_bfm": {
        "path": "marl_runs/sb3_bfm_0611_1628",
        "action_mode": "bfm",
        "mappo": False,
    },
    "mappo": {
        "path": "marl_runs/mappo_run_0611_1434",
        "action_mode": "continuous",
        "mappo": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize air combat agent")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_REGISTRY.keys()),
        default="sb3_continuous",
        help="Which trained model to evaluate",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int, default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: results/<model_name>_<timestamp>)",
    )
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    output_dir = args.output_dir or f"results/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    print(f"{'='*55}")
    print(f"Air Combat Evaluation")
    print(f"  Model:     {args.model}")
    print(f"  Action:    {cfg['action_mode']}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*55}\n")

    # ── Load model ───────────────────────────────────────────────────────
    model = None
    path = cfg["path"]

    # Dynamically resolve single_pursuit path
    if path is None and cfg["action_mode"] == "pursuit":
        import glob as _glob
        candidates = sorted(_glob.glob("marl_runs/single_pursuit_*"))
        if not candidates:
            print("  No single_pursuit model found! Train first: python scripts/train_single_pursuit.py")
            return
        path = candidates[-1]  # latest run
        print(f"  Using latest run: {path}")

    if cfg["mappo"]:
        model = load_mappo_model(path)
    else:
        if cfg["action_mode"] == "pursuit":
            for name in ["best_model", "model", "final_model"]:
                model_path = os.path.join(path, f"{name}.zip")
                if os.path.exists(model_path):
                    break
            if not os.path.exists(model_path):
                print(f"  No model found at {path}")
                return
            from stable_baselines3 import PPO
            print(f"  Loading: {model_path}")
            model = PPO.load(model_path)
        else:
            model = load_sb3_model(path, cfg["action_mode"])

    # ── Run evaluation ───────────────────────────────────────────────────
    if cfg["action_mode"] == "pursuit":
        episodes, env = run_pursuit_evaluate(model, n_episodes=args.episodes)
    else:
        episodes, env = run_evaluate(
            model, cfg["action_mode"], cfg["mappo"],
            n_episodes=args.episodes,
        )

    # ── Export Tacview ───────────────────────────────────────────────────
    if cfg["action_mode"] == "pursuit":
        tacview_path = os.path.join(output_dir, f"{args.model}_engagement.txt.acmi")
        env.export_tacview(tacview_path)
        print(f"\n  Tacview exported: {tacview_path}")
        print(f"  Open in Tacview: File → Open → {os.path.abspath(tacview_path)}")

        # Plot best episode
        best_ep = max(range(len(episodes)), key=lambda i: episodes[i]["total_reward"])
        _plot_pursuit_episode(episodes[best_ep], os.path.join(output_dir, f"{args.model}_trajectory_best.png"))
    elif hasattr(env, '_tacview_frames') and env._tacview_frames:
        tacview_path = os.path.join(output_dir, f"{args.model}_engagement.txt.acmi")
        env.export_tacview(tacview_path)
        print(f"\n  Tacview exported: {tacview_path}")
        print(f"  Open in Tacview: File → Open → {os.path.abspath(tacview_path)}")

        # Plot best episode (highest attacker reward)
        best_ep = max(range(len(episodes)), key=lambda i: episodes[i]["total_reward"])
        print(f"\n── Best episode (ep {best_ep+1}, reward={episodes[best_ep]['total_reward']:+7.1f}) ──")
        plot_path = os.path.join(output_dir, f"{args.model}_trajectory_best.png")
        plot_trajectory(episodes[best_ep], plot_path)

    # ── Summary ──────────────────────────────────────────────────────────
    if cfg["action_mode"] == "pursuit":
        reasons = [ep.get("reasons", {}).get("reason", "timeout") for ep in episodes]
        n_success = sum(1 for r in reasons if r == "success")
        rewards = [ep["total_reward"] for ep in episodes]
        min_dists = [ep.get("reasons", {}).get("min_dist", -1) for ep in episodes]

        print(f"\n── Summary ({args.episodes} episodes) ──")
        print(f"  Capture rate:  {n_success}/{args.episodes} = {100*n_success/args.episodes:.0f}%")
        print(f"  Avg reward:    {np.mean(rewards):+.1f} ± {np.std(rewards):.1f}")
        print(f"  Avg min dist:  {np.mean(min_dists):.0f} ± {np.std(min_dists):.0f}m")
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  Tacview:  {args.model}_engagement.txt.acmi")
        print(f"  Plots:    {args.model}_trajectory_best.png")
    else:
        reasons = [ep["reasons"].get("attacker_0", "timeout") for ep in episodes]
        n_success = sum(1 for r in reasons if r == "success")
        rewards = [ep["total_reward"] for ep in episodes]

        print(f"\n── Summary ({args.episodes} episodes) ──")
        print(f"  Success rate:  {n_success}/{args.episodes} = {100*n_success/args.episodes:.0f}%")
        print(f"  Avg reward:    {np.mean(rewards):+.1f} ± {np.std(rewards):.1f}")
        print(f"  Avg duration:  {np.mean([ep['duration_s'] for ep in episodes]):.0f}s")
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  Tacview:  {args.model}_engagement.txt.acmi")
        print(f"  3D plot:  {args.model}_trajectory_best.png")
        print(f"  Altitude: {args.model}_trajectory_best_altitude.png")


if __name__ == "__main__":
    import logging
    import warnings
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("ray").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)

    # Move helper into module scope
    if '_plot_pursuit_episode' not in globals():
        from inspect import getsource
        exec('', globals())
    main()


def _plot_pursuit_episode(episode: dict, output_path: str):
    """Plot a SinglePursuitEnv trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    p_traj = episode["pursuer_ned"]
    t_traj = episode["target_ned"]

    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    ax3d.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2],
              "r-", linewidth=1.5, alpha=0.8, label="Pursuer")
    ax3d.plot(t_traj[:, 0], t_traj[:, 1], t_traj[:, 2],
              "b-", linewidth=1.5, alpha=0.8, label="Target")
    ax3d.scatter(p_traj[0, 0], p_traj[0, 1], p_traj[0, 2], color="darkred", s=80, marker="o", label="P start")
    ax3d.scatter(p_traj[-1, 0], p_traj[-1, 1], p_traj[-1, 2], color="red", s=80, marker="x", label="P end")
    ax3d.scatter(t_traj[0, 0], t_traj[0, 1], t_traj[0, 2], color="darkblue", s=80, marker="o", label="T start")
    ax3d.scatter(t_traj[-1, 0], t_traj[-1, 1], t_traj[-1, 2], color="blue", s=80, marker="x", label="T end")
    ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title("3D Trajectory"); ax3d.legend(fontsize=9)

    ax2d.plot(p_traj[:, 0], p_traj[:, 1], "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax2d.plot(t_traj[:, 0], t_traj[:, 1], "b-", lw=1.5, alpha=0.8, label="Target")
    ax2d.scatter(p_traj[0, 0], p_traj[0, 1], color="darkred", s=80, marker="o")
    ax2d.scatter(p_traj[-1, 0], p_traj[-1, 1], color="red", s=80, marker="x")
    ax2d.scatter(t_traj[0, 0], t_traj[0, 1], color="darkblue", s=80, marker="o")
    ax2d.scatter(t_traj[-1, 0], t_traj[-1, 1], color="blue", s=80, marker="x")
    d0 = np.linalg.norm(p_traj[0] - t_traj[0]); d1 = np.linalg.norm(p_traj[-1] - t_traj[-1])
    ax2d.annotate(f"Start: {d0:.0f}m", xy=(p_traj[0, 0], p_traj[0, 1]), fontsize=9, color="gray")
    ax2d.annotate(f"End: {d1:.0f}m", xy=(p_traj[-1, 0], p_traj[-1, 1]), fontsize=9, color="gray")
    ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)"); ax2d.set_title("Top-Down View")
    ax2d.legend(fontsize=9); ax2d.axis("equal")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(10, 4))
    times = episode.get("times", np.arange(len(p_traj)) * 0.5)
    ax.plot(times[:len(p_traj)], p_traj[:, 2], "r-", lw=1.5, label="Pursuer")
    ax.plot(times[:len(t_traj)], t_traj[:, 2], "b-", lw=1.5, label="Target")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)"); ax.set_title("Altitude Profile")
    ax.legend(); ax.grid(True, alpha=0.3)
    alt_path = output_path.replace(".png", "_altitude.png")
    fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Trajectory plot: {output_path}")
    print(f"  Altitude plot:   {alt_path}")
