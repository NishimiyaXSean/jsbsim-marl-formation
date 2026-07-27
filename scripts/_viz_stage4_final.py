"""Final Stage 4 visualization: 3D trajectory + top-down + ACMI for best model."""
import os, sys, warnings, logging, pickle
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask

# ══════ Academic styling ══════
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

COLORS = {"p0": "#377eb8", "p1": "#ff7f00", "target": "#e41a1c",
          "p0_light": "#a6cee3", "p1_light": "#fdbf6f", "grid": "#e0e0e0"}


class EvalModel(nn.Module):
    """Mirror AttentionFormationActor + RLlibAttentionActor for inference."""
    def __init__(self):
        super().__init__()
        self.sp = nn.Linear(15, 128); self.tp = nn.Linear(14, 128); self.mp = nn.Linear(10, 128)
        self.tte = nn.Parameter(torch.zeros(1, 3, 128))
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.apq = nn.Parameter(torch.zeros(1, 1, 128))
        self.fg = nn.Linear(2, 256); self.fb = nn.Linear(2, 256)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(256, 3), nn.Linear(256, 5), nn.Linear(256, 3)])

    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs.get('obs', obs.get('observation', obs))
        B = obs.shape[0]
        t_self = self.sp(obs[:, :15]).unsqueeze(1)
        t_target = self.tp(obs[:, 15:29]).unsqueeze(1)
        t_mate = self.mp(obs[:, 29:39]).unsqueeze(1)
        tokens = torch.cat([t_self, t_target, t_mate], dim=1) + self.tte
        ao, _ = self.attn(tokens, tokens, tokens)
        pq = self.apq.expand(B, -1, -1)
        pooled, _ = self.attn(pq, ao, ao)
        pooled = pooled.squeeze(1)
        agent_id = obs[:, 27:29]
        feat = self.mlp(pooled) * self.fg(agent_id) + self.fb(agent_id)
        return [h(feat) for h in self.heads]


def load_model(ckpt_dir):
    pkl = os.path.join(ckpt_dir, 'policies', 'shared_policy', 'policy_state.pkl')
    with open(pkl, 'rb') as f:
        w = pickle.load(f)['weights']
    m = EvalModel()
    rmap = {
        'sp.weight': 'actor.self_proj.weight', 'sp.bias': 'actor.self_proj.bias',
        'tp.weight': 'actor.target_proj.weight', 'tp.bias': 'actor.target_proj.bias',
        'mp.weight': 'actor.mate_proj.weight', 'mp.bias': 'actor.mate_proj.bias',
        'tte': 'actor.token_type_embed', 'apq': 'actor.attn_pool_query',
        'attn.in_proj_weight': 'actor.attention.in_proj_weight',
        'attn.in_proj_bias': 'actor.attention.in_proj_bias',
        'attn.out_proj.weight': 'actor.attention.out_proj.weight',
        'attn.out_proj.bias': 'actor.attention.out_proj.bias',
        'mlp.0.weight': 'actor.mlp_head.0.weight', 'mlp.0.bias': 'actor.mlp_head.0.bias',
        'mlp.2.weight': 'actor.mlp_head.2.weight', 'mlp.2.bias': 'actor.mlp_head.2.bias',
        'fg.weight': 'actor.film_gamma.weight', 'fg.bias': 'actor.film_gamma.bias',
        'fb.weight': 'actor.film_beta.weight', 'fb.bias': 'actor.film_beta.bias',
        'heads.0.weight': '_heads.0.weight', 'heads.0.bias': '_heads.0.bias',
        'heads.1.weight': '_heads.1.weight', 'heads.1.bias': '_heads.1.bias',
        'heads.2.weight': '_heads.2.weight', 'heads.2.bias': '_heads.2.bias',
    }
    m.load_state_dict({ek: torch.from_numpy(w[rk]) for ek, rk in rmap.items() if rk in w}, strict=False)
    m.eval()
    return m


def run_episode(model, seed):
    """Run one episode and capture all trajectory data."""
    env = BaseEnv(task=FormationTask({'curriculum_stage': 1, 'difficulty_level': 0.0}), env_config={})
    obs, _ = env.reset(seed=seed)

    p0_pos, p1_pos, t_pos = [], [], []
    p0_alt, p1_alt = [], []
    distances_p0, distances_p1 = [], []
    pincer_angles = []
    total_r = 0

    for st in range(500):
        acts = {}
        for aid in env._agent_ids:
            o = obs[aid]
            if isinstance(o, dict):
                o = o.get('obs', o.get('observation', o))
            x = torch.from_numpy(o).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            acts[aid] = np.array([l.argmax(-1).item() for l in logits], dtype=np.int64)

        obs, rews, terms, truncs, info = env.step(acts)

        p0_pos.append(env.pursuers[0].aircraft.position_ned.copy())
        p1_pos.append(env.pursuers[1].aircraft.position_ned.copy())
        t_pos.append(env.targets[0].aircraft.position_ned.copy())
        p0_alt.append(float(env.pursuers[0].aircraft.state["alt_m"]))
        p1_alt.append(float(env.pursuers[1].aircraft.state["alt_m"]))

        tp = t_pos[-1]
        d0 = float(np.linalg.norm(p0_pos[-1] - tp))
        d1 = float(np.linalg.norm(p1_pos[-1] - tp))
        distances_p0.append(d0)
        distances_p1.append(d1)

        pincer = getattr(env.task, '_last_pincer', 0.0) or 0.0
        pincer_angles.append(pincer)

        for r in rews.values():
            total_r += r

        if terms.get('__all__') or truncs.get('__all__'):
            break

    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    env.close()
    return {
        'p0_pos': np.array(p0_pos), 'p1_pos': np.array(p1_pos), 't_pos': np.array(t_pos),
        'p0_alt': np.array(p0_alt), 'p1_alt': np.array(p1_alt),
        'distances_p0': np.array(distances_p0), 'distances_p1': np.array(distances_p1),
        'pincer_angles': np.array(pincer_angles),
        'n_steps': st + 1, 'total_reward': total_r,
        'd0_min': float(np.min(distances_p0)), 'd1_min': float(np.min(distances_p1)),
        'termination': reason, 'seed': seed,
    }


def run_with_acmi(model, seed, acmi_path):
    """Run episode with ACMI export."""
    env = BaseEnv(task=FormationTask({'curriculum_stage': 1, 'difficulty_level': 0.0}), env_config={})
    obs, _ = env.reset(seed=seed)
    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()

    total_r, min_d0, min_d1 = 0, 99999, 99999
    for st in range(500):
        acts = {}
        for aid in env._agent_ids:
            o = obs[aid]
            if isinstance(o, dict):
                o = o.get('obs', o.get('observation', o))
            x = torch.from_numpy(o).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            acts[aid] = np.array([l.argmax(-1).item() for l in logits], dtype=np.int64)

        obs, rews, terms, truncs, info = env.step(acts)
        env.log_acmi_step()
        for r in rews.values():
            total_r += r
        tp = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - tp))
        d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - tp))
        if d0 < min_d0: min_d0 = d0
        if d1 < min_d1: min_d1 = d1
        if terms.get('__all__') or truncs.get('__all__'):
            break

    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    env.close()
    return st + 1, total_r, min_d0, min_d1, reason


# ═══════════════════════════════════════════════════════════════════════
#  3D Trajectory plot
# ═══════════════════════════════════════════════════════════════════════

def plot_3d_trajectory(ep, save_path):
    p0 = ep['p0_pos']
    p1 = ep['p1_pos']
    t = ep['t_pos']

    # Center on target start
    origin = t[0].copy()
    p0_rel = p0 - origin
    p1_rel = p1 - origin
    t_rel = t - origin

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    T = len(p0_rel)
    alpha_curve = np.linspace(0.35, 1.0, T)

    for i in range(T - 1):
        ax.plot(p0_rel[i:i+2, 1], p0_rel[i:i+2, 0], -p0_rel[i:i+2, 2],
                color=COLORS["p0"], linewidth=1.4, alpha=alpha_curve[i])
        ax.plot(p1_rel[i:i+2, 1], p1_rel[i:i+2, 0], -p1_rel[i:i+2, 2],
                color=COLORS["p1"], linewidth=1.4, alpha=alpha_curve[i])
        ax.plot(t_rel[i:i+2, 1], t_rel[i:i+2, 0], -t_rel[i:i+2, 2],
                color=COLORS["target"], linewidth=1.0, alpha=alpha_curve[i], linestyle="--")

    # Start/end markers
    for label, pos, color, marker in [
        ("P0", p0_rel[0], COLORS["p0"], "o"),
        ("P1", p1_rel[0], COLORS["p1"], "o"),
        ("P0", p0_rel[-1], COLORS["p0"], "s"),
        ("P1", p1_rel[-1], COLORS["p1"], "s"),
        ("T", t_rel[-1], COLORS["target"], "X"),
    ]:
        ax.scatter(pos[1], pos[0], -pos[2], c=color, marker=marker,
                   s=50, edgecolors="white", linewidth=0.5, zorder=5)

    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Down (m)")
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

    legend = [
        Line2D([0], [0], color=COLORS["p0"], linewidth=2, label="P0 (Striker)"),
        Line2D([0], [0], color=COLORS["p1"], linewidth=2, label="P1 (Interceptor)"),
        Line2D([0], [0], color=COLORS["target"], linewidth=1.5, linestyle="--", label="Target"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.9, edgecolor="none")
    ax.set_title(
        f"Stage 4 Formation Pursuit — Shared-Attention MAPPO (CTDE)\n"
        f"seed={ep['seed']}  min_d0={ep['d0_min']:.0f}m  min_d1={ep['d1_min']:.0f}m  "
        f"rew={ep['total_reward']:+.0f}  [{ep['termination']}]",
        fontsize=9, pad=15,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] 3D: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Top-down view
# ═══════════════════════════════════════════════════════════════════════

def plot_topdown(ep, save_path):
    p0 = ep['p0_pos']
    p1 = ep['p1_pos']
    t = ep['t_pos']

    origin = t[0].copy()
    p0_rel = p0 - origin
    p1_rel = p1 - origin
    t_rel = t - origin

    fig, ax = plt.subplots(figsize=(8, 7))

    T = len(p0_rel)
    alpha_curve = np.linspace(0.3, 1.0, T)

    for i in range(T - 1):
        ax.plot(p0_rel[i:i+2, 1], p0_rel[i:i+2, 0], color=COLORS["p0"],
                linewidth=1.5, alpha=alpha_curve[i])
        ax.plot(p1_rel[i:i+2, 1], p1_rel[i:i+2, 0], color=COLORS["p1"],
                linewidth=1.5, alpha=alpha_curve[i])
        ax.plot(t_rel[i:i+2, 1], t_rel[i:i+2, 0], color=COLORS["target"],
                linewidth=1.2, alpha=alpha_curve[i], linestyle="--")

    # Markers
    ax.scatter(p0_rel[0, 1], p0_rel[0, 0], c=COLORS["p0"], marker="o", s=80,
               edgecolors="white", linewidth=0.8, zorder=5, label="P0 start")
    ax.scatter(p1_rel[0, 1], p1_rel[0, 0], c=COLORS["p1"], marker="o", s=80,
               edgecolors="white", linewidth=0.8, zorder=5, label="P1 start")
    ax.scatter(p0_rel[-1, 1], p0_rel[-1, 0], c=COLORS["p0"], marker="s", s=80,
               edgecolors="white", linewidth=0.8, zorder=5, label="P0 end")
    ax.scatter(p1_rel[-1, 1], p1_rel[-1, 0], c=COLORS["p1"], marker="s", s=80,
               edgecolors="white", linewidth=0.8, zorder=5, label="P1 end")
    ax.scatter(t_rel[-1, 1], t_rel[-1, 0], c=COLORS["target"], marker="X", s=100,
               edgecolors="white", linewidth=0.8, zorder=5, label="Target end")

    # 300m / 800m rings around target end position
    for radius, style, label in [(300, ':', '300m capture'), (800, '--', '800m AND gate')]:
        circle = plt.Circle((t_rel[-1, 1], t_rel[-1, 0]), radius,
                            fill=False, color='gray', linestyle=style,
                            linewidth=0.8, alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(label, (t_rel[-1, 1] + radius, t_rel[-1, 0]),
                    fontsize=6, color='gray', alpha=0.7)

    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_title(
        f"Stage 4 Formation Pursuit — Top-Down View\n"
        f"seed={ep['seed']}  min_d0={ep['d0_min']:.0f}m  min_d1={ep['d1_min']:.0f}m  "
        f"rew={ep['total_reward']:+.0f}  [{ep['termination']}]",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Top-down: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Combined distance + altitude panel
# ═══════════════════════════════════════════════════════════════════════

def plot_metrics(ep, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    t_axis = np.arange(ep['n_steps']) * 0.2  # seconds

    # ── Distances ──────────────────────────
    ax = axes[0, 0]
    ax.plot(t_axis, ep['distances_p0'], color=COLORS["p0"], linewidth=1.2, label="P0")
    ax.plot(t_axis, ep['distances_p1'], color=COLORS["p1"], linewidth=1.2, label="P1")
    ax.axhline(y=800, color='gray', linestyle='--', alpha=0.4, label='AND gate (800m)')
    ax.axhline(y=300, color='gray', linestyle=':', alpha=0.4, label='Capture (300m)')
    ax.set_ylabel("Distance to Target (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(framealpha=0.8, fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Pincer angle ───────────────────────
    ax = axes[0, 1]
    ax.plot(t_axis, ep['pincer_angles'], color='#4daf4a', linewidth=1.2)
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.4, label='AND threshold (30°)')
    ax.fill_between(t_axis, 30, ep['pincer_angles'].clip(min=0),
                    alpha=0.15, color='#4daf4a', label='Above threshold')
    ax.set_ylabel("Pincer Angle (°)")
    ax.set_xlabel("Time (s)")
    ax.legend(framealpha=0.8, fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Altitude ───────────────────────────
    ax = axes[1, 0]
    ax.plot(t_axis, ep['p0_alt'], color=COLORS["p0"], linewidth=1.2, label="P0")
    ax.plot(t_axis, ep['p1_alt'], color=COLORS["p1"], linewidth=1.2, label="P1")
    t_alt = ep['t_pos'][:, 2]
    ax.plot(t_axis[:len(t_alt)], -t_alt, color=COLORS["target"], linewidth=1.0,
            linestyle='--', label="Target")
    ax.axhline(y=2500, color='red', linestyle=':', alpha=0.4, label='Min altitude (2500m)')
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(framealpha=0.8, fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Reward summary text ────────────────
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f"Seed: {ep['seed']}",
        f"Steps: {ep['n_steps']} / 500",
        f"Total Reward: {ep['total_reward']:+.0f}",
        f"Min P0 Distance: {ep['d0_min']:.0f} m",
        f"Min P1 Distance: {ep['d1_min']:.0f} m",
        f"Max Pincer: {float(np.max(ep['pincer_angles'])):.1f}°",
        f"Termination: {ep['termination']}",
        "",
        "Model: Shared-Attention MAPPO (CTDE)",
        "Action: MultiDiscrete([3 speed, 5 hdg, 3 alt])",
        "Checkpoint: rllib_base_0727_1414_finetune_s42/best",
    ]
    for i, line in enumerate(lines):
        color = 'black'
        if 'Termination' in line and 'low_altitude' in line:
            color = 'red'
        ax.text(0.05, 0.95 - i * 0.08, line, transform=ax.transAxes,
                fontsize=9, color=color, fontfamily='monospace',
                verticalalignment='top')

    fig.suptitle("Stage 4 Formation Pursuit — Evaluation Metrics\n"
                 "Conservative Fine-tune (lr=5e-5, ent=0.002, batch=8192, clip=0.1)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Metrics: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        'marl_runs/rllib_base_0727_1414_finetune_s42/checkpoints/best'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    out_dir = sys.argv[3] if len(sys.argv) > 3 else 'results/stage4_final'

    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt)

    # 1. Collect trajectory data
    print(f"Running seed={seed}...")
    ep = run_episode(model, seed)

    # 2. Generate ACMI
    acmi_path = os.path.join(out_dir, f'formation_s{seed}.acmi')
    print(f"Generating ACMI: {acmi_path}")
    st, rew, d0, d1, reason = run_with_acmi(model, seed, acmi_path)
    print(f"  ACMI: {st} steps, rew={rew:+.0f}, d0_min={d0:.0f}m, d1_min={d1:.0f}m [{reason}]")

    # 3. Generate plots
    plot_3d_trajectory(ep, os.path.join(out_dir, f'fig_3d_trajectory_s{seed}.png'))
    plot_topdown(ep, os.path.join(out_dir, f'fig_topdown_s{seed}.png'))
    plot_metrics(ep, os.path.join(out_dir, f'fig_metrics_s{seed}.png'))

    # 4. Save trajectory data
    npz_path = os.path.join(out_dir, f'trajectory_s{seed}.npz')
    np.savez(npz_path,
             p0_positions=ep['p0_pos'], p1_positions=ep['p1_pos'],
             target_positions=ep['t_pos'],
             distances_p0=ep['distances_p0'], distances_p1=ep['distances_p1'],
             pincer_angles=ep['pincer_angles'],
             p0_alt=ep['p0_alt'], p1_alt=ep['p1_alt'],
             n_steps=ep['n_steps'], total_reward=ep['total_reward'],
             d0_min=ep['d0_min'], d1_min=ep['d1_min'],
             termination=ep['termination'], seed=seed)
    print(f"[OK] NPZ data: {npz_path}")

    print(f"\n=== Stage 4 Final Output ===")
    print(f"  Seed {seed}: {ep['n_steps']} steps, rew={ep['total_reward']:+.0f}")
    print(f"  min_d0={ep['d0_min']:.0f}m  min_d1={ep['d1_min']:.0f}m")
    print(f"  max_pincer={float(np.max(ep['pincer_angles'])):.1f}°")
    print(f"  termination: {ep['termination']}")
    print(f"  Outputs: {out_dir}/")


if __name__ == '__main__':
    main()
