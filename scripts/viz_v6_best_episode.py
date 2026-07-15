"""Generate Figure 1: Time-series for best V6 episode (+31,854) + P1 action distribution."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                     "font.size":9,"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})

# --- Load V6 autopsy data ---
d = np.load("data/viz/v6_autopsy.npz", allow_pickle=True)
n_eps = int(d['n_episodes'])

# Find best episode
best_idx = max(range(n_eps), key=lambda i: float(d[f'ep{i}_rew']))
rew = float(d[f'ep{best_idx}_rew'])
steps = int(d[f'ep{best_idx}_steps'])
d1_arr = d[f'ep{best_idx}_d1']
pincer_arr = d[f'ep{best_idx}_pincer']
d0_arr = d[f'ep{best_idx}_d0']
reason = str(d[f'ep{best_idx}_reason'])

# Compute pincer reward per decision step
AND_ANGLE = 35.0; COEFF = 35.0; DT = 0.2
T = min(len(d1_arr), len(pincer_arr))
pincer_reward_step = np.zeros(T)
for t in range(T):
    shaped = min(pincer_arr[t], AND_ANGLE)
    pincer_reward_step[t] = COEFF * shaped * DT * 0.5  # split between both agents

# AND gate checks
both_in_1200 = (d0_arr[:T] < 1200) & (d1_arr[:T] < 1200)
and_met = both_in_1200 & (pincer_arr[:T] >= 35)

# --- Figure 1: Time-series ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Top: P1 distance + AND gate
ax1.plot(range(T), d1_arr[:T], color='#377eb8', lw=1.5, label='P1 distance to target')
ax1.plot(range(T), d0_arr[:T], color='#e41a1c', lw=1.2, alpha=0.6, label='P0 distance to target')
ax1.axhline(y=1200, color='green', ls=':', lw=0.8, alpha=0.5, label='AND gate (1200m)')
ax1.axhline(y=1440, color='orange', ls=':', lw=0.8, alpha=0.3, label='Pincer gate (1440m)')
# Shade AND-met regions
for t in range(T-1):
    if and_met[t]:
        ax1.axvspan(t, t+1, alpha=0.15, color='green')
ax1.set_ylabel('Distance to Target (m)')
ax1.set_title(f'V6 Best Episode (idx={best_idx}, rew={rew:.0f}, {steps} steps, {reason})\n'
              'P0/P1 Distance to Target (green shade = AND condition met)',
              fontweight='bold')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.2)

# Bottom: Pincer reward per step
ax2.plot(range(T), pincer_reward_step, color='#ff7f00', lw=1.2, label='Pincer shaping reward/step')
ax2.fill_between(range(T), 0, pincer_reward_step, alpha=0.2, color='#ff7f00')
ax2.axhline(y=122, color='gray', ls='--', lw=0.5, alpha=0.5, label='Max (35°, both in gate)')
ax2.set_xlabel('Decision Step (0.2s each)')
ax2.set_ylabel('Pincer Reward per Step')
ax2.set_title('Per-Step Pincer Shaping Reward', fontweight='bold')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.2)

# Stats annotation
active_pincer = (d1_arr[:T] < 2000) & (d0_arr[:T] < 2000)
fig.text(0.02, 0.01,
    f'Pincer active: {active_pincer.sum()}/{T} steps ({100*active_pincer.mean():.0f}%) | '
    f'AND met: {and_met.sum()}/{T} steps ({100*and_met.mean():.1f}%) | '
    f'P1 flew AWAY: t=50→200 (2243→2862m)',
    fontsize=7, color='gray')

plt.tight_layout()
fig.savefig('results/viz/fig_v6_best_ep_timeseries.pdf', facecolor='white')
plt.close()
print('[OK] Figure 1: results/viz/fig_v6_best_ep_timeseries.pdf')

# --- Now collect P1 action distribution via eval ---
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from src.environment.formation_rllib_env import FormationRLlibEnv, COOP_PHASE_AND
from src.models.formation_rllib_model import RLlibAttentionActor

register_env('formation_2v1_rllib', lambda c: FormationRLlibEnv(c))
ModelCatalog.register_custom_model('attention_formation', RLlibAttentionActor)

CKPT = "/home/sean/jsbsim-marl-formation/marl_runs/rllib_formation_0715_1055_s42/checkpoints/checkpoint_final"
algo = PPO.from_checkpoint(os.path.abspath(CKPT))

S1 = {"and_dist": 1200.0, "and_angle": 35.0, "bearing_min": -30.0, "bearing_max": 30.0,
      "target_dist_min": 1600.0, "target_dist_max": 2200.0, "sustain_steps": 2}

p1_turns = []
p1_speeds = []
p0_turns = []
p0_speeds = []

for ep in range(50):
    env = FormationRLlibEnv({'difficulty_level': 0.0, 'lock_altitude': True,
                             'record_tacview': False, 'cooperative_mode': True})
    env.set_coop_phase(COOP_PHASE_AND)
    env.set_curriculum_stage_full(1, S1["and_dist"], S1["and_angle"],
        S1["bearing_min"], S1["bearing_max"],
        S1["target_dist_min"], S1["target_dist_max"], S1["sustain_steps"])
    obs_dict, _ = env.reset(seed=42 + ep)
    done, step = False, 0
    while not done:
        actions = {}
        for aid in env._agent_ids:
            if aid in obs_dict:
                raw = algo.compute_single_action(obs_dict[aid], policy_id=f"{aid}_policy", explore=False)
                actions[aid] = raw
        obs_dict, rewards, terms, truncs, _ = env.step(actions)
        if 'p1' in actions:
            p1_turns.append(int(actions['p1'][0]))
            p1_speeds.append(int(actions['p1'][1]))
        if 'p0' in actions:
            p0_turns.append(int(actions['p0'][0]))
            p0_speeds.append(int(actions['p0'][1]))
        done = terms.get('__all__', False) or truncs.get('__all__', False)
        step += 1
        if step >= 600: break

algo.stop()

TURN_LABELS = ['HardLeft', 'SoftLeft', 'Straight', 'SoftRight', 'HardRight']
SPEED_LABELS = ['Slow(180)', 'Cruise(250)', 'Fast(320)']

# --- Figure 2: Action distribution ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

colors_turn = ['#d73027','#fc8d59','#fee090','#91bfdb','#4575b4']

for ax, data, labels, title, agent_name in [
    (ax1, p1_turns, TURN_LABELS, 'P1 (Interceptor) — Turn', 'P1'),
    (ax2, p1_speeds, SPEED_LABELS, 'P1 (Interceptor) — Speed', 'P1'),
    (ax3, p0_turns, TURN_LABELS, 'P0 (Striker) — Turn', 'P0'),
    (ax4, p0_speeds, SPEED_LABELS, 'P0 (Striker) — Speed', 'P0'),
]:
    counts = np.bincount(data, minlength=len(labels))
    pct = 100 * counts / counts.sum()
    bars = ax.bar(range(len(labels)), counts, color=colors_turn[:len(labels)], edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Count')
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(i, c + max(counts)*0.02, f'{p:.0f}%', ha='center', fontsize=8, fontweight='bold')

fig.suptitle('V6 Eval Action Distribution (50 episodes, checkpoint_final)\n'
             f'P1 total actions: {len(p1_turns)}, P0 total actions: {len(p0_turns)}',
             fontsize=10, y=1.01)
plt.tight_layout()
fig.savefig('results/viz/fig_v6_action_dist.pdf', facecolor='white')
plt.close()
print('[OK] Figure 2: results/viz/fig_v6_action_dist.pdf')

# Print summary
print(f'\n=== P1 Action Summary ===')
print(f'  Turn:  ' + ' | '.join(f'{l}: {np.bincount(p1_turns, minlength=5)[i]/len(p1_turns)*100:.0f}%' for i,l in enumerate(TURN_LABELS)))
print(f'  Speed: ' + ' | '.join(f'{l}: {np.bincount(p1_speeds, minlength=3)[i]/len(p1_speeds)*100:.0f}%' for i,l in enumerate(SPEED_LABELS)))
print(f'\n=== P0 Action Summary ===')
print(f'  Turn:  ' + ' | '.join(f'{l}: {np.bincount(p0_turns, minlength=5)[i]/len(p0_turns)*100:.0f}%' for i,l in enumerate(TURN_LABELS)))
print(f'  Speed: ' + ' | '.join(f'{l}: {np.bincount(p0_speeds, minlength=3)[i]/len(p0_speeds)*100:.0f}%' for i,l in enumerate(SPEED_LABELS)))
