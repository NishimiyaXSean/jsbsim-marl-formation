"""Evaluate fine-tuned FormationTask model — direct torch load, no Ray actors."""
import os, sys, warnings, logging, pickle, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask
from src.models.attention_actor import AttentionFormationActor


class EvalAttentionModel(nn.Module):
    """Mirror AttentionFormationActor + RLlibAttentionActor structure."""
    def __init__(self):
        super().__init__()
        # Token projections
        self.self_proj = nn.Linear(15, 128)
        self.target_proj = nn.Linear(14, 128)
        self.mate_proj = nn.Linear(10, 128)
        self.token_type_embed = nn.Parameter(torch.zeros(1, 3, 128))
        # Self-Attention
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.attn_pool_query = nn.Parameter(torch.zeros(1, 1, 128))
        # FiLM
        self.film_gamma = nn.Linear(2, 256)
        self.film_beta = nn.Linear(2, 256)
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        # Action heads [speed=3, heading=5, altitude=3]
        self.heads = nn.ModuleList([
            nn.Linear(256, 3),
            nn.Linear(256, 5),
            nn.Linear(256, 3),
        ])
        self.log_std = nn.Parameter(torch.zeros(1, 2))
        self.mean_layer = nn.Linear(256, 2)

    def forward(self, obs):
        """
        obs: [B, 39] flat observation
         - self token: [0:15]
         - target token: [15:29]
         - mate token: [29:39]
        Returns: tuple of logits for each head
        """
        B = obs.shape[0]
        # Tokenize
        t_self = self.self_proj(obs[:, :15]).unsqueeze(1)     # [B, 1, 128]
        t_target = self.target_proj(obs[:, 15:29]).unsqueeze(1) # [B, 1, 128]
        t_mate = self.mate_proj(obs[:, 29:39]).unsqueeze(1)    # [B, 1, 128]
        tokens = torch.cat([t_self, t_target, t_mate], dim=1)  # [B, 3, 128]
        tokens = tokens + self.token_type_embed

        # Self-Attention
        attn_out, _ = self.attention(tokens, tokens, tokens)   # [B, 3, 128]
        # Learned pooling
        pool_query = self.attn_pool_query.expand(B, -1, -1)
        pooled, _ = self.attention(pool_query, attn_out, attn_out)
        pooled = pooled.squeeze(1)  # [B, 128]

        # FiLM modulation
        agent_id = obs[:, 27:29]  # FiLM identity (indices 27-29 in obs)
        gamma = self.film_gamma(agent_id).unsqueeze(0)  # [B, 256]
        beta = self.film_beta(agent_id).unsqueeze(0)    # [B, 256]

        feat = self.mlp_head(pooled)
        feat = feat * gamma.squeeze(0) + beta.squeeze(0)  # FiLM

        # Action heads
        logits = [head(feat) for head in self.heads]
        return logits  # [ (B,3), (B,5), (B,3) ]


def load_weights(model, weights):
    """Map RLlib state dict to eval model."""
    state = model.state_dict()
    rllib_to_eval = {
        'self_proj.weight': 'actor.self_proj.weight',
        'self_proj.bias': 'actor.self_proj.bias',
        'target_proj.weight': 'actor.target_proj.weight',
        'target_proj.bias': 'actor.target_proj.bias',
        'mate_proj.weight': 'actor.mate_proj.weight',
        'mate_proj.bias': 'actor.mate_proj.bias',
        'token_type_embed': 'actor.token_type_embed',
        'attn_pool_query': 'actor.attn_pool_query',
        'attention.in_proj_weight': 'actor.attention.in_proj_weight',
        'attention.in_proj_bias': 'actor.attention.in_proj_bias',
        'attention.out_proj.weight': 'actor.attention.out_proj.weight',
        'attention.out_proj.bias': 'actor.attention.out_proj.bias',
        'mlp_head.0.weight': 'actor.mlp_head.0.weight',
        'mlp_head.0.bias': 'actor.mlp_head.0.bias',
        'mlp_head.2.weight': 'actor.mlp_head.2.weight',
        'mlp_head.2.bias': 'actor.mlp_head.2.bias',
        'film_gamma.weight': 'actor.film_gamma.weight',
        'film_gamma.bias': 'actor.film_gamma.bias',
        'film_beta.weight': 'actor.film_beta.weight',
        'film_beta.bias': 'actor.film_beta.bias',
        'heads.0.weight': '_heads.0.weight',
        'heads.0.bias': '_heads.0.bias',
        'heads.1.weight': '_heads.1.weight',
        'heads.1.bias': '_heads.1.bias',
        'heads.2.weight': '_heads.2.weight',
        'heads.2.bias': '_heads.2.bias',
        'mean_layer.weight': 'actor.mean.weight',
        'mean_layer.bias': 'actor.mean.bias',
    }
    mapped = {}
    for eval_key, rllib_key in rllib_to_eval.items():
        if rllib_key in weights:
            mapped[eval_key] = torch.from_numpy(weights[rllib_key])
    model.load_state_dict(mapped, strict=False)
    print(f"Loaded {len(mapped)}/{len(rllib_to_eval)} parameter tensors")


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else \
        'marl_runs/rllib_base_0727_1414_finetune_s42/checkpoints/best'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    acmi_out = sys.argv[3] if len(sys.argv) > 3 else 'results/finetune_eval.acmi'

    # Load weights
    pkl_path = os.path.join(ckpt_dir, 'policies', 'shared_policy', 'policy_state.pkl')
    with open(pkl_path, 'rb') as f:
        weights = pickle.load(f)['weights']
    model = EvalAttentionModel()
    load_weights(model, weights)
    model.eval()

    # Run env
    env = BaseEnv(task=FormationTask({'curriculum_stage': 1, 'difficulty_level': 0.0}), env_config={})
    obs, _ = env.reset(seed=seed)
    env.enable_acmi_logging(acmi_out)
    env.log_acmi_step()

    totals = {f.__class__.__name__: {'p0': 0, 'p1': 0} for f in env.task.reward_functions}
    # Disable ACMI for batch evals
    env.disable_acmi_logging()
    total_r, min_d0, min_d1 = 0, 99999, 99999
    for st in range(500):
        acts = {}
        for aid in env._agent_ids:
            o = obs[aid]
            # Handle dict observations (RLlib wraps dict obs)
            if isinstance(o, dict):
                o = o['obs'] if 'obs' in o else o['observation']
            x = torch.from_numpy(o).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            # Argmax (deterministic)
            act = np.array([l.argmax(-1).item() for l in logits], dtype=np.int64)
            acts[aid] = act

        obs, rews, terms, truncs, info = env.step(acts)
        env.log_acmi_step()

        for r in rews.values():
            total_r += r
        tp = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - tp))
        d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - tp))
        if d0 < min_d0: min_d0 = d0
        if d1 < min_d1: min_d1 = d1

        # Accumulate rewards
        bd = getattr(env.task, '_reward_breakdown', {})
        for name, vals in bd.items():
            totals[name]['p0'] += vals.get('p0', 0)
            totals[name]['p1'] += vals.get('p1', 0)

        if terms.get('__all__') or truncs.get('__all__'):
            break

    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    print(f'Seed {seed}: {st+1} steps  TotalRew={total_r:+.0f}  '
          f'min_d0={min_d0:.0f}m  min_d1={min_d1:.0f}m  {reason}')
    print(f'{"Module":<30} {"P0":>12} {"P1":>12}')
    print('-' * 56)
    for name, vals in totals.items():
        print(f'{name:<30} {vals["p0"]:12.1f} {vals["p1"]:12.1f}')
    print('-' * 56)
    tp0 = sum(v['p0'] for v in totals.values())
    tp1 = sum(v['p1'] for v in totals.values())
    print(f'{"TOTAL":<30} {tp0:12.1f} {tp1:12.1f}')

    prog = totals.get('ProgressReward', {'p1': 0})
    adp = totals.get('AltitudeDeviationPenalty', {'p1': 0})
    asym = totals.get('DistanceAsymmetryPenalty', {'p1': 0})
    pincer = totals.get('PincerShapingReward', {'p1': 0})
    print(f'\nP1 diagnosis: Progress={prog["p1"]:.0f}  AltPen={adp["p1"]:.0f}  '
          f'Asym={asym["p1"]:.0f}  Pincer={pincer["p1"]:.0f}')
    print(f'ACMI saved to: {acmi_out}')
    env.close()


if __name__ == '__main__':
    main()
