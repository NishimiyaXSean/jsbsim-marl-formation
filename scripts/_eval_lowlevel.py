"""Lightweight deterministic eval — loads policy state_dict, no Ray workers."""
import numpy as np, sys, os, warnings, logging, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim','gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)

from src.environment.base_env import BaseEnv
from src.environment.heading_task import HeadingTrackingTask

# Load trained model weights (RLlib default FullyConnectedNetwork state_dict)
model_path = sys.argv[1] if len(sys.argv) > 1 else \
    'data/models/lowlevel_controller.pt'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
target_hdg = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0

state_dict = torch.load(os.path.abspath(model_path), map_location='cpu', weights_only=True)

# Build a simple MLP+LSTM that matches the RLlib model architecture
# RLlib default: fcnet_hiddens [512,512,256], lstm_cell_size 128
import torch.nn as nn

class EvalModel(nn.Module):
    """Matches RLlib default model: MLP [12→512→512→256] → LSTM(256,128) → Linear(128,78)."""
    def __init__(self):
        super().__init__()
        # Policy branch (matches RLlib _hidden_layers + _logits_branch)
        self.fc1 = nn.Linear(12, 512); self.fc2 = nn.Linear(512, 512); self.fc3 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.logits = nn.Linear(128, 78)  # 21+21+21+15 = 78 flat logits

    def forward(self, x, hx=None):
        x = torch.tanh(self.fc1(x)); x = torch.tanh(self.fc2(x)); x = torch.tanh(self.fc3(x))
        x, hx = self.lstm(x.unsqueeze(1), hx if hx else None); x = x.squeeze(1)
        logits = self.logits(x)  # [B, 78]
        a0 = logits[:, 0:21].argmax(-1); a1 = logits[:, 21:42].argmax(-1)
        a2 = logits[:, 42:63].argmax(-1); a3 = logits[:, 63:78].argmax(-1)
        return torch.stack([a0, a1, a2, a3], dim=-1), hx

model = EvalModel()
# Map RLlib nested keys to flat EvalModel keys
key_map = {
    '_hidden_layers.0._model.0.weight': 'fc1.weight', '_hidden_layers.0._model.0.bias': 'fc1.bias',
    '_hidden_layers.1._model.0.weight': 'fc2.weight', '_hidden_layers.1._model.0.bias': 'fc2.bias',
    '_hidden_layers.2._model.0.weight': 'fc3.weight', '_hidden_layers.2._model.0.bias': 'fc3.bias',
    '_logits_branch._model.0.weight': 'logits.weight', '_logits_branch._model.0.bias': 'logits.bias',
    'lstm.weight_ih_l0': 'lstm.weight_ih_l0', 'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
    'lstm.bias_ih_l0': 'lstm.bias_ih_l0', 'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
}
mapped = {key_map.get(k, k): v for k, v in state_dict.items() if key_map.get(k) is not None}
model.load_state_dict(mapped, strict=False)
model.eval()

# Run deterministic eval
env = BaseEnv(task=HeadingTrackingTask({'target_heading': target_hdg}), env_config={})
obs, _ = env.reset(seed=seed)

print(f'Target: {target_hdg:.0f}°  Start hdg: {float(env.pursuers[0].aircraft.state["yaw_deg"]):.0f}°')
print(f'step  time   hdg    err   alt_m  roll   pitch  spd    rew')

log = []; total_r = 0; hx = None
for st in range(500):
    x = torch.from_numpy(obs['p0']).float().unsqueeze(0)
    with torch.no_grad():
        act_tensor, hx = model(x, hx)
    act = act_tensor.squeeze(0).numpy().astype(np.int64)
    if hx is not None:
        hx = (hx[0].detach(), hx[1].detach())
    obs, rews, terms, truncs, info = env.step({'p0': act})
    s = env.pursuers[0].aircraft.state
    hdg = float(s['yaw_deg']); alt = float(s['alt_m']); spd = float(s['airspeed_mps'])
    roll = float(s['roll_deg']); pitch = float(s['pitch_deg'])
    err = abs((target_hdg - hdg + 180) % 360 - 180)
    r = rews.get('p0', 0); total_r += r
    log.append({'step':st, 't':st*0.2, 'hdg':hdg, 'err':err, 'alt':alt,
                'roll':roll, 'pitch':pitch, 'spd':spd, 'rew':r})
    if st % 50 == 0:
        print(f'{st:4d}  {st*0.2:5.1f}s  {hdg:5.1f}° {err:5.1f}° {alt:5.0f}m {roll:5.1f}° {pitch:5.1f}° {spd:5.0f}m/s {r:+.3f}')
    if terms.get('__all__') or truncs.get('__all__'): break

n = len(log)
errs = [d['err'] for d in log]
print(f'\n--- Summary (n={n}) ---')
print(f'Total reward: {total_r:+.1f}')
print(f'Heading MAE: {np.mean(errs):.1f}° (last 100: {np.mean(errs[-100:]):.1f}°)')
print(f'Final heading: {log[-1]["hdg"]:.1f}° (target {target_hdg:.0f}°, err {log[-1]["err"]:.1f}°)')
print(f'Altitude: {np.mean([d["alt"] for d in log]):.0f}m ±{np.std([d["alt"] for d in log]):.0f}m')
print(f'Reason: {info.get("p0",{}).get("termination_reason","timeout")}')
env.close()
