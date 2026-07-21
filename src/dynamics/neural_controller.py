"""NeuralFlightController — LAG-trained MLP+GRU flight controller.

Loads the LAG BaselineActor (MLPBase→GRU→ACTLayer) and adapts it to the
jsbsim-marl-formation state interface.

Input convention (matches LAG HeadingTask.get_obs exactly):
  dim 0:  delta_altitude / 1000          [km]   (target_alt_m - current_alt_m) / 1000
  dim 1:  delta_heading / π              [rad]  (target_hdg - current_hdg) * π/180
  dim 2:  delta_u / 340                  [mh]   (target_spd - body_vel_u) / 340
  dim 3:  altitude / 5000                [5km]  current_alt_m / 5000
  dim 4:  sin(roll_rad)
  dim 5:  cos(roll_rad)
  dim 6:  sin(pitch_rad)
  dim 7:  cos(pitch_rad)
  dim 8:  body_vel_u / 340               [mh]   u_fps * 0.3048 / 340
  dim 9:  body_vel_v / 340               [mh]   v_fps * 0.3048 / 340
  dim 10: body_vel_w / 340               [mh]   w_fps * 0.3048 / 340
  dim 11: airspeed_mps / 340             [mh]

Output: MultiDiscrete([41 aileron, 41 elevator, 41 rudder, 30 throttle])
  → normalized to ControlSurfaces ([-1,1] / [0,1])
"""

from __future__ import annotations

import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .controller_base import BaseController, ControlSurfaces, FlightTarget


# ═══════════════════════════════════════════════════════════════════════════════
#  LAG BaselineActor (minimal reproduction — sufficient for inference)
# ═══════════════════════════════════════════════════════════════════════════════

def _check(input):
    return torch.from_numpy(input) if isinstance(input, np.ndarray) else input


class _MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        sizes = [input_dim] + list(map(int, hidden_size.split(' ')))
        layers = []
        for j in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                nn.ReLU(),
                nn.LayerNorm(sizes[j + 1]),
            ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class _MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = _MLPLayer(input_dim, hidden_size)

    def forward(self, x):
        return self.mlp(x)


class _GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hxs):
        # x=[N, input_size], hxs=[N, L, hidden_size]
        x, hxs = self.gru(x.unsqueeze(0), hxs.transpose(0, 1).contiguous())
        x = x.squeeze(0)
        hxs = hxs.transpose(0, 1)
        x = self.norm(x)
        return x, hxs


class _Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.logits_net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.logits_net(x)
        return torch.distributions.Categorical(logits=logits).probs.argmax(dim=-1, keepdim=True)


class _ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dims, use_mlp_actlayer=False):
        super().__init__()
        self._mlp_actlayer = use_mlp_actlayer
        if self._mlp_actlayer:
            self.mlp = _MLPLayer(128, '128 128')
        action_outs = []
        for action_dim in action_dims:
            action_outs.append(_Categorical(input_dim, action_dim))
        self.action_outs = nn.ModuleList(action_outs)

    def forward(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)
        actions = []
        for action_out in self.action_outs:
            action = action_out(x)
            actions.append(action)
        return torch.cat(actions, dim=-1)


class _BaselineActor(nn.Module):
    """Exact clone of LAG's BaselineActor for inference."""

    def __init__(self, input_dim=12, use_mlp_actlayer=False):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.base = _MLPBase(input_dim, '128 128')
        self.rnn = _GRULayer(128, 128, 1)
        self.act = _ACTLayer(128, [41, 41, 41, 30], use_mlp_actlayer)
        self.to(torch.device('cpu'))

    def forward(self, obs, rnn_states):
        x = _check(obs).to(**self.tpdv)
        h_s = _check(rnn_states).to(**self.tpdv)
        x = self.base(x)
        x, h_s = self.rnn(x, h_s)
        actions = self.act(x)
        return actions, h_s


# ═══════════════════════════════════════════════════════════════════════════════
#  NeuralFlightController
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralFlightController(BaseController):
    """LAG-trained MLP+GRU neural flight controller.

    Loads baseline_model.pt, builds the 12-dim heading observation from
    jsbsim-marl-formation aircraft state, and outputs MultiDiscrete control
    surface deflections.
    """

    # LAG action space: [41 aileron, 41 elevator, 41 rudder, 30 throttle]
    ACTION_DIMS = [41, 41, 41, 30]

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "models", "baseline_model.pt")
        self._model = _BaselineActor(input_dim=12)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._rnn_states = np.zeros((1, 1, 128), dtype=np.float32)

    @property
    def controller_type(self) -> str:
        return "neural"

    def reset(self, initial_speed_mps: float = 200.0) -> None:
        self._rnn_states = np.zeros((1, 1, 128), dtype=np.float32)

    def predict(self, state: dict, target: FlightTarget, dt: float) -> ControlSurfaces:
        """Build 12-dim LAG HeadingTask observation → forward BaselineActor → normalize."""

        # ── Extract current state ────────────────────────────────────────
        alt_m = float(state["alt_m"])
        roll_rad = math.radians(float(state["roll_deg"]))
        pitch_rad = math.radians(float(state["pitch_deg"]))

        # Body-frame velocities (LAG uses m/s, aircraft exports ft/s)
        u_mps = float(state["u_fps"]) * 0.3048
        v_mps = float(state["v_fps"]) * 0.3048
        w_mps = float(state["w_fps"]) * 0.3048

        # Calibrated airspeed (already m/s in our state dict)
        vc_mps = float(state["airspeed_mps"])

        # ── Deltas from target ───────────────────────────────────────────
        delta_alt = target.altitude_m - alt_m
        current_hdg = float(state["yaw_deg"])
        delta_hdg = (target.heading_deg - current_hdg + 180.0) % 360.0 - 180.0
        delta_u = target.speed_mps - u_mps

        # ── Build 12-dim obs (matches LAG HeadingTask.get_obs) ───────────
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = delta_alt / 1000.0          # km
        obs[1] = delta_hdg * np.pi / 180.0   # rad
        obs[2] = delta_u / 340.0             # mh
        obs[3] = alt_m / 5000.0              # 5km
        obs[4] = np.sin(roll_rad)
        obs[5] = np.cos(roll_rad)
        obs[6] = np.sin(pitch_rad)
        obs[7] = np.cos(pitch_rad)
        obs[8] = u_mps / 340.0               # mh
        obs[9] = v_mps / 340.0               # mh
        obs[10] = w_mps / 340.0              # mh
        obs[11] = vc_mps / 340.0             # mh

        obs = np.expand_dims(obs, axis=0)  # [1, 12]

        # ── Forward through BaselineActor ────────────────────────────────
        with torch.no_grad():
            _action, _rnn_states = self._model(obs, self._rnn_states)
            action = _action.detach().cpu().numpy().squeeze(0)
            self._rnn_states = _rnn_states.detach().cpu().numpy()

        # ── Normalize to ControlSurfaces ─────────────────────────────────
        # LAG normalization: aileron_idx/20 - 1, elevator/20 - 1, rudder/20 - 1, throttle/58 + 0.4
        return ControlSurfaces(
            aileron=float(action[0]) / 20.0 - 1.0,
            elevator=float(action[1]) / 20.0 - 1.0,
            rudder=float(action[2]) / 20.0 - 1.0,
            throttle=float(action[3]) / 58.0 + 0.4,
        )
