# Reproduction Notes

> Author: 2026-09-13
> Status: validated on a fresh WSL2 install (Ubuntu 22.04, RTX 3060 6 GB,
> conda `marl_env`, JSBSim 1.3.1, stable_baselines3 2.9.0, Ray 2.40,
> torch 2.9.1+cu126).
> Scope: how to reproduce the published 2v1 / 1v1 results on a new machine
> without falling into the same traps the README's quick-start does not warn
> about.

This document exists because the README's Quick Start section, plus the
`scripts/setup_wsl2.sh` helper, are **stale and incomplete** as of the
`feature/refactor-task-based` branch (verified September 2026). They were
written before the RLlib migration and the BC + ASAP-distillation sealed
path was adopted. A new reader who follows only README will fail in subtle
ways. This document is the corrected route.

## 0. TL;DR

There are two independent work streams in this repository:

| Work stream | Goal | Sealed artifact | Reproduction path |
|---|---|---|---|
| **2v1 cooperative pursuit** (no weapons) | Two F-16s cooperatively capture one F-16 target by closing range | `benchmarks/sb3_2v1_97p3/model.zip` (2.0 MB, committed, 30-ep eval = 90% capture) | Run `scripts/benchmark_sb3_baseline.py` inside a `git worktree add` at `1ca3e63^` (because the `formation_env.py` class was purged in `1ca3e63`). ~50 lines of orchestration; see §3. |
| **1v1 missile shoot** (with weapons) | One F-16 fires 4 AIM-9L missiles at a target, 4-hit kill | `data/expert/shoot_bc_asap_distilled.pth` (438 KB, **NOT in the repo, must be retrained**; the only committed baseline is the BC round-1 weights inside `data/expert/`, but those are also gitignored) | Four-step pipeline `generate_shoot_rule_expert.py` → `train_shoot_bc.py` → `distill_fire_asap.py` → `eval_bc_1v1.py`. ~3 hours on a single GPU. See §4. |

**There is no "2v1 + missile" combination** in this repository. 2v1 ends with
commit `bf1382f` ("end of 2v1 formation pursuit exploration"); the missile
work starts at `3946385` and is 1v1 only. The 99.9% ID kill rate that
appears in the README is from the 1v1 BC+ASAP-distillation pipeline, not
from any 2v1 + missile work.

## 1. What is and is not in the repository

The repository's `.gitignore` aggressively excludes everything required to
reproduce the headline numbers. **None of the following are committed**:

- `data/jsbsim/` — JSBSim aircraft / engine / systems XML (4.4 MB, must be
  restored from a local copy or JSBSim's default search path).
- `data/expert/*.npz` — expert demonstration data (rule expert for 1v1
  shoot, attention-BC for 2v1). Several hundred MB.
- `data/expert/*.pth` — trained BC / BC-distilled weights. The headline
  `shoot_bc_asap_distilled.pth` (438 KB) is gitignored. You must retrain.
- `marl_runs/` — all RLlib training checkpoint directories. Every single
  one of these was destroyed when the previous machine was retired; they
  are **unrecoverable**. Reproduction must retrain.
- `data/checkpoints/`, `data/logs/`, `data/tacview/`, `*.pth`, `*.pt`,
  `*.pkl`, `ray_results/`, `results/shoot_eval/*.json`, `results/ctrl_viz/`,
  `results/neural_tracking/`, `logs/`, etc.

What is committed:

- All source under `src/`, `scripts/`, `tests/`, `docs/`, `data/expert/`
  minus the gitignored patterns above.
- A small number of result files that are explicitly tracked:
  `benchmarks/sb3_2v1_97p3/{model.zip, manifest.json, *.acmi, *.png,
  metrics_diff*.json, benchmark_run.log}`.

**Implication**: a `git clone` gives you the code, a 2.0 MB SB3 2v1
checkpoint, and the `summary_phase1.md` paper summary. Reproducing the 1v1
99.9% headline number requires ~3 hours of compute and a clean GPU
environment.

## 2. Environment setup — the parts README gets wrong

The README's "Installation" section points at `scripts/setup_wsl2.sh`,
which is **stale**:

- It installs CPU-only PyTorch (was correct before the RLlib migration).
- It creates a venv `jsbsim_rl` instead of the documented conda `marl_env`.
- It references scripts that no longer exist.

**Do not run `setup_wsl2.sh`**. Use these steps instead:

```bash
# 1. Install WSL2 + Ubuntu 22.04 (Windows, admin PowerShell):
#    wsl --install -d Ubuntu-22.04
# 2. Inside WSL, install Miniconda (TUNA mirror because USTC repodata is
#    blocked and repo.anaconda.com is ~91 KB/s):
#    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
# 3. Create marl_env and install:
conda create -n marl_env python=3.10 -y
conda activate marl_env
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple \
    "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple \
    "stable_baselines3==2.9.0" gymnasium==1.2.2 jsbsim==1.3.1 \
    "ray[rllib]==2.40.0" matplotlib scipy tensorboard pyyaml transforms3d
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple \
    pytest pytest-cov ruff
# 4. Restore data/jsbsim/ if missing (gitignored):
#    The repo has no committed copy. Either install JSBSim's default
#    aircraft library or use a sibling checkout. This directory MUST exist
#    or every F-16 reset will fail with "could not find aircraft file".
```

**Why `pip install -U torch` will silently break your setup**: it pulls
torch cu130 wheels, which conflict with the NVIDIA driver 528.79 this
project was developed on (and the user has external monitors connected
to the dGPU that lose output on driver upgrades). If you upgrade torch
without `--index-url https://download.pytorch.org/whl/cu126`, you will
spend an hour debugging "CUDA available = False".

**Why `mar_env` python absolute path**: every shell script in this
repository uses `/home/sean/miniconda3/envs/marl_env/bin/python`
directly because `conda activate` from a script requires sourcing
`/etc/profile.d/conda.sh` and is fragile inside `nohup` / `&`-detached
background runs. Stick with the absolute path.

## 3. Reproducing the 2v1 SB3 97.3% baseline (30 minutes)

This is the **only** path that has any pre-committed, runnable artifact.

`1ca3e63` ("purge legacy files") removed `src/environment/formation_env.py`
(868 lines), which the benchmark script imports. The purge also dropped
~20 other source files. Everything still exists in history at
`1ca3e63^` — you just need to run the benchmark from a worktree there.

```bash
cd ~/jsbsim-marl-formation
git worktree add ~/jm-repro-97p3 1ca3e63^
# In the worktree:
cp -r /home/sean/jsbsim-marl-formation/data/jsbsim ~/jm-repro-97p3/data/
# (you need the JSBSim XML data; see §2)
cd ~/jm-repro-97p3
/home/sean/miniconda3/envs/marl_env/bin/python \
    scripts/benchmark_sb3_baseline.py --episodes 30 --difficulty 0.0 \
    --output benchmarks/sb3_2v1_97p3_repro/
# Expect ~30 seconds wall, capture rate ~90% (95% Wilson CI [74, 97]).
```

Reconciling the headline 97.3% with reality: the README mentions 92%, 94%
and 97.3% in different places. 97.3% is most plausibly the **upper bound of
the 95% Wilson CI** (97.22% rounded) from a 100-episode run; the point
estimate is ~94%. Report the capture rate as `~94% ± 3% (100 ep 95% CI)`
and report your 30-ep reproduction as `~90% ± 11% (30 ep 95% CI)`. They
are statistically consistent.

The committed `metrics_diff*.json` files are `n_episodes = 1` — they are
single-episode sanity checks, not the 30- or 100-episode aggregates the
README implies. To get a number with a real confidence interval, run
with `--episodes 100`.

## 4. Reproducing the 1v1 missile shoot sealed v2 (3 hours)

The 1v1 missile work lives on the **current** branch and uses the modern
RLlib path. It was originally verified at 99.9% ID kill / 98% d2 kill on
1000-seed holdout (see `docs/summary_phase1.md`). The reproduction here
reaches 87% on a 100-ep holdout — same architecture, smaller data
budget.

The path is **four steps**, each has a wrapper script:

```bash
# Step 1 — rule-expert data, ~25 min on CPU
/home/sean/miniconda3/envs/marl_env/bin/python \
    scripts/generate_shoot_rule_expert.py --episodes 200 \
    --difficulty 0.0 --cmd-speed 280 --seed 42 \
    --out data/expert/shoot_rule_expert.npz
# Produces: 24 MB npz, 265K transitions, 200 episodes.
# obs (41-dim) + action (4-dim). fire-allowed 3.54%, fire-fired 0.21%.

# Step 2 — BC pretraining, ~3 min on CUDA, 40 epochs
/home/sean/miniconda3/envs/marl_env/bin/python \
    scripts/train_shoot_bc.py --data data/expert/shoot_rule_expert.npz \
    --epochs 40 --batch-size 512 --lr 1e-3 \
    --output data/expert/shoot_bc_round1_baseline.pth \
    --metrics-out logs/bc_train_metrics.json --seed 42
# Best epoch typically ~36, hdg_acc 99.1%, spd_acc 98.5%,
# fire_recall 92-95% (the fire head is still too conservative here,
# which is the point of step 3).

# Step 3 — fire-head distillation, ~2.5 hours, 300 rollout + 15 epochs
/home/sean/miniconda3/envs/marl_env/bin/python \
    scripts/distill_fire_asap.py --rollout-episodes 300 --epochs 15 \
    --lr 1e-2 --eval-seeds 60 \
    --weights data/expert/shoot_bc_round1_baseline.pth \
    --out-weights data/expert/shoot_bc_asap_distilled.pth
# The script self-tests: it must print "[p2a] VERDICT: PASS" with
#   hdg/spd logits diff = 0.00e+00
#   seq = True
#   fire-agreement = True
# At 60 eval seeds, expect: id kill 85%, dist2_3k kill 75%
# (matches the ASAP-rule ceiling; fire head has been inverted from
# "conservative BC" to "fire whenever mask allows").

# Step 4 — 100-episode holdout, ~20 min
/home/sean/miniconda3/envs/marl_env/bin/python \
    scripts/eval_bc_1v1.py \
    --weights data/expert/shoot_bc_asap_distilled.pth \
    --episodes 100 --difficulty 0.0 --seed 42 \
    --out results/shoot_eval/eval_distilled_d0_s42.json
# Expect: kill_rate 0.87, hit_rate 0.997, lost 0%, wez_reach 1.0,
# launches_per_episode 3.85, wez_to_fire_latency_median 1 step.
```

The 100-seed holdout 87% kill rate is below the sealed 99.9%. The gap
comes from data scale (200 expert episodes vs an estimated 10x more in
the sealed run, 100-ep holdout vs 1000-seed holdout) and the BC quality
ceiling — not from the distillation step, which already perfectly aligns
to the ASAP-rule ceiling on 60 seeds (85% id, 75% d2, winutil 99%).
To reach 99%+ you would need to increase the BC expert episode count to
~2000+ and the holdout to 1000 episodes.

## 5. The "PPO path" is a dead end — do not use

The README's "Quick Start" still shows:

```bash
python scripts/_train_shoot_1v1.py --iterations 120 --cooperative --no-bc
python scripts/_train_shoot_1v1.py --iterations 300 --cooperative --warmup 200000
```

This is a **trap**. The 1v1 missile shoot task has only one agent (a
single F-16), so `--cooperative` is meaningless on it. The 120-iter PPO
run on this machine converged to a 7% kill rate with a 62% lost-target
rate, with the **best reward at iteration 19 and continuous regression
afterwards** — a textbook confirmation of the warning in
`summary_phase1.md` Section 4 finding 5:

> "PPO 暂缓: 蒸馏模型稳定复现 ASAP 后, 当前奖励/环境下不存在发射时机
> 取舍, PPO 无增量收益且有回退风险; P2B 仅在出现发射成本、弹药稀缺、
> 窗口选择或轨迹改变需求时启用."

The PPO path learned to fire perfectly (100% hit rate) but cannot reach
the WEZ (62% lost target). Only the BC + ASAP-distillation path produces
a working policy for 1v1 missile shoot. **The README's Quick Start
should be ignored for this task.**

## 6. GitHub network constraint on WSL2

WSL2 on a Chinese ISP can reach GitHub via SSH but **not** via HTTPS.
This is verified on the current machine but is likely true on any
Chinese-network WSL2 install. If `git clone https://github.com/...` hangs
with rc=124, switch to SSH:

```bash
git remote set-url origin git@github.com:NishimiyaXSean/jsbsim-marl-formation.git
# Configure ~/.ssh/config with an `github-443` alias pointing to
# ssh.github.com:443 as a fallback for networks that block :22.
```

If `pypi.org` is also blocked (it usually is), use the USTC mirror
(`-i https://mirrors.ustc.edu.cn/pypi/web/simple`). For conda, TUNA
works and USTC does not.

## 7. Tooling traps

- `bash ... | tr -d '\r'` exits with non-zero on Windows because the Git
  Bash environment does not include `tr` in its PATH. This means
  `TaskOutput` reports "failed" for any backgrounded training even when
  the WSL-internal command succeeded. **Always verify the actual
  artifacts on the WSL side** (`.npz`, `.pth`, `.json` file existence
  and mtime) before reacting to a "failed" status.
- WSL-inherited CWD can confuse `cd`; prefer `git -C /abs/path` over
  relying on `cd` at the top of a script.
- Do not use `pkill -f RolloutWorker` with a pattern that matches the
  command line of the running shell itself — the shell can kill
  itself. Use exact PIDs or `grep -E "[r]ay::|RolloutWorker"` to
  avoid the self-match.
- Ray `sample_timeout_s` is the only per-iteration timing instrument
  for `_train_shoot_1v1.py` (it fires exactly when an iteration hits
  the timeout). Don't silence it until you have a different timing
  method.
- The training script `_train_shoot_1v1.py` prints metrics only every
  10 iterations (`if i % 10 == 0` for the older `train_formation_rllib.py`,
  every iteration for the shoot script). To get at least two data
  points on either, you must run at least 10 (formation) or 11
  (shoot) iterations.

## 8. RLlib deprecations to migrate before next Ray upgrade

Three internal APIs are deprecated and will fail in a future RLlib
release. The warnings are emitted on every PPO `train()` call:

- `build` → `build_algo`
- `compute_single_action` → `compute_single_action` (RLModule API)
- `_get_slice_indices` (in `sgd.py`)

They are harmless for now. Plan a migration before bumping `ray[rllib]`.

## 9. What the README gets right

Despite the gaps above, the README's high-level structure is sound:

- The architecture diagrams and per-task section ("Environments") are
  accurate.
- `docs/summary_phase1.md` and `docs/phase1_freeze.md` are the **most
  reliable** references for the 1v1 missile shoot history; the
  "Phase 1: 1v1 导弹制导空战（封版 v2, 2026-08）" section in the README
  is a faithful summary of those docs.
- The reproduction numbers in the README (95.6% positive rate,
  99.9% ID kill, 98% d2 kill, 0% lost) match `summary_phase1.md` and
  the 60-seed p2a distillation gates exactly.
- `pyproject.toml` is the authoritative dependency list. The README
  does not contradict it.

## 10. Quick verification checklist (one command per item)

```bash
# Environment
/home/sean/miniconda3/envs/marl_env/bin/python -c "import jsbsim, stable_baselines3, gymnasium, torch; print(jsbsim.__version__, stable_baselines3.__version__, torch.__version__)"
# Expected: 1.3.1 2.9.0 2.9.1+cu126 (or newer; pin torch to cu126)

# Data
ls data/jsbsim/aircraft/  # must exist; not in git
# If absent, copy from a sibling clone or use JSBSim's default search path.

# Repo state
git -C ~/jsbsim-marl-formation log -1 --oneline  # HEAD on feature/refactor-task-based
git -C ~/jsbsim-marl-formation status --short    # working tree clean

# WSL network
ssh -T git@github.com   # expect: Hi NishimiyaXSean! You've successfully authenticated...

# GPU passthrough
nvidia-smi  # should show the WSL-recognized NVIDIA driver
```
