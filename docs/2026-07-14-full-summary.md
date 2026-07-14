# 2026-07-14 完整工作总结：AND-gate 协同攻坚 — 从诊断到架构突破

> **日期**: 2026-07-14
> **主题**: AND-gate 协同死锁诊断 → CTDE 能力边界探明 → 架构层三连突破 → V5 完整配置启动
> **提交**: 12 commits
> **核心成果**: 探明 CTDE 隐式协同天花板，通过 Reward × Communication × Identity 三层突破，启动史上最强训练配置

---

## 1. AND-gate 协同死锁全链条诊断

### 1.1 发现阶段：Sustain Counter 60Hz Bug

AND-gate 的 `_coop_sustain_counter` 在 physics sub-step (60Hz) 层累加，导致 COOP_SUSTAIN_STEPS=6 仅需 0.1 秒——形同虚设。修复为 per-decision-step (5Hz) 累加。

### 1.2 探索阶段：收紧课程后的"真面目"

修复 counter 后，Stage 1 (AND 1200m/35°, target 1600-2200m) 下 sync rate 峰值仅 15%，终值 0%。三种训练 (BC hotstart old, resume, fresh V4) 一致证明：收紧后的 AND-gate 超过了 CTDE 架构的隐式协同能力边界。

### 1.3 定量阶段：奖励经济学分析

| 行为 | Reward | 与协同击杀差距 |
|------|:------:|:------:|
| 摸鱼挂机 | −3,120 | 7,120 |
| 进攻逼近 (400m) | −1,040 | 5,040 |
| 协同击杀 | +5,000+ | — |

单个 pursuer 闭合 100m = +75 pts。基线 time bleed 仅 −0.2/step/agent——时间压力太弱。而 Pincer 奖励阈值 60°，在 AND 角度 35° 之间存在巨大的 dead zone。

---

## 2. 奖励层突破："三叉戟" (V4)

### 2.1 Progress 1.5→1.0 (削弱单刷)

闭合 100m 收益从 75 降到 50——让"个人英雄主义"不如"团队配合"值钱。

### 2.2 OOC 惩罚 (消灭挂机)

per-agent, d_i > AND_dist+400m 持续 6s 后每步 −2.0。独立判定——P0 冲进去了 P1 还在外面，只罚 P1。

### 2.3 防自杀经济学

时间压力增大后的关键防护：
- REWARD_CRASH: −200 → **−3,000**
- REWARD_OOB: 0 → **−3,000** (新增)
- REWARD_LOST_TARGET: −200 → **−3,000**
- FORMATION_COLLISION: −3,000 (不变)

```
自杀 (−6,000) << 摸鱼 (−3,120) < 进攻 (−1,040) << 协同 (+5,000)
```

自杀是绝对最差策略。唯一切实可行的正收益路径是协同击杀。

### 2.4 OR-gate 劫持三修复

| Fix | 问题 | 方案 |
|-----|------|------|
| 预热期协同 shaping | OR warmup 只教"冲到 200m" | OOC 在 OR 阶段也生效 (阈值 2400m) |
| 平滑阶段过渡 | 硬切换导致 Critic 震荡 | 15 轮线性插值所有连续参数 |
| One-shot OR fallback | AND 阶段冲到 200m 毫无奖励 | +1,000 一次性奖励，Flag 锁死，不终止回合 |

---

## 3. 架构层突破：Communication + Identity (V5)

### 3.1 显式通信信道 (Mate Broadcast)

CTDE 的死锁根因之一：P1 只能看到 P0 的物理状态 (rel_pos, rel_vel)，无法预判 P0 的下一个动作。0.2s 决策延迟 = P1 永远慢半拍。

在 mate token 中新增 4 维广播：

```
mate 6→10 dims:
  原始: rel_pos(3) + rel_vel(3)
  新增: cmd_turn_rate/20.0, cmd_speed_norm, cos(ref_hdg), sin(ref_hdg)
```

BC 兼容性：旧 `mate_proj.weight [128,6]` → 新 `[128,10]`。Columns 0-5 保留 BC 权重，6-9 零初始化。`self_proj` 同理 (13→15)。

### 3.2 Agent One-Hot ID (打破对称)

参数共享 MAPPO 的根本矛盾：网络不知道"我是 P0 还是 P1"。角色分化只能被动依赖几何位置，无法主动规划。

在 observation 中拼接 one-hot ID:

```
base 27→29 dims:  [original 27] + [P0=10 or P1=01]
```

Self token 13→15 dims。`self_proj` surgical padding (cols 0-12 from BC, 13-14 zero)。

### 3.3 Potential Pincer Shaping (密集梯度)

旧 pincer 阈值 60°，AND-gate 35° 之间存在 dead zone (35-60°)。替换为：

```
r = 35.0 × min(θ_pincer, AND_angle) × dt  (both in range, θ > 0)
```

从 0° 到 AND angle 线性推拉，到阈值即停——真正的 payoff 来自 cooperative_success。

### 3.4 Temporal Relaxation (课程动态维持步数)

| Stage | Sustain | 物理含义 |
|:-----:|:------:|------|
| 1 (温室) | **2 步** (0.4s) | 瞬时触发——先让 agent 尝到协同击杀的正反馈 |
| 2 (逼近) | **4 步** (0.8s) | 短暂维持——强迫近战动态微调 |
| 3 (实战) | **6 步** (1.2s) | 全要求——高精度三维协同控制稳定性 |

---

## 4. V5 完整配置：CTDE 架构最强形态

```
┌─────────────────────────────────────────────────────────┐
│                    V5 Stack                              │
│                                                          │
│  架构层                                                  │
│    ✅ One-hot Agent ID (2 dims) — 打破对称               │
│    ✅ Mate Broadcast (4 dims) — 显式通信                 │
│    ✅ Speed-dependent Turn Rates — 物理自适应            │
│                                                          │
│  奖励层                                                  │
│    ✅ Progress 1.0 / Pincer Shaping 35.0 / OOC -2.0     │
│    ✅ One-shot OR fallback +1,000 (不终止)               │
│    ✅ Suicide-proof penalties -3,000                     │
│                                                          │
│  课程层                                                  │
│    ✅ 3-stage curriculum with performance-based gates    │
│    ✅ Temporal relaxation (2→4→6 sustain)                │
│    ✅ Smooth transitions (15-iter interpolation)         │
│    ✅ Warmup cooperative shaping (OOC in both phases)    │
│                                                          │
│  工程层                                                  │
│    ✅ BC surgical padding (self_proj + mate_proj)        │
│    ✅ Sustain counter per decision-step (5Hz)            │
│    ✅ Early termination for hopeless episodes            │
│    ✅ Transition drive every iteration                   │
└─────────────────────────────────────────────────────────┘
```

### 启动命令

```bash
python scripts/train_formation_rllib.py \
  --iterations 500 --cooperative --warmup 200000 \
  --lr 3e-4 --entropy-coeff 0.05 \
  --checkpoint-freq 25 --eval-interval 25 --eval-episodes 20 \
  --seed 42
```

---

## 5. Git 提交 (12 commits)

```
ebc0bb5 feat: potential pincer shaping + temporal relaxation on sustain steps
40b9385 fix: dense pincer shaping (20→150deg) + suicide-proof crash penalties (-3000)
ae2e332 feat: add agent one-hot ID to observation (+2 dims) + surgical pad self_proj
7f1f217 fix: surgical padding of mate_proj.weight for BC compatibility (6→10 dims)
5ffa905 feat: add explicit mate broadcast channel to observation (+4 dims)
b2bdb70 fix: drive curriculum transition every iteration, not just eval
9e75438 feat: three-pronged fix for OR-gate hijacking + stage shock + warmup shaping
a7368fe feat: reward rebalance — progress 1.0, pincer 30.0, per-agent OOC penalty
e981b75 fix: resume now applies curriculum Stage 1 params (was OR-only)
6820468 feat: AND-gate Stage 1 cooperative success trajectory visualization
376227b fix: tighten AND-gate curriculum + add early termination for hopeless episodes
6d86b6d fix: AND-gate sustain counter now ticks per decision step (5Hz), not physics sub-step (60Hz)
```

---

## 6. 下一步

V5 训练已启动 (2026-07-14 22:00)，预计 2026-07-15 凌晨完成。关键观测窗口：
- Iter ~25: AND-gate Stage 1 激活 (sustain=2)
- Iter 50-100: 温室 sync rate 能否破冰
- Iter 200+: 是否触发 Stage 2 晋级 (sync >50%)
