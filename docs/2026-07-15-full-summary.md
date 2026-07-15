# 2026-07-15 完整工作总结：V6 尸检 → V7 六项优化 → 正式训练

> **日期**: 2026-07-15
> **主题**: V6 独立 Actor 深度尸检 → 奖励漏洞诊断与修复 → V7 六项优化整合 → 正式训练启动
> **提交**: 13 commits
> **核心成果**: 发现 P1 "79% Slow 巡航观众" 策略 → 指数 pincer 衰减 + 增强反摸鱼 + 软碰撞塑形 → V7 启动

---

## 1. V6 独立 Actor 深度尸检

### 1.1 尸检数据 (50 集, `diagnose_v6_autopsy.py`)

| 指标 | 值 | 诊断 |
|------|:--:|------|
| 脱靶率 | **86%** | P1 根本没有参与追击 |
| P1 平均终点距离 | **8,441m** | 飞到了 8.4 公里外 |
| P0 平均终点距离 | 3,519m | 也在远离 |
| 协同成功率 | 8% | 瞬触 AND 2 步触发 |
| 碰撞率 | 2% | 极低——P1 主动避战 |

### 1.2 P1 动作分布 (50 集 eval)

| 动作 | P0 (Striker) | P1 (Interceptor) |
|------|:---:|:---:|
| **Slow (180 m/s)** | 42% | **79%** 🔥 |
| Cruise (250) | 22% | 5% |
| Fast (320) | 36% | 16% |
| SoftLeft | 23% | **44%** 🔥 |
| Straight | 24% | 30% |

**诊断**: P1 的策略是 "79% Slow + 44% SoftLeft = 在安全距离外缓缓画圈当观众"。不是直线逃跑，而是巡航摸鱼。

### 1.3 最佳 episode (+31,854) 时间线分析

```
t=0-50:   P1 2243→2828m   ← 远离目标
t=50-200: P1 2648-2862m   ← 持续在 2.6km+ 徘徊
t=200-250: P1 开始回归
t=250-308: P1 进入 1200m  → AND 瞬间触发 2 步 → cooperative_success
```

pincer 奖励实际仅贡献 2,940 pts (9%)——早期可视化脚本因忽略距离门虚增了 12 倍。

---

## 2. 奖励漏洞诊断与修复

### 2.1 Pincer 距离门过宽

**根因**: `PINCER_DIST_MAX = 2000m` 是固定值，比 Stage 1 AND 距离 (1200m) 宽 67%。两机可在 1500-2000m 巡航刷 pincer 而不逼近。

**修复**: 动态软包络门——`pincer_gate = AND_dist * 1.2` (Stage 1: 1440m)

### 2.2 硬阈值 → 指数衰减

**修复**: 将二元 mask 替换为连续指数衰减：
```
R_pincer = 35.0 × min(θ, AND_angle) × exp(-d0/τ) × exp(-d1/τ)
τ = AND_dist
```
- d=1200m (1τ): 37% 奖励
- d=2400m (2τ): 2% → 经济上不可能刷分
- 全程有梯度，无 dead zone

### 2.3 增强反摸鱼惩罚

**修复**: 目标 P1 的 "79% Slow 巡航" 行为——
```
if d > AND_dist AND (speed < 200 m/s OR velocity > 90° away from target):
    penalty = -10.0/step
```
- Slow 头朝向目标：仍然惩罚（低速下无法有效建包抄）
- Cruise/Fast 45-90° 包抄机动：不惩罚

### 2.4 软碰撞塑形

**根因**: -3000 + 立即终止 → P1 害怕碰撞 → 躲到安全区

**修复**: 
- 10-100m: 平滑排斥力 `-10 × (100 - d_mate)²` per second
- <10m: 仍 -3000 + 终止（真实碰撞）
- 连续梯度让 agent 学会安全逼近而非恐惧躲避

### 2.5 NaN 防护

分母加 epsilon guard (`+1e-6`): pincer cos 除法 + fleeing check 除法

### 2.6 Ego-centric Critic 修复

**Bug**: P1 的 Critic 收到的 global_state 始终是 [P0, P1, Target] 固定顺序——token 0 是 P0 而非自己。Self 和 Mate 在 P1 视角下被交换。

**修复**: 为每 agent 构建以自我为中心的 [Self, Mate, Target] 顺序。

---

## 3. V7 完整架构

```
┌─────────────────────────────────────────────────────────┐
│                    V7 Final Stack                         │
│                                                          │
│  执行层                                                  │
│    ✅ Independent Actor MAPPO (policy_p0 / policy_p1)    │
│    ✅ Ego-centric global_state per agent                 │
│    ✅ BC surgical padding (self_proj + mate_proj)        │
│    ✅ 23/23 keys each, zero skipped                      │
│                                                          │
│  奖励层                                                  │
│    ✅ Exponential pincer decay (连续梯度, 无硬门)        │
│    ✅ Progress 1.0 + Potential pincer 35.0               │
│    ✅ Enhanced anti-loiter -10.0/step                    │
│    ✅ Soft collision shaping (10-100m 平滑排斥)          │
│    ✅ One-shot OR fallback +1,000 (不终止回合)          │
│    ✅ Suicide-proof penalties -3,000                     │
│    ✅ OOC -2.0/step per-agent                            │
│                                                          │
│  课程层                                                  │
│    ✅ 3-stage curriculum with performance gates          │
│    ✅ Temporal relaxation (sustain 2→4→6)                │
│    ✅ Smooth transitions (15-iter interpolation)         │
│    ✅ Sustain_steps in transition (bug fixed)            │
│                                                          │
│  信息层                                                  │
│    ✅ One-hot Agent ID (2 dims)                          │
│    ✅ Mate broadcast channel (4 dims: turn, speed, hdg)  │
│    ✅ Speed-dependent turn rates (Slow +33%, Fast -20%)  │
│                                                          │
│  容错层                                                  │
│    ✅ NaN epsilon guards (pincer cos + fleeing check)    │
│    ✅ Early termination for hopeless episodes            │
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

### V7 vs V6 关键差异

| | V6 | V7 |
|------|------|------|
| Actor | 独立 | 独立 |
| Pincer 距离门 | 固定 2000m | 指数衰减 (τ=AND_dist) |
| 反摸鱼 | closing velocity penalty | **-10.0/step** (Slow 或 逃逸) |
| 碰撞 | -3000 + 终止 | 平滑排斥力 (10-100m) |
| Critic global_state | 绝对顺序 (Bug) | **Ego-centric** |
| BC policy 命名 | policy_0/1 (Bug) | **policy_p0/p1** |

---

## 4. 全部 AND-gate 实验演进

| 版本 | 架构 | 关键创新 | Sync 峰值 | Best Eval |
|------|------|------|:---:|:---:|
| V2 | 共享 | AND-gate 课程 | 15% | −4,452 |
| V4 | 共享 | 三叉戟奖励 | 5% | −4,770 |
| V5 | 共享 | 通信 + ID + 全栈 | 15% | **+178** |
| V6 | 独立 | 独立 Actor | 5% | −3,815 |
| **V7** | **独立** | **指数pincer+反摸鱼+软碰撞+egoCritic** | **训练中** | **训练中** |

---

## 5. Git 提交 (13 commits)

```
27ad218 fix: policy naming — policy_0/1 → policy_p0/p1 to match agent IDs
7dd50ff fix: sustain_steps in transition + ego-centric global_state + log fix
308ffc3 fix: epsilon guards against NaN in pincer cos and fleeing check divisions
c359453 refactor: V7 — switch to independent actors (policy_0 / policy_1)
ab263f7 docs: update README with V5-V7 experiment results and AND-gate findings
e42daae fix: revert to shared Actor for V7 (independent Actor was V6 experiment)
772672b feat: enhanced anti-loiter + soft collision shaping for V7
3171869 feat: exponential pincer decay + closing velocity anti-loiter penalty
708a827 fix: hard multiplicative mask for pincer reward + corrected V6 autopsy figures
67f8623 fix: soft-envelope pincer gate — 1.2× AND_dist (was 2000m fixed)
6012ab6 fix: tighten pincer shaping distance gate from fixed 2000m to dynamic AND_dist
680bb83 feat: V6 autopsy diagnostic — reward breakdown, termination analysis, geometric metrics
c608772 feat: V6 — Independent Actor MAPPO (no parameter sharing)
```

---

## 6. 下一步

V7 训练已启动 (2026-07-15 20:30)，预计 2026-07-16 凌晨完成。关键观测：
- 增强反摸鱼能否根除 P1 的 "79% Slow 巡航" 策略
- 软碰撞能否降低 P1 的避战倾向
- 指数 pincer 能否提供充分的协同梯度
- V7 sync 能否突破 15% 天花板
