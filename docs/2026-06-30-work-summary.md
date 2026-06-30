# 2026-06-30 工作总结：手写 MAPPO CTDE — 从零到验证

> **日期**: 2026-06-30  
> **主题**: 纯 PyTorch MAPPO 训练循环 + 1v1/2v1 CTDE 验证  
> **提交**: ~18 commits

---

## 1. 背景

6/29 完成的 SB3 2v1 共享策略达到 97.3% 成功率，但 RLlib (Ray 不支持 Windows) 和 Tianshou (Python 3.10 vs 2.x 需 3.11+) 均无法使用。决定手写纯 PyTorch MAPPO。

---

## 2. MAPPO 1v1：从零构建到 SB3 等价

### Phase 5a：初始管道 (2v1, 失败)

200 epoch, rew -40~-65, 无改善。定位四个 Bug:

| Bug | 修复 |
|-----|------|
| log_std=-0.5 (std=0.6) | → 0.0 (std=1.0) |
| GAE timeout 未 bootstrap | → timeout 用 V(s_T) |
| obs/act/rew 跨 episode 错位 | → 严格对齐 |
| 2v1 global≠local | → 降级 1v1 |

### Phase 5b：1v1 降级验证

```
100K steps, diff=0.0, 3 轮调参

v1 (baseline): rew -7005→-3659, entropy 3→9 爆炸
v3 (balanced): rew -3806→-2949, entropy 3→8
```

核心问题: 熵持续增长, Critic 学习速度远落后于 Actor。

### Phase 5c：四项 SB3 级优化

| 优化 | 效果 |
|------|------|
| **P1 正交初始化** (gain=0.01 actor out) | 熵 3→4 受控 (vs 3→9) |
| P2 非对称 LR (actor=1e-4, critic=5e-4) | Critic 加速 |
| P3 逐 minibatch adv 标准化 | 梯度稳定 |
| P4 线性退火 | 后期收敛 |

```
v4 (4-opt): rew -2769→-2769, entropy 2.97→4.42 (+1.45 vs +6.29)

P1 正交初始化是最关键的单点优化。
Actor 输出层 gain=0.01 使初始策略接近零动作, 防止动作方差爆炸。
```

### Phase 5d：护城河补齐 + SB3 热启动

```
P0: Reward Scaling (/100) + Explained Variance 监控
P1: Value Clipping (SB3-style)
P2: 延长训练到 500K

Cold-start 500K: ev 0.11→0.83 (Critic 在学习!), 但熵仍 3→10.7
→ 结论: Actor 的 ortho init gain=0.01 过于保守, 需要热启动
```

### Phase 5e：SB3 Phase 3.6 热启动

```
SB3 [256,27] → MAPPO [256,33] (前 27 维相同, 后 6 维 padding)

Hot-start 200K: rew +3,966→+7,437, entropy 2.838 完全锁定!

vs cold-start 500K:
  Start rew: +3,966 vs -3,450
  Best rew: +7,746 vs -2,515
  Entropy: 2.84 vs 10.67 (锁定 vs 爆炸)
```

### Phase 5f：两阶段解冻 + KL 保护

```
0-50K:  Actor frozen (lr=0), Critic lr=5e-4
50K+:   Actor lr=1e-5, Critic lr=1e-4
KL 熔断: approx_kl > 0.015 → 跳过 minibatch
ent_coef=0 (SB3 权重不需要探索)

200K fine-tune: rew +7,700 稳定, entropy 2.838 零漂移
→ MAPPO 1v1 达到 SB3 同等性能! 手写管道完全验证通过。
```

---

## 3. MAPPO 2v1：CTDE 编队追猎

### 架构

```
共享 Actor(33→2) × 2 pursuers
CTDE Critic(21→1) — 随机初始化 (SB3 Critic 66-dim 不兼容)
SB3 Phase 4.1 权重平铺: [256,66]→[256,33], [4,256]→[2,256]
```

### Phase 6a：基础 CTDE

```
200K, SB3 hot-start, Actor frozen 50K

Warmup: rew +5,600 avg
Fine-tune: rew +5,000~+6,800, ev 0.10→0.49
Critic 从零自学到 ev=0.49 (健康!)
```

### Phase 6b：橡皮筋 + 助攻门控

```
针对 "懒散 pursuer" 问题:
  1. 橡皮筋惩罚: spacing > 1500m → -0.01·(d-1500)
  2. 助攻门控: 仅 dist < 1500m 的 pursuer 分享击杀奖
  3. 个体进度奖励: 每架独立 delta-distance bonus

100K fine-tune: rew +5,900, 3/4 集双方紧密协同 (间距 400-600m)
```

### Phase 6c：EV 门控超长预热

```
WARMUP_STEPS: 50K → 150K
EV 门控: 仅当 EV >= 0.6 时解冻 Actor

300K 训练:
  EV 从 0.20 爬到 0.655
  151K 时 EV 短暂回落 → 自动延长预热
  EV 达标后解冻 → rew 稳定, ev 0.60-0.66
```

### 根因分析：CTDE 平铺的信息不对称

```
SB3 共享策略 (Box(4), 66-dim):
  → 单网络看到全部 66 维, 学到了 "一人追一人歇" 的分工模式

CTDE 平铺 (Box(2)×2, 33-dim each):
  → P1 的 Actor 丢失了 P0 的位置信息
  → mate 特征 (obs[27:33]) 存在但 SB3 权重未独立依赖它们
  → P1 系统性飞偏, 无法恢复

结论: SB3 权重平铺到 CTDE 不是有效的迁移路径。
SB3 共享策略 (97.3% @ 2v1) 仍是当前最佳基线。
```

---

## 4. 技术债务与已知问题

| 问题 | 状态 |
|------|------|
| Ray 2.x Windows 不支持 | 已诊断, WSL2 是唯一解 |
| Tianshou 2.x 需 Python 3.11+ | 降级到 0.5.1, API 不兼容 |
| Tacview 绿线渲染 | 经多次迭代, 回退到最简飞机输出 |
| CTDE 平铺信息不对称 | 根因已定位, 需从头训练或架构改造 |

---

## 5. 全版本对比

| 版本 | 类型 | 架构 | 最佳 rew | 备注 |
|------|------|------|---------|------|
| SB3 Phase 3.6 | 1v1 | Box(2) | 83% success | 平滑微调 |
| **MAPPO 1v1 hot** | 1v1 | Actor(33→2) | **+7,700** | SB3 等价 ✅ |
| SB3 Phase 4.1 | 2v1 | Box(4) 共享 | **97.3%** | 分段间距 |
| MAPPO 2v1 CTDE | 2v1 | Actor×2(33→2) | +6,800 | 懒散 pursuer ❌ |

---

## 6. 输出文件

### 模型

| 文件 | 描述 |
|------|------|
| `mappo_1v1_0630_1649_s42/best_policy.pth` | MAPPO 1v1 最佳 (rew +7,700) |
| `mappo_2v1_0630_1828_s42/final_policy.pth` | MAPPO 2v1 基础 CTDE |
| `mappo_2v1_0630_2047_s42/final_policy.pth` | MAPPO 2v1 EV 门控 |

### 可视化

| 目录 | 内容 |
|------|------|
| `results/mappo_hot_viz/` | 1v1 热启动: 6 ACMI + 3D 轨迹图 |
| `results/mappo_2v1_viz/` | 2v1 CTDE: 4 ACMI + 间距分析 |

### 文档

| 文件 | 内容 |
|------|------|
| `docs/2026-06-30-mappo-progress.md` | MAPPO 1v1 调试详细记录 |
| `docs/ray-windows-startup-failure-diagnosis.md` | Ray Windows 故障诊断 |

---

## 7. 提交清单 (18 commits)

```
226cd0b feat: 2v1 MAPPO — 150K Critic warmup + EV-gated unfreeze
45be693 feat: 2v1 MAPPO — rubber band + engagement zone fine-tune
6417f4a feat: 2v1 MAPPO — rubber band penalty + engagement zone gating
f9572e9 viz: 2v1 fine-tuned trajectories — rubber band analysis
f38a1d5 viz: MAPPO 2v1 CTDE — 4 success trajectories
1c1facc feat: MAPPO 2v1 — shared Actor CTDE, rew +6,800 stable
08e3940 feat: MAPPO 2v1 CTDE — shared Actor, SB3 Phase 4.1 hot-start
c47c6ca viz: MAPPO hot-start — 6 success trajectories
2ee6abe feat: MAPPO 1v1 SB3 hot-start — rew +7,700 stable
863e8fd feat: MAPPO 1v1 — 2-stage warmup + KL early stopping + ent=0
8c3a595 feat: MAPPO 1v1 — SB3 Phase 3.6 weight hot-start
1f392dc fix: remove REWARD_SCALE double-counting
e20e056 feat: MAPPO 1v1 — P0 Reward Scaling + P1 Value Clipping
880d6f2 feat: MAPPO 1v1 — 4 SB3-verified optimizations
8b3738b tune: MAPPO 1v1 hyperparameter sweep — 3 runs
b4a8031 feat: MAPPO 1v1 baseline — working GAE+PPO with 4 bug fixes
4e5aa8f docs: 2026-06-30 MAPPO 1v1 progress
```

---

## 8. 下一步

1. **2v1 CTDE 从头训练**: 不用 SB3 平铺, 让 CTDE Actor 从零学习协调
2. **WSL2 + Ray**: 解锁 RLlib MAPPO 多机分布式训练
3. **注意力聚合**: 用 Self-Attention 替代平铺, 让 Actor 动态聚合 mate 信息
4. **SB3 97.3% 基线守卫**: 当前最佳 2v1 方案, 可用于论文消融实验
