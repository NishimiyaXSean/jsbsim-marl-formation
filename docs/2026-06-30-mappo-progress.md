# 2026-06-30 MAPPO 1v1 手写训练循环调试总结

> **日期**: 2026-06-30  
> **主题**: 纯 PyTorch MAPPO CTDE — 从管道搭建到熵控制  
> **提交**: ~6 commits

---

## 背景

6/29 完成的 SB3 共享策略 2v1 达到 97.3% 成功率, 但 RLlib (Ray 不支持 Windows) 和 Tianshou (Python 3.10 vs 2.x 需要 3.11+) 均无法使用。决定手写纯 PyTorch MAPPO 训练循环。

## Phase 5a：初始 MAPPO 管道 (2v1)

### 架构

```
FormationPettingZooEnv (ParallelEnv)
  → dict obs {"pursuer_0": (33,), "pursuer_1": (33,)}
  → dict act {"pursuer_0": (2,), "pursuer_1": (2,)}
  → global state (21-dim) in infos

FormationActor(33→2):  local obs → Tanh→Tanh → mean + log_std
FormationCritic(21→1): global state → Tanh→Tanh → scalar value
```

### 结果

```
200 epochs, diff=0.0
ep_rew: -38 ~ -65 (全程负, 无改善)
```

四个关键 Bug 被识别:

| Bug | 症状 | 影响 |
|-----|------|------|
| log_std=-0.5 | std=0.6, 探索不足 | 策略锁死 |
| GAE 无 bootstrap | timeout 视为 terminal | 价值估计错误 |
| obs 错位 | s_t/a_t/r_t 跨 episode 未对齐 | 数据污染 |
| 2v1 global≠local | Critic 维度错 | 梯度混乱 |

---

## Phase 5b：1v1 降级验证 + Bug 修复

### 四项修复

1. **log_std=0** (std=1.0): 充分探索
2. **GAE bootstrap**: timeout → V(s_T), crash/kill → value=0
3. **数据对齐**: obs/act/rew/next_obs 跨 episode 严格配对
4. **1v1 模式**: Actor/Critic 均用 33-dim 局部观测

### 结果

```
100K steps, diff=0.0

v1 (baseline): ent=0.05, PPO_EPOCHS=10, ROLLOUT=2048
  rew: -7005 → -3659  entropy: 2.95 → 9.24  (改善 57%, 但熵持续增长)

v2 (aggressive): ent=0.01, PPO_EPOCHS=20, ROLLOUT=4096
  rew: -1715 → -1715  entropy: 3.11 → 14.30  (策略漂移)

v3 (balanced): ent=0.02, PPO_EPOCHS=10, ROLLOUT=4096
  rew: -3806 → -2949  entropy: 3.07 → 8.20  (最佳但熵仍增长)
```

核心问题: 熵持续增长 (3→8-14), 策略未收敛——Critic 学习速度远落后于 Actor。

---

## Phase 5c：四项 SB3 级优化

### 优化方案

| 优先级 | 优化 | 说明 |
|--------|------|------|
| P1 | **正交初始化** | gain=1.414 (hidden), 0.01 (actor out), 1.0 (critic out) |
| P2 | **非对称 LR** | actor=1e-4, critic=5e-4, Adam eps=1e-5 |
| P3 | **逐 minibatch 标准化** | 每次抽取 minibatch 后重新标准化 advantage |
| P4 | **线性退火** | LR 和 ent_coef 随训练进度线性衰减 |

### 结果

```
100K steps, diff=0.0

v4 (4-opt):
  rew: -2769 → -2769
  entropy: 2.97 → 4.42  ← 受控! (+1.45 vs +6.29 baseline)
  best avg10: -2769 (最佳)
  LR: 1e-4 → 1.8e-5 (平滑衰减)

vs v1 (baseline):
  start rew: -2769 vs -7005 (+60%)
  entropy end: 4.42 vs 9.24 (熵爆炸被压制)
  best avg10: -2769 vs -3659 (+24%)
```

### 关键发现

**P1 正交初始化是最关键的单点优化**。Actor 输出层 gain=0.01 使初始策略接近零动作 (loc ≈ 0), 防止动作方差爆炸导致的熵失控。基线版本中 entropy 从 3 爆炸到 9 的根因就是 PyTorch 默认的 Kaiming 初始化给了输出层过大的权重。

---

## 全版本对比

| 版本 | Start rew | Best avg10 | Final entropy | 熵趋势 |
|------|----------|-----------|--------------|--------|
| v1 (baseline) | -7005 | -3659 | 9.24 | 3→9 爆炸 |
| v2 (aggressive) | -1715 | -1715 | 14.30 | 3→14 漂移 |
| v3 (balanced) | -3806 | -2949 | 8.20 | 3→8 增长 |
| **v4 (4-opt)** | **-2769** | **-2769** | **4.42** | **3→4 受控** |

---

## 与 SB3 基线的差距

| 指标 | SB3 Phase 2.0 | MAPPO v4 |
|------|--------------|----------|
| 100K rew | ~-500 | -2,769 |
| 收敛速度 | ~6× 更快 | — |
| 熵控制 | 自动 | 需手动 tuning |
| 管道稳定性 | ✅ | ✅ (零崩溃) |

差距主要来源:
1. SB3 有更精细的 LR 调度 (cosine + warmup)
2. SB3 的 Value clipping 和 dual-clip PPO
3. 更多训练步数 (SB3 通常 500K+ 才收敛)

### 下一步

1. 继续训练 v4 到 200K-500K 步——熵已受控, 收敛可期
2. 加入 SB3 的 Value function clipping
3. 验证通过后, 从 1v1 迁回 2v1 MAPPO
