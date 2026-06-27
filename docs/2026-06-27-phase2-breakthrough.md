# Phase 2 连续动作 BC→PPO 突破性训练总结

> **日期**: 2026-06-27  
> **主题**: 从 0% 到 65.4% success — 行为克隆 + 动态决策率 + 渐进奖励三重奏  
> **提交**: 4 commits  
> **总计训练步数**: ~1,300,000（跨 3 个训练版本）

---

## 1. 起点：Phase 1 死锁

6 月 26 日 Phase 1 离散 BFM 训练以 **0% 真 success 率** 告终。经过 7 个版本、220 万步后，Agent 从未完成一次真正的拦截。

**根因链**：
1. Discrete(9) 动作空间粒度不足（60° 坡度转弯，无法 5-15° 微调）
2. 探索空间 9^120，随机碰到 success 的概率趋近于零
3. 熵坍缩 → Agent 学会 "不做任何事以避免失速"

---

## 2. 破局三阶段

### Phase A：纯连续 Box(2) 环境（Commit `c2acf70`）

**架构**：
```
Discrete(9) BFM → Box(2) [turn_rate, speed] → FlightController
```

**关键修复**：
- **Bug 1**: Warmup 在 2D 模式下走 BFMAutopilot 导致高度漂移 → 改用 FlightController
- **Bug 2**: 目标高度随机偏移 → 2D 模式强制同高度
- **Bug 3**: BFM 机动时高度掉落 → FC 接管升降舵，BFM 管横滚+油门

**效果**：ep_rew 从 -3,000 改善到 +886，entropy 不再坍缩，但 success 仍为 0%。

---

### Phase B：CARW + 渐进奖励（Commit `1bff327`）

**CARW (Close-in Adaptive Rate Window)**：
```
dist >= 500 m →  2 Hz (0.5s, 30 微步) — 巡航
dist <  500 m → 10 Hz (0.1s,  6 微步) — 末端引导
```

**渐进末端奖励**：200-500m 线性引力场，桥接奖励荒漠。

**效果**：Value 网络 explained_variance 0→0.87，ep_len 翻倍至 135，但仍 0% success——Agent 学会了靠近但过不了最后 200m。

---

### Phase C：行为克隆预热（Commit `7f139c3`, `2416f08`）

**PN 专家数据生成**：
- PN 导引 (N=3, Kp=0.7) + FlightController
- 500 成功轨迹 / 38,344 (obs, action) 对
- Difficulty 0.0→0.3，总体成功率 39%

**关键 Bug 修复**：
- `ANTI_STALL_MIN_DIST` 300→200（消除 200-300m 死区）
- Success 判定改用 episode 起始距离（非宏动作起始距离），防止 CARW 每步门控

**BC 预训练**：
- 50 epoch MSE 监督，val_loss 0.044→0.006
- Actor 学会拦截基础模式

---

## 3. 最终结果：BC→PPO 微调 500K 步

| 指标 | Phase 1 离散 | Phase 2.0 连续 | **Phase 2.2 BC→PPO** |
|------|-------------|---------------|---------------------|
| 最终 success | 0% | 0% | **51.9%** (峰值 65.4%) |
| 最高 difficulty | 0.00 | 0.00 | **0.28** |
| 最终 stall | 98.7% | 89.3% | **48.1%** |
| entropy | 坍缩 | 稳定 | **稳定 (-2.83)** |
| explained_var | ~0 | ~0 | **0.905** |
| ground_crash/lost/oob | 0% | 0% | **0%** |

### Success 率演进曲线

```
 10K:   2.1%  ← 史上首次 success!
 40K:  10.2%  ← 稳定学习
 80K:  25.0%  ← 突破四分之一
150K:  38.4%  ← difficulty 自动推进到 0.06
260K:  49.4%  ← 接近过半，difficulty=0.12
340K:  55.6%  ← 过半，difficulty=0.16
420K:  59.0%  ← difficulty=0.22
490K:  65.4%  ← 峰值，difficulty=0.26
500K:  51.9%  ← 终值，difficulty=0.28
```

### 课程学习自动推进

Difficulty 从 0.00 自动推进到 0.28——Agent 持续适应更难的目标（更大初始距离、更大方位角偏移、目标更快的转弯速率）。

---

## 4. 关键工程决策

| 决策 | 理由 |
|------|------|
| **新建 ContinuousPursuitEnv** | 保留 BFM 离散基线用于论文消融实验对比 |
| **继承 BFMPursuitEnv** | 复用观测/奖励/终止/Tacview，只改变动作空间 |
| **纯连续 Box(2)** | SB3 原生支持，避免 5+1 参数化空间的 argmax 梯度断裂 |
| **FlightController 全权接管** | 消除 BFM 飞控的能量-高度耦合，真正解耦为 2D 问题 |
| **BC 在动作空间确定后执行** | PN 专家生成的动作标签必须匹配最终动作空间 |

---

## 5. 下一步方向

1. **扩大 BC 数据集**：覆盖 difficulty 0.0–0.5，增加轨迹多样性
2. **更高难度训练**：从 BC 预热直接启动 difficulty=0.15+ 的训练
3. **多机编队迁移**：将经过验证的 ContinuousPursuitEnv 架构迁移到 MAPPO 多机框架
4. **论文消融实验**：Phase 1 离散 vs Phase 2 连续的受控对比

---

## 提交清单

```
2416f08 feat: BC pretraining pipeline — PN expert data + PPO actor warm-start
7f139c3 fix: ANTI_STALL_MIN_DIST=200 + episode start_dist for CARW success gate
1bff327 feat: CARW dynamic decision rate + terminal-pull reward gradient
c2acf70 feat: Phase 2 continuous Box(2) pursuit — 2D altitude-lock fixes + FlightController takeover
```
