# 2026-06-27 全日训练总结：从 0% 死锁到 65.4% 突破

> **日期**: 2026-06-27  
> **主题**: Phase 2 连续动作环境 → BC 破局 → Phase 3 智能前置追踪  
> **提交**: 8 commits  
> **总计训练步数**: ~2,300,000（跨 5 个训练版本）

---

## 目录

1. [起点：Phase 1 死锁](#1-起点phase-1-死锁)
2. [Phase 2.0：纯连续 Box(2) 环境](#2-phase-20纯连续-box2-环境)
3. [Phase 2.1：CARW + 渐进奖励](#3-phase-21carw--渐进奖励)
4. [Phase 2.2：BC 破局 — 首次 success](#4-phase-22bc-破局--首次-success)
5. [Phase 3v1：智能前置追踪（首次尝试）](#5-phase-3v1智能前置追踪首次尝试)
6. [Phase 3v2：重校准恢复](#6-phase-3v2重校准恢复)
7. [全部训练版本对比](#7-全部训练版本对比)
8. [关键工程决策](#8-关键工程决策)
9. [遗留问题与下一步](#9-遗留问题与下一步)

---

## 1. 起点：Phase 1 死锁

6 月 26 日 Phase 1 离散 BFM 训练以 **0% 真 success 率** 告终。

| 版本 | 动作空间 | 最终 success | 核心问题 |
|------|---------|-------------|---------|
| Phase 1 v1–v7 | Discrete(9) BFM | **0%** | 熵坍缩、动作粒度不足、探索空间 9^120 |

**根因链**：离散 9 动作无法做 5-15° 微调 → 60° 坡度转弯消耗过量能量 → 30s 内失速 → Agent 学会 "什么都不做" → Decelerate/Level Flight 惰性策略。

---

## 2. Phase 2.0：纯连续 Box(2) 环境

**Commit**: `c2acf70`

### 架构

```
Discrete(9) → Box(2) [turn_rate_factor, speed_factor] → FlightController
```

Agent 输出连续转向率 (±15°/s) 和速度目标 (150–350 m/s)，FlightController 负责所有底层飞控。

### 三个关键 Bug 修复

| Bug | 症状 | 修复 |
|-----|------|------|
| Warmup 高度漂移 | lock_altitude 下 warmup 仍走 BFM，前 3 秒高度偏移 | warmup 也用 FlightController |
| 目标高度随机 | 2D 模式下目标仍生成在不同高度 | lock_altitude 时强制 target_alt_offset=0 |
| BFM 机动高度掉落 | 只有 Action 0 走 FC，转弯时高度仍由 BFM 控制 | BFM 管横滚+油门，FC 接管升降舵 |

### 训练结果

```
500K 步, difficulty=0.0, lock_altitude=True

ep_rew:  -928 → +886    (正向!)
ep_len:   47 →   90     (存活时间翻倍)
entropy: 稳定 -2.85      (不坍缩)
stall:   98.7% → 89.3%
success: 0%             (仍未突破)
```

**结论**：连续动作解决了熵坍缩和能量管理，但 reward signal 仍太稀疏——Agent 从未随机撞到 success。

---

## 3. Phase 2.1：CARW + 渐进奖励

**Commit**: `1bff327`

### CARW (Close-in Adaptive Rate Window)

```
dist >= 500 m →  2 Hz (0.5s hold, 30 微步) — 巡航节能
dist <  500 m → 10 Hz (0.1s hold,  6 微步) — 末端精确引导
```

### 渐进末端奖励

200–500m 线性引力场，bridge 奖励荒漠与 5000 分 success 悬崖之间。

### 训练结果

```
300K 步, difficulty=0.0

ep_rew:           -928 → -26.9
ep_len:            125 → 135
explained_variance: 0 → 0.87    ← Value 网络活了!
stall:             97.5% → 98.7%
success:           0%
```

**结论**：Value 网络有史以来首次可训练（explained_var=0.87），但 success 仍为 0%。Agent 学会了靠近目标赚 shaping 分，但过不了最后 200m。

---

## 4. Phase 2.2：BC 破局 — 首次 success

**Commits**: `7f139c3`, `2416f08`, `6a2dc92`

### 关键修复

| 修复 | 内容 |
|------|------|
| ANTI_STALL_MIN_DIST | 300m → 200m（消除 200-300m 死区） |
| Success 判定 | 宏动作 start_dist → episode start_dist（CARW 兼容） |

### BC 数据生成

- PN 导引专家 (N=3, Kp=0.7) + ContinuousPursuitEnv
- **500 成功轨迹 / 38,344 (obs, action) 对**
- Difficulty 0.0–0.3, 总体成功率 39%

### BC 预训练

- 50 epoch MSE 监督 (lr=1e-4)
- val_loss: 0.044 → **0.006**

### BC→PPO 微调结果

```
500K 步, difficulty=0.0, BC warm-start

Success 进化:
  10K:   2.1%  ← 项目史上首次 success!
  40K:  10.2%
  80K:  25.0%
 150K:  38.4%  (diff=0.06)
 260K:  49.4%  (diff=0.12)
 340K:  55.6%  (diff=0.16)
 490K:  65.4%  (diff=0.26) ← 峰值
 500K:  51.9%  (diff=0.28) ← 终值

最终指标:
  success:         51.9%
  peak success:    65.4%
  peak difficulty:  0.28
  stall:           48.1%
  timeout:           0%
  explained_var:    0.905
  entropy:         -2.83 (稳定)
```

### 失速根因分析

录制了 diff=0.28 下的成功 vs 失速轨迹对比：

| 诊断 | 成功轨迹 | 失速轨迹 |
|------|---------|---------|
| ATA 末端 | **6.1°** | **49.3°** |
| Alpha max | 0.8° | 4.2° |
| 末端转弯幅度 | 3.8 dps | 12.3 dps |
| 速度变化 | 217→254 ↑ | 251→239 ↓ |
| 距离变化 | 1203→204 ↓ | 437→501 ↑ |

**结论**：**不是拉爆 Alpha，是纯追踪退化**（Tail-Chase Degeneration）。Agent 丢失前置角 → 退化为追尾 → 越转 ATA 越大 → 越转越慢 → 关闭率归零。

---

## 5. Phase 3v1：智能前置追踪（首次尝试）

**Commit**: `47d1d3a`

### 三大优先级改造

| 优先级 | 改造 | 内容 |
|--------|------|------|
| P1 观测 | LOS rate (λ̇) | 25→27 维观测，显式提供 PN 感知 |
| P1 记忆 | 高难度 BC | 300 轨迹, diff=0.2–0.5, 27-dim |
| P2 奖励 | ATA 门控 | terminal_pull × max(0, cos(ATA)³) |
| P2 奖励 | ATA 退化惩罚 | dist<1000m & \|ATA\|>20° → **-5.0·dt** |
| P2 奖励 | Lead boost | LEAD_PREDICT_WEIGHT 25→50, Vc≤0 屏蔽 |
| P3 宽容 | 反失速放宽 | ANTI_STALL_WINDOW 30→50 |

### 训练结果

```
500K 步, difficulty=0.15, BC warm-start, 27-dim obs

 10K:   4.6%    ← 有 success
 20K:  11.3%    ← 短暂峰值
 50K:   1.2%    ← 暴跌
500K:   4.1%    ← 终值，difficulty 卡死在 0.00

timeout:  96%
stall:     0%
```

### 失败根因

**ATA 退化惩罚 -5.0·dt 太重**。Agent 从 BC 继承的策略在 diff=0.15 下初始 ATA 不够完美，一旦进入 1000m 内就持续被扣分。Agent 学到了 "别靠近目标"——在远距离巡航 60 秒拿 timeout 的 -500 分，比靠近后被 ATA 惩罚扣几千分更划算。

---

## 6. Phase 3v2：重校准恢复

**Commit**: `aa74f8d`

### 四项调整

| 调整 | 旧值 | 新值 |
|------|------|------|
| ATA 退化惩罚 | **-5.0·dt** | **-1.0·dt** |
| ATA 惩罚课程 | 无 | 0–100K: w=0, 100K–300K: w=0→1 |
| 难度起始 | 0.15 | **0.0** |
| 反失速窗口 | 50 (25s) | **35 (17.5s)** |

### 训练结果

```
500K 步, difficulty=0.0, BC warm-start, 27-dim obs

Success 进化:
  10K:  26.7%  (diff=0.00)
  50K:  23.4%  (diff=0.02)
 150K:  ~30%   (diff=0.08)
 200K:  ~35%   (diff=0.14)
 300K:  22.9%  (diff=0.18)  ← penalty 全开后短暂下降
 400K:  31.5%  (diff=0.24)  ← 恢复并超越
 470K:  47.8%  (diff=0.30)  ← 峰值!
 500K:  27.3%  (diff=0.32)  ← 终值

最终指标:
  peak success:    47.8% (@ diff=0.30)
  final success:   27.3% (@ diff=0.32)
  peak difficulty:  0.32
  timeout:          70%
  stall:             0%
  entropy:         -2.82 (稳定)
```

### v1→v2 恢复对比

| 指标 | Phase 3v1 | Phase 3v2 |
|------|-----------|-----------|
| 10K success | 4.6% | **26.7%** |
| 终值 success | 4.1% | **27.3%** |
| 峰值 success | 8.8% | **47.8%** |
| 最终 difficulty | 0.00 | **0.32** |

**结论**：重校准方向完全正确。Agent 在 penalty 课程下学会了真正的前置追踪——300K 步 penalty 全开后 success 短暂下降（22.9%），然后自适应回升并超越（47.8%）。最终 difficulty=0.32 是所有版本中最高的。

---

## 7. 全部训练版本对比

| 版本 | 动作空间 | Success 峰值 | 最终 diff | Stall | Timeout | 核心突破 |
|------|---------|-------------|-----------|-------|---------|---------|
| Phase 1 | Discrete(9) | 0% | 0.00 | 98.7% | 1.3% | — |
| Phase 2.0 | Box(2) | 0% | 0.00 | 89.3% | 10.7% | entropy 稳定 |
| Phase 2.1 | Box(2)+CARW | 0% | 0.00 | 98.7% | 1.3% | Value 网络活了 |
| **Phase 2.2** | Box(2)+BC | **65.4%** | 0.28 | 48.1% | 0% | **首次 success** |
| Phase 3v1 | Box(2)+LOS+ATA | 8.8% | 0.00 | 0% | 96% | 架构创新 |
| **Phase 3v2** | Box(2)+LOS+ATA+cal | **47.8%** | **0.32** | 0% | 70% | **最高难度** |

### 关键趋势

1. **从 0 到 1**：Phase 2.2 的 BC 预热是整天的转折点——打破了 "从未见过 success" 的探索死锁
2. **质量 vs 数量**：Phase 3v2 的 47.8% @ diff=0.30 比 Phase 2.2 的 65.4% @ diff=0.26 更有价值——Agent 在更难的环境下用更好的几何（ATA-gated）拦截
3. **课程是必须的**：Phase 3v1 的失败证明 reward shaping 不能 "一步到位"，需要渐进课程让 Agent 适应

---

## 8. 关键工程决策

| 决策 | 理由 | 结果 |
|------|------|------|
| **新建而非修改** ContinuousPursuitEnv | 保留 BFM 离散基线用于论文消融 | ✅ 两个版本可独立对比 |
| **继承 BFMPursuitEnv** | 复用观测/奖励/终止/Tacview | ✅ 只改动作空间，变量单一 |
| **纯连续 Box(2)** 而非 5+1 参数化 | SB3 原生支持，避免 argmax 梯度断裂 | ✅ 训练稳定 |
| **FlightController 全权接管** | 消除 BFM 能量-高度耦合 | ✅ 2D 锁高度零 crash |
| **BC 在动作空间确定后执行** | 标签必须匹配动作格式 | ✅ Phase 2.2 一举破局 |
| **ATA 惩罚渐进课程** | 不能让 Agent 恐慌性逃避 | ✅ Phase 3v2 恢复成功 |

---

## 9. 遗留问题与下一步

### 当前最好模型

- **成功率最高**：Phase 2.2 (`phase2_continuous_0627_1551_s42_bc`) — 65.4% @ diff=0.26
- **难度最高**：Phase 3v2 (`phase2_continuous_0627_1852_s42_bc`) — 47.8% @ diff=0.30

### 遗留问题

1. **Timeout 率仍高 (70%)**：Phase 3 的反失速放宽让无望 episode 跑满 60 秒。需要更智能的 "绝望检测"——当 ATA 持续 >30° 且 Vc<0 时快速截断。
2. **Phase 3v2 还有上升空间**：ATA penalty 课程在 300K 才全开，继续训练到 1M 步可能进一步提升。
3. **BC 数据可扩展**：当前只有 300 条高难度轨迹。增加 diff=0.3–0.5 的 PN 专家数据可让 Actor 更好应对大偏差初始条件。
4. **Phase 2.2 的高 success 率部分来自 "非门控" 奖励**——Agent 可能学会了一些不依赖良好 ATA 的 "蛮力拦截"。需要在 Phase 2.2 模型上重新评估 ATA 质量。

### 下一步优先级

1. 继续训练 Phase 3v2 到 1M 步，观察 penalty 全开后的长期趋势
2. 扩大高难度 BC 数据集 (diff 0.3–0.5)
3. 实施 "绝望检测" 快速截断以减少无效 timeout
4. 在 Phase 2.2 模型上做 ATA 质量回溯分析

---

## 提交清单

```
aa74f8d fix: Phase 3 recalibrated — gentle ATA penalty, curriculum ramp, tighter anti-stall
47d1d3a feat: Phase 3 — intelligent lead pursuit (LOS rate, ATA gating, anti-stall relax)
e3afcfe Store test results
6a2dc92 docs: Phase 2 breakthrough — BC→PPO 0%→65.4% success, difficulty 0→0.28
2416f08 feat: BC pretraining pipeline — PN expert data + PPO actor warm-start
7f139c3 fix: ANTI_STALL_MIN_DIST=200 + episode start_dist for CARW success gate
1bff327 feat: CARW dynamic decision rate + terminal-pull reward gradient
c2acf70 feat: Phase 2 continuous Box(2) pursuit — 2D altitude-lock fixes + FlightController takeover
```
