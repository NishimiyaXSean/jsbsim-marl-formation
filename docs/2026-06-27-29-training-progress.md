# 2026-06-27~29 训练进度总结：从 0% 到 50%@diff=0.58

> **日期**: 2026-06-27 至 2026-06-29  
> **主题**: 连续动作空间 + BC 破局 + 智能前置追踪  
> **提交**: 11 commits（Phase 2 ~ Phase 3.4）  
> **总训练步数**: ~4,800,000（跨 7 个训练版本）

---

## 1. 全版本演进总览

| 版本 | 动作空间 | 最终 success | 峰值 success | 最终 diff | 核心特征 |
|------|---------|-------------|-------------|-----------|---------|
| Phase 1 (6/26) | Discrete(9) BFM | **0%** | 0% | 0.00 | 熵坍缩, 98.7% stall |
| Phase 2.0 | Box(2) FC | 0% | 0% | 0.00 | 连续动作, 2D 锁高度 |
| Phase 2.1 | Box(2)+CARW | 0% | 0% | 0.00 | 动态决策率, Value 网络活了 |
| **Phase 2.2** | Box(2)+BC | **51.9%** | **65.4%** | 0.28 | **首次 success**, BC 破局 |
| Phase 3v1 | Box(2)+LOS+ATA | 4.1% | 8.8% | 0.00 | ATA 惩罚过重(-5.0·dt) |
| Phase 3v2 | Box(2)+recal | 27.3% | 47.8% | 0.32 | 惩罚课程, diff=0 重启 |
| **Phase 3v3** | Box(2)+prog | **50.0%** | **61.7%** | **0.58** | **最高难度**, 渐进时间压力 |
| Phase 3.4 | Box(2)+hot | 4.9% | ~13% | 0.00 | 自我蒸馏, diff=0.4 热启动 (失败) |
| **Phase 3.5** 🏆 | Box(2)+self | **81.3%** | **81.3%** | **0.58** | **大规模自我蒸馏 + diff=0 重启** |

### 最佳模型

| 维度 | 模型 | 路径 |
|------|------|------|
| **最高成功率+最高难度** | Phase 3.5 | `phase2_continuous_0629_1253_s42_bc` (81.3% @ diff=0.58) |
| 最高纯成功率 | Phase 2.2 | `phase2_continuous_0627_1551_s42_bc` (65.4% @ diff=0.26) |
| 历史参考 | Phase 3v3 | `phase2_continuous_0627_2326_s42_bc` (50.0% @ diff=0.58) |

---

## 2. Phase 3v3 最佳模型详细指标

### TensorBoard 面板（1M 步）

| 指标 | 起始 | 中期 (500K) | 终值 (1M) | 趋势 |
|------|------|------------|----------|------|
| `termination_rate/success` | 25.0% | 22.9% | **50.0%** | 📈 |
| `phase2/difficulty` | 0.00 | 0.26 | **0.58** | 📈 |
| `train/explained_variance` | 0.00 | 0.91 | **0.97** | 📈 |
| `train/entropy_loss` | -2.84 | -2.75 | -2.77 | ➡️ 零坍缩 |
| `rollout/ep_len_mean` | 118 | 200 | **246** | 📈 |
| `train/value_loss` | 1.6M | 305K | **164K** | 📉 10× 改善 |
| `termination_rate/stall` | 0% | 0% | **0%** | ✅ |
| `termination_rate/ground_crash` | 0% | 0% | **0%** | ✅ |
| `train/std` | 1.00 | 0.96 | 0.97 | ➡️ 健康探索 |

### 难度递进

```
 10K: diff=0.00 (success 25%)
 50K: diff=0.02 (success 23%)  ← ATA penalty=0
100K: diff=0.08 (penalty ramp start)
200K: diff=0.14 (penalty ramp mid)
300K: diff=0.18 (penalty=1.0 full)
400K: diff=0.24 (success 31%)
500K: diff=0.28 (success 35%)
600K: diff=0.34
700K: diff=0.40
800K: diff=0.46 (success 52%)
900K: diff=0.52 (success 56%)
1.0M: diff=0.58 (success 50%)  ← 终极
```

### 难度 0.58 的含义

| 参数 | diff=0 | diff=0.58 |
|------|--------|-----------|
| 初始距离 | 900–1300m | **1538–2286m** |
| 方位角偏移 | ±0° | **±26.1°** |
| 航向差 | ±0° | **±17.4°** |
| 目标机动 | 直线平飞 | 直线平飞 (2D 模式) |

### 末端能量诊断 (@ diff=0.56)

| 指标 | 数值 |
|------|------|
| 攻击机速度 @ kill | **279 ± 21 m/s** |
| 目标机速度 @ kill | **216 ± 6 m/s** |
| 闭合率 @ kill | **+63 ± 22 m/s** (全为正值) |
| 最小闭合率 | +22 m/s (从不 overshoot) |
| 失速 (速度 < 150) | **0%** |
| Alpha max | 正常范围 |

### 控制平滑度

| 指标 | 数值 | 评价 |
|------|------|------|
| 转弯率均值 | 8.1 dps | ~54% 最大 |
| 转弯率 std | 7.5 dps | 有大幅机动也有微调 |
| \|turn jerk\| mean | 6.9 dps/s | 平滑，无高频振荡 |
| 速度命令 std | 24 m/s | 策略稳定 |
| 末端转弯率 | 3-5 dps | 自动柔和 |

---

## 3. 关键架构决策

| 决策 | 时间点 | 结果 |
|------|--------|------|
| Box(2) 连续而非 Discrete(9) | Phase 2.0 | ✅ 从 0%→65.4% |
| FlightController 全权接管 | Phase 2.0 | ✅ 2D 锁高度零 crash |
| 新建 ContinuousPursuitEnv 继承 BFMPursuitEnv | Phase 2.0 | ✅ 保留离散基线用于论文消融 |
| ANTI_STALL_MIN_DIST = 200 | Phase 2.2 | ✅ 消除 200-300m 死区 |
| Episode start_dist 修复 | Phase 2.2 | ✅ CARW 每步门控修复 |
| BC 预热 (PN 专家 500 条轨迹) | Phase 2.2 | ✅ 首次 success (2.1%@10K) |
| CARW (2→10 Hz 动态决策率) | Phase 2.1 | ✅ Value 网络可训练 |
| Observation 25→27 (λ̇ + bearing err) | Phase 3 | ✅ PN 感知能力 |
| ATA-gated terminal pull ×cos³(ATA) | Phase 3 | ✅ 消除 "白嫖距离" |
| Lead reward 25→50 + Vc 保护 | Phase 3 | ✅ 鼓励前置追踪 |
| ATA 退化惩罚 + 课程 (0→1) | Phase 3v2 | ✅ 温和引导 |
| 渐进时间压力 (loiter 探测) | Phase 3v3 | ✅ 打破舒适区 |
| 自我蒸馏 (DAgger) | Phase 3.4 | 实验性 |

---

### Phase 3.5 关键结果（新增）

| 指标 | Phase 3v3 | Phase 3.5 | 提升 |
|------|-----------|-----------|------|
| 最终 success | 50.0% | **81.3%** | +31pp |
| 最终 timeout | 50.0% | **18.8%** | -31pp |
| 最终 difficulty | 0.58 | 0.58 | — |
| 最终 ep_rew | -2,790 | **+3,434** | 翻正 |
| explained_variance | 0.97 | 0.96 | 均优秀 |
| entropy | -2.77 | -2.79 | 均稳定 |
| stall/crash | 0% | 0% | — |

**Success 率演进**：
```
100K: 16.7% (diff=0.00)
300K: 32.6% (diff=0.12)  ← ATA penalty 全开
600K: 53.7% (diff=0.32)  ← 突破 50%
700K: 72.9% (diff=0.38)  ← 爆发
900K: 76.2% (diff=0.52)  ← 峰值
1.0M: 81.3% (diff=0.58)  ← 终值 🏆
```

**成功因素**：
1. 1500 条 BC 轨迹 (800 PN + 700 self-play) 覆盖 diff 0.20–0.70
2. diff=0.0 重启（Phase 3.4 的 diff=0.4 热启动失败证明了这步的关键性）
3. 课程系统从简单到困难自然推进

---

## 4. 残余问题

### 4.1 Timeout 率 18.8% (Phase 3.5 已大幅改善)

在 diff=0.58 下，18.8% episode 以 timeout 结束（Phase 3v3 为 50%）。剩余 timeout 主要来自极端的初始几何（bearing > 25° 且 pursuer 需要大幅转弯）。

### 4.2 极端几何 BC 数据生成困难

PN 专家在 diff > 0.6（bearing > 27°）时成功率骤降至 ~3%，受限于：
- FlightController 实际转弯率仅 ~4 dps（命令 15 dps）
- 大偏角 + 大距离下 60s 时限不足
- 混合专家 (Bang-Bang + PN) 未能突破 FC 物理限制

### 4.3 Phase 3.4 热启动未达预期

从 diff=0.4 直接热启动导致 success 率仅 4.9%——增量 BC 更新太温和 (lr=3e-5) 不足以适应高难度。应从 diff=0.0 逐步推进。

---

## 5. 输出文件清单

### 模型文件

| 文件 | 描述 |
|------|------|
| `marl_runs/phase2_continuous_0627_1551_s42_bc/phase2_final.zip` | 最高成功率 (65.4%) |
| `marl_runs/phase2_continuous_0627_2326_s42_bc/phase2_final.zip` | 最高难度 (50%@0.58) |
| `marl_runs/phase2_continuous_0627_2326_s42_bc/checkpoint_*.zip` | Phase 3v3 检查点 |

### BC 数据

| 文件 | 描述 |
|------|------|
| `data/expert_phase3_combined/pn_expert_800ep_89289steps.npz` | PN 专家 (diff 0.2–0.5) |
| `data/expert_phase3_high/pn_expert_500ep_56498steps.npz` | 高难度 PN (diff 0.3–0.5) |
| `data/expert_selfplay/selfplay_200ep_18534steps.npz` | 自我蒸馏 (diff 0.5–0.65) |
| `data/expert_combined_all/pn_selfplay_1000ep_107823steps.npz` | 全量混合数据集 |

### 分析图表

| 文件 | 描述 |
|------|------|
| `results/phase3v3_eval/success_profiles.png` | 成功拦截轨迹曲线 |
| `results/phase3v3_eval/timeout_profile.png` | Timeout 轨迹分析 |
| `results/phase2_eval/best_success_diff028.acmi` | Tacview 成功案例 |
| `results/phase2_eval/worst_stall_diff028.acmi` | Tacview 失速案例 |

### 文档

| 文件 | 描述 |
|------|------|
| `docs/2026-06-27-full-training-summary.md` | 6/27 全日总结 |
| `docs/2026-06-27-phase2-breakthrough.md` | Phase 2 突破报告 |

---

## 6. 下一步优先级

1. **极端几何 BC 数据**：需要更高效的生成策略（自我蒸馏已部分缓解）
2. **多机编队迁移**：将经过验证的 ContinuousPursuitEnv 架构迁移到 MAPPO 框架
3. **论文消融实验**：Phase 1 离散 vs Phase 2/3 连续的受控对比
4. **目标机动引入**：放开 2D lock_altitude，让目标做规避机动

---

## 7. 提交清单

```
2507e94 feat: Phase 3.4 — self-distillation + hot-start BC pipeline
8166e44 feat: Phase 3.3 final — progressive time pressure + extreme-geometry BC attempt
56fe1f9 feat: Phase 3.3 — high-difficulty BC + progressive time pressure
77ad3aa docs: 2026-06-27 full training summary — 0% deadlock to 65.4% breakthrough
aa74f8d fix: Phase 3 recalibrated — gentle ATA penalty, curriculum ramp, tighter anti-stall
47d1d3a feat: Phase 3 — intelligent lead pursuit (LOS rate, ATA gating, anti-stall relax)
e3afcfe Store test results
6a2dc92 docs: Phase 2 breakthrough — BC→PPO 0%→65.4% success, difficulty 0→0.28
2416f08 feat: BC pretraining pipeline — PN expert data + PPO actor warm-start
7f139c3 fix: ANTI_STALL_MIN_DIST=200 + episode start_dist for CARW success gate
1bff327 feat: CARW dynamic decision rate + terminal-pull reward gradient
c2acf70 feat: Phase 2 continuous Box(2) pursuit — 2D altitude-lock fixes + FlightController takeover
```
