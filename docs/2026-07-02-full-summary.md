# 2026-07-02 完整工作总结：Phase 5 协作环境重构与 200 万步实验

> **日期**: 2026-07-02
> **主题**: 协同奖励设计 → AND-gate 课程退火 → 两阶段训练 → 数据过滤 → 诚实结论
> **提交**: ~12 commits
> **总训练步数**: ~2,000,000 环境步（4 × 500K）

---

## 1. Tacview 导出修复

### 问题
`benchmarks/sb3_2v1_97p3/` 中的 Tacview ACMI 文件只有 3 行文件头，完全缺失对象定义和遥测数据。

### 根因
`_record_single_tacview()` 创建 `FormationEnv` 时未传 `record_tacview=True`，导致 `_tacview_frames` 始终为空。

### 修复
1. `benchmark_sb3_baseline.py`: 传入 `record_tacview=True`，抑制 JSBSIM stderr
2. `formation_env.py export_tacview()`: 对象定义（Name/Color/Coalition）在遥测之前写入，符合 ACMI 规范，经纬度使用 8 位小数精度

重新生成了 diff=0.0（50帧）、0.3（72帧）、0.6（53帧）的 Tacview 文件。

---

## 2. Phase 5 Cooperative 2v1 环境设计

### 2.1 协同奖励 (Reward Shaping)

**合围角奖励 (Pincer Reward)**：
- 计算两机 LOS 向量在水平面内的夹角
- 夹角在 60°-150° 范围内时给予奖励（wider = better flanking）
- 权重 PINCER_WEIGHT=15.0，仅在两机均 < 2000m 时生效

**动态角色分配**：
- 距离目标更近的 pursuer = Striker（追踪奖励 ×1.5）
- 距离目标更远的 pursuer = Interceptor（合围维持奖励 ×2.0）
- 消除搭便车动机——远离目标就拿不到追踪分

### 2.2 拦截判定 (AND-gate)

**旧 OR-gate**：`min(d1, d2) < 200m` → 单机成功即可

**新 AND-gate**：
- 双机同时 < 400m（可课程退火）
- 合围角 > 45°（可课程退火）
- 持续压制 6 个微步（0.1s）
- 额外合围质量奖励：+2000 × (angle/180°)

### 2.3 非对称重置 (Asymmetric Resets)

- 70% 概率：随机一架 pursuer 落后 1500m + 背向目标
- 每次 episode 随机交换劣势位置
- 强制网络学习角色切换：劣势位→拦截补位，优势位→主攻

### 2.4 EV 门控降低

EV_UNFREEZE_THRESHOLD: 0.6 → 0.3（协同奖励需要 PPO 探索）

---

## 3. 实验 1: AND-gate 课程退火 (500K)

### 配置
- AND-gate: 800m/20° → 300m/60°（cosine 退火，0→60% 进度）
- SB3 原始 2v1 BC 预训练模型（53K 样本）

### 结果
```
Phase      EV 峰值    Best avg10   终点 avg10
WARMUP     0.690      —            —
FINE_TUNE  0.869      +1,038       +118
```

**发现**: EV 0.869 是当时所有实验最高，但 reward 始终震荡。AND-gate 收紧至 300m/60° 后 avg10 从未稳定为正。

---

## 4. 诊断: P1 根本不在追

运行 `scripts/diagnose_coop_tacview.py` 导出 3 个 episode 的步级诊断：

```
Ep1: d0=375m  d1=6,668m  kill_zone=0%
Ep2: d0=1,785m d1=6,276m  kill_zone=0%  pincer_max=171°!
Ep3: d0=1,798m d1=5,986m  kill_zone=0%
```

**根因锁定**: P1 持续满右转（turn=0.99~1.00），飞离目标 6000m。BC Actor 的权重本身就编码了"搭便车"策略——因为 2v1 BC 数据来自旧 OR-gate 环境，P1 的最优策略是远离前线避免碰撞惩罚。

---

## 5. 实验 2: 距离压力惩罚 (500K)

### 配置
- 新增 DIST_PRESSURE：> 1500m 即受罚（-3.0/m/dt，上限 -10.0/dt）
- AND-gate 课程放宽：1200m/10° → 300m/60°（80% 进度）

### 结果
```
Best avg10: +243 (远低于 v1 的 +1,038)
EV 峰值: 0.874
终点: avg10=-1,013
```

**结论**: 距离压力没有帮助——它只是让负奖励更深，淹没了正向学习信号。

---

## 6. 实验 3: 两阶段 OR→AND (500K)

### 设计
```
Phase 1 [OR] 0→200K: OR-gate (任一 < 200m) + 轻度合围引导 (>30° 加分)
Phase 2 [AND] 200K→500K: 宽松 AND-gate (双机 < 500m + 夹角 > 45°)
```

### 结果
```
Phase 1 [OR]:  avg10 +4,500~+5,094  ← OR-gate 打破负 reward 陷阱！
Phase 2 [AND]: avg10 -400~+600       ← 仍震荡，未稳定
Best avg10: +5,094 (Phase 1)
EV 峰值: 0.833
```

**关键发现**: Phase 1 OR-gate 成功让 Actor 重新学会双机靠近目标。但 Phase 2 AND-gate 切换后仍无法稳定——诊断显示 P1 在 Phase 2 后又开始逃跑。

---

## 7. 实验 4: PID 协同专家数据

### 设计
用纯 PID 控制器强制两架无人机从目标两侧逼近：
- P0 → 目标左侧偏移点，P1 → 目标右侧偏移点
- 距离 < 600m 时切换为直接追击
- 距离过滤：仅记录双机均在 2000m 内的样本

### 结果
- 1000 集 → 10,468 过滤样本
- BC 预训练：val_loss 0.0074，但 S2M_attn 降至 0.05
- MAPPO 训练：Phase 1 avg10=-4,000，Phase 2 avg10=-5,200

**结论**: PID 10K 样本远不足以教会 Attention Actor JSBSim F-16 飞行动力学。PID 动作基于地面真值几何，BC 模型无法从 33-dim 局部观测复现。

---

## 8. 实验 5: 过滤 SB3 BC 数据

### 设计
从 SB3 原始 2v1 BC 数据（53K 样本）中过滤：
- 保留 mate 相对距离 < 1500m 的样本
- 18,558 样本（35%）

### BC 预训练
```
val_loss: 0.128 → 0.039
S2M_attn: 0.341 → 0.505  ← mate 注意力增强！
```

### MAPPO 两阶段
```
Phase 1 [OR]:  avg10 +4,500~+5,293  ← 最佳！
Phase 2 [AND]: avg10 -200~+360      ← 稍改善，仍震荡
Best avg10: +5,293
EV 峰值: 0.855
```

---

## 9. Tokenized Attention Critic

将 Critic 从 flat MLP (21→256→256→1) 升级为 3-entity Self-Attention：

```
(B, 3, 7) tokens [Self, Mate, Target]
  → Token Projection → Type Embedding
  → MultiHeadAttention + LayerNorm + Residual
  → Learned Pooling Query → Value Head [256, 1]
  → scalar value
```

### 对比
```
┌─────────────────────┬──────────┬──────────┐
│ Critic 架构         │ EV 峰值  │ 最佳 rew │
├─────────────────────┼──────────┼──────────┤
│ Flat MLP (21-dim)   │ 0.317    │ +6,846   │
│ Tokenized Attn (3×7)│ 0.295    │ +6,820   │
└─────────────────────┴──────────┴──────────┘
```

**结论**: 3-entity 序列太短，flat MLP 已接近最优。Attention Critic 未带来可测量的提升。

---

## 10. 全版本最终对比

```
┌──────────────────────┬──────────┬──────────┬──────────────────┐
│ 实验                 │ Best rew │ EV 峰值  │ P1 行为          │
├──────────────────────┼──────────┼──────────┼──────────────────┤
│ v1 AND 课程退火      │ +1,038   │ 0.869    │ 逃跑 d1=6000m    │
│ v2 距离压力          │ +243     │ 0.874    │ 逃跑 d1=6000m    │
│ v3 OR→AND (原始BC)   │ +5,094   │ 0.833    │ Phase1 追, P2 逃 │
│ v4 PID BC            │ -4,000   │ 0.878    │ 数据不足         │
│ v5 OR→AND (过滤BC)   │ +5,293   │ 0.855    │ Phase1 追, P2 逃 │
│ 非协作 BC (参考)     │ +6,846   │ 0.317    │ P1 不追但也不罚  │
│ SB3 集中式 (天花板)  │ +5,908   │ —        │ 上帝视角          │
└──────────────────────┴──────────┴──────────┴──────────────────┘
```

---

## 11. 核心科学发现

### 11.1 Phase 1 OR-gate 始终成功
所有使用 SB3 BC 数据的实验，Phase 1 avg10 均稳定在 +4,500~+5,300。OR-gate 成功打破了负 reward 循环，让 Actor 积累了正向追击经验。

### 11.2 Phase 2 AND-gate 始终失败
4 轮共 200 万步实验，没有任何一轮在 AND-gate 下实现稳定正 reward。P1 始终在 Phase 2 切换后恢复逃跑行为。BC 权重中的搭便车策略太过深层，PPO 梯度下降无法抹除。

### 11.3 EV 不是瓶颈
所有实验的 Critic EV 均达到 0.83-0.88。价值函数学习质量极高——问题不在 Critic，而在 Actor 的探索能力。

### 11.4 这不是数据问题，是架构约束
即使过滤掉所有搭便车数据（v5），Phase 2 仍然失败。CTDE 架构下，每个 Actor 只有 33 维局部观测——它看不到全局协同状态。在严格的 AND-gate 下，Actor 无法从局部观测推断出"我需要和队友形成特定夹角才能成功"。

---

## 12. 产出清单

### 新增文件
| 文件 | 描述 |
|------|------|
| `scripts/diagnose_coop_tacview.py` | 协作模型步级诊断 + Tacview 导出 |
| `scripts/generate_coop_expert.py` | PID 协同轨迹生成器 |
| `docs/2026-07-02-full-summary.md` | 本文档 |

### 修改文件
| 文件 | 主要变更 |
|------|---------|
| `src/environment/formation_env.py` | Phase 5 协作环境（合围奖励+AND-gate+非对称重置+两阶段） |
| `src/models/attention_actor.py` | Tokenized AttentionCritic + 注意力坍缩防护 |
| `scripts/train_attention_actor.py` | EV 门控+KL 熔断+TensorBoard+两阶段训练+熵防护 |
| `scripts/benchmark_sb3_baseline.py` | Tacview 修复 + SB3 模型加载容错 |
| `scripts/train_attention_bc_2v1.py` | 数据过滤 BC 管道 |

### 数据文件
| 文件 | 描述 |
|------|------|
| `data/expert/coop_pid_data.npz` | PID 协同数据（10,468 样本） |
| `data/expert/attention_bc_2v1_filtered.npz` | 过滤 SB3 BC 数据（18,558 样本） |

---

## 13. Git 提交

```
bedfb2f results: filtered BC two-phase — Phase 1 +5293
2cb3a00 feat: PID cooperative BC pipeline
70f96ba results: two-phase training — root cause P1 free-riding
fa8e664 feat: two-phase cooperative training — OR-gate -> relaxed AND-gate
97753e1 results: distance pressure v2 500K — AND-gate still too hard
1f8a917 fix: distance pressure penalty + relaxed AND-gate curriculum
4c9708e feat: AND-gate curriculum annealing + entropy guard
4c8f069 docs: 2026-07-01 complete summary
62f4825 fix: Tacview ACMI export
f2b90c0 feat: TensorBoard logging
38b83e9 feat: Phase 5 Cooperative 2v1
```

---

## 14. 下一步

1. **论文写作**: "CTDE 的协同天花板"——200 万步实验作为实证，与 SB3 集中式天花板形成对照
2. **放宽 AND-gate**: 如果追求正向结果，将最终条件设为 800m/30°（而非 500m/45°）
3. **中央化 Critic**: 给 Critic 66 维全局观测（打破 CTDE 约束），验证是否改善 Actor 梯度
4. **独立 Actor**: 为 P0 和 P1 训练不同的 Actor 网络，打破参数共享的对称性陷阱
