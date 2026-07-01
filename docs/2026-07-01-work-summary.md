# 2026-07-01 工作总结：封存 SB3 97.3% 基线与自注意力 Actor 架构设计

> **日期**: 2026-07-01
> **主题**: 确立集中式上限 + Self-Attention CTDE Actor 架构创新
> **优先级**: 先确立理论闭环，再突破架构上限

---

## 1. 战略决策

基于 6/30 的成果矩阵（SB3 97.3% vs CTDE 平铺失败），决定不再通过超参微调修补 CTDE 平铺模型，转而执行以下路线：

1. **封存 SB3 97.3% 基线** → 确立"集中式上限"（Global Optimum）
2. **重构 Actor 信息流** → 引入 Self-Attention 跨越信息不对称鸿沟

---

## 2. Priority 1：封存 SB3 97.3% 基线

### 2.1 基准模型归档

- **源模型**: `marl_runs/formation_2v1_0629_1721_s42/formation_2v1_final.zip`
- **归档位置**: `benchmarks/sb3_2v1_97p3/model.zip`
- **基准文档**: `benchmarks/sb3_2v1_97p3/README.md`

### 2.2 基准评估脚本

- **脚本**: `scripts/benchmark_sb3_baseline.py`
- **功能**:
  - Wilson CI 成功率统计（可配置 episode 数和难度级别）
  - Tacview ACMI 导出（最佳/最差 episode）
  - 3D 轨迹 + 编队间距时序图
  - JSON manifest 输出（论文表格可直接引用）

### 2.3 为什么这是"天花板"

SB3 共享策略（Box(4), 66-dim obs）的单一神经网络能同时看到两个 pursuer 的完整状态，学到了"一人追一人歇"的分工模式。CTDE 中每个 Actor 仅 33 维，必须通过仅 6 维的 mate 特征推断协调策略——这是信息不对称的根本瓶颈。

---

## 3. Priority 2：Self-Attention FormationActor

### 3.1 核心创新

放弃简单 MLP，在 FormationActor 中引入轻量级 Multi-Head Self-Attention。

**数据流**:
```
obs[33] → segment → [Self(13), Target(14), Mate(6)]
  → Linear projection → 3 tokens × d_model
  → + Token-Type Embedding (可学习的位置编码)
  → MultiHeadSelfAttention(QKV over 3 tokens)
  → Residual connection
  → Learned Attention Pooling (加权求和)
  → MLP head → action_mean(2) + log_std(2)
```

### 3.2 观测分割

| Token | 维度 | 内容 |
|-------|------|------|
| **Self** | 13 | own_vel(3), attitude(3), ang_vel(3), height(1), alpha(1), airspeed(1), pad(1) |
| **Target** | 14 | target_rel_pos(3), target_vel(3), placeholder(3), tac_geo(3), los_rate(1), bearing_err(1) |
| **Mate** | 6 | mate_rel_pos(3), mate_rel_vel(3) |

### 3.3 关键设计特性

1. **可解释性**: Attention weights 可直接读出 Actor 在每个决策步对 Self/Target/Mate 的注意力分配
2. **Curriculum 兼容**: `mate_scale` 参数可从 0→1 平滑退火，支持 1v1→2v1 渐进训练
3. **Token-Type Embedding**: 可学习的位置编码让注意力层区分 token 的语义角色
4. **Learned Pooling**: 用可学习的 query vector 而非简单 mean pooling 聚合 3 个 token

### 3.4 参数对比

| 模型 | 参数量 |
|------|--------|
| 基线 MLP Actor (33→256→256→2) | ~75K |
| Attention Actor (3 tokens, d=128, 4 heads) | ~170K |
| 增幅 | ~2.3× |

### 3.5 实现文件

- **网络定义**: `src/models/attention_actor.py`
  - `AttentionFormationActor`: 自注意力 Actor
  - `AttentionCritic`: 标准集中式 Critic（不变）
  - 自测通过：forward pass、attention weights、pool weights 均正常

---

## 4. 训练策略

### 4.1 Cold-Start（从零训练）

```
Phase 1 (1v1, mate_scale=0): 基础追猎技能引导
  → 网络学会 Self↔Target 注意力分配
  → mate token 被置零，注意力机制学会忽略它

Phase 2 (2v1, mate_scale 0→1 cosine ramp): 编队协调涌现
  → 200K steps 内 mate_scale 从 0 平滑升至 1.0
  → 网络自发发现 mate 特征的战术价值
  → 无需 SB3 权重平铺，避免信息不对称陷阱
```

### 4.2 训练脚本

- **脚本**: `scripts/train_attention_actor.py`
- **模式**:
  - `--mode 1v1`: 单机追猎（mate_scale=0）
  - `--mode 2v1`: 直接 2v1 冷启动（mate_scale ramp）
  - `--mode curriculum`: 1v1 预训练 → 2v1 微调（推荐）

### 4.3 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| ACTOR_LR | 3e-4 | Actor 学习率（冷启动需较高） |
| CRITIC_LR | 1e-3 | Critic 学习率（从零学全局价值） |
| ENT_COEF | 0.01 | 熵系数（鼓励探索，冷启动必须） |
| MATE_SCALE_RAMP | 0→1 over 200K | Cosine schedule |
| ROLLOUT_STEPS | 4096 | 每轮采样步数 |

---

## 5. 产出清单

### 新增文件

| 文件 | 描述 |
|------|------|
| `benchmarks/sb3_2v1_97p3/README.md` | 基准文档 |
| `benchmarks/sb3_2v1_97p3/model.zip` | 归档模型权重 |
| `scripts/benchmark_sb3_baseline.py` | 基准评估脚本 |
| `src/models/attention_actor.py` | Self-Attention Actor 网络定义 |
| `scripts/train_attention_actor.py` | 冷启动训练脚本 |

### 修改文件

| 文件 | 修改 |
|------|------|
| — | 无现有文件修改，全部新增 |

---

## 6. 下一步行动计划

1. **[立即] 运行基准评估**: `python scripts/benchmark_sb3_baseline.py -n 100 -d 0.0,0.3,0.6`
2. **[核心] 启动 1v1 冷启动训练**: `python scripts/train_attention_actor.py --mode 1v1 --steps 200000`
3. **[核心] 启动 2v1 冷启动训练**: `python scripts/train_attention_actor.py --mode 2v1 --steps 500000`
4. **[分析] 注意力权重可视化**: 训练过程中监控 Self→Mate 注意力权重的涌现
5. **[对比] vs SB3 97.3% 基线**: 衡量 Attention Actor 逼近集中式上限的程度

---

## 7. 理论贡献总结

本次架构创新的核心论点：

> **CTDE 中的信息不对称不是超参问题，是结构问题。**
> 平铺的 MLP Actor 缺乏动态分配注意力的机制——它只能把所有 33 维输入等同对待。
> Self-Attention 通过 Query-Key-Value 机制赋予 Actor 在每个时间步自主决定
> "看多少自己、看多少目标、看多少队友"的能力，从而在去中心化执行约束下
> 逼近集中式策略的协调水平。
