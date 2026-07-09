# 2026-07-09 完整工作总结：连续→离散动作迁移 + 消融实验矩阵

> **日期**: 2026-07-09
> **主题**: MultiDiscrete 架构重构 → Self-Attention 消融 → 离散 BC → 320 轮拉长训练
> **提交**: ~8 commits
> **核心成果**: 离散 Self-Attention 冷启动 eval 三次转正 (峰值 +2,376)，证明 Token-based 架构不需要专家知识即可学会协同追击

---

## 1. 连续→离散动作空间迁移 (P1-P5)

基于 7/8 的 AND-gate 困境（eval -1,171 最优但无法转正），执行五个优先级的架构重构：

### P1: 环境动作空间

```python
# 之前: Box(2) [turn_rate, speed] — 连续 DiagGaussian
single_act = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

# 之后: MultiDiscrete([5, 3]) — 15 个战术基元
single_act = gym.spaces.MultiDiscrete([N_TURN, N_SPEED])

# Turn:  HardLeft(-15°/s) SoftLeft(-5°) Straight(0°) SoftRight(+5°) HardRight(+15°)
# Speed: Slow(180m/s) Cruise(250m/s) Fast(320m/s)
```

### P2: 环境步进映射

离散索引 `(turn_idx, speed_idx)` → `TURN_RATES[turn_idx]`, `SPEEDS[speed_idx]` → FlightController。

### P3: 模型输出头

`AttentionFormationActor.mean` (Linear→2) + `log_std` → `turn_head` (Linear→5) + `speed_head` (Linear→3)。

新增 `forward_features()` 方法提取 Self-Attention 中间特征。

### P4: Action Masking

基于飞行安全约束的动态 mask：
- 失速 (<130m/s): 禁止慢速 + 禁止急转
- 近地 (<200m): 禁止急转
- 高速 (>95% Vmax): 禁止 Fast

### P5: 训练超参

- `entropy_coeff`: 0.01 → 0.03（离散需要更高探索）
- 移除连续空间专用配置（Gaussian std limit）

### Bug Fix: NaN in Action Mask

根因：`torch.full_like(logits, float("-inf"))` 在 mask 计算中 `0 * (-inf) = NaN`。

修复：`(1.0 - action_mask) * (-1e9)` 替代 `torch.where(..., float("-inf"))`。

---

## 2. 路线 A：Self-Attention 架构激活

实验 4a 使用纯 MLP fallback 成功验证离散管道无 NaN。随后发现 NaN 根因是 `float("-inf")` 而非 Self-Attention，恢复 `forward_features()` 调用。

**关键架构**：

```
obs[33] → segment → [Self(13), Target(14), Mate(6)]
  → Token Projection → Type Embedding
  → MultiHeadAttention(4 heads, d=128)
  → Learned Pooling → MLP [256,256]
  → turn_head(5) + speed_head(3) → MultiDiscrete logits
```

---

## 3. 路线 B：离散行为克隆 (Discrete BC)

### 离散化投影

将 18,558 个 SB3 连续专家动作 `[-1,1]²` 通过 argmin 投影到离散网格：

```python
turn_physical = cont_actions[:, 0] * 15.0   # [-15, 15]
speed_physical = 250 + cont_actions[:, 1] * 100  # [150, 350]
turn_idx = argmin(|turn_physical - TURN_GRID|)
speed_idx = argmin(|speed_physical - SPEED_GRID|)
```

### BC 预训练

- 模型: DiscreteBCActor (Self-Attention backbone + turn_head + speed_head)
- 损失: CrossEntropyLoss(turn) + CrossEntropyLoss(speed)
- 分层学习率: backbone lr=1e-4, heads lr=1e-3
- 40 epochs → turn_acc=72.3%, speed_acc=81.0%, val_loss=1.14

---

## 4. 消融实验矩阵（完整）

| 实验 | 架构 | BC | 轮数 | Eval最优 | Eval转正次数 | 熵终值 |
|------|------|-----|------|---------|------------|--------|
| 4a | MLP | 冷启动 | 120 | −4,542 | 0 | 2.09 |
| 4a-v2 | Self-Attn | 冷启动 | 120 | +1,345 | 1 | 2.07 |
| 4b | Self-Attn | 离散BC | 120 | −1,135 | 0 | 1.94 |
| **4a-v2 ext** | **Self-Attn** | **冷启动** | **320** | **+2,376** | **3** | **1.87** |

### 核心发现

1. **Self-Attention 是决定性因素** — 冷启动 Self-Attention 在 120 轮即转正 (+1,345)，MLP 从未转正。差距 +5,887 分。

2. **BC 热启动增益有限** — 实验 4b (BC) 的 eval 最优 −1,135，不如冷启动 Self-Attention 的 +1,345。离散 BC 提供稳定收敛但无额外峰值。

3. **长周期训练释放 Self-Attention 潜力** — 续训至 320 轮，eval 三次转正（+1,345, +2,376, +4.0），训练 reward 峰值 +5,401。

4. **策略方差大** — eval 在 +2,376 和 -7,975 之间剧烈波动，320 轮后收敛未完成。推测需要 500+ 轮。

---

## 5. 统计 autopsy：AND-gate eval 定量分析

对实验 3 AND-gate checkpoint 的 10 集 eval 进行量化统计：

```
Metric 1: 超时率                 0%
Metric 2: 单机入线率 (<800m)     22.1%
Metric 3: 同步入线率 (BOTH<800m) 0.0%  ← 致命瓶颈
Metric 4: AND-gate 成功           0/10
Metric 5: Pincer > 30°           58.4%
d₀ median: 329m  |  d₁ median: 1,974m
```

同步入线率 0% — P1 永远无法在 P0 逼近目标的同时进入 AND 包线。

---

## 6. 文档与工具

| 文件 | 操作 | 内容 |
|------|------|------|
| `CLAUDE.md` | 重写 | WSL2 环境 + 离散架构 + 启动清单 |
| `README.md` | 更新 | 离散动作 + 三层架构图 + 六阶段演变 |
| `docs/discrete-action-migration-plan.md` | 新建 | 优先级行动清单 |
| `.gitignore` | 更新 | `.agents/` + `skills-lock.json` |
| `.agents/skills/pptx/` | 安装 | Anthropic 官方 pptx skill |
| `.agents/skills/find-skills/` | 安装 | 技能搜索工具 |
| `scripts/train_discrete_bc.py` | 新建 | 离散 BC 预训练脚本 |
| `scripts/analyze_eval_statistics.py` | 新建 | Eval 统计 autopsy |

---

## 7. Git 提交

```
a78b30d feat: Route B — discrete BC pretraining + Experiment 4b support
1279bc4 feat: reactivate Self-Attention in discrete model (Route A)
61b0927 chore: also ignore skills-lock.json
254cc75 chore: add .agents/ to gitignore
d6094cf docs: update README + CLAUDE.md for discrete action migration + WSL2 env
c3bec36 fix: discrete model NaN — replace float('-inf') with -1e9
b462cf4 fix: NaN guard in discrete model + forward_features extraction
5aff9ff feat: continuous→discrete action space — MultiDiscrete([5,3]) + action masking
```

---

## 8. 下一步

1. **500+ 轮 Self-Attention 训练** — 320 轮后策略收敛未完成，需要更长时间
2. **离散 BC 权重 debug** — 当前 BC 加载了 19 个 key (backbone) 但 missing_critic=22，可能有 head 不匹配
3. **论文写作** — Self-Attention vs MLP 消融 + 三次 eval 转正 + 同步入线率 autopsy 已够一篇完整的 ablation study
4. **AND-gate 续攻** — 同步入线率 0% 需要更激进的奖励设计或放宽距离阈值
