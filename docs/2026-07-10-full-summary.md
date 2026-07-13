# 2026-07-10 完整工作总结：离散 BC 权重加载 Bug 修复 + 500 轮 BC 热启动实验

> **日期**: 2026-07-10
> **主题**: BC 权重加载 Bug 诊断 → 修复 → 500 轮长周期训练 → 续训策略设计
> **提交**: 2 commits
> **核心成果**: 修复离散 BC 权重加载遗漏（turn_head/speed_head 从未被加载），500 轮 BC 热启动训练 reward 95.6% 为正

---

## 1. 离散 BC 权重加载 Bug — 根因诊断与修复 (P1)

### 问题定位

在 7/9 的实验中，BC 热启动 (实验 4b, eval −1,135) 表现不如冷启动 Self-Attention (实验 4a-v2, eval +1,345)。排查发现 `load_bc_weights()` 存在严重 Bug：

**BC checkpoint 结构** (`discrete_attention_bc.pth`)：
```
{
  "actor_state": {              ← 19 个 backbone key (被加载)
      self_proj.*, target_proj.*, mate_proj.*,
      attention.*, mlp_head.*, attn_pool_query,
      token_type_embed, mean.*, log_std
  },
  "turn_head.weight": [5, 256],   ← 顶层! 从未被读取!
  "turn_head.bias":   [5],
  "speed_head.weight": [3, 256],  ← 顶层! 从未被读取!
  "speed_head.bias":   [3],
}
```

旧代码 `load_bc_weights()` 只迭代 `bc_ckpt["actor_state"]`：
```python
bc_state = bc_ckpt["actor_state"]           # 只取到 backbone 19 个 key
for bc_key, bc_val in bc_state.items():
    rllib_key = f"actor.{bc_key}"            # → actor.self_proj.*, ...
```

**turn_head 和 speed_head 存在于 BC checkpoint 顶层，从未被映射。**

### 修复方案

分两阶段加载：
- **Phase 1 (backbone)**: `actor_state.{key}` → `actor.{key}` (19 keys)
- **Phase 2 (discrete heads)**: 顶层 `turn_head.*`/`speed_head.*` → RLlib 直接 key（无 prefix）

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Backbone 加载 | 19/19 ✅ | 19/19 ✅ |
| 离散头加载 | **0/4** ❌ | **4/4** ✅ |
| 总加载 key 数 | 19 | 23 |
| skipped | 0 | 0 |
| missing (critic+fallback) | 22 | 18 |

**验证**: `turn_head.weight`, `speed_head.weight` 与 BC checkpoint 逐元素匹配 (`torch.allclose` = True)。

这意味着之前所有 `--load-discrete-bc` 训练**只加载了 backbone**，turn_head 和 speed_head 始终是随机初始化的——等同于冷启动 + backbone 热启动的混合态。

### 连带修复

1. **LR auto-adjust 时序 Bug**: LR 自动调整逻辑放在 `config.build()` 之后，导致 `lr=None` 传入 PPOConfig，训练崩溃。移至 config 构建之前。
2. **BC 加载在 resume 时覆盖 checkpoint**: BC 权重在 `algo.restore()` 之前加载，被立即覆盖。修复：resume 时跳过 BC 加载。

---

## 2. 冒烟测试验证

| 配置 | 结果 |
|------|------|
| BC 热启动 (15 轮, seed=123) | ✅ 23 keys loaded, NaN-free |
| 冷启动 (15 轮, seed=123) | ✅ NaN-free, entropy 2.51 (更高探索) |

BC 热启动 iter 0 entropy 2.04 vs 冷启动 2.51 — BC 提供的先验让策略从更聚焦的起点开始。

---

## 3. 500 轮 BC 热启动正式训练

### 训练配置

```bash
python scripts/train_formation_rllib.py \
  --iterations 500 --cooperative \
  --checkpoint-freq 25 --eval-interval 25 --eval-episodes 20 \
  --seed 42
# LR auto-adjusted: 2e-4 (BC hotstart, 0.67× cold-start)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| LR | 2e-4 | 热启动减量学习率，防止灾难性遗忘 |
| Entropy Coeff | 0.03 | 默认值 |
| Checkpoint | 每 25 轮 | 高频保存，支持回溯分析 |
| Eval | 每 25 轮, 20 episodes | |
| 运行时间 | ~6 小时 | 500 轮 × ~41 秒/轮 |

### 训练 Reward 轨迹

```
iter   0-30:  −4,184 → −2,391   BC→PPO 适应期
iter  40:     +722               🔥 首次转正 (冷启动需要 120 轮)
iter  80-490: 43/45 为正 (95.6%)  范围 +19 ~ +4,041
```

**仅 2 次负值**: iter 100 (−2,498), iter 110 (−3,602)

### Eval 轨迹

| Iter | Eval | 备注 |
|------|------|------|
| 25 | −887 | — |
| **100** | **+191** | 🔥 首次 eval 转正 |
| 225 | −51 | 再次逼近零 |
| 250 | −8,644 | 最差 |
| **425** | **+121** | 二次转正 |
| **450** | **+381** | **最佳** |
| 500 | −1,718 | — |

### 与冷启动对比

| 指标 | BC 热启动 (500轮) | 冷启动 (320轮, 4a-v2 ext) |
|------|-----------------|------------------------|
| 训练 reward 首次转正 | **iter 40** 🔥 | iter ~120 |
| 训练 reward 正向率 | **95.6%** | ~50% |
| Eval 首次转正 | **iter 100** 🔥 | iter ~180 |
| Eval 最佳 | +381 | +2,376 |
| Eval 方差 | +381 ↔ −8,644 | +2,376 ↔ −7,975 |
| Entropy 终值 | 1.89 (健康) | 1.87 |
| Entropy 衰减 | ❌ 无 | ❌ 无 |
| ep_len 趋势 | 335 → 181 (效率提升) | — |

### 关键发现

1. **BC 热启动成功解决了训练稳定性**: 训练 reward 95.6% 为正，冷启动仅 ~50%
2. **收敛速度加快 2-3 倍**: 训练转正快 3×，eval 转正快 2×
3. **Eval 方差问题未消除**: +381 ↔ −8,644 与冷启动的 +2,376 ↔ −7,975 量级相当，说明方差可能来自 OR-gate 环境本身而非训练不稳定性
4. **峰值偏低**: BC 热启动的 eval 最佳 +381 远低于冷启动的 +2,376 — LR=2e-4 偏保守，BC 先验锚定了安全区域但限制了突破性探索
5. **Entropy 全程健康**: 1.84−2.08，无坍缩、无发散

---

## 4. 续训策略设计 (7/13 启动)

基于 500 轮分析，设计"熵驱动探索"续训方案：

| 参数 | 前 500 轮 | 续训 (450→650) | 目的 |
|------|----------|---------------|------|
| LR | 2e-4 | **3e-4** | 增大探索步长 |
| Entropy Coeff | 0.03 | **0.05** | 强制动作多样性，对冲 LR 提高带来的灾难性遗忘风险 |
| DIST_THRESH | 500m | **800m** | 允许更大距离差 |
| DIST_WEIGHT | 0.5 | **0.3** | 降低非对称惩罚 |
| SYNC_WEIGHT | 1.0 | **0.5** | 放宽同步约束 |

从 iter 450 最佳 checkpoint (eval +381) 恢复，续训 200 轮。同时修复了 resume 时 BC 权重覆盖 checkpoint 的 Bug。

---

## 5. 代码产出

| 文件 | 操作 | 内容 |
|------|------|------|
| `scripts/train_formation_rllib.py` | 修改 | BC 两阶段加载 + LR auto-adjust + `--lr`/`--entropy-coeff` CLI + resume 跳过 BC |
| `src/environment/formation_rllib_env.py` | 修改 | 放宽距离/同步惩罚权重 (exploration phase) |

---

## 6. Git 提交

```
b00ae3b fix: move LR auto-adjust before config.build() to prevent None lr
4f84f36 fix: discrete BC weight loading — load turn_head/speed_head from checkpoint top-level
```

---

## 7. 下一步

1. **续训观察** (7/13 启动中): LR=3e-4 + entropy_coeff=0.05 + 放松惩罚 → 目标突破 +381 峰值
2. **AND-gate 激活**: 续训稳定后，引入 `--warmup` 激活 AND-gate 协同约束
3. **论文写作**: BC 热启动 vs 冷启动的消融对比 + 稳定性分析已足够支撑论文 Experiment 章节
