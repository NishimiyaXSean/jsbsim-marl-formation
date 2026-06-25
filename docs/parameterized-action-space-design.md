# 参数化动作空间设计 (Parameterized Action Space)

> **日期**: 2026-06-25  
> **状态**: 设计提案 — 供 MAPPO 迁移阶段参考

---

## 动机

当前 9 动作离散 BFM 空间存在两个根本局限：

1. **粒度不足**：转弯只有 60° 坡度可选，无法做 15° 的精细前置追踪
2. **缺乏高度管理**：爬升/下降是独立的动作，不能与转弯同时进行

MAPPO 连续空间可以输出任意坡度，但纯连续空间（4-dim Box）的探索维度爆炸（throttle × elevator × aileron × rudder 的全排列）。

**参数化动作空间**在两者之间取得平衡：Agent 先选择**宏观战术意图**（离散），再输出该意图下的**连续微调参数**。

---

## 空间定义

### 离散动作（战术意图）

| 索引 | 语义 | 含义 |
|------|------|------|
| 0 | TRACK | 纯追踪 — 指向目标，用于末端拦截 |
| 1 | LEAD | 前置追踪 — 指向目标前方拦截点 |
| 2 | LAG | 滞后追踪 — 高侧保持能量优势 |
| 3 | PURE_CLIMB | 纯爬升 — 积攒势能 |
| 4 | PURE_DESCENT | 纯下降 — 释放势能换速度 |

### 连续参数（每个离散动作附带 1 个浮点数）

| 离散动作 | 连续参数 p ∈ [0, 1] | 含义 |
|----------|---------------------|------|
| TRACK | `p × 30°` = 目标坡度 | p=0→平飞, p=0.5→15°坡度, p=1→30° 柔和追踪 |
| LEAD | `p × 45°` = 前置坡度 | p=0.33→15°, p=0.67→30°, p=1→45° 前置拦截 |
| LAG | `p × 60°` = 滞后坡度 | LAG 需要更大的坡度来保持高侧位置 |
| PURE_CLIMB | `p × 4G + 1G` = 目标 G | p=0→1G 缓升, p=0.5→3G 标准, p=1→5G 急跃升 |
| PURE_DESCENT | `p × 3G` = 卸载 G | p=0→1G, p=1→-2G 俯冲 |

### 空间维度

```
Gymnasium 空间:
  Tuple(Discrete(5), Box(1, 0, 1))
  → 总维度: 1 (离散) + 1 (连续) = 2
  → 对比纯连续 4-dim: 探索空间减小 ~100×
```

---

## 宏动作封装

Agent 每次决策后，动作被**强制保持 1.5-2.0 秒**（90-120 帧 @ 60Hz）。

```python
# env.step() 伪代码
action_discrete, action_continuous = agent.act(obs)
hold_steps = int(MACRO_HOLD_S / physics_dt)  # 90-120 steps

for _ in range(hold_steps):
    n_n, mu = _decode_action(action_discrete, action_continuous)
    autopilot.step(n_x=0, n_n=n_n, mu=mu, ...)
    physics.step()
```

**效果**：Agent 从 "10Hz 微操抖动" 变为 "每 2 秒一次坚定决策"。

---

## 奖励函数配合

### 动作平滑惩罚

```python
reward_action_smoothness(prev_action, curr_action, is_discrete=True)
```
离散动作切换时给予惩罚，鼓励 Agent 连续多步保持同一意图。

### 前置追踪奖励

```
R_lead = w · exp(-|lead_angle_error| / σ)  # 鼓励在正确的前置角上
```

### 能量管理

```
R_energy = w · (Es / Es_ref)  # 保持能量高度
```

---

## 与现有架构的关系

```
参数化动作
  → 解码为 (n_x, n_n, mu)
  → FlightEnvelope (V-n 限制 + G-smoothing + 高度保护)
  → BFMAutopilot (λ-G 飞控律)
  → JSBSim F-16 FDM
```

现有管线无需改动——参数化动作空间仅替换 RL Agent 的输出接口。

---

## 实施路径

1. **Phase 1** (当前): 在 `src/environment/` 中实现 `ParameterizedPursuitEnv`
2. **Phase 2**: 训练 SB3 PPO (支持 MultiDiscrete + Box 的混合空间)
3. **Phase 3**: 迁移到 MAPPO (RLlib 原生支持参数化动作空间)
4. **Phase 4**: 对比离散 9 动作 vs 参数化 5+1 动作的追踪效率
