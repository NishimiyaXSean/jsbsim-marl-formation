# Phase 1 离散 BFM 强化学习训练总结

> **日期**: 2026-06-26  
> **主题**: Phase 1 SB3 PPO 训练全流程 —— 从飞控手术到奖励函数反狙击  
> **提交**: 11 commits  
> **总计训练步数**: ~2,200,000（跨 7 个训练版本）

---

## 目录

1. [训练版本进化全记录](#1-训练版本进化全记录)
2. [核心 Bug 发现与修复](#2-核心-bug-发现与修复)
3. [架构升级](#3-架构升级)
4. [奖励函数演进](#4-奖励函数演进)
5. [Tacview 导出格式修复](#5-tacview-导出格式修复)
6. [训练监控基础设施](#6-训练监控基础设施)
7. [最终状态与遗留问题](#7-最终状态与遗留问题)

---

## 1. 训练版本进化全记录

| 版本 | 训练步数 | 关键配置 | 真 success 率 | 最终状态 |
|------|---------|----------|---------------|----------|
| **v1** | 3M | 不公平目标 (AltitudeStabilizer) | 0% | Decelerate 100% 坍缩 |
| **v2** | 1M | 公平对决 + delta-ATA + ent_coef=0.03 | 0% | Decelerate 100% → Climb 转移 |
| **v3** | 200K | 同上 + 终止率监控 | 0% | Climb 80% + TurnRight |
| **v4** | 510K | ent_coef=0.08 + stall=500 + VecNormalize | 0% | 假 success 率 10-32% |
| **v5** | 200K | 2D锁高度 + Action0=FC + 60s episode | 0% | timeout 率提升至 30% |
| **v6** | 370K | 奖励重平衡 (soften stall, boost progress) | 0% | timeout 40%, 仍无 success |
| **v7** | 启动中 | 距离门控ATA + 基线放血 + 5000击杀奖 | — | 训练中 |

> **关键发现**: v1-v4 训练日志中 10-32% 的 "success" 全部为假阳性——warmup 随机生成位置过近 (<200m) 被误判为成功。修复 success 判定条件 (`start_dist > 400m`) 后，真实成功率为 0%。

---

## 2. 核心 Bug 发现与修复

### Bug 1: Warmup 假阳性 (CRITICAL)

**发现**: success 判定 `current_dist < 200m` 在 warmup 结束后的第一帧就可能触发。
warmup 随机生成的追猎者和目标之间的初始距离可能 <200m，导致 Agent 未采取任何行动即被判定 "success"。

**修复** (`f493865`): 
```python
# 旧: if current_dist < 200.0:
# 新: if current_dist < 200.0 and start_dist > 400.0:
```

**影响**: v1-v4 所有训练指标的 "success rate" 全部作废。

### Bug 2: 策略熵坍缩

**发现**: v3 训练到 200K 步时 `train/entropy_loss` 坍缩至 1e-10（均匀分布为 -2.197）。
Agent 收敛到 100% Decelerate 策略。

**修复** (`36b21e1`, `908e645`):
- `ent_coef`: 0.01 → 0.03 → 0.08
- 添加 `VecNormalize(norm_reward=True)` 防止 reward scale 导致的梯度爆炸
- 添加终止率监控 callback，及时发现策略异常

**效果**: entropy 恢复至 -2.05~-2.19（接近均匀分布）。

### Bug 3: Reward Scale 爆炸

**发现**: `REWARD_SUCCESS=5000` 在 VecNormalize 的 running statistics 中产生异常值，
导致 NaN 梯度传播和训练崩溃。

**修复** (`136bdbe`): 移除 VecNormalize，使用手动调好的 reward scale。

---

## 3. 架构升级

### 3.1 公平对决协议 (`f9f14e6`)

**问题**: 目标机使用 `AltitudeStabilizer`（具备高度保持），攻击机使用 `BFMAutopilot`（无高度反馈）。
目标机天然保持平飞而攻击机持续爬升，形成不公平的对决条件。

**修复**: 目标机和攻击机使用**完全相同的飞控栈**:
```
攻击机: FlightEnvelope → BFMAutopilot
目标机: FlightEnvelope → BFMAutopilot (Action 0: Level Flight)
```

### 3.2 混合飞控路由 (`6ebb59c`)

**概念**: 不同的战术意图需要不同的控制模式。

```
Action 0 (Level Flight)  → FlightController (3通道: altitude+speed+heading)
Actions 1-8 (机动)       → BFMAutopilot via FlightEnvelope
```

**效果**: Action 0 提供真正的 "平飞恢复模式"，不再有 BFMAutopilot 的持续爬升积弊。

### 3.3 2D 锁高度模式 (`6ebb59c`)

**问题**: 3D 空间中动能(速度)和势能(高度)的耦合是航空动力学中最复杂的部分。
Agent 在学习追踪的同时还要管理能量，导致探索空间过大。

**方案**: 添加 `lock_altitude` 参数。双方使用 `FlightController` 锁死高度至 3000m。
Agent 只需学习 2D 水平面追踪。

```python
env = BFMPursuitEnv(lock_altitude=True)
# → 双方高度被 FlightController 锁定在 3000m
# → Agent 纯粹学习水平面的切半径和前置追踪
```

### 3.4 Episode 时间缩短

**问题**: 180s 的 episode 过长。失速通常在 30-50s 内发生，剩余 130s 产生无效梯度。

**修复**: `MAX_EPISODE_TIME`: 180s → 60s。

---

## 4. 奖励函数演进

### 4.1 增量奖励 (`f9f14e6`)

新增三个即插即用的奖励组件:

**Delta-ATA (potential-based shaping)**:
```python
R = 8.0 * (exp(-|ATA_cur|/30°) - exp(-|ATA_prev|/30°)) * dt
```
ATA 改善时得正分，恶化为负分。

**闭合率奖励**:
```python
if Vc > 0: R = 6.0 * (Vc / 30.0) * dt
```
鼓励缩小距离，惩罚拉开距离。

**渐进低速警告**:
```python
if speed < 130 m/s: penalty = 1.0 * (deficit / 130) * dt
```
在失速截断触发前给 Agent 早期警告。

### 4.2 第一轮重平衡 (`df2452d`)

| 参数 | 修改前 | 修改后 | 逻辑 |
|------|--------|--------|------|
| `REWARD_PROGRESS` | 0.5 | **1.5** | 3× 更强闭合信号 |
| `REWARD_ATA` | 5.0 | **8.0** | 机头指向优先 |
| `STEP_PENALTY` | 1.0 | **0.25** | 允许活更久 |
| `ANTI_STALL_PENALTY` | 500 | **200** | 别吓瘫 Agent |
| `ANTI_STALL_SPEED_WARN` | 160 m/s | **130 m/s** | 更晚触发 |
| `REWARD_SUCCESS` | 2000 | **3000** | 更大胡萝卜 |
| **`REWARD_TIMEOUT`** | 无 | **-500** | 拖时间是失败 |

### 4.3 反狙击手术 (`1a68e04`)

**距离门控 ATA**:
```python
dist_factor = max(0.0, 1.0 - current_dist / 3000.0)
ata_r = REWARD_ATA * cos_ata * dt * dist_factor
```
距离 >3000m 时 ATA 奖励归零。Agent 无法在远处 "偷看" 刷分。

**基线放血**:
```python
total_reward -= 1.0 * dt  # 每秒扣 1.0 分
```
无论什么动作，每秒固定扣分。必须主动靠近目标才能回本。

**击杀大奖**: `REWARD_SUCCESS`: 3000 → **5000**

---

## 5. Tacview 导出格式修复

共计发现并修复了 **4 个 Tacview ACMI 格式 Bug**:

| Bug | 表现 | 修复 (commit) |
|-----|------|--------------|
| **经纬度反转** | 纬度写成 120°（地理上不可能） | `T=Lat\|Lon` → `T=Lon\|Lat` (`9896a61`) |
| **时间戳重复** | 每帧写两次 `#time` | 每帧只写一次 (`9896a61`) |
| **对象 ID=0** | 占用全局环境保留 ID | 攻击机→101, 目标机→102 (`5d37ac7`) |
| **UTF-8 BOM 缺失** | Windows 下编码错误 | `encoding="utf-8-sig"` (`5d37ac7`) |
| **缺少机型标签** | 渲染为方块而非战斗机 | `Type=Air+Fighter` (`36f1d6b`) |

修复后 Tacview 输出格式:
```
﻿FileType=text/acmi/tacview
0,ReferenceTime=2024-01-01T00:00:00Z
101,Name=F-16 Pursuer
101,Type=Air+Fighter
102,Name=F-16 Target
102,Type=Air+Fighter
#0.00
101,T=120.001(经度)|30.005(纬度)|3007m|0.0°|4.2°|321.9°
102,T=119.997(经度)|30.004(纬度)|2984m|-0.0°|-2.1°|328.4°
```

---

## 6. 训练监控基础设施

### 6.1 终止率监控 (`908e645`)

`Phase1Callback` 每 10K 步记录各终止原因的分布:

```
[Phase1] step=10000 terms:[success=10 stall=143 timeout=0]
```

TensorBoard 指标:
- `termination_rate/success` — 真实成功率
- `termination_rate/stall` — 失速率
- `termination_rate/timeout` — 超时率
- `train/entropy_loss` — 策略熵（防坍缩预警）

### 6.2 定期 Checkpoint (`6ebb59c`)

每 200K 步自动保存模型，支持训练中断后恢复和阶段性评估。

---

## 7. 最终状态与遗留问题

### 7.1 当前状态

| 项目 | 状态 |
|------|------|
| 飞控管线 | ✅ BFMAutopilot + FlightController 混合路由 |
| 2D 训练模式 | ✅ `lock_altitude=True` |
| Episode 时长 | ✅ 60s |
| Success 判定 | ✅ `start_dist > 400m` 防假阳性 |
| 奖励函数 | ✅ 距离门控 + 基线放血 + 5000 击杀奖 |
| Tacview 导出 | ✅ 格式完全合规 |
| 训练监控 | ✅ 终止率 + entropy 实时日志 |

### 7.2 核心问题

**真 success 率始终为 0%** — 经过 7 个训练版本、~2.2M 步后，Agent 从未在 `start_dist > 400m` 的条件下完成一次真正的拦截。

根因链:
1. 离散 9 动作空间粒度不足（60° 坡度转弯无法精细追踪）
2. BFMAutopilot 在 2D 模式下转弯仍消耗能量导致失速
3. 探索空间过大 (9^120)，随机探索撞到成功序列的概率趋近于零
4. 即使有距离门控和基线放血，正向奖励信号仍然过于稀疏

### 7.3 下一步方向

1. **行为克隆预热**: 用启发式纯追踪策略生成 500 条成功轨迹，预训练 Actor 网络
2. **参数化动作空间**: Discrete(5) 意图 × Box(1) 连续参数，允许 15° 精细坡度
3. **分阶段课程**: Stage 1(直线目标) → Stage 2(缓慢转弯) → Stage 3(完全体)
4. **连续动作 MAPPO**: 迁移到连续空间，Agent 可输出任意坡度而非只有 0°/60°

---

## 提交清单

```
1a68e04 fix: anti-sniper — distance-gated ATA + baseline bleed + 5000 kill
df2452d fix: reward rebalance — risk-seeking pursuit incentives
6ebb59c feat: Phase 1 v5 — 2D lock_altitude + Action0=FC + 60s episodes
f493865 fix: success condition requires start_dist > 400m (anti-warmup-fraud)
36f1d6b feat: Tacview Type=Air+Fighter label on all aircraft exports
5d37ac7 fix: Tacview Object ID 0 -> 101/102 + utf-8-sig encoding
9896a61 fix: Tacview ACMI lat/lon swap + duplicate timestamps
136bdbe fix: eval env stays raw (no VecEnv) for callback compatibility
36b21e1 fix: anti-entropy-collapse — ent_coef=0.08, stall=500, VecNormalize
908e645 feat: termination-rate logging in Phase1Callback + TensorBoard
f9f14e6 fix: fair-fight protocol — target uses BFMAutopilot + reward shaping
```
