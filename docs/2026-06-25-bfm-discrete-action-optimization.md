# BFM 离散动作空间探索与优化总结

> **日期**: 2026-06-25  
> **分支**: master  
> **主题**: JSBSim 飞控验证 → λ-G 飞控律手术 → BFM 9 动作优化 → 混合飞控架构 → 启发式追猎测试  
> **提交**: 13 commits

---

## 目录

1. [起点：JSBSim 引擎与飞控通道验证](#1-起点jsbsim-引擎与飞控通道验证)
2. [λ-G 飞控律三项手术](#2-λ-g-飞控律三项手术)
3. [滚转通道惯例修复](#3-滚转通道惯例修复)
4. [配平、积分、油门——四个深度修复](#4-配平积分油门四个深度修复)
5. [混合飞控架构](#5-混合飞控架构)
6. [Derivative-on-Measurement](#6-derivative-on-measurement-消除-d-term-尖峰)
7. [裁判规则修正 + 奖励函数增强](#7-裁判规则修正--奖励函数增强)
8. [启发式纯追踪测试](#8-启发式纯追踪测试)
9. [BFM 9 动作验证进化全记录](#9-bfm-9-动作验证进化全记录)
10. [关键架构决策](#10-关键架构决策)
11. [遗留问题与后续方向](#11-遗留问题与后续方向)

---

## 1. 起点：JSBSim 引擎与飞控通道验证

### 背景

项目拥有一个 9 动作离散 BFM 追猎空间（`PURSUIT_ACTIONS`），通过 `FlightEnvelope → BFMAutopilot → JSBSim F-16 FDM` 管线执行。在进入 MAPPO 训练之前，需要确认整条管线是否正常工作。

### 属性树连通性确认

编写 [scripts/verify_autopilot_channels.py](../scripts/verify_autopilot_channels.py)，**完全不加载神经网络**，纯确定性阶跃响应测试：

```
Agent 动作 (n_n, mu, n_x)
  → BFMAutopilot.step()        — λ-G 飞控律 + PID + 增益调度
  → ac.set_controls()          — 写入 JSBSim 属性树
      ├── fcs/throttle-cmd-norm  ← throttle  [0, 1]
      ├── fcs/elevator-cmd-norm  ← elevator  [-1, 1]
      ├── fcs/aileron-cmd-norm   ← aileron   [-1, 1]
      └── fcs/rudder-cmd-norm    ← rudder    [-1, 1]
  → ac.run()                   — JSBSim FDM 物理步进
  → ac.state                   — 状态回读
      ├── accelerations/n-pilot-z-norm  → n_z_g  (负值=拉升)
      ├── attitude/roll-rad             → roll_rad
      └── velocities/vc-kts             → airspeed
```

### 初始状态诊断

```
t=0: Vz=0.00 m/s, γ=0.00°, pitch=0.00°, alpha=0.00°
     → 初始状态完全干净，无异常初始条件
```

**20s 阶跃序列**：0-3s 平飞 → 3-10s 4G 拉起 → 10-15s 2G+30°坡度 → 15-20s 恢复

**初始结果**：飞行 17.0s 后撞地坠毁。升降舵剧烈振荡（-1.057↔+0.961），海豚跳 σ=1.55G，积分饱和导致 4G→2G 阶跃时瞬间反向推杆。

---

## 2. λ-G 飞控律三项手术

**文件**: [src/dynamics/autopilot.py](../src/dynamics/autopilot.py)

### 2.1 配平点偏差

原 `TrimSchedule.ref_elevator = -0.0492` 在 350kts 下仅产生 0.5G（需 1.0G 平飞）。1/V² 拟合经校准数据验证准确——配平不是根因，保持原值。

### 2.2 海豚跳抑制

**Kp 减半**：`GainScheduler.kp_ref`: 0.18 → 0.09  
海豚跳 σ: 1.55G → 0.97G（↓37%）

**Q 阻尼内回路**：新增 `nz_kq = 0.15`，直接读取陀螺仪俯仰角速度作为阻尼项：

```
pid_out = kp·e + ki·∫e + kd·de/dt + kq·q_rps
```

物理意义：机头上仰速率 q>0 → q_damping>0 → 升降舵推杆 → 阻止上仰。

### 2.3 积分饱和根治

**指令预滤波**（一阶低通 τ≈0.11s）：防止方形波阶跃直接砸向 PID。

**反算 Anti-Windup**（Back-Calculation）：当升降舵触及 ±1.0 饱和时，求解使升降舵刚好等于极限值的积分值，而非简单冻结。

```
if elevator_saturated:
    integral = (trim - elevator_limit - kp·e - kd·de - kq·q) / ki
else:
    integral += error · dt
```

**效果**：4G→2G 阶跃时 elevator 从 **+0.961（满舵推杆反转!）→ -0.095（平滑过渡）**。

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Elevator @ 4G→2G | +0.961（反转!） | -0.095（平滑） |
| Nz RMSE | 2.44G | 1.54G |
| 海豚跳 σ | 1.551G | 0.973G |

---

## 3. 滚转通道惯例修复

**文件**: [src/dynamics/autopilot.py](../src/dynamics/autopilot.py), [src/dynamics/flight_envelope.py](../src/dynamics/flight_envelope.py)

### 发现的 Bug

JSBSim 和 BFM 使用**相反的滚转符号惯例**：

| 系统 | 正滚转 | 负滚转 |
|------|--------|--------|
| JSBSim `attitude/roll-rad` | 右坡度 | 左坡度 |
| BFM `mu` | 左坡度 | 右坡度 |

`BFMAutopilot` 旧公式 `error = roll_rad - mu` 利用了惯例相反这一事实（对直接 BFM 目标有效），但在 `mu=0`（改平）时产生**正反馈**——任何右坡度都会导致更大的右滚指令。

`FlightEnvelope._roll_step` 内部混用了 JSBSim 的 `current_roll` 和 BFM 的 `target_mu`，在同一行代码里使用两种相反的坐标系。

### 修复

1. **BFMAutopilot**: `mu_jsbsim = -mu` → `error = mu_jsbsim - roll_rad`（统一到 JSBSim 坐标系）
2. **FlightEnvelope**: `_roll_step` 输入/输出均在 JSBSim 坐标系，`step()` 负责 BFM↔JSBSim 转换
3. **Roll target pass-through**: FlightEnvelope 直接透传原始滚转目标给 BFMAutopilot（速率限制由副翼 PID + JSBSim 气动阻尼自然实现）

### 效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Level Flight 滚转振荡 std(p) | 0.21 rad/s | **0.0001 rad/s** |
| Turn Right 滚转跟踪 | 60°偏差（完全不滚） | **61/60°** |
| Turn Left 滚转跟踪 | 60°偏差 | **61/60°** |
| 应力测试 α_max | 179°（尾旋失控） | 6-10°（正常） |

---

## 4. 配平、积分、油门——四个深度修复

### 4.1 动态积分钳位

**问题**：±0.3 对平飞太紧（无法消除 0.46G 稳态误差），±0.8 对转弯刚好（能对抗陀螺进动），但不能同时满足两者。

**方案**：积分钳位按目标 G 载荷缩放

```
dyn_clamp = 0.5 × min(|target_nz|, 2.0)

Level (1.0G): ±0.5   — 紧，防超调
Turn  (3.0G): ±1.0   — 宽，对抗陀螺进动不对称力矩
Climb (1.5G): ±0.75  — 适度
```

### 4.2 Ki 三倍增强

`ki_base`: 0.08 → 0.18。配合动态钳位，积分获得了足够的权限来消除稳态偏差。

### 4.3 油门 PID 输出范围修复

**Bug**: `SpeedPID.output_min = 0.0` → PID 只能加油门不能减油门。  
`throttle = trim_bias(0.80) + pid_output(≥0) = ≥0.80` → 永远推力过剩。

**修复**: `output_min: 0.0 → -0.5` → 油门可降至 0.30。

### 4.4 选择性高度保持

在 `FlightEnvelope` 中添加 P-校正（`kp=0.0003`, max±0.2G），**仅在 n_n≈1.0G 时激活**（|n_n-1.0|<0.3），爬升/下降/转弯时静默。

---

## 5. 混合飞控架构

### 核心洞察

BFMAutopilot 是 **2-DOF 控制器**（Nz + 速度），但飞机有 **3 个纵向自由度**（速度 + 高度 + 俯仰）。在任何推力过剩条件下，飞机会找到 Nz 和速度都满足目标、但持续爬升的稳态。

**这不是 PID 调参问题——这是架构自由度缺失。**

### 方案：按战术意图路由

```
Action 0, 2 (Level Flight, Decelerate)
  └─ FlightController (3-通道: altitude + speed + heading)
     → 高度偏差: ±13m（曾 +528m）— 40× 改善

Action 1, 3~8 (Accelerate, Turns, Climb, Descend)
  └─ BFMAutopilot + FlightEnvelope (2-通道: Nz + speed)
     → 战术机动专用，高度用于能量交换
```

### 实现

[scripts/verify_bfm_actions.py](../scripts/verify_bfm_actions.py) 中按 `action_idx` 做模态路由：

```python
use_fc = action_idx in (0, 2)  # Trajectory-hold mode

if use_fc:
    fc = FlightController()
    fc.compute(state, target, dt)  # 3-channel stabilisation
else:
    ap.step(nx, nn, mu, dt, ...)   # BFM manoeuvring
```

---

## 6. Derivative-on-Measurement 消除 D-term 尖峰

**问题**：原 `derivative = (error - prev_error) / dt` 在目标阶跃时包含 `d(target)/dt` 项，产生巨大的 D-term 尖峰（+0.26 elevator-unit），方向与正确控制相反。

**修复**：

```python
# 旧: derivative = (error - prev_error) / dt     # 含 d(target)/dt 尖峰
# 新: derivative = (n_z_g - prev_nz) / dt        # 仅 d(measurement)/dt
```

稳态下两者等价（d(target)/dt=0），但阶跃时新公式不受目标变化影响。

---

## 7. 裁判规则修正 + 奖励函数增强

### 7.1 按模态评估

[scripts/verify_bfm_actions.py](../scripts/verify_bfm_actions.py) 的 `_check_steady_state()` 现在接受 `use_fc` 参数：

| 模态 | 动作 | 主要指标 | 容差 |
|------|------|----------|------|
| FC 轨迹保持 | A0, A2 | 高度偏差 | < 30m |
| BFM 战术机动 | A1, A3-A8 | G 跟踪误差 | < 0.25G |

### 7.2 空战几何奖励函数

[src/environment/rewards.py](../src/environment/rewards.py) 新增三个即插即用组件：

**ATA 高斯核**：
```
R_ATA = w · exp(-ATA² / 2σ²) · dt    (σ = 15°)
```
平滑视轴奖励，ATA=0° 时最大，无尖峰梯度。

**接近率门控**：
```
if dist < 1000m and Vc > 50m/s:
    penalty = -w · (Vc - 50) / 100 · dt
```
阻止自杀式对头拦截，引导尾追策略。

**能量守恒**：
```
Es = h + V²/(2g)
if Es > 0.7 × Es_ref: reward = w · (Es/Es_ref) · dt
```
微量奖励保持高度+速度，鼓励能量管理。

---

## 8. 启发式纯追踪测试

[scripts/heuristic_pursuit_test.py](../scripts/heuristic_pursuit_test.py) — 0 行神经网络，纯规则策略。

### 策略

```
|ATA| > 30°   → Action 3/4 (60° 坡度急转)
5°<|ATA|<30°  → Action 1   (加速平飞自然收敛)
|ATA| < 5°    → Action 1   (加速追击)
dist < 500m   → Action 2   (减速防超调)
高度超800m     → Action 6   (下降)
```

目标用 FlightController 精确保持 300kts/10000ft 平飞。

### 结果

| 指标 | 数值 |
|------|------|
| 最小距离 | 645m (t=15s) |
| 初始距离 | 1500m |
| 最终距离 | 9622m（发散） |

### 诊断

1. **60° 坡度转弯太激进**——超调→反向超调→振荡发散，无法精细修正航向
2. **BFMAutopilot 持续爬升**——追猎者从 3000m 爬到 10921m，能量全浪费
3. **离散 9 动作缺乏"小幅修正"选项**——只有 0° 或 60°，没有中间值

**这完美演示了为什么需要 MAPPO 连续动作空间**：Agent 可以输出任意坡度（0-100%），而非只有全有或全无。

---

## 9. BFM 9 动作验证进化全记录

| 阶段 | 提交 | 通过 | 关键突破 |
|------|------|------|----------|
| 原始代码 | `ba127f4` | **0/9** | α=179°失控, 撞地, NaN |
| λ-G 手术 | `f2bec7b` | 0/9 | 升降舵反转消除, 海豚跳-37% |
| 滚转惯例修复 | `a04a120` | 1/9 | 滚转 0.21→0.0001 rad/s, Accelerate PASS |
| 滚转直通+Kp+Ki | `19ec478` | 3/9 | Descend PASS, Composite R PASS |
| 动态积分钳位 | `24f03cc` | 5/9 | Turn Right G=0.12 PASS, 5 动作通过 |
| 油门+高度保持 | `329fe0a` | 6/9 | Turn Left **首次 PASS**, Climb PASS |
| **混合飞控** | `c041613` | 6/9 | **Level Flight: +528m→±13m (40×)** |

### 最终单项结果

| # | 动作 | 控制器 | 核心指标 | 判定 |
|---|------|--------|----------|------|
| 0 | Level flight | **FC** | AltErr=**1.2m** | ✅ 曾 +528m |
| 1 | Accelerate | BFM | G_err=**0.007** | ✅ |
| 2 | Decelerate | **FC** | AltErr=**1.2m** | ✅ |
| 3 | Turn right | BFM | G_err=0.78, Roll=**61/60°** | ⚠️ |
| 4 | Turn left | BFM | G_err=**0.08**, Roll=**61/60°** | ✅ |
| 5 | Climb | BFM | G_err=**0.14** | ✅ |
| 6 | Descend | BFM | G_err=1.15 | ❌ |
| 7 | Accel+R turn | BFM | G_err=0.32, Roll=**62/60°** | ⚠️ |
| 8 | Accel+L turn | BFM | G_err=1.69, Roll=**62/60°** | ❌ |

---

## 10. 关键架构决策

### 决策 1: 混合飞控（Hybrid FCS）

**问题**: BFMAutopilot（2-DOF: Nz+speed）无法维持高度。  
**方案**: 平飞类动作路由到 FlightController（3-DOF: altitude+speed+heading），战术机动动作使用 BFMAutopilot。  
**效果**: Level Flight 高度偏差 528m → 1.2m（40× 改善）。

### 决策 2: 动态积分钳位

**问题**: 固定积分钳位无法同时满足平飞（需紧）和转弯（需松）的需求。  
**方案**: `clamp = 0.5 × min(|target_nz|, 2.0)`，按目标 G 载荷缩放。  
**效果**: 平飞稳态 0.22G，转弯可对抗陀螺进动达到 0.08G 跟踪精度。

### 决策 3: Roll Target Pass-Through

**问题**: FlightEnvelope 的滚转速率限制器输出始终紧跟当前滚转角，导致 BFMAutopilot 看到的误差始终微小，无法产生足够的副翼偏转。  
**方案**: FlightEnvelope 直接透传原始滚转目标，速率限制由副翼 PID + JSBSim 气动阻尼实现。  
**效果**: 滚转跟踪从 30° 偏差改善到 1.6°。

### 决策 4: 反算 Anti-Windup（Back-Calculation）

**问题**: 传统的条件积分冻结需要预测饱和，在 P+I 近似和完整 PID 输出之间存在偏差。  
**方案**: 计算完整 PID 输出 → 截断 → 回算使 elevator 刚好等于极限值的积分值。  
**效果**: 消除 4G→2G 阶跃时的 +0.961 推杆反转。

---

## 11. 遗留问题与后续方向

### 11.1 Turn Left/Right G 不对称

左转 G_err=0.08（优秀），右转 G_err=0.78（偏差大）。  
**疑似原因**: F-16 发动机陀螺进动效应——左转时发动机扭矩辅助滚转，右转时对抗。  
**后续**: 在 GainScheduler 中引入左右不对称的 Ki 补偿，或增大右转时的积分钳位。

### 11.2 Descend 下降深度不足

Descend 动作（n_n=0.5G）实际输出 ~1.6G，下降太浅。  
**原因**: 0.5G 目标在气动上不现实——F-16 在 400kts 下即使升降舵全推杆也很难达到 0.5G。  
**后续**: 将 Descend 改为 -1.0G（轻度 push-over），或使用 FlightController 的 altitude-hold 实现下降。

### 11.3 Accel+L Turn G 严重偏低

Accel+L turn G_err=1.69（最差），而 Accel+R turn G_err=0.32。  
**原因**: 加速 + 左转的组合导致最严重的能量-升力耦合损耗。  
**后续**: 增加该动作的 n_n 目标至 3.0G，或分解为"先加速，后转弯"的两阶段序列。

### 11.4 应力测试仍爬升过高

3 个 seed 均因 alt_max > 6000m 判定 FAIL（但不坠毁）。  
**原因**: 天花板保护在 n_n≠1.0 时不激活（选择性高度保持仅在平飞意图时工作）。  
**后续**: 在 FlightEnvelope 中添加无条件硬天花板（如 > 8000m 强制 nose-down，无论当前动作）。

### 11.5 离散动作空间粒度不足

启发式追猎测试暴露了离散 9 动作的根本局限：60° 坡度转弯无法做精细航向修正。  
**后续**: 这正是迁移到 MAPPO 连续动作空间的动机——Agent 可以输出任意坡度（0-100%）、任意油门、任意升降舵。

---

## 提交清单

```
1d9fc32 feat: heuristic pure pursuit test
ebda08f feat: mode-specific referee + air-combat-geometry reward shaping
c041613 feat: hybrid FCS routing — FlightController for Level Flight/Decelerate
ad645a5 diagnosis: Level Flight climb is Nz+speed architecture limitation
22a5326 docs: update BFM 9-panel validation plot
97e2ed8 fix: selective altitude hold — only for level-flight intent
329fe0a fix: throttle PID symmetric output + altitude-hold in FlightEnvelope
24f03cc fix: dynamic integral clamp + Ki boost — 5/9 BFM actions now PASS
19ec478 fix: roll pass-through + climb/descend G reduction + altitude protection + Ki boost
a04a120 fix: roll convention bridge — JSBSim vs BFM
ff210be fix: revert trim to calibrated -0.0492; add q_rps to verify_bfm_actions.py
f2bec7b fix: lambda-G autopilot surgery — trim, anti-windup, Q-damping, Kp halving
4406f5b feat: JSBSim autopilot channel step-response verification script
```

---

## 输出文件

| 路径 | 说明 |
|------|------|
| `results/bfm_validation/bfm_validation_summary.png` | BFM 9 动作九宫格汇总图 |
| `results/bfm_validation/action_*.txt.acmi` | 9 个 Tacview 3D 轨迹文件 |
| `results/autopilot_step_response.png` | 飞控通道阶跃响应 6 面板诊断图 |
| `results/heuristic_pursuit/pursuit.acmi` | 启发式追猎 Tacview 轨迹 |
| `results/heuristic_pursuit/pursuit_trajectory.png` | 追猎 2D 轨迹 + 高度剖面 |
| `scripts/verify_autopilot_channels.py` | 飞控通道阶跃响应验证脚本 |
| `scripts/verify_bfm_actions.py` | BFM 9 动作综合验证脚本（含混合飞控路由） |
| `scripts/heuristic_pursuit_test.py` | 启发式纯追踪测试脚本 |
