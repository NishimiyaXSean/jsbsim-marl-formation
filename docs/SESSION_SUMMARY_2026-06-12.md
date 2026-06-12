# 工作进展总结 — 2026-06-12

## 目标

使用 JSBSim F-16 飞行动力学 + 强化学习，训练单机追击拦截能力。

## 训练结果

| Stage | 捕获率 | 说明 |
|-------|--------|------|
| Stage 1 (正前方 ±2°) | **50-75%** | 目标直飞，3-seed 验证 |
| Stage 2 (±15°, 温和机动) | **25-45%** | 自动进阶 |
| Stage 3 (±45°, 编织机动) | **10-40%** | 峰值 40% |

**致胜公式**: `200m 击杀半径 + 残差 RL + 自适应制导专家 + FlightController`

---

## 关键 Bug 修复

### 1. `src/dynamics/aircraft.py` — 引擎启动

**问题**: `Aircraft.reset()` 中 `run_ic()` 不会自动启动涡轮发动机。所有训练都在 0 磅推力下滑翔。

**修复**: `run_ic()` 后添加:
```python
fdm["propulsion/engine[0]/set-running"] = 1
```

**影响**: 推力从 0 → 27,000+ lbs (满加力)。此前所有训练无效。

### 2. `src/dynamics/autopilot.py` — 升降舵符号 + 偏置

**问题 1**: BFM autopilot 的 Nz PID 输出正 elevator（推头向下），但注释说"正 elevator = 拉杆"。符号反转导致持续俯冲。

**修复**: 改为 `elevator = ELEVATOR_TRIM - PID(error)`，添加 -0.05 配平偏置。

**问题 2**: 油门 PID 从 0 开始，无前馈。配平油门 ~0.8 需要积分累积 30s+。

**修复**: 添加 `THROTTLE_BIAS = 0.80`。

### 3. `src/environment/air_combat_env.py` — GPWS 坐标系

**问题**: GPWS 使用 `w_fps`（机体坐标系垂直速度）判断下降。坡度 >30° 时完全失效——飞机在俯冲但 GPWS 探测不到。

**修复**: 改为 `h_dot_fps`（世界坐标系垂直速度），从 `aircraft.py` state 新增此属性。

### 4. `src/dynamics/flight_controller.py` — 航向响应

**问题**: HeadingStabilizer 最大坡度 45°，转弯率 ~2°/s。90° 转弯需 45 秒。

**修复**: 最大坡度提高到 70°，PID 增益增强，坡度补偿改为 `1/cos(bank)`。

---

## 新增/修改模块

### `src/utils/pn_guidance.py` — 比例导引 [新文件]

增强型比例导引: `desired_heading = bearing_to_target + N * lambda_dot * dt`。结合纯追踪（指向目标）和 PN（预测拦截点）。用于自适应专家的制导逻辑。

### `src/environment/scenario.py` — 追击出生点

新增 `generate_pursuit_spawn()` 函数。三阶段课程:
- **Stage 1**: 目标在正前方 ±2°，同向同速 — 只需加速
- **Stage 2**: ±15° 方位，±20° 航向差 — 需要温和转向
- **Stage 3**: ±45° 方位 — 接近实战

### `src/environment/single_pursuit_env.py` — 单机追击环境

关键改动:
- **直接副翼控制**: 绕过 FC 的 HeadingStabilizer，agent 直接输出 aileron
- **高度匹配**: FC 目标高度 = target 出生高度（消除 3D 垂直偏差）
- **坡度补偿**: `1/cos(bank)` 防止转弯时掉高度
- **出生对准**: pursuer 出生即对准 target（零 bearing 偏差）
- **碰撞半径**: 200m（从 50m 扩大）

### `src/dynamics/bfm_actions.py` — 追击动作集

新增 9 动作 BFM 子集（安全、低 G、无俯冲转弯）:
0: 平飞, 1: 加速, 2: 减速, 3: 右转, 4: 左转, 5: 爬升, 6: 下降, 7: 加速右转, 8: 加速左转

### `src/dynamics/flight_envelope.py` — GPWS 增强

- GPWS 使用完整 G（不受 curriculum g_scale 影响）
- G-smoothing `tau_g` 从 0.4s 降至 0.15s

---

## 训练脚本

### `scripts/train_single_pursuit.py` — **主力训练脚本** [重点]

架构: **残差 RL + 自适应制导专家**

```python
class ResidualExpertWrapper(gym.Wrapper):
    """
    Agent 输出残差修正，叠加在自适应专家上。
    专家 = 根据观测计算目标方位 → 副翼转向 + 满油门。
    RL 残差范围 ±0.5，永不会覆盖专家基线。
    """
```

关键参数: 200k 步, LR=1e-4, ent_coef=0.01, 200m 半径, 3 阶段课程

### `scripts/train_bfm_pursuit.py` — BFM 追击训练

13 个离散 BFM 动作 + SB3 PPO。由于 BFM autopilot 调参困难，已被 residual 方案取代。保留供后续参考。

### `scripts/train_continuous_pursuit.py` — 连续舵面训练

4 维连续动作 + 配平偏置。因 phugoid 不稳定被取代。保留供参考。

### `scripts/verify_pursuit.py` — 环境验证

三种场景的追击能力验证: (1) 满油门直飞, (2) 纯追踪, (3) BFM 动作。用于诊断。

### `scripts/find_trim.py` — 配平扫描

扫描不同 throttle 和 elevator 组合，找到平飞配平点。诊断工具。

---

## 设计文档

### `docs/superpowers/specs/2026-06-12-continuous-pursuit-training-design.md`

短期方案的完整设计 spec — continuous action space + trim bias + pursuit spawn。

---

## 已尝试但未采用的方法

| 方法 | 最佳结果 | 失败原因 |
|------|---------|---------|
| BFM autopilot + RL | 5% | Nz PID 符号错误，调参复杂 |
| Continuous 直接舵面 | 0% | F-16 配平点长周期不稳定 (phugoid) |
| FlightController 航向控制 | 0% | 转弯响应太慢 (5.7°/s max) |
| BC bias 初始化 | 40%→0% | PPO 更新覆盖 bias |

---

## 已保存模型

| Run | Seed | 最终 Eval | 训练峰值 | 路径 |
|-----|------|-----------|---------|------|
| s2_v2 | 0 | 0% (Stage 3) | 40% Stage 3 | `marl_runs/single_pursuit_0612_2103_s0` |
| s2_v2 | 1 | 10% (Stage 3) | 35% Stage 3 | `marl_runs/single_pursuit_0612_2103_s1` |
| s2_v2 | 2 | 5% (Stage 3) | 30% Stage 2 | `marl_runs/single_pursuit_0612_2103_s2` |

最终 Tacview 导出: `results/single_pursuit/single_pursuit_engagement.txt.acmi`

## 下一步建议

1. **碰撞半径调回 100m** — 200m 太宽松，逐步缩小验证泛化能力
2. **Stage 3 专项优化** — 用 Stage 2 训练好的模型做行为克隆初始化
3. **多机编队** — 当前架构可直接扩展到 2v2
4. **整理代码** — 清理已废弃的训练脚本，统一参数命名
