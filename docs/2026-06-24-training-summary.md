# 2026-06-24 工作总结：反海豚跳 → 飞控改造 → BFM 离散动作

## 执行摘要

今日完成了 **9 次代码提交**，涉及 **15 个文件**（新建 10 个，修改 5 个）。工作分为三条主线：
1. **奖励函数修复**（V10.5 → V11）：彻底重构 V_c 耦合机制和海豚跳抑制
2. **底层飞控全面改造**（Phase 1–4）：从开环扫掠到增益调度，赋予 autopilot 航空学常识
3. **BFM 离散动作空间**（Discrete(9) + Action Masking）：将 RL 从连续飞行控制升级为战术机动决策

核心认识：**连续动作空间（V11 Box(3)）是单机追踪的最终方案；离散 BFM（Discrete(9)）应留给真正的 1v1 空战格斗。**

---

## 主线一：奖励函数修复（4 次提交）

### V10.5 — 反海豚跳两招 (`86a878a`)

海豚跳根因：俯冲→动能→V_c 飙升→guidance 奖励暴增 的套利循环。收益上限远高于惩罚成本。

| 杀招 | 公式 | 效果 |
|---|---|---|
| **二次方高度惩罚** | `-30 × (Δh/1000)²` | 100m 偏离仅 -0.3，800m 偏离爆炸至 -19.2（64×） |
| **压低 V_c 饱和点** | sigmoid(K=0.3, MID=15) | V_c=30 m/s 即达 0.99 乘数，平飞满油门即可吃满奖励 |

文件：`src/environment/ablation_wrappers.py`（LeadPursuitRewardWrapper）

### V11 — 反策略坍塌 (`40dd7c3`)

策略坍塌根因：低速转弯时 V_c→0 → 所有 guidance 奖励归零 → 梯度消失 → log_std 归零。

| 优化 | 公式 | 效果 |
|---|---|---|
| **V_c 最低工资** | `0.3 + 0.7 × clamp(V_c/30, 0, 1)` | 即使 V_c≤0，仍有 30% guidance 信号 |
| **ent_coef 心脏起搏器** | 0.015 → **0.02** | 强制维持探索噪声，阻止 log_std 坍塌 |

文件：`src/environment/ablation_wrappers.py`, `scripts/train_single_pursuit.py`, `scripts/run_v10_5_training.py`, `scripts/run_ablation_study.py`

### 训练结果

V10.5 + V11 训练（3 seeds × 5M steps, seed 0 完成）：
- 峰值：80% CR @ diff=0.15（step 45K）
- 坍塌：diff=0.20 后全部归零
- **策略坍塌问题未被奖励函数修复完全解决**——坍塌过程与 V7 一致

---

## 主线二：底层飞控全面改造（4 次提交）

### 核心目标

将飞行稳定性交还给经典控制理论（PID/前馈），让 RL 只做战术决策。

### Phase 1: 开环气动特性扫掠 (`6920fc8`)

- `scripts/sweep_elevator.py`：在 150/200/250 m/s 三个速度点，施加 -0.2→+0.2 升降舵阶跃
- 产出：`data/trim_table.json`，验证 **1/V² 配平定律**（RMS error = 0.00366）
- 实测配平：300 kts → -0.0813，400 kts → -0.0492，500 kts → -0.0328

### Phase 2: 单通道解耦 PD 调参 (`6920fc8`)

| 通道 | kp | kd | 约束 |
|---|---|---|---|
| Roll | 1.5 | 0.08 | 外环 kp ≤ 0.5× JSBSim 内环（roll-rate kp=3.0） |
| Pitch (Nz) | 0.18 | 0.012 | 外环 kp ≤ 0.6× JSBSim 内环（G-load kp=0.3） |
| Speed | 0.02 | 0.0 | 慢动态，仅 P 控制 |

关键发现：**JSBSim 内环 PID 已经非常激进**，外环增益必须保守，否则双环级联必然震荡。

### Phase 3: 注入物理先验 (`6920fc8`, `58a6b34`)

- **动态配平 TrimSchedule**：`trim(V) = ref_elevator × (V_ref/V)²`，替换硬编码常量
- **重力补偿**：倾斜时 Nz 目标自动增加 `1/cos(μ) - 1` G
- **Alpha/G 限制器**：alpha > 25° 强制推杆，n_z > 9G 锁死拉升
- **积分重新启用**：保守 anti-windup（30% 输出范围）

### Phase 3.5: 严格检测 + 增益调度 (`2701a48`, `d03f143`, `58a6b34`)

**三项严格检测**（`scripts/verify_bfm_actions.py`）：

| 检测项 | 阈值 | 目的 |
|---|---|---|
| G 跟踪误差 | < 0.1G | 稳态 Nz 收敛到目标（含倾斜补偿） |
| 滚转角跟踪误差 | < 3° | 倾角锁定精度 |
| 震荡检测 std(q)/std(p) | < 0.05 rad/s | 捕捉俯仰/滚转抖动（残留标准差） |

**GainScheduler** — 速度 + 目标 Nz 自适应增益：
- kp(V) = kp_ref × (V_ref/V)²，钳位在 [0.08, 0.30]
- ki 通过 tanh 平滑过渡：平飞 ki=0.08，大 G 指令 ki=0.14
- 效果：Climb (3G) G_err **0.65→0.07** ✓，Descend (-2G) G_err **0.49→0.09** ✓

**迎角重力补偿实验** (`58a6b34`)：
- `alpha_gravity_loss = 1 - cos(α)` ≈ **0.004G at α=5°**
- 结论：效应可忽略，不是 0.6G 静态偏差的根因
- 回退到 Phase 1 实测配平值

### Phase 4: BFM 9 动作全验证 (`7154166`, `2701a48`)

9 个 PURSUIT_ACTIONS 全部通过包线安全检测（alpha < 28°, nz < 9.5G, 无撞地/超速）。每个动作独立生成 Tacview 文件 + 3×3 汇总图。

文件产出：
- `src/dynamics/autopilot.py` — TrimSchedule, GainScheduler, 增益更新
- `src/dynamics/flight_controller.py` — 共享 TrimSchedule
- `src/environment/air_combat_env.py` — 传递 alpha_deg
- `scripts/sweep_elevator.py`, `scripts/tune_roll.py`, `scripts/tune_pitch.py`, `scripts/tune_speed.py` — Phase 1/2 工具
- `scripts/verify_bfm_actions.py` — Phase 4 验证
- `data/trim_table.json` — 配平查表

---

## 主线三：BFM 离散动作空间（2 次提交）

### BFMPursuitEnv (`c26eb7a`)

将动作空间从连续 Box(3) 切换为 Discrete(9)：

| 动作 | n_x | n_n | μ | 含义 |
|---|---|---|---|---|
| 0 | 0.0 | 1.0 | 0° | 平飞 |
| 1 | +2.0 | 1.0 | 0° | 加速 |
| 2 | -2.0 | 1.0 | 0° | 减速 |
| 3 | 0.0 | 2.0 | -60° | 右转 |
| 4 | 0.0 | 2.0 | +60° | 左转 |
| 5 | 0.0 | 3.0 | 0° | 爬升 |
| 6 | 0.0 | -2.0 | 0° | 下降 |
| 7 | +1.0 | 2.0 | -60° | 加速右转 |
| 8 | +1.0 | 2.0 | +60° | 加速左转 |

- 决策频率 2 Hz（0.5s/机动），通过 Phase 3.5 autopilot 全链路稳定
- 67% 的代码与 SinglePursuitEnv 共享（观测、奖励、终止逻辑）

**训练结果（seed=0, 5M steps, 1.5M 处停止）：完全失败**
- 103 次 eval，capture rate **始终 0%**
- avg_min_dist ~1200m，从未接近目标
- 每集 ~15 秒坠毁，奖励 -4000 到 -5000
- 根因：离散 BFM 动作缺乏精确拦截所需的连续微调能力

### Action Masking (`ba127f4`)

引入无效动作屏蔽，防止智能体选择自毁动作：

**5 条安全规则**：
1. **硬甲板**（alt < 1500m）→ 禁止 Descend
2. **防失速**（spd < 160 m/s）→ 禁止 Decelerate, Climb
3. **防超速**（spd > 350 m/s）→ 禁止 Accelerate, Accel+turns
4. **防高迎角**（alpha > 20°）→ 禁止 Turns, Climb, Accel+turns
5. **兜底**：全部屏蔽时保留 Level flight

**自定义策略网络**（`src/environment/masked_policy.py`）：
- `MaskedFeatureExtractor`：从 Dict 观测中提取 "obs" 特征
- `MaskableActorCriticPolicy`：将无效动作 logits 设为 -1e9

首次 eval 仍 0%（30/30 ground_crash），但掩码强制智能体避免最糟糕的选择。

文件产出：
- `src/environment/bfm_pursuit_env.py` — 新环境
- `src/environment/masked_policy.py` — 自定义 SB3 策略
- `scripts/train_bfm_pursuit.py` — 训练脚本
- `scripts/quick_tacview_bfm.py` — 可视化

---

## 文件变更清单

| 文件 | 新建/修改 | 涉及提交 |
|---|---|---|
| `src/environment/ablation_wrappers.py` | 修改 | V10.5, V11 |
| `src/environment/rewards.py` | — | （未改动，仅诊断） |
| `src/dynamics/autopilot.py` | 修改 | Phase 3, 3.5, GainScheduler |
| `src/dynamics/flight_controller.py` | 修改 | Phase 3 |
| `src/environment/air_combat_env.py` | 修改 | Phase 3 |
| `src/environment/bfm_pursuit_env.py` | **新建** | BFM Discrete(9), Action Masking |
| `src/environment/masked_policy.py` | **新建** | Action Masking |
| `tests/test_dynamics/test_autopilot.py` | 修改 | Phase 3 |
| `tests/test_environment/test_ablation_wrappers.py` | 修改 | V10.5, V11 |
| `scripts/train_single_pursuit.py` | 修改 | V11, Phase 3 |
| `scripts/run_v10_5_training.py` | **新建** | V10.5, V11 |
| `scripts/run_ablation_study.py` | 修改 | V11 |
| `scripts/train_bfm_pursuit.py` | **新建** | BFM Discrete(9) |
| `scripts/quick_tacview.py` | **新建** | V10.5 |
| `scripts/quick_tacview_bfm.py` | **新建** | BFM Discrete(9) |
| `scripts/sweep_elevator.py` | **新建** | Phase 1 |
| `scripts/tune_roll.py` | **新建** | Phase 2 |
| `scripts/tune_pitch.py` | **新建** | Phase 2 |
| `scripts/tune_speed.py` | **新建** | Phase 2 |
| `scripts/verify_bfm_actions.py` | **新建** | Phase 4 |
| `data/trim_table.json` | **新建** | Phase 1 |

---

## Git 提交记录

```
ba127f4 feat: action masking for BFM discrete pursuit — 5 safety rules
c26eb7a feat: BFMPursuitEnv — Discrete(9) BFM action space for tactical pursuit
58a6b34 fix: revert to Phase 1 calibrated trim refs — alpha-gravity effect negligible
d03f143 feat: GainScheduler — speed-scheduled Nz gains with target-Nz integral boost
2701a48 feat: strict BFM validation metrics + Phase 3.5 PID/trim tuning
7154166 fix: generate 9 individual Tacview files with spatial tracking per BFM action
6920fc8 feat: BFMAutopilot low-level control law overhaul — 4-phase plan
40dd7c3 feat: V11 anti-collapse — V_c minimum-wage floor + ent_coef pacemaker
86a878a feat: V10.5 anti-dolphin fixes — quadratic altitude penalty + lowered V_c saturation ceiling
```

---

## 关键认识

1. **连续动作空间（Box(3)）是单机追踪平飞目标的最优方案。** V11 在训练 eval 中达到了 80% CR（虽然离线评估因策略坍塌归零）。连续控制提供精确拦截所需的微调能力。

2. **离散 BFM 动作（Discrete(9)）不适合精确追击。** 9 个固定机动指令是能量空战格斗指令，不是拦截引导。2 Hz 决策率 × 15 秒存活时间 = 30 个动作就坠毁。应留给真正的 1v1 对抗（Phase 2），届时目标是抢占角度而非精确瞄准。

3. **底层飞控改造方向正确。** Phase 1-4 的工作赋予了 autopilot 航空学常识：动态配平、重力补偿、alpha/G 限制器、增益调度。BFM 验证确认 9 个战术动作全部稳定执行。飞控本身不再是瓶颈。

4. **策略坍塌是独立于奖励函数和飞控的深层问题。** V10.5 和 V11 的修复未阻止坍塌——它仍然在 45K 步后发生。坍塌的根因可能在于 PPO 的 advantage 估计在高斯探索消退后归零，需要从算法层面（如 SAC 替代 PPO、或持续注入探索噪声）解决。

5. **严格检测基础设施已建立。** G 跟踪误差、滚转跟踪误差、震荡检测三套自动化指标可以持续监控飞控改进效果，无需人工检查 Tacview。
