# 2026-06-23 工作总结：V7→V10 训练迭代

## 执行摘要

今日完成了从 V7 分析到 V10 实施的 5 个主要版本迭代，共 **8 次代码提交**，涉及 **5 个文件**。发现并修复了 1 个关键 Bug。核心问题——**飞机海豚跳（porpoising）**——最终未被完全解决，需要在架构层面重新考虑 V_c 耦合奖励机制。

---

## 版本迭代历史

### V7 分析 (@ 9a5e2a8)
对 `ablation_0619_1241` 进行了完整的训练后分析，关键发现：
- 峰值捕获率 84.4%，但所有 seed 在 step ~45K 后断崖崩溃
- **approx_kl 在 ~200K 后归零** → 策略永久冻结
- 线性 V_c 耦合 `V_c/50` 在 V_c=15 m/s 时仍给予 30% guidance → agent 学会低速漂移
- 静态滚油区 (800m) → agent 在 801m 处巡航
- `CONSECUTIVE_ADVANCE_REQUIRED=2` 从未满足 → 课程无法推进

### V8 (1 commit: `7657d9b`)
**目标：打破 V7 的局部最优陷阱**

| 文件 | 改动 |
|:---|:---|
| `ablation_wrappers.py` | Sigmoid V_c: `1/(1+e^(-0.2*(V_c-25)))` |
| `single_pursuit_env.py` | 动态滚油区: `800 + d*400` m |
| `train_single_pursuit.py` | `ent_coef=0.03`, `sde_sample_freq=2`, `CONSECUTIVE=1`, `TOTAL_TIMESTEPS=5M` |
| `run_ablation_study.py` | 同上 |

**结果**: 训练 eval 显示 77% 捕获率 @ diff=0.15，但离线评估得 **0%**。模型过拟合了 eval 环境。

### V9 (1 commit: `2c84ce9`)
**目标：将探索从"帕金森抖动"改为"人类驾驶节奏"**

| 参数 | V7 | V8 | V9 |
|:---|---:|---:|---:|
| `sde_sample_freq` | 4 | 2 | **10** (1.0s) |
| `ent_coef` | 0.01 | 0.03 | **0.015** |
| `SMOOTHNESS_WEIGHT` | 2.0 | 2.0 | **4.0** |

**结果**: 仍然 0% 捕获率，海豚跳持续。确认问题不在探索参数。

### 诊断阶段
运行了 4 项控制链稳定性测试：

| 测试 | 结果 | 关键发现 |
|:---|:---|:---|
| BFMAutopilot 平飞 | ❌ 10秒掉高 **-391m**，滚转 **180°** | PID 符号映射或增益错误 |
| FlightController 静态保持 | ✅ -4.8m/10s | FC 自身稳定 |
| **10Hz RL 抖动** | ⚠️ 滚转 **70°**，标准差 22° | **帕金森效应确认** |
| 决策频率分析 | ⚠️ RL 10Hz vs 飞行员 1-2Hz | 决策频率过高 |

**关键洞察**: 问题不在 PID，而在 RL 每 0.1 秒改一次主意，FC 只有 6 个微步来响应。

### V10 (4 commits: `14b56df`, `ac30130`, `143b7d5`, `4cee8a4`)
**目标：消灭海豚跳的多层防御**

```
防御层 1: ActionRepeatWrapper (2Hz 决策，5 帧重复)
   ├─ RL 每 0.5s 决策一次，FC 有 30 个微步执行
   └─ 文件: ablation_wrappers.py + build_env() + train()

防御层 2: 动作幅度惩罚 r_action_mag = -1.0 × |a|²
   ├─ 惩罚不必要的操纵输入
   └─ 文件: ablation_wrappers.py (LeadPursuitRewardWrapper)

防御层 3: 垂直速度惩罚 r_vz = -15.0 × |V_z/50|
   ├─ 俯冲加速直接扣分
   └─ 文件: ablation_wrappers.py

防御层 4: 高度差惩罚 r_alt_delta = -20.0 × |Δalt|/1000
   ├─ 强制共面飞行
   └─ 文件: ablation_wrappers.py

防御层 5: 按难度保存模型
   ├─ best_model_diff_X.XX.zip + entry_model_diff_X.XX.zip
   └─ 文件: train_single_pursuit.py (AutoCurriculumCallback)
```

**结果**: 训练 eval 显示 **100% 捕获率 (30/30)** @ diff=0.15, **77%** @ diff=0.20。这是训练历史上首次达到 30/30。

### 关键 Bug 发现 (`aa0c397`)
`ActionRepeatWrapper` 没有显式 `difficulty_level` 属性委托。Python 的 `eval_env.difficulty_level = 0.20` 创建了实例属性在 wrapper 上，**未传递到 SinglePursuitEnv**。

**影响**: 所有 eval 实际运行在 `difficulty=0.0`（900-1300m 直线平飞目标），而非显示的 diff=0.15/0.20。100% 捕获率是对静止目标的，不是对抗机动目标。

---

## 核心未解决问题：海豚跳 (Porpoising)

### 物理机制
```
飞机爬升 → 获得高度优势
    ↓
转为俯冲 → 重力势能 → 动能 → V_c 飙升
    ↓
sigmoid V_c_norm 从 0.5→0.99
    ↓
所有 guidance 奖励暴增 (r_lead_vel + r_lead_pred + r_los_rate)
    ↓
飞机爬升 → 重新获得高度 → 循环往复
    ↓
结果: 500-1000m 振幅的简谐运动
```

### 已尝试的修复
1. **Sigmoid V_c** (V8) — 降低了低速区耦合，但高速俯冲仍能骗到满额 guidance
2. **ActionRepeat 2Hz** (V10) — 滑了轨迹但仍不能阻止策略选择俯冲
3. **动作幅度惩罚** (V10) — 确实让动作更平滑 (`r_smoothness ≈ 1e-7`)，但策略仍选择周期性俯冲-爬升
4. **垂直速度惩罚** (V10) — 直接惩罚 V_z，但 guidance 收益 (+30~50/step 额外) 可以覆盖惩罚 (-7.5~15/step)
5. **高度差惩罚** (V10) — 惩罚偏离目标高度，同上被 guidance 收益覆盖

### 为什么这些防御层都失败了
V_c 耦合奖励的 **收益上限太高** —— 一次成功的俯冲可以额外产出 ~240 点 guidance reward，而所有惩罚的总和只有 ~150。**数学上，俯冲-爬升循环是正期望收益的。**

### 可能的解决方向
1. **将 V_c 耦合改为纯乘法门控**：`V_c < 30 → 0, V_c ≥ 30 → 1`（二值化），消除中间梯度的"部分奖励"
2. **惩罚 V_c 的剧烈变化**：`-k * |dV_c/dt|`，惩罚"通过改变俯仰来加速"的行为
3. **能量惩罚**：`-k * |ΔE|`（势能+动能变化），直接堵住势能→动能转换的漏洞
4. **完全移除 V_c 耦合**，改用纯距离倒数奖励 + ATA alignment（回到更传统的 guidance reward）

---

## 文件修改清单

| 文件 | V8 | V9 | V10 | BugFix | 总改动 |
|:---|:---|:---|:---|:---|:---|
| `src/environment/ablation_wrappers.py` | Sigmoid V_c | SMOOTHNESS=4.0 | ActionRepeat + action_mag + VZ + alt_delta | difficulty delegation | 核心文件 |
| `src/environment/single_pursuit_env.py` | 动态 ZONE_DEATH | — | — | — | 环境层 |
| `scripts/train_single_pursuit.py` | ent_coef, sde_freq, CONSEC, steps | sde_freq=10, ent=0.015 | ActionRepeat in train() | per-difficulty checkpoint | 训练+课程 |
| `scripts/run_ablation_study.py` | ent_coef, sde_freq, steps | sde_freq=10, ent=0.015 | ActionRepeat in build_env() | — | 实验运行 |
| `src/dynamics/autopilot.py` | — | P-only (ki=kd=0) | — | — | 预防性修复 |
| `src/environment/__init__.py` | — | — | ActionRepeat 导出 | — | 包管理 |
| `tests/test_environment/test_ablation_wrappers.py` | sigmoid 测试 | — | — | — | 14/14 pass |

---

## 产出文件

| 路径 | 内容 |
|:---|:---|
| `marl_runs/ablation_0619_1241/V7_TRAINING_ANALYSIS.md` | V7 完整分析报告 |
| `results/v10_s0_eval/` | V10 best_model diff=0.15 Tacview + 轨迹图 |
| `results/v10_s0_diff020/` | V10 diff=0.20 Tacview + 轨迹图 |
| `results/v10_s0_diff020_best/` | V10 best_model_diff_0.20 离线评估 |
| `marl_runs/ablation_0623_*/` | 5 轮完整训练产出 (1058, 1449, 1640, 1758, 1940) |

---

## Git 提交记录

```
aa0c397 fix: ActionRepeatWrapper blocks difficulty_level property delegation
4cee8a4 feat: per-difficulty model checkpointing  
143b7d5 feat: V10 — altitude delta penalty + normalised vertical velocity penalty
ac30130 feat: V10 — action magnitude penalty + vertical velocity penalty
14b56df feat: V10 — ActionRepeatWrapper (2Hz decisions) + BFMAutopilot P-only stabilisation
2c84ce9 feat: V9 — aerodynamic exploration rhythm + smoothness enforcement
7657d9b feat: V8 training improvements — sigmoid V_c coupling + dynamic zone-of-death + exploration tuning
```

---

## 结论

今天的工作系统性地排查了训练失败的多个潜在原因——探索参数、决策频率、飞控 PID、奖励漏洞。虽然 100% 捕获率的突破被 Bug 冲淡，但 **V10 的多层防御框架（ActionRepeat + action_mag + VZ + alt_delta）是正确的方向**。海豚跳的根因是 V_c 耦合奖励的结构性缺陷——它天然奖励"势能→动能→guidance"的物理转换。下一轮迭代应聚焦于**重新设计 V_c 在奖励函数中的角色**，从可微分耦合改为硬门控，或引入能量守恒惩罚。
