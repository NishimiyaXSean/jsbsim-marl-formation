# 单机追踪训练优化 — 一日总结 (2026-06-13)

## 目标

基于 Phase 1 训练优化报告，继续完善单机追击任务的 RL 训练，在 jsbsim_rl 虚拟环境中调试并修复训练问题。

---

## 训练版本演进

| 版本 | 时间 | 种子数 | 配置 | Stage 1.0 最佳 | Stage 1.5 最佳 | 存活率 | 关键发现 |
|---|---|---|---|---|---|---|---|
| v1 (Phase 1) | 上午 | 3 | PN Expert + ent_coef=0.01 | 50% | 23% | 2/3 | clip_fraction=0 全程 |
| v3 | 下午 | 3 | Free agent + raw aileron | 43% | — | 0/3 | 策略坍缩 |
| v5 | 下午 | 3 | FC heading + ent_coef=0 + σ=1.0 | **57%** | **30%** | 2/3 | 首次真正学习 |
| v5-final | 下午 | 5 | 同 v5 × 5 seeds | **53%** | **27%** | 3/5 | 多种子验证 |
| v6 | 下午 | 3 | FC PI heading（太精确） | 30% | — | 1/3 | FC 太好反而不利 |
| v7 (MARL) | 下午 | 3 | 7类密集奖励 | 53%→0% | — | 1/3 | 密集奖励适得其反 |
| **v8** | 傍晚 | 3 | **v5 + 分段递进奖励** | **53%** | **30%** | **3/3** | 🏆 **最佳** |
| v9 | 傍晚 | 3 | v8 + 目标平滑化 | **87%** | 0% | 3→0 | 「87%是假象」 |
| v10 | 傍晚 | 3 | v9 + 课程软化 | 87% | 0% | 3→0 | 同 v9 |
| v11 | 晚上 | 3 | v10 + min 40k步/级 | 87%→0% | — | 0/3 | 训练破坏默认策略 |

---

## 已修复的 Bug（4 项确认）

### 1. `ent_coef=0.01` 淹没策略梯度 🔴

**症状**：clip_fraction=0, policy_gradient_loss≈1e-9, σ 永远=1.0

**根因**：连续控制下 entropy bonus = `0.01 × 4.26 = 0.0426` 远大于 policy_loss ≈ 1e-9。优化器被驱动去最大化熵而非改进策略。

**修复**：`ent_coef=0.0`（连续控制任务不需要 entropy bonus）

### 2. Expert 锁死速度通道 🔴

**症状**：Agent 无法控制速度，总是满油门

**根因**：`ResidualExpertWrapper._compute_expert()` 输出 `[ail, 0.0, 1.0]`：
- d_alt=0 → 高度不变
- d_spd=1.0 → 速度始终 +10m/s/决策，锁死在 250m/s 上限

**修复**：Expert 输出归零 `[0.0, 0.0, 0.0]`，Agent 获得三通道完整控制权

### 3. raw aileron 控制无法学习 🔴

**症状**：Agent 输出 raw aileron 值，F-16 中性滚转稳定性导致飞机不断翻滚

**根因**：`SinglePursuitEnv.step()` 将 action[0] 直接用作副翼值 ×0.3，绕过了 FlightController 的航向稳定器。Agent 需要学习复杂的「滚→等→反滚」时序控制。

**修复**：action[0] 改为 `d_heading`，由 FC HeadingStabilizer 自动处理滚转-转弯动力学

### 4. `log_std_init=-1` 导致策略过早坍缩 🔴

**症状**：σ=0.368 冻结，策略输出趋近零，捕获率 0%

**根因**：初始 σ 太小（0.37），策略无法充分探索。近零的输出恰好让飞机直飞，但遇到随机初始朝向偏离目标时就永远追不上。

**修复**：`log_std_init=0.0`（σ=1.0），让策略从充分探索开始

### 5. 目标机高度震荡 🟡（部分修复）

**症状**：Tacview 轨迹图中目标机高低起伏

**根因**：`_move_target()` 使用 kp=0.002 的弱 P 控制器跟踪高度

**修复**：替换为 FlightController 的 `AltitudeStabilizer`（kp=0.008 + PID）

**副作用**：目标飞得过好反而降低了训练效果（详见下文核心发现 #3）

---

## 核心发现

### 1. Policy gradient 消失的完整诊断链

```
ent_coef >> policy_loss → 熵主导优化 → σ 不收敛 → 策略永远是随机的
                                                              ↓
                                            clip_fraction = 0（策略从未改变）
```

**解决**：`ent_coef=0` + `log_std_init=0` → 策略梯度恢复 100,000×(1e-9 → 1e-4)

### 2. Stage 1.0 的 87% 是假象

v9/v10 中 Stage 1.0 的 87% 捕获率来自**默认物理行为**而非 RL 学习：
- 初始策略输出近零 → 飞机直飞 180m/s
- 目标直飞 130m/s → 速度差 50m/s
- 初始距离 800-1800m → 16-36s 自然追上
- 证据：entropy=-4.26（最大熵），clip_fraction=0（策略未变）

**这意味着 RL 训练在 Stage 1.0 阶段实际上是在破坏默认策略**，不是在学习。

### 3. 目标机震荡是一个「意外特征」

v8（弱 P 控制器，目标震荡）效果最好（3/3 存活），而 v9/v10（目标平滑飞行）全部坍缩。原因：
- 震荡目标创造了自然的探索多样性
- 目标意外靠近 pursuer 时触发 proximity bonus → 学习信号
- 平滑目标太「完美」→ agent 从未进入奖励圈 → 无信号

### 4. 密集奖励（MARL 风格）适得其反

7 类奖励（靠近率/追踪/高度/能量/末端/趋势/超时）让 agent 学会「刷分」——维持高度优势 + 机头对准就能拿分，不需要真正逼近目标。

### 5. FC 航向 PI 控制器探索记录

| 尝试 | kp | ki | kd | 输出 | 结果 |
|---|---|---|---|---|---|
| 原始 | 0.10 | 0.03 | 0.03 | ±0.50 | 10s 转 42° |
| BFM激进 | 0.15 | 0.005 | 0.06 | ±0.80 | 震荡 |
| BFM宽比例 | 0.026 | 0.005 | 0.002 | ±1.0 | 稳态误差 |
| PI外环 | 2.0(P) | 0.15(I) | — | ±1.0 | 20s 收敛但 RL 坍缩 |

**结论**：原始 P-only 控制器（稳态误差 42°）恰好提供了 RL 需要的探索随机性。太精确的控制器反而有害。

---

## 当前最可靠的 Baseline（v8）

```python
# 奖励函数
REWARD_PROGRESS = 5.0        # 距离缩短
REWARD_ATA = 5.0             # 机头指向
REWARD_SUCCESS = 500.0       # dist < 200m
REWARD_CRASH = -200.0
REWARD_LOST_TARGET = -200.0
PROXIMITY_TIERS = [(800, 25), (500, 50), (300, 100)]  # 分段递进

# PPO 超参
learning_rate = 3e-4
ent_coef = 0.0
log_std_init = 0.0
n_steps = 2048
batch_size = 256
net_arch = [128, 128]

# 动作接口
FC-based: d_heading ±30°, d_alt ±15m, d_spd ±10m/s
Free agent（Expert 中立）

# FC 航向（原始 P-only）
ROLL_PER_DEG_HEADING = 2.5
Roll PID: kp=0.10, ki=0.03, kd=0.03, output=±0.50
```

**结果**：3/3 存活，Stage 1.0 最高 53%，Stage 1.5 最高 30%

---

## 最大教训

1. **简单 > 复杂**：v5 的简单奖励（3 项 + 1 个终端）效果远好于 v7 的 7 类密集奖励
2. **Bug 可能是 Feature**：目标机震荡在 RL 探索中起了正面作用
3. **先验证默认策略**：87% 捕获率是「飞直线」的默认行为，不代表学会了
4. **RL 对超参极度敏感**：`ent_coef` 从 0.01 到 0.0，改变了 100,000× 的梯度信号
5. **太精确的控制器不利于 RL**：FC PI 比 P-only 在航向跟踪上更好，但损害了 RL 探索

---

## 下一步建议

1. **延长训练**：200k → 500k 步，给策略更多时间收敛
2. **增加课程密度**：5 级 → 7-8 级，平滑难度梯度
3. **解决 Stage 1.0 假学习问题**：增加 Stage 1.0 目标速度或初始距离
4. **探索自适应 σ**：随训练步数逐渐降低 log_std
5. **尝试 MAPPO 1v1**：当前单机框架稳定后，迁移到多机对抗训练

---

## 修改的文件

| 文件 | 改动 |
|---|---|
| `scripts/train_single_pursuit.py` | ResidualExpertWrapper 中立化、ent_coef=0、log_std_init、lr=3e-4、MIN_STEPS_PER_STAGE |
| `src/environment/single_pursuit_env.py` | 动作接口 FC化、初始速度 180m/s、分段递进奖励、目标高度控制器、出生对齐 |
| `src/dynamics/flight_controller.py` | 航向控制器多次调参（最终回到接近原始值） |
| `scripts/diagnose_dynamics.py` | 新增：飞机动力学与控制链诊断脚本 |
| `scripts/eval_multi_seed.py` | 新增：多种子 Wilson CI 评估脚本 |

## 生成的输出

- `results/single_pursuit/single_pursuit_engagement.txt.acmi` — Tacview 轨迹
- `results/single_pursuit/single_pursuit_trajectory_best.png` — 3D 轨迹图
- `results/phase1a_v8_eval.csv` — v8 评估报告
- `marl_runs/single_pursuit_0613_1814_s{0,1,2}/` — v8 最佳模型
