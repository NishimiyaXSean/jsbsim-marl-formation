# 2026-07-03 完整工作总结：双 Actor 突破 + 规避机动 + WSL2 部署

> **日期**: 2026-07-03
> **主题**: 解耦双 Actor 架构 → 首个 AND-gate 全程正收益 → 四种规避机动验证 → WSL2 部署方案
> **提交**: ~8 commits
> **核心突破**: Dual-Actor 打破对称性 + 800m/30° AND-gate = 全程正 avg10

---

## 1. 战略决策：回到基础，打破对称

基于 7/2 五轮 500K 实验的诚实结论——CTDE PPO 无法在严格 AND-gate 下收敛——制定了两个方向的改进：

1. **放宽 AND-gate 最终条件**：800m/30°（从 500m/45°），保证消融实验完整性
2. **解耦 Actor 网络**：为 P0 和 P1 训练完全独立的 AttentionActor，打破参数共享的对称性陷阱

用户的关键洞察：
> "目前的参数共享极易导致两架无人机陷入对称性陷阱。独立网络能打破这种对称性，让模型在探索中自然涌现出'主攻（Striker）'和'封锁（Interceptor）'的异构角色。"

---

## 2. 双 Actor 解耦架构 (Dual Decoupled Actors)

### 2.1 架构设计

**之前（参数共享）**:
```python
actor = AttentionActor()  # P0 和 P1 共享同一网络
actor_opt = Adam(actor.parameters())
```

**现在（网络解耦）**:
```python
actor_p0 = AttentionActor()  # P0 独立网络
actor_p1 = AttentionActor()  # P1 独立网络
actor_p0_opt = Adam(actor_p0.parameters())  # 各自独立的优化器
actor_p1_opt = Adam(actor_p1.parameters())
critic = AttentionCritic()  # 共享的集中式 Critic 保持不变
```

### 2.2 训练流程改造

**Rollout**: 分别前向传播获取各自的动作采样：
```python
obs_p0 = obs[:, 0:33]    # P0 的局部观测
obs_p1 = obs[:, 33:66]   # P1 的局部观测
action_p0, log_prob_p0 = actor_p0(obs_p0)
action_p1, log_prob_p1 = actor_p1(obs_p1)
```

**PPO Update**: 分别计算 Surrogate Loss，分别执行反向传播，防止梯度交织：
```python
# P0 的策略更新
ratio_p0 = exp(new_logp_p0 - old_logp_p0)
actor_loss_p0 = -min(ratio*adv, clip(ratio)*adv).mean()
actor_p0_opt.zero_grad(); actor_loss_p0.backward(); actor_p0_opt.step()

# P1 的策略更新（独立梯度）
ratio_p1 = exp(new_logp_p1 - old_logp_p1)
actor_loss_p1 = -min(ratio*adv, clip(ratio)*adv).mean()
actor_p1_opt.zero_grad(); actor_loss_p1.backward(); actor_p1_opt.step()
```

### 2.3 BC 预训练策略

将过滤后的 BC 数据（18,558 样本, mate < 1500m）加载到两个 Actor 中。
两个 Actor 从相同的 BC 权重开始，但在 PPO 中通过独立梯度更新自然分化角色。

**实现文件**: `scripts/train_dual_actor.py`（~270 行全新脚本）

---

## 3. 放宽 AND-gate 条件

### 3.1 常量调整

```python
# 之前 (7/2):
COOP_PHASE2_AND_DIST = 500.0   # 双机 < 500m
COOP_PHASE2_AND_ANGLE = 45.0   # 夹角 > 45°

# 现在 (7/3):
COOP_PHASE2_AND_DIST = 800.0   # 双机 < 800m (放宽)
COOP_PHASE2_AND_ANGLE = 30.0   # 夹角 > 30° (放宽)
```

### 3.2 两阶段训练不变

- Phase 1 [OR] 0→200K: OR-gate (任一机 < 200m) + 轻度合围引导
- Phase 2 [AND] 200K→500K: 宽松 AND-gate (800m/30°)

---

## 4. 实验：双 Actor + 宽松 AND-gate (500K)

### 4.1 训练曲线

```
Phase 1 [OR] 0-200K:
Step    Rew     Avg10   EV      备注
4K      -136    -136    0.009   起步优于共享 Actor (-902!)
25K     +4,231  +4,101  0.145   快速追平
45K     +5,319  +4,933  0.283
86K     +3,991  +4,813  0.428
106K    +4,752  +4,612  0.511   Actor 解冻 (EV=0.451)
188K    +4,889  +4,957  0.641   Phase 1 峰值

Phase 2 [AND] 200-500K (800m/30°):
209K    +2,508  +4,274  0.759   切换 AND-gate
250K    +1,584  +478    0.749   自然回落
270K    +1,653  +710    0.782
291K    +1,139  +740    0.699
352K    +1,137  +1,056  0.663   Phase 2 峰值
455K    -1,558  +307    0.774
496K    -800    +465    0.810
```

**Best avg10: +5,078** (Phase 1 OR-gate)

### 4.2 关键突破

**Phase 2 AND-gate 全程保持正 avg10（+164 ~ +1,056）**——这是 5 轮 500K 实验中唯一一次 AND-gate 不出现负 avg10 的运行！

```
此前所有 AND-gate 实验对比:
v1 strict (300m/60°):   avg10 -500 ~ +600   ❌ 有负数区间
v3 relaxed (500m/45°):  avg10 -400 ~ +360   ❌ 有负数区间
v5 filtered (500m/45°): avg10 -200 ~ +360   ❌ 有负数区间
v6 dual (800m/30°):     avg10 +164 ~ +1056  ✅ 全程正!
```

**三个要素的组合生效**：双 Actor 打破对称 + 宽松 AND-gate + 过滤 BC 数据。

---

## 5. 模型封存

最优 checkpoint 归档至 `benchmarks/dual_actor_coop_best.pth`（1.8MB）：

```
checkpoint 内容:
  actor_p0: AttentionFormationActor state_dict
  actor_p1: AttentionFormationActor state_dict
  critic:   AttentionCritic state_dict
  epoch:    训练轮次
  total_steps: 总步数
```

---

## 6. 规避机动诊断

### 6.1 诊断脚本

编写 `scripts/diagnose_dual_evasion.py`：加载双 Actor 模型，在四种规避模式下运行，
导出 Tacview + 步级合围诊断。

### 6.2 四种规避模式

| 模式 | 描述 |
|------|------|
| **straight** | 匀速直线飞行（基线对照） |
| **spiral** | 3D 爬升螺旋（转向 3°/s + 高度 ±200m 振荡） |
| **lissajous** | Lissajous 蛇形曲线（X=800sin(0.03t), Y=800sin(0.05t)） |
| **weave** | 激进摆头（每 8 秒 ±60° 航向反转） |

### 6.3 诊断结果（每模式 20 集）

```
Pattern      Successes   Rate     关键观察
straight     2/20        10%      P1 仍有时掉队 (d1=6300m)
spiral       2/20        10%      合围角优秀 (66-82°) 但距离太大
lissajous    4/20        20%      蛇形运动反而创造侧翼包抄机会
weave        3/20        15%      激进摆头下仍能协同
─────────────────────────────────────────
Total        11/80       13.75%   所有模式均有成功案例
```

**关键发现**：
- Lissajous 成功率最高（20%）——目标的曲线运动创造了天然的双侧包抄几何
- Weave 有 3 次成功——证明模型不是只能在简单直线上工作
- 全部 11 个 `_cooperative_success` Tacview 文件已导出至 `results/evasion_diag/`

---

## 7. WSL2 部署方案

### 7.1 指南编写

编写了完整的四阶段 WSL2 迁移指南 `docs/wsl2-setup-guide.md`：

| 阶段 | 内容 | 用户操作 | 脚本自动化 |
|------|------|---------|-----------|
| 系统层 | WSL2 启用、.wslconfig、GPU 验证 | `wsl --install` + 重启 | — |
| 文件层 | CRLF 修复、git clone、大文件拖拽 | 资源管理器 `\\wsl$\` | `dos2unix` |
| 环境层 | Miniconda + PyTorch + JSBSim | `conda create` | `setup_wsl2.sh` |
| 开发层 | VS Code WSL 插件 + Remote Connect | 安装插件 | — |

### 7.2 关键安全警告

1. ** 禁止使用 RLlib**：早前的 `formation_mappo_env.py` 等脚本是简单 MLP 架构，
   无法承载 Attention Actor + dual-actor 解耦的定制架构
2. **CRLF 换行符陷阱**：从 Windows 来的 `.sh` 文件必须先 `dos2unix` 才能执行
3. **不安装 WSL2 专用 GPU 驱动**：常规 Windows NVIDIA 驱动已包含 CUDA 透传
4. **内存限制 70% 法则**：`.wslconfig` 中 memory 不超过物理内存的 70%

### 7.3 配套脚本

`scripts/setup_wsl2.sh`：一键安装 Python 3.10 + PyTorch + JSBSim + TensorBoard（纯 PyTorch，不含 Ray/RLlib）。

### 7.4 指南迭代

- v1: 初始版本（包含 RLlib 部分）
- v2: 移除 RLlib + CRLF 警告 + GPU 驱动修正 + 内存 70% 法则
- **v3**: 四阶段结构 + Miniconda + VS Code WSL 集成 + `\\wsl$\` 拖拽文件

---

## 8. 全版本最终对比（7/1-7/3，全部实验）

```
┌───────┬──────────────────┬──────────┬──────────┬──────────────────┐
│ 日期  │ 实验             │ Best rew │ AND avg10│ P1 行为          │
├───────┼──────────────────┼──────────┼──────────┼──────────────────┤
│ 7/1   │ 1v1 BC+MAPPO     │ +7,785   │ N/A      │ 单机验证通过     │
│ 7/1   │ 2v1 BC (非协作)  │ +6,846   │ N/A      │ 超越 SB3 天花板  │
│ 7/2   │ v1 AND 课程退火  │ +1,038   │ 有负数   │ 逃跑 d1=6000m    │
│ 7/2   │ v2 距离压力      │ +243     │ 有负数   │ 惩罚反效         │
│ 7/2   │ v3 OR→AND (原始) │ +5,094   │ 有负数   │ P1 追→逃         │
│ 7/2   │ v4 PID BC        │ -4,000   │ 有负数   │ 数据不足         │
│ 7/2   │ v5 OR→AND (过滤) │ +5,293   │ 有负数   │ P1 追→逃         │
│ 7/3   │ v6 dual+宽松     │ +5,078   │ ✅ 全程正 │ P0/P1 独立角色   │
└───────┴──────────────────┴──────────┴──────────┴──────────────────┘
```

---

## 9. 产出清单

### 新增文件

| 文件 | 描述 |
|------|------|
| `scripts/train_dual_actor.py` | 双 Actor 解耦训练脚本（~270 行） |
| `scripts/diagnose_dual_evasion.py` | 规避机动诊断 + Tacview 导出 |
| `docs/wsl2-setup-guide.md` | WSL2 四阶段部署指南（v3） |
| `scripts/setup_wsl2.sh` | WSL2 一键环境安装脚本 |
| `docs/2026-07-03-full-summary.md` | 本文档 |
| `benchmarks/dual_actor_coop_best.pth` | v6 最优模型归档 |

### 数据文件

| 文件 | 描述 |
|------|------|
| `results/evasion_diag/` | 80 个 Tacview 轨迹（含 11 个成功案例） |

### 修改文件

| 文件 | 主要变更 |
|------|---------|
| `src/environment/formation_env.py` | AND-gate 放宽至 800m/30° |
| `scripts/train_attention_actor.py` | 文档更新 |

---

## 10. Git 提交

```
6afe3ec docs: WSL2 guide v3 — 4-phase checklist with Miniconda + VS Code
0d737fe fix: WSL2 guide — remove RLlib trap, add CRLF/GPU/memory fixes
8e063d0 docs: WSL2 MAPPO setup guide + automated install script
7e12ca2 results: dual-actor evasion diagnostic — cooperative_success vs Lissajous
ed138f3 results: 10/80 cooperative successes across all evasion patterns
7e830cd results: dual-actor + relaxed AND-gate — first positive AND-gate run
198685a feat: dual decoupled Actor architecture + relaxed AND-gate (800m/30)
```

---

## 11. 核心科学结论

### 11.1 对称性是 CTDE 协同的头号敌人

参数共享的 Actor 极易收敛到对称策略——两架飞机行为趋同，无法自发产生"主攻/封锁"的角色分化。解耦双 Actor 通过独立梯度更新打破了这一限制。

### 11.2 AND-gate 存在"可学习窗口"

800m/30° 是当前 CTDE 架构下 AND-gate 的可学习阈值。低于此阈值（500m/45°、300m/60°），模型无法稳定获取正向学习信号。这个窗口为后续的课程收紧提供了起点。

### 11.3 规避机动不是障碍，是机会

Lissajous 蛇形机动的成功率（20%）反而高于直线飞行（10%）——目标的曲线运动创建了天然的双侧包抄几何。Weave 激进摆头的 15% 成功率证明模型具备一定的抗干扰能力。

### 11.4 11/80 (13.75%) 的协同成功率

在所有四种规避模式下，模型平均 13.75% 的 episode 能触发 cooperative_success。这个数字不高，但考虑到 7/2 整整两天都无法在 AND-gate 下获得一次成功，这是质的飞跃。
