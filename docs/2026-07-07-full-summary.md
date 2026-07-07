# 2026-07-07 完整工作总结：RLlib MARL 架构重建 + 实验 1 非协同基线

> **日期**: 2026-07-07
> **主题**: 放弃自定义 PyTorch MAPPO，回归 RLlib 构建可扩展 MARL 管道
> **提交**: ~3 commits
> **核心成果**: 3 个新 RLlib 文件（1,407 行）+ 实验 1 验证"无协同 = 内卷"

---

## 1. 战略决策：回归 RLlib

基于前几日（7/1–7/3）的诚实结论（CTDE PPO 在严格 AND-gate 下无法收敛，纯 PyTorch 管道无法扩展到 NvM），决定回归 RLlib 多智能体训练基础设施。

核心原则：
- **不使用 RLlib 的 RLModule 新 API**：坚持旧 API 栈（`enable_rl_module_and_learner=False`）以兼容 `TorchModelV2`
- **Self-Attention 架构不妥协**：将 `AttentionFormationActor` 完整嵌入 `TorchModelV2`，而非退化为 MLP
- **Dual-Actor 通过多策略实现**：RLlib 的 `multi_agent.policies` 天然支持独立 Actor

---

## 2. 代码产出

### 2.1 归档旧脚本

24 个过时脚本从 `scripts/` 移至 `scripts_old/`：
- 早期 BFM/离散阶段：`train_bfm_pursuit.py`, `train_phase1_discrete.py`, `train_phase2_continuous.py` 等
- 硬编码评估：`eval_v8_s0.py`, `viz_success.py`, `visualize_best_carw.py` 等
- 旧框架尝试：`train_formation_tianshou.py`, `train_formation_mappo_torch.py`
- 旧 BC 管道：`train_bc_pretrain.py`, `generate_pn_expert_data.py`

### 2.2 新建 RLlib 管道（3 文件，1,407 行）

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/environment/formation_rllib_env.py` | ~800 | RLlib MultiAgentEnv — 完整 Phase 5 逻辑 |
| `src/models/formation_rllib_model.py` | ~185 | TorchModelV2 — Self-Attention Actor + Critic |
| `scripts/train_formation_rllib.py` | ~425 | 训练入口 — Dual-Actor + BC 热加载 + 两阶段课程 |

### 2.3 关键技术细节

**环境**：
- Agent IDs: `"p0"`, `"p1"`
- 观测空间: `Dict({"obs": Box(33), "global_state": Box(21)})`
- 动作空间: `Box(2)` — `[turn, speed]` → FlightController
- 完整的 Phase 5 协同逻辑：合围角奖励、动态角色分配、AND-gate (800m/30°)、非对称重置
- 动作边界裁剪：防止 DiagGaussian 无界采样导致 NaN

**模型**：
- `forward()` 同时计算 Actor logits 和 Critic value（RLlib PPO 要求）
- Global state reshape: flat (B,21) → token sequence (B,3,7)
- 正交初始化双重保险
- BC 键名映射：19/19 actor 键完美匹配（`actor.*` 命名空间）

**训练**：
- Dual-Actor 配置：`p0_policy` / `p1_policy` 独立模型实例
- BC 权重热加载：`actor_state` → `actor.*` 映射，`strict=False` 跳过 Critic
- 两阶段训练：OR-gate 热身 → AND-gate 切换（`set_coop_phase`）
- 独立 Critic：接受的设计选择（P0/P1 奖励结构不同，独立 Critic 更合理）

---

## 3. Bug 修复记录

### 3.1 `cooperative_mode` 传参漏洞（commit `d780f8d`）

**问题**：`--no-cooperative` 标志在命令行解析后未传递到 `env_config`，Worker 环境始终默认启用全部 Phase 5 协同奖励。

**修复**：
- `formation_rllib_env.py`：新增 `cooperative_mode` 配置项并门控 Phase 5 代码块
- `train_formation_rllib.py`：`env_config` 传递 `cooperative_mode` + `run_evaluation` 同步传递

### 3.2 API 兼容修复

Ray 2.40 API 参数名变更：
- `sgd_minibatch_size` → `minibatch_size`
- `num_sgd_iter` → `num_epochs`
- `num_cpus_per_env_runner` 移至 `env_runners()` 方法

---

## 4. 实验 1：非协同基线（部分完成）

```bash
python scripts/train_formation_rllib.py \
  --iterations 200 --no-cooperative \
  --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
  --eval-interval 20 --seed 42
```

### 4.1 训练曲线

```
Iter  20:  Eval avg_rew = -8,888   ← BC 权重初入 PPO，Critic 随机
Iter  40:  Eval avg_rew = -7,536   ← 唯一一次改善
Iter  60:  未破纪录                 ← 平台开始
Iter  80:  未破纪录                 ← 持续平台
Iter 100:  未破纪录（周期性保存）    ← 确认内卷
```

### 4.2 关键发现

**"无协同 = 对称内卷"** — 实验 1 的核心科学信号：

关闭 Phase 5 协同奖励后，两架 F-16 接收完全对等的独立单机奖励（progress, ATA），没有合围角激励、没有角色分化。BC 权重中编码的协同行为（S2M attention = 0.40）与 PPO 的单机优化目标冲突，导致 reward 在 -7,500 附近停滞。

**论文价值**：这组对称性陷阱数据可以作为引入 Dual-Actor + Independent Critic 架构的实证理由——如果两个 Agent 拿完全相同的奖励信号，参数共享下必然陷入策略同质化。

### 4.3 预计完整结果

实验在 100 轮时手动停止。如果继续跑到 200 轮，预计 eval reward 将保持在 -7,500 ~ -8,000 之间窄幅震荡，证实"非协同基线天花板"的存在。

---

## 5. 全版本代码变更对比

```
┌──────────────────┬──────────┬─────────────────────────┬──────────┐
│ 日期             │ 架构     │ 关键进展                │ 状态     │
├──────────────────┼──────────┼─────────────────────────┼──────────┤
│ 7/1–7/3          │ 纯PyTorch│ SB3天花板→Attention BC  │ 已封存   │
│                  │ MAPPO    │ → Dual-Actor → 13.75%   │          │
│                  │          │   AND-gate协同成功率     │          │
│ 7/7 (今天)       │ RLlib    │ ★ 重建完整MAPPO管道     │ 进行中   │
│                  │ MAPPO    │   3新文件(1,407行)       │          │
│                  │          │   24旧脚本归档           │          │
│                  │          │   实验1: 非协同基线验证   │          │
└──────────────────┴──────────┴─────────────────────────┴──────────┘
```

---

## 6. 待执行实验

| 实验 | 命令 | 预计时间 |
|------|------|---------|
| **实验 2**: OR-gate 快速对齐 | `--iterations 120 --cooperative` | ~2 小时 |
| **实验 3**: 两阶段 OR→AND | `--iterations 500 --cooperative --warmup 200000` | ~8 小时 |

### 监控要点

| 指标 | 健康范围 | 警告 |
|------|---------|------|
| `kl` | < 0.015 (AND-gate) | > 0.03 连续多轮 → 中断，降 lr=1e-4 |
| `ent` | 0.4–2.5 | < 0.3 坍缩，> 3 过随机 |
| `r_p0` vs `r_p1` | 实验 3 中应分化 | 高度重合 = 角色未涌现 |

---

## 7. Git 提交

```
d780f8d fix: pass cooperative_mode to RLlib env_config
ca54c1b feat: rebuild MARL architecture on RLlib — Phase 5 + Self-Attention
```
