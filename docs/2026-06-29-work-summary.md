# 2026-06-29 工作总结：2v1 编队追猎 + MAPPO 架构迁移

> **日期**: 2026-06-29  
> **提交**: 20 commits  
> **主题**: 从单机 SOTA 到 2v1 编队协同, RLlib MAPPO 迁移

---

## 目录

1. [Phase 3.5：大规模自我蒸馏](#1-phase-35大规模自我蒸馏)
2. [Phase 3.6：动作平滑微调](#2-phase-36动作平滑微调)
3. [Phase 4：2v1 编队追猎](#3-phase-42v1-编队追猎)
4. [Phase 4.1：分段编队间距](#4-phase-41分段编队间距)
5. [Phase 5：RLlib MAPPO 迁移](#5-phase-5rllib-mappo-迁移)
6. [Tacview 导出修复](#6-tacview-导出修复)
7. [Ray Windows 启动故障](#7-ray-windows-启动故障)
8. [全版本对比](#8-全版本对比)
9. [提交清单](#9-提交清单)

---

## 1. Phase 3.5：大规模自我蒸馏

### 策略

用 Phase 3v3 最佳模型在 diff=0.55–0.70 下收集自身成功轨迹 (DAgger 风格), 合并 PN 专家数据, 从 diff=0.0 重启 PPO。

### 数据集

| 来源 | 轨迹数 | 步数 | 难度范围 |
|------|--------|------|---------|
| PN 专家 | 800 | 89,289 | 0.20–0.50 |
| Self-play v1 | 200 | 18,534 | 0.50–0.65 |
| Self-play v2 | 500 | 48,084 | 0.55–0.70 |
| **合计** | **1,500** | **155,907** | 0.20–0.70 |

### 结果

```
1M 步, diff=0.0 起步, BC 热启动

Success: 16.7% @ 10K → 81.3% @ 1M (diff=0.58)
Peak: 76.2% @ 900K (diff=0.52)

vs Phase 3v3 (50.0%): +31pp success
vs Phase 3v3 (-2,790 ep_rew): +3,434 ep_rew (翻正!)
```

**关键突破**: 自我蒸馏让 Agent 在 diff=0.58 下的 timeout 率从 50% 降到 18.8%。OOD 盲区大幅缩小。

---

## 2. Phase 3.6：动作平滑微调

### 问题

Phase 3.5 模型存在 68% 的转弯方向切换率 (PN dithering 残留), 轨迹有可见锯齿。

### 方案

- `SMOOTHNESS_WEIGHT` 4→8, 配合课程退火 (0–50K: w=2, 50K–150K: 2→8)
- 从 Phase 3.5 权重热启动, 300K 步微调

### 结果

```
300K 步, diff=0.0

Final: 83.0% @ diff=0.20 (300K 限制, 未推进到更高难度)
Sign changes: 79% → 72% (-7pp)
FC actual_std: 0.6 dps (持续压制)
```

**结论**: 平滑惩罚部分缓解了 dithering, 但 BC 继承的模式需要更大训练预算才能根除。FC 的低通滤波效果使实际飞行轨迹振幅控制在可接受水平。

---

## 3. Phase 4：2v1 编队追猎

### 架构

```
FormationEnv (SB3 共享策略):
  Obs: Box(66) = [P0 33-dim | P1 33-dim]
  Act: Box(4) = [turn_p0, spd_p0, turn_p1, spd_p1]
  每架: 独立 Aircraft + FlightController (Box(2))
  队友观测: 6 维 (相对位置 + 相对速度, body frame)
```

### 权重平铺

Phase 3.6 [256,33]→[256,66] 对称复制, 动作输出 [2,256]→[4,256] 复制。Actor 冻结前 50K 步让 Critic 先行收敛。

### 分阶段训练

| 阶段 | 步数 | Actor | Critic | 结果 |
|------|------|-------|--------|------|
| S1 | 0–50K | 冻结 | LR=5e-4 | success=3%, Critic explained_var=0.96 |
| S2 | 50K–150K | 解冻 LR=1e-5 | LR=5e-4 | success 2%→93% (爆发) |
| S3 | 150K–200K | 全训练 | LR=5e-4 | **92.5% success** |

**关键**: 分阶段训练防止了 "奖励休克"——如果直接解冻, Critic 的随机初始化会洗掉 Phase 3.6 的 Actor 权重。

---

## 4. Phase 4.1：分段编队间距

### 分段间距奖励

```
d < 50m:   危险区, 固定惩罚 -5.0/dt
50–200m:    Coulomb 排斥缓冲区, 惩罚线性递减
200–500m:   理想协同区, 小正奖励 (+2.0/dt cap)
            ⚠️ 门控: 仅当双方都在闭合目标时发放
d > 500m:   太分散, 无奖励
```

### 权重退火

```
0–50K:    w=0      (保护 Phase 4 成功率)
50K–150K: w=0→1   (线性 ramp)
150K+:    w=1.0    (full enforcement)
```

### 结果

| 指标 | Phase 4 (无 spacing) | Phase 4.1 | 变化 |
|------|---------------------|-----------|------|
| Success | 92.5% | **97.3%** | +5pp |
| Collision | 5.8% | **2.2%** | **-62%** |
| ep_rew | ~+2,500 | **~+6,500** | +160% |

Collision 从 8 次/10K 步 (w=1.0 初期) 降到 1 次 (终值), Coulomb 排斥力生效。

---

## 5. Phase 5：RLlib MAPPO 迁移

### 交付物

| 文件 | 内容 |
|------|------|
| `formation_mappo_env.py` | MultiAgentEnv, 2 agents, per-agent Box(2) + Dict obs |
| `formation_mappo_model.py` | CTDE TorchModelV2, Actor(33→2), Critic(21→1) |
| `train_formation_mappo.py` | RLlib MAPPO 训练入口 |

### 架构

```
Agents: pursuer_0, pursuer_1
Policy: shared_pursuer (shared weights)
Actor:  33-dim local → 256→Tanh→256→Tanh → action_mean(2)
Critic: 21-dim global → 256→Tanh→256→Tanh → scalar
Target: scripted straight-and-level
```

### 状态

✅ 代码完备, 语法验证通过, 环境烟雾测试通过  
❌ 训练未运行 — Ray 2.x 不支持 Windows (见第 7 节)

---

## 6. Tacview 导出修复

经历了多次迭代 (301 绿线 / Target= / Text= / TextLocation=), 最终回退到最简版本:

```
仅输出 101/102/201 的 T= 位置数据
无 301 对象, 无 Target=, 无额外线条
```

**原因**: `Type=Static+Minor` 在本地 Tacview 版本中导致对象完全不可见; polyline T= 格式未被正确解析; `Target=` 原生锁定线因 P1 始终飞得更远而从不切换。

已生成 6 个成功 episode 的 ACMI 文件 + 3D 轨迹图。

---

## 7. Ray Windows 启动故障

### 诊断

```
Ray 2.55.1 + Python 3.10.20 + Windows 10 Pro

ray.init() → GCS (gRPC) ✅
          → raylet (C++ 调度器) ❌ 崩溃
          → raylet.out: "unknown" (静默失败)
          → GCS 等待注册 → 60s 超时
```

### 根因

Ray 2.x 不再支持原生 Windows。`raylet` 二进制依赖 Linux 系统调用。

### 解决路径

| 方案 | 说明 |
|------|------|
| **SB3 IPPO** (短期) | 独立策略 + 去中心化执行, 无需 Ray |
| **WSL2 + Ray** (长期) | 完整 CTDE / MAPPO 支持 |

已输出详细诊断报告: `docs/ray-windows-startup-failure-diagnosis.md`

---

## 8. 全版本对比

| 版本 | 类型 | 成功率 | 难度 | 碰撞率 | 架构 |
|------|------|--------|------|--------|------|
| Phase 3.5 | 1v1 | **81.3%** | 0.58 | N/A | Box(2) SB3 |
| Phase 3.6 | 1v1 | 83.0% | 0.20* | N/A | Box(2) + smooth |
| Phase 4 | 2v1 | 92.5% | 0.00 | 5.8% | Box(4) 共享策略 |
| **Phase 4.1** | 2v1 | **97.3%** | 0.00 | **2.2%** | Box(4) + spacing |
| Phase 5 | 2v1 | — | — | — | MAPPO (代码完备) |

\* Phase 3.6 仅 300K 步, 未推进到更高难度

---

## 9. 提交清单

```
51e263f docs: Ray Windows startup failure diagnosis
ff1d93e feat: Phase 5 — RLlib MAPPO 2v1 formation (CTDE architecture)
fe3f910 fix: Tacview export — bare aircraft only
6091d5e fix: revert Tacview to working version
ebcce8f fix: Tacview — 301 polyline + Text distance per frame
6925dc9 viz: regenerate Tacview with corrected Target= direction
74f030a fix: Target (201) locks nearest pursuer
e439841 fix: Tacview — native Target= lock line
1f72495 fix: Tacview — single green line only
3cbdea3 fix: add distance label (Text=) to green engagement line
a0303fb fix: Tacview lines — green=nearest pursuer→target, yellow=spacing
c691dca viz: add green formation spacing line to Tacview exports
5dda75f viz: Phase 4.1 2v1 — 6 success trajectories (Tacview + 3D plot)
bfb629d feat: Phase 4.1 final — piecewise formation spacing
6094f80 feat: Phase 4.1 — piecewise formation spacing + weight annealing
618bfff feat: Phase 4 — 2v1 cooperative pursuit (92.5% success)
34283d6 feat: FormationEnv 1v1 validation
fae76f0 feat: FormationEnv — 2v1 cooperative pursuit
478d212 feat: Phase 3.6 — smoothness tuning
a30aafb viz: clean Phase 3.5 trajectory visualization
```

---

## 核心成就

1. **1v1 SOTA**: 81.3% @ diff=0.58 (自我蒸馏)
2. **2v1 编队**: 97.3% success + 2.2% collision (分段间距)
3. **MAPPO 就绪**: CTDE 代码完备, 待 WSL2 环境运行
4. **工程基础**: FormationEnv 双模式 (1v1/2v1), 权重平铺, 分阶段训练
