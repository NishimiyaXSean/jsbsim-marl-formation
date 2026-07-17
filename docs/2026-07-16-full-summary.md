# 2026-07-16 完整工作总结：V7 尸检 → V10 孵化器破局 → AND-gate sync 40%

> **日期**: 2026-07-16
> **主题**: V7 死亡诊断 → V8/V9 架构迭代 → V10 孵化器突破 15% 天花板 → V11 降门
> **提交**: 14 commits
> **核心成果**: AND-gate sync 从历史 15% 天花板飙升至 40%，孵化器证明"出生在包线内"是正确的起点

---

## 1. V7 深度尸检——独立 Actor 惨败

### 1.1 死亡数据

| 指标 | 值 |
|------|:--:|
| 30 集 lost_target | **100%** |
| 最差 episode (−23,323) | P1: 2,728→10,001m (124 步飞离 7.3km) |
| AND met | 0/124 steps |

### 1.2 尸检图表

- `fig_v7_crash_diagnosis.pdf`: 3-panel (distances, penalties, pincer angle)
- `fig_v7_entropy.pdf`: 熵健康 (1.65→1.97) 但 reward 崩盘 (−1,930→−16,725)
- `fig_v7_kl_entropy_reward.pdf`: KL 0.005 几乎不更新——梯度消失

### 1.3 根因

独立 Actor + 高压惩罚 (−10/step loiter) = P1 宁可受罚也要逃跑。熵正常探索但 reward 全景无正收益路径。

---

## 2. V8——共享 Actor 回归 + Loiter AND 修复

### 改动

1. **Loiter 判定**: `is_slow OR fleeing` → `is_slow AND fleeing`
   Slow 朝向目标的 head-on 逼近 (180m/s) 不再被误罚
2. **Stage 1 OOC grace**: gate = max(AND+400, spawn_upper) = 2200m
   保证 1600-2200m 出生不被即死
3. **共享 Actor 回归**: V7 证明独立 Actor 不 work

### 结果

熔断于 iter 58——数值过载 (−25,904)。比 V7 多活了 33 轮但惩罚累积仍致命。

---

## 3. V9——FiLM 深层身份调制

### 架构创新

```python
# 在 MLP 两层之间插入 FiLM
feat = F.tanh(self.mlp_head[0](pooled))
gamma = self.film_gamma(agent_id)   # [B, 256] from onehot [B, 2]
beta  = self.film_beta(agent_id)
feat = gamma * feat + beta          # 深层特征调制
feat = F.tanh(self.mlp_head[2](feat))
```

初始化 gamma→1 (恒等), beta→0 (无偏置)——BC 权重不受影响。碰撞力场加 50/step cap 防 Critic 爆炸。

### 结果

存活 155 轮 (V8 的 2.7×)。训练 reward 改善到 −1,418。sync 闪现 5%。熔断于"协同荒漠"。

### V9 诊断图

`fig_v9_diagnostic.pdf`: 7,105 steps, 0% AND met。死因确认——不是尾追死锁，是空间分离。每集必有一机飞到 10,000m。Pincer 角度健康 (均值 37.4°)但两机从不同时进入 AND 区域。

---

## 4. V10——孵化器突破 🔥🔥🔥

### 核心洞察

V9 尸检证明：1600-2200m 的初始距离迫使两机在长途接近中固化成"并排尾随"姿态。抵达 1200m 时夹角已锁定在 0-10°。

### 孵化器设计

```
Stage 1 (孵化器):
  AND 距离: 1200 → 1600m (放宽)
  AND 角度: 35 → 25° (降低)
  出生距离: 1600-2200 → 1000-1400m (必然在 AND 门内!)
  维持步数: 2 → 1 (瞬触即发)
  初始偏航: [-30,+30] → [-20,+20] (更窄)
```

两机睁眼就在 AND 包线内——不需要长途赶路。唯一任务：分离到 25° 夹角触发协同击杀。

### 结果：ALL-TIME RECORD

| 指标 | 历史天花板 | V10 |
|------|:---:|:---:|
| sync 峰值 | 15% (V2/V5) | **40%** 🔥🔥🔥 |
| eval 正值 | +178 (V5) | **+784** 🔥 |
| 500 轮完成 | — | ✅ 无熔断 |

```
iter  25: sync=15%
iter  50: sync=20%
iter  75: sync=35%  ← 突破 15% 天花板
iter 225: sync=40%  eval=+784  ← 历史最高 sync
iter 450: sync=40%  eval=+443
iter 500: sync=25%
```

3-eval MA 峰值 33%，距 30% 降门仅差一步未触达。

---

## 5. V11——降门 + 熵衰减尝试

### 改动

1. Stage 1→2 gate: 60% → **30%** (V10 已达 33% MA)
2. Stage 2→3 gate: 50% → **45%**
3. 熵衰减 schedule (失败——RLlib 旧 API 不支持，回退)

### 结果

熔断于 iter 225 协同荒漠。entropy_coeff=0.03 压制了 V10 的高探索 (0.05)，sync 从 30% 衰减到 0%。

### 教训

**V10 的 entropy=0.05 是孵化器的正确搭档。** 孵化器需要高探索来尝试不同的分离方向。降低 entropy 反而让策略锁定在次优行为。

---

## 6. 全部 AND-gate 实验演进全景

| 版本 | 架构 | 关键创新 | Sync 峰值 | Best Eval | 500 轮 | 熔断 |
|------|------|------|:---:|:---:|:---:|:---:|
| V2 | 共享 | AND-gate 课程 | 15% | −4,452 | ✅ | ❌ |
| V5 | 共享 | 通信+ID+全栈 | 15% | +178 | ✅ | ❌ |
| V6 | 独立 | 独立 Actor | 5% | −3,815 | ✅ | ❌ |
| V7 | 独立+高压 | 反摸鱼+软碰撞 | 5% | −8,891 | ❌ | 数值过载 |
| V8 | 共享 | AND-loiter+OOC | 0% | −8,223 | ❌ | 数值过载 |
| V9 | 共享+FiLM | 深层身份调制 | 5% | −6,602 | ❌ | 协同荒漠 |
| **V10** | **共享+FiLM+孵化器** | **出生即入门** | **40%** 🔥 | **+784** 🔥 | ✅ | ❌ |
| V11 | 共享+FiLM+降门 | 30%降门+低熵 | 30% | −1,774 | ❌ | 协同荒漠 |

---

## 7. 基础设施

| 组件 | 版本 | 功能 |
|------|:---:|------|
| TrainingHealthMonitor | V8+ | 3 道熔断 + 自动 Markdown 尸检报告 |
| AirCombatMetricsCallbacks | V8+ | TensorBoard 实时几何指标 |
| FiLM 调制层 | V9+ | 深层身份破缺 (gamma*feat+beta) |
| 孵化器 Stage 1 | V10+ | 出生在 AND 包线内 |
| 降门课程 | V11+ | Stage 1→2: 30% |
| warmup=0 bypass | V8+ | 跳过 OR 预热直入 AND |

---

## 8. Git 提交 (14 commits)

```
430f6e2 fix: revert entropy schedule — RLlib old API only accepts float
9f83080 feat: V11 — lowered stage gates + entropy decay schedule
90f1042 feat: V10 incubator Stage 1 — spawn inside AND envelope
5461cf1 feat: V9 diagnostic — distance-pincer scatter + 3D trajectories
c7bb1ef feat: V9 — FiLM deep identity modulation + collision shaping cap
020421d fix: initialize pincer_angle=0.0 before inner loop for info dict safety
ecaf011 fix: warmup=0 AND-gate deadlock + pincer 1m→1mm threshold
bba6a87 feat: training health monitor + circuit breakers + custom callbacks
082a784 fix: rotate velocity vector when bearing constraint clamps heading
3c3683d feat: V8 — shared Actor + AND-loiter + Stage 1 OOC grace
36ab431 feat: V7 KL divergence + entropy + reward 3-panel diagnostic
12c5896 feat: V7 entropy curve — healthy exploration, catastrophic reward
3d9570c feat: V7 crash diagnosis — P1 flees 2,728→10,001m in 124 steps
```

---

## 9. 下一步：V12

- **基座**: V10 全栈 (孵化器 + FiLM + 共享 Actor + ego Critic)
- **仅改降门**: Stage 1→2: 60% → 30%
- **entropy_coeff**: 保持 0.05 (V10 证明优于 0.03)
- **预期**: Stage 1 孵化器触达 30% MA → 晋级 Stage 2 → 熵逐渐固化
