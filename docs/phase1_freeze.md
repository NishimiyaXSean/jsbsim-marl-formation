# Phase 1 封版声明 (v2) — 2026-08-10

## 1. 预注册统计结论 (final holdout, 2026-08-09)

```
Overall preregistered verdict: FAIL
Reason: T1 overshoot gate exceeded
  (fire3 后 min_range < 1500m 约 30%, 预注册阈值 <= 1%)
```

按封版纪律: 预注册失败就是失败, 不因事后分析(如 "1500m 定义过宽")回头修改本轮判据;
该批 holdout seeds 已被使用, 不再用于任何调参或判据修改。

## 2. 工程/任务级验收 (mission-level acceptance): PASS

| 指标 | 结果 |
| --- | --- |
| ID kill (1000) | **99.9%** (teacher 99.8%, paired +0.1pp) |
| dist2_3k kill (400) | **98.0%** (teacher 98.5%, paired -0.5pp) |
| OOD 13 格 | 96-100% |
| L0-L4 stress (17x100) | 97-100% |
| lost | **0 observed** (ID 0/1000, 95% Wilson 上界 0.3%) |
| bad fire | **0 observed** |
| 3-hit timeout | 基本消失 (ID 0, d2 2/400) |
| 第4合法窗口出现率 | 97-100% |

原始故障链已切断:

```
近距过冲 -> ATA翻转 -> 无法重建第4窗口 -> 3-hit timeout     (已切断)
```

现在变为:

```
部分局仍近距穿越 -> 快速恢复 -> 第4窗口仍出现 -> 正常击杀
```

## 3. Known limitation: residual close-pass / overshoot

- ID 13.3% / dist2_3k 20% 的局在第三发命中后仍有 <100m 近距穿越 (最小 5-7m);
- 准确表述: **在当前仿真动力学、碰撞模型、传感器与控制延迟下, 尚未观察到任务后果** — 不是 "resolved";
- 若未来引入以下任一因素, 可能重新成为问题: 碰撞/近碰惩罚, 更真实机体尺寸, wake/气动干扰,
  传感器瞬时丢失, 控制执行延迟, 更强目标机动, terrain/altitude constraint, 多机避碰。

## 4. 冻结清单 (三层基线)

今后任何新模型只与 v2 比较, 不再以 v19 为低门槛基线。

| 层级 | 版本 | 说明 | 冻结标识 |
| --- | --- | --- | --- |
| L0 | v19 | 历史 PPO 基线 (kill 10% / lost 32%) | 历史 commit 75fd2ff |
| L1 | BC round1 / ASAP 蒸馏 | 诊断链路与可解释中间基线 | shoot_bc_round1_baseline.pth / shoot_bc_asap_distilled.pth |
| **L2** | **v2 (正式当前最优)** | rule-free neural policy; heading/encoder 未解冻; 无 PPO | 见下 |

v2 冻结标识:

```text
student 权重:  data/expert/shoot_bc_speed_distilled_v2.pth
student SHA256: a94176861f71e521c21183463c8edc34c8bffff4664c76a9acea4871dd8960d3
代码/训练 commit: fb1ac46
评测脚本 commit: af81378 (final_holdout.py)
判定文档 commit: bfc3bb5
holdout seed 范围: ID 40000-40999, d2 41000-41399, cells 42000+, stress 44000+
最终 JSON: results/shoot_eval/final_holdout.json
预注册判据: scripts/final_holdout.py 顶部 (冻结后未修改)
```

## 5. Phase 2 触发条件 (不自动开启)

1. 新仿真加入近碰/碰撞模型, close-pass 开始产生真实代价;
2. 更高 difficulty 下 <100m 穿越与 lost/timeout 重新相关;
3. 新 OOD 场景中 3-hit timeout 明显回升;
4. time-to-kill 成为正式优化指标;
5. 任务要求最低 separation (如始终 >500m);
6. 多机作战使近距穿越产生战术风险。

否则残留 close-pass 作为 backlog, 不构成 blocker。

## 6. 下一周期的门禁定义 (仅用于下一套开发集 + 第二套全新 holdout, 绝不回溯本轮)

```text
Proximity risk : min_range < 100m / 250m / 500m
Harmful overshoot : close pass AND (第4窗口长期不出现 OR 3-hit timeout)
```

不再使用 min_range < 1500m 作为 T1 (已证明包含大量无害近距机动)。

## 7. 路线总结 (分层归因)

```text
接近失败         -> BC 解决 (heading平衡 + fire窗口采样 + 掩码CE)
发射过保守       -> ASAP 蒸馏解决 (合法即发, kill 43% -> 91%)
3-hit timeout    -> 定位到高速近距过冲 (失败局归因 + hit3分型 100% T1)
过冲后缺第4窗口  -> 无状态 anti-overshoot speed 策略解决 (v2, kill -> 98-99.9%)
```

成果核心: 从 v19 (kill 10% / lost 32%) 到 v2 (kill 98-99.9% / lost 0 observed),
全程未解冻 heading/encoder、未使用 PPO 微调, 每一步都有因果诊断与门禁支撑。
