# Phase 1 总结：1v1 导弹制导空战（规则专家 → BC → ASAP → anti-overshoot v2）

> 分支: `feature/refactor-task-based` · 更新: 2026-08-19 · 状态: 已封版 (v2)

## 1. 路线回顾

目标：1v1 战斗机携带导弹制导任务（`singlecombat_shoot_task.py`），训练"接近 → 进入发射区 → 发射 → 击杀"的完整闭环，并以 v19 PPO 作为对比基线。

```
离散规则专家(无状态标签)
  → BC 模仿(heading平衡 + fire B/C窗口采样 + 掩码CE)
  → 500配对seed验证(BC击杀+7.4pp显著优于专家)
  → ID/OOD/压力三套件回归(lost全0)
  → 几何修复(可seed复现 + 场景参数化, 单测10/10)
  → fire oracle 诊断(dist2_3k瓶颈=fire过保守, 非不可达)
  → ASAP规则对照组(全场景全局上限)
  → P0加载门禁 / P1独立critic warm-up / P2A fire-only ASAP蒸馏
  → 失败归因(3-hit timeout缺第4窗口) / hit3分型(100%高速近距过冲)
  → anti-overshoot: 无状态几何减速规则(B2) + speed head蒸馏(v2)
  → 最终封版holdout(预注册): 工程PASS / T1门禁FAIL(已知限制)
```

## 2. 版本与关键指标对比

ID 分布 = 尾追 2-5km、初始偏置 30-60°、目标直飞；所有数值均为 difficulty=0、确定性动作。

| 模型 | ID 500 lost | ID 击杀率 | dist2_3k 击杀 | 发射/局 | bad 发射 | 关键依据 |
| --- | --- | --- | --- | --- | --- | --- |
| v19 (PPO 基线) | 32% | 10% | — | 0.48 | — | 旧训练产物 |
| 离散规则专家 | 0% | 35.8% | 1% | 2.89 | 0 | 500 配对 seed |
| BC round1 | 0% | 43.2% | 6.7% | 3.01 | 0 | 配对差值 +7.4pp (95%CI [+3.8,+11.0]) |
| ASAP rule override | 0% | 90.6% | 74% | 3.89 | 0 | 规则上限 |
| **ASAP 蒸馏** | 0% | 91.2% | 77% | 3.90 | 0 | 完整回归通过 |
| **v2 speed-蒸馏（封版）** | **0%** | **99.9%** (ID1000) | **98.0%** | ~3.9 | **0** | 预注册 holdout (ID1000 / d2 400 / OOD+stress 97-100%) |

## 3. 三套件回归（ASAP 蒸馏模型）

| 套件 | lost | 击杀率 | 备注 |
| --- | --- | --- | --- |
| ID 500 seeds | 0% | 91.2% | 命中 99.9%, premium+good 100% |
| 13 格 OOD（key 100 / other 50 seeds） | 0% | 77-100% | bias90_120 / dist5_8k / 分离 100% |
| 压力 L0-L4 | 0% | 81.7-91% | L4 hdg_k20 危险 5% 全部自行恢复 |

冻结 BC 在各套件的 lost 也为 0%，但击杀率显著更低（ID 43.2%、dist2_3k 6.7%、压力 30-38%）。

## 4. 关键发现

1. **几何勘误**：`set_geometry()` 实际被 `env.reset() → SingleCombatShootTask.reset` 覆盖；真实初始几何为偏置 30-60°（`max_heading_bias_deg=60`），数据中不存在 90-120°。已修复为可 seed 复现 + 场景参数化（偏置/距离/横向/高度差/速度差/目标行为），单元测试 10/10。
2. **fire oracle（dist2_3k）**：固定 BC 航向/速度轨迹，枚举发射时序——"合法即发"（asap）击杀 73.3%，延迟/DLZ 选择性策略全部更差。证明 dist2_3k 不是不可达，而是 BC fire 策略过于保守（等 premium 窗口，错过约 1 步宽的近距窗口）。
3. **ASAP 全局上限**：fire 策略是全局瓶颈（不只是近距）；ASAP 在全场景 lost=0/bad=0，击杀普遍比 BC 高 40-60pp。
4. **蒸馏技术要点**：BC fire head 的 no-fire logit 约 -10（深度先验），正样本 CE 在 lr 1e-4 下收敛极慢；由于推理时 action mask 强制 disallowed=0，蒸馏目标=把 fire head 翻转为"全发"，单头 lr 1e-2、10-15 epochs、早停 99.5% 即可精确复现 ASAP。
5. **PPO 暂缓**：蒸馏模型稳定复现 ASAP 后，当前奖励/环境下不存在发射时机取舍，PPO 无增量收益且有回退风险；P2B 仅在出现发射成本、弹药稀缺、窗口选择或轨迹改变需求时启用。
6. **失败归因**：非击杀局 84-85% 为 3-hit timeout，第 4 合法窗口缺失率 95-100%；冷却结束后几何(ATA/range/closure)从未再满足——根因是第三发后的接近/几何保持，而非 fire。
7. **hit3 分型**：20/20 T1 近距过冲；首因时序 range 在 hit3+1 步失效(7-36m) → ATA +35-44 步翻转 → closure 随后恶化；反事实 oracle 证明 speed-first(提前 16s 减速至 240 可救回 95%)，heading 干预无效。
8. **anti-overshoot 与 v2**：Gate S0 证明无状态规则 memoryless_decel 优于 latch(盲集 ID 98%/d2 94.5%)；precursor 几何规则闭环救回 90%、成功保留 100%；S3/S4 speed head 蒸馏闭环等价 teacher；封版 holdout ID 99.9%/d2 98%/lost 0，预注册 T1 门禁 FAIL(残留 <100m 穿越 13-20%，无任务后果，登记 known limitation)。

## 5. 统计与门禁

- **500 配对 seed**：BC vs 专家 delta_kill +7.4pp（bootstrap 95%CI [+3.8,+11.0]），胜/负/平 62/25/413；lost 差值 0。
- **P0 加载门禁**：BC 权重装入 RLlib policy，零更新 500 seeds 逐 seed 100% 复现，logits max diff < 1e-4。
- **P1 critic warm-up**：actor state-dict hash 不变、logits diff 0、100 seeds 逐动作序列一致；critic explained variance 0.736，kill 局价值排序正确。
- **P2A 蒸馏门禁**：hdg/spd logits diff 0、共享前缀动作序列一致、fire-agreement（窗口利用率 100%）、lost=0、bad=0。

## 6. 产物清单

| 类型 | 路径 |
| --- | --- |
| 冻结 BC 基线 | `data/expert/shoot_bc_round1_baseline.pth` |
| P1 独立 critic | `data/expert/shoot_critic_p1.pth` |
| **最优：ASAP 蒸馏** | `data/expert/shoot_bc_asap_distilled.pth` |
| RLlib checkpoint | `marl_runs/shoot_bc_asap_distilled/checkpoints/best` |
| 500 配对评测 | `results/shoot_eval/paired_bc_vs_expert_500.json` |
| ASAP 规则基线 | `results/shoot_eval/asap_baseline.json` |
| 场景矩阵 | `results/shoot_eval/scenario_matrix_expanded.json` / `scenario_matrix_distilled.json` |
| 压力评测 | `results/shoot_eval/stress_L0..L4.json`（BC 与蒸馏版） |
| fire oracle | `results/shoot_eval/fire_oracle_dist2_3k_60.json` |
| 可视化 | `results/ctrl_viz/`（轨迹 / ACMI / 面板 / reward 分布） |
| 计划与执行记录 | `docs/plan_bc_rule_expert.md` |

## 7. 关键复现命令

```bash
# 专家数据审计（含确定性复算与 split C 统计）
python scripts/audit_expert_data.py --skip-validate

# 500 配对评测（BC vs 专家）
python scripts/eval_paired_bc_vs_expert.py --seeds 500

# ASAP 规则对照组
python scripts/eval_asap_baseline.py --mode all

# P1 critic warm-up
python scripts/ppo_p1_critic_warmup.py --rollout-episodes 400 --verify-seeds 100

# P2A fire-only ASAP 蒸馏
python scripts/distill_fire_asap.py --rollout-episodes 300 --eval-seeds 60

# 蒸馏模型完整回归
python scripts/eval_bc_1v1.py --weights data/expert/shoot_bc_asap_distilled.pth --episodes 500
python scripts/eval_scenario_matrix.py --weights data/expert/shoot_bc_asap_distilled.pth --eps-key 100 --eps-other 50
python scripts/stress_eval.py --level L0..L4 --weights data/expert/shoot_bc_asap_distilled.pth
```

## 8. 封版与后续

- **Phase 1 已封版 (v2)**：rule-free 神经策略，heading/encoder 未解冻、无 PPO；预注册统计结论 = FAIL(T1 门禁)，工程/任务级验收 = PASS(ID 99.9% / d2 98% / lost 0 observed / bad 0 / 3-hit≈0 / 第4窗口 97-100%)；
- 三层基线冻结：v19(历史) / BC+ASAP(诊断中间) / v2(当前最优)；今后新模型只与 v2 比较；
- 已知限制：ID 13.3% / d2 20% 的局仍存在 <100m 近距穿越(当前动力学下无任务后果)，非 resolved；
- Phase 2 触发条件(不自动开启)：近碰代价模型 / 高难度下 <100m 重新相关 / 新 OOD 中 3-hit 回升 / TTK 成为正式指标 / 最低 separation 要求 / 多机避碰需求；
- 详见 [docs/phase1_freeze.md](phase1_freeze.md)（封版声明与冻结清单）。

