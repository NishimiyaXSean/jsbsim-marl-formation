# geomA 重训（heading bias [0,60]）vs 冻结基线 —— kill rate 对比报告

生成时间：2026-09-14 22:35 +0800
驱动：`scripts/_run_geomA_pipeline.sh`（老板于 2026-09-14 19:20:29 在 Windows 终端手动启动）
几何改动：commit `fb48155`，`bias_lo` 由 `max(5.0, 0.5*max_bias)=30` 改为 `0.0` ⇒ 初始航向偏置 `U(30,60)×±1` → `U(0,60)×±1`

---

## 0. 结论

**管线已完整跑完。kill rate 点估计为提升（95/100 vs 基线 87/100，+8.0 pp），但该差值落在统计噪声内，且与"新几何本身更容易"完全混淆 —— 因此判为「持平偏改善，不可归因为策略变强」。既不能说确证提升，也不构成退步。**

- 按老板给定的判定门槛（≥93/100 计为真实提升）：95/100 **越过**该门槛。
- 按更严格的单臂分辨率（MDE ≈ 9.4 pp，需 ≥96/100 才使 Wilson 下界越过 0.87）：95/100 **未达标**。
- 两种口径都远没有到 p<0.05：Fisher 精确检验 **p = 0.081**。

---

## 1. 管线执行（已验证事实）

`logs/geomA_driver.log` 四阶段 `rc=` 齐全，末行 `===== RUN DONE 2026-09-14T21:29:00 =====`。

| 阶段 | 起止 | 耗时 | 产物 |
|---|---|---|---|
| 1 规则专家 200 ep | SKIP（复用） | — | `data/expert/shoot_rule_expert_geomA.npz` 24,684,175 B |
| 2 BC 40 ep | SKIP（复用） | — | `data/expert/shoot_bc_geomA.pth` 438,085 B |
| 3 ASAP 蒸馏 | 19:20:30 → 21:17:25 | 1 h 57 m | `data/expert/shoot_bc_asap_geomA.pth` 438,633 B |
| 4 100-ep holdout | 21:17:25 → 21:29:00 | 11 m 35 s | `results/shoot_eval/eval_geomA_d0_s42.json` 1,823 B |

- 全流程 **2 h 08 m**；此前 2.8 h / 3.2 h 的估算偏保守。
- **本次未被 WSL 关机杀掉。** `last -x` 权威记录显示 `reboot 19:14 … still running`，且 `sean pts/1` 自 19:14 一直登录 —— 老板那个终端全程保持附着，正是既定操作规则所要求的条件。第 4 次 60 s 空闲关机未发生。

---

## 2. kill rate 对比（已验证事实）

| 指标 | 冻结基线 `eval_distilled_d0_s42.json` | geomA `eval_geomA_d0_s42.json` | 差值 |
|---|---|---|---|
| **kill_rate** | **87/100 = 0.870** | **95/100 = 0.950** | **+8.0 pp**（相对 +9.2%） |
| episodes | 100 | 100 | — |
| termination_reasons | target_killed 87, timeout 13 | target_killed 95, timeout 5 | timeout −8 |
| hit_rate | 0.99740（384/385） | **1.00000（394/394）** | +0.26 pp，零脱靶 |
| lost_target_rate | 0.0 | 0.0 | 持平 |
| wez_reach_rate | 1.0 | 1.0 | 持平 |
| launches_per_episode | 3.85 | 3.94 | +2.3% |
| launch_quality | premium 218 / good 167 / bad 0 | premium 224 / good 170 / bad 0 | 结构基本持平（premium 占比 56.6% → 56.9%） |
| mean_steps | 965.7 | 884.1 | **−8.4%**（结束更快） |
| 发射时 aa_deg 均值 | 14.27° | 11.23° | **−21%**（离轴角更小） |
| 发射时 ata_deg 均值 | 7.44° | 7.11° | 基本持平 |
| 发射时 range 均值 | 2529 m | 2506 m | 基本持平 |
| time_to_wez_steps 均值/中位 | 87.08 / 70 | 58.08 / 61 | 更快 |
| time_to_wez_steps **最小值** | **56 步** | **0 步** | 出现"开局即可发射"的 episode |
| first_fire_steps **最小值** | **57 步** | **0 步** | 同上 |

一个正向信号：`wez_to_fire_latency_steps` 中位数两组都是 1 步，`lost_after_wez` 两组都是 0 —— 策略没有为了刷 kill 而牺牲发射纪律或丢失目标。

---

## 3. 统计支撑（已验证事实）

- 冻结基线 87/100 → Wilson 95% CI **[0.790, 0.922]**，半宽 ±6.6 pp
- geomA 95/100 → Wilson 95% CI **[0.888, 0.979]**，半宽 ±4.5 pp
- 差分 +8.0 pp；Wald 95% CI **[+0.2, +15.9] pp**；Newcombe 95% CI **[−0.1, +16.5] pp**（下界跨 0）
- Fisher 精确检验（非配对，双侧）**p = 0.081**
- n=100/组、非配对、80% power 的最小可检测差异 **≈ 13.3 pp** ⇒ **+8.0 pp 在噪声带之内**
- 若把基线当作精确参照（单臂口径），MDE ≈ **9.4 pp**，判定阈值 ≥96/100；k=93/100 对应 Fisher p=0.238
- 要把分辨率压到 5 pp 需 **n≈587/组**；3 pp 需 **n≈1772/组**

---

## 4. 门禁与产物完整性（已验证事实）

- `logs/p2a_distill.json` mtime = **2026-09-14 21:17:21**，晚于 19:20:30 ⇒ 确系本次 geomA 运行的产物，**不是**旧几何遗留。（旧结果已备份为 `logs/p2a_distill_geomOLD_backup.json`，mtime 09-14 09:50。）
- gates 全 PASS：`hdg_spd_logits_identical=true`、`max_diff=0.0`、`hdg_spd_sequence_identical=true`、`fire_allowed_agreement=true`；`best_val_allowed_acc=0.9982`。
- 60-seed 门禁场景中 distilled 与 rule **逐场景完全相同**（id 93.3% / dist2_3k 78.3%），BC 仅 id 43.3% / dist2_3k 11.7% —— 蒸馏忠实度确证。
- **冻结产物零改动**，mtime 全部仍为 2026-09-13：`shoot_rule_expert.npz` 20:22、`shoot_bc_round1_baseline.pth` 20:25、`shoot_bc_asap_distilled.pth` 23:01、`eval_distilled_d0_s42.json` 23:19。（驱动脚本启动时也会断言基线 JSON = 1845 B。）

---

## 5. 关键混淆因素（我的判断，但有代码级证据）

`fb48155` 只改了一行：

```
- bias_lo = max(5.0, 0.5 * max_bias)
+ bias_lo = 0.0
```

抽样语句 `rng.uniform(bias_lo, bias_hi) * rng.choice([-1, 1])` 的**抽样次数与顺序未变**，且它是 `reset()` 中最后一次 RNG 消耗。由此可推出两个事实：

1. **同一 seed 下，两组的 t_hdg、t_spd、chase_dist、lateral、目标/追击者初始位置、p_spd、alt_diff 逐位相同**，唯一差别是追击者初始航向偏置。所以这份对比比"完全非配对"要更接近配对 —— 但**偏置本身就是被研究的处理变量**，所以仍然无法做配对检验。
2. 同一个均匀数 u 映射到：旧偏置 `30 + 30u`，新偏置 `60u`。因为 `60u ≤ 30 + 30u` 对一切 u∈[0,1] 成立，**新几何每个 episode 的初始偏置都小于或等于旧几何，均值小 15°**。

⇒ **geomA 组的场景在初始对准意义上被系统性地放宽了。+8.0 pp 中有多少来自"策略变强"、多少来自"开局更容易"，在现有评估产物下无法分离。** 这是本次结论不能宣称"提升"的首要原因，比统计功效问题更根本。

旁证：`time_to_wez_steps` 与 `first_fire_steps` 的最小值都由 56/57 步降到 **0 步**。旧几何下 ATA 在出生点必然 ≥30°（远超 ~15° 的发射角门限），结构上不可能第 1 步就满足发射包线；新几何下出现了这种 episode。**至少存在 1 个"零成本"episode 是可证的；精确条数无法从只存聚合值的 JSON 反推。**

---

## 6. 局限（不可回避）

1. **非配对**：`eval_bc_1v1.py` 用 `env.reset(seed=args.seed + ep)`，两次评估各自可复现，但几何改动把同一 RNG 流映射到了不同分布，且评估 JSON 只存聚合值、无逐 episode 记录 ⇒ **无法事后做 McNemar / 配对检验**。
2. **分辨率不足**：n=100/组只能分辨 ~13 pp（非配对）、~9 pp（单臂）。+8.0 pp 处于任何口径的噪声边缘。本次实验**设计上就不可能给出确证结论**。
3. **难度不等价**（见 §5），这是根本性的。
4. `time_to_wez_steps` 的口径不一致：`logs/geomA_4_eval.log` 里的 `[TIMING] WEZ_first`（fire-mask 判据、1-based、且 `ep=` 恒为 1）均值 91.1，与 JSON 的 `time_to_wez_steps`（`_is_valid_launch_envelope`、0-based）均值 58.08 不是同一个量。次要计时指标只应在同一文件内比较，跨文件比较无效。

---

## 7. 建议（未执行）

1. **做 2×2 对照**才能回答"策略是否变强"：{旧几何, 新几何} × {基线权重, geomA 权重}。当前 `bias_lo` 是硬编码常量、不是 config key，复现旧几何需要打补丁 —— 建议先把 `bias_lo` 提为配置项（例如 `min_heading_bias_deg`），使几何可配置、对照可复现。
2. **提分辨率**：要判定 5 pp 级别的差异需 n≈600/组（100 ep × 6 组，约 1 h）。当前 100 ep 的 holdout 只适合做粗筛。
3. **把逐 episode 结果落盘**（至少 seed / bias / kill 三个字段）到评估 JSON，后续即可做配对检验，成本极低。
4. `distill_fire_asap.py` 加中途 checkpoint + `python -u`，消除"任何一次被杀就丢 2 h"的结构性风险（本次虽然跑完，但风险未消除）。
