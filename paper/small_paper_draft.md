# Small Paper Draft v2 — 2026-09-16

> **⚠ v1 → v2 重大更正（2026-09-16）**
> v1 的核心表格把三个不同的实验混为一谈。经逐文件核对 `results/shoot_eval/*.json`：
> - `eval_distilled_d0_s42.json` 与 `eval_2x2_base_geoOld_s42.json` 是**同一次运行**（weights 与 seed 完全相同）→ 87% 是 **SPC 模型在旧几何 U(30,60) 下**的击杀率，**不是 BC**。
> - `eval_geomA_d0_s42.json` 的 weights 是 `shoot_bc_asap_geomA.pth`（**另一个重训模型**）→ 95% 是 **geomA 重训模型在新几何下**的击杀率，**不是 fire-head 修正的收益**。
> - 因此 v1 的「BC 87% → SHD 95%，+8 pp 来自 fire-only fix」**不成立**。这一点项目自己的 `results/shoot_eval/geom2x2_n400_report.md` 第 25、117 行早已判定：*「最初『87 → 95，+8 pp』高估了策略增益约 2.5 倍」*。
> - 真正的 fire-only 干预效应是 **BC 43.2% → SPC 91.2%（ID 500 配对种子，+48.0 pp）**。
> v2 按此重写。所有数字标注来源与证据级别（见 §6）。

---

## Title Candidates

**首选（Sean 2026-09-17 定稿方向）**：
*Diagnosing and Surgically Correcting Conservative Decision Bias in Imitation-Learned Air Combat Policies*

**备选**：
- *Surgical Correction of Expert-Induced Decision Bias in Imitation-Learned Air Combat Agents*（更短）
- *Diagnosing and Surgically Correcting Conservative Engagement Gating in Imitation-Learned Air Combat Policies*（前版，机制词更具体）

> **标题演进记录**：v1 "Hidden Suboptimalities…" → 2026-09-16 Sean 拍板突出机制（conservative engagement gating）→ 2026-09-17 Sean 定稿方向 **conservative decision bias**（比 gating 更通用、比 suboptimality 更具体，且直接呼应论文真正证明的链条：expert-induced bias → diagnosis → correction）。
>
> 定位（AAMAS）：**interacting-agent environments 中 expert-induced policy bias** 这一层。卖点：规则专家的**局部**决策规则（一条比环境合法性更严的发射门）在**闭环**对抗中产生**全局**性能损失，被模仿学习忠实继承 —— 这是 interactive 场景特有的失效模式，且可诊断、可局部修正。

---

## Abstract (~170 words)

Behavior cloning (BC) from rule-based experts is a standard bootstrap for air-combat policies, under the implicit assumption that the expert's decisions are worth imitating. We test that assumption on a JSBSim F-16 within-visual-range (WVR) 1v1 missile-engagement benchmark. We introduce the **Conditional Launch Rate (CLR)** — the probability that a policy commands launch at a step where the environment permits it, `P(a_fire=1 | m_fire=1)` — and show that the hand-designed rule expert commands launch at only **6.07%** of permitted steps and BC lands in the same band at **7.26%** (400 episodes, fresh env per episode, identical seeds). The cause is a launch-quality gate the expert applies *on top of* the environment's legality condition — a conservative engagement preference the environment does not require, not a deficit in the imitation, which is behaving as intended. Two interventions isolate its cost. **(A)** Holding the BC maneuver trajectory frozen and enumerating fire policies on top of it, a mask-permissive policy (launch whenever the environment-level launch mask permits) achieves **86.7%** kills where the inherited launch policy achieves **15.0%**, and every range- or delay-selective alternative is worse; the maneuver is fixed by construction, so the gap is attributable to the launch decision alone. **(B)** We then apply **Surgical Policy Correction (SPC)**, which re-trains *only* the launch head while every other parameter stays bit-identical (heading/speed logits `max diff < 1e-9`). Under a matched d=0, deterministic-argmax evaluation on **400 paired seeds**, replacing the conservative launch decision alone raises the kill rate from **46.50% to 90.75%** (**+44.25 pp**, exact McNemar **p = 1.04e-53**); SPC's CLR of **100.00%** is true by construction, not a finding. The pairing is uniform, not merely significant: of 177 discordant seeds, **all 177 favour the correction and none favour the original**. Under an evading target (`difficulty_level = 0.3`) the benefit grows rather than decays: SPC is unchanged at **90.75%** while the baseline falls to **42.25%**, widening the gap to **+48.50 pp** with all 194 discordant seeds again favouring the correction. We do not claim that the mask-permissive policy is optimal in general; we claim that a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime.

> **措辞纪律（Sean 2026-09-16 要求）**：不得写 "SPC improves performance by 48 pp" 这类泛化句式。必须始终绑定四个限定：**matched setting / d=0 / deterministic argmax / isolated intervention**。否则 reviewer 的第一反应是「为什么只改一个 head 能提升这么多？」—— 答案正是「因为轨迹冻结，所以差异只能来自这个 head」，但这个因果必须自己讲出来，不能被追问。
>
> **E7 已落地（2026-09-16）**：n=400，seeds 20000–20399，几何 U(0,60)，d=0，确定性 argmax。BC=186/400=**46.50%**、SPC=363/400=**90.75%**、paired **+44.25 pp**、discordant **177 : 0**、exact McNemar **p=1.04e-53**。
> **⚠ 数字已更新**：文档记录的 43.2% / 91.2%（500 seeds、旧几何）**被本次实测取代** —— 因为只有本次把 kill rate、CLR、几何放在**同一 seed 集**上，才是自洽可比的一对。论文一律采用 46.50% / 90.75% / +44.25 pp。

---

## 1. Introduction (1 page)

### 1.1 Background
- Imitation learning (IL) bootstraps control policies in domains where online trial-and-error is expensive or unsafe: autonomous driving, manipulation, air combat.
- BC is the simplest IL: supervised regression/classification on expert demonstrations.
- Dominant assumption: the expert is a *sufficient* teacher; fidelity to the expert is the objective.

### 1.2 Problem: expert-induced bias, amplified by closed-loop interaction
- Rule/heuristic experts are designed, not optimized. Their decision rules encode engineering judgement that is *locally* plausible but never validated in the closed loop.
- In interactive settings (opponent/evader present), a locally reasonable restriction — "only launch when the shot is high quality" — changes the entire engagement trajectory, so its cost is not locally observable.
- BC propagates the bias faithfully, so downstream RL or evaluation inherits it as a *floor*, not a *fluctuation*.
- Existing BC evaluation reports aggregate task success only; it does not attribute the residual to a specific decision head.

### 1.3 Contributions

> 结构按 Sean 2026-09-17 定型。**第一贡献是诊断，不是性能提升**；叙事主线：*专家含有一个隐藏的、局部但灾难性的决策偏差；BC 忠实复制它；一个局部策略修正即可在不改变机动策略的前提下恢复性能。* 措辞纪律：不说 "expert 很差"，说 *the expert contains a conservative engagement preference that is not required by the environment constraints*（环境允许 ≠ 专家愿意 —— 这正是机制所在）。

1. **C1 — Diagnosis: expert-induced conservative bias, quantified.** 我们引入 **Conditional Launch Rate（CLR）**，`P(a_fire=1 | m_fire=1)` —— 一个无需 oracle、可迁移到任何「离散提交型决策」（发射 / 交接 / 急停 / 变道）的指标 —— 并发现规则专家在环境允许发射的步上只发射约 6%，BC 忠实复现该偏好（约 7%）。我们把原因定位到一个**设计出来的**发射质量门：它严格地比环境合法性掩码更严（§3.2）。这不是专家「能力不足」，而是专家携带了一个**环境并不要求的保守偏好**。⚠ 数字来源纪律：C1 的 expert CLR 必须**只**采用 fresh-env 测量（E3 CLR run，进行中）；reused-env 的 6.04%/5.43% 已剔除，不得回流。
2. **C2 — Surgical Policy Correction: isolating the behavioral dimension responsible.** SPC 重训**单个二值动作头**使其对齐环境合法性策略，**冻结其余一切参数**，并以 logit 与动作序列双门验证冻结的逐比特性（§3.3）。方法论要点不是「只训一个 head」这个实现细节，而是 **isolate the behavioral dimension responsible for the failure**：轨迹保持不变、修正局部化。没有这个隔离，「你只是 fine-tune」的质疑就无法反驳。
3. **C3 — Causal validation: two interventions, one conclusion.** 两个互补的因果论证（§4.3, §4.4）：
   - **Intervention A（frozen-trajectory oracle replacement）**：15.0% → 86.7% —— *the fire decision alone can explain the performance gap*；
   - **Intervention B（SPC training）**：46.50% → 90.75%（配对，177:0，p≈1e-53）—— *the correction can be internalized into the policy*。
   - 效果在规避目标下保持并扩大（+44.25 → +48.50 pp），且被一套独立的识别方法复现（§4.3 E6 matched 2×2：+40.0 → +50.0 pp）。

**叙事结构**（正文按此展开，比普通 ablation 高一层）：
```
Observation（CLR 异常低）
   → Where is the failure? → 冻结轨迹干预 ⇒ 发射决策是因果因素
   → Can the policy be repaired? → SPC ⇒ 修正可被内化进策略
```

---

## 2. Related Work (1 page)

> 结构按 Sean 2026-09-17 拍板：**三个关键词，不堆 RL 文献**。本文不是 air-combat RL 论文，related work 必须服务三个贡献（诊断 / 局部修正 / 因果验证），不要让 §2 变成 MARL 综述。所有引用标注 `[verify]` 的须在投稿前核对真实性 —— **宁可留空也不编造**。

### 2.1 Imitation learning and expert demonstration

- **BC and its standard pathology.** Cloning a demonstrated policy by supervised learning goes back to ALVINN (Pomerleau, NIPS 1:305-313, 1989 [verify year]) and was scaled to end-to-end driving by Bojarski et al. (arXiv:1604.07316, 2016). The canonical account of its failure mode is DAgger (Ross, Gordon & Bagnell, AISTATS 2011), which attributes the failure to **distribution shift / compounding error** between the learner's and the expert's state visitation.
- **Core sentence.** *Existing IL methods usually assume the demonstrations are informative and near-optimal; the learner's objective is fidelity to the expert.*
- **The literature that does relax optimality relaxes it as a scalar.** T-REX learns a reward from ranked suboptimal demonstrations (Brown, Goo, Nagarajan & Niekum, ICML 2019, pp. 783-792); LERP models suboptimality as reward noise (Huo, Wang & Xu, AAAI 2023, pp. 7953-7961); IRLEED models a demonstrator's suboptimality as *reward bias plus action variance* (Beliaev & Pedarsani, arXiv:2402.01886, 2024); IL-from-imperfection reweights supplementary suboptimal data (Li, Xu, Qin, Yu & Luo, NeurIPS 2023). In every case suboptimality is a **scalar quality** to be estimated, ranked, or down-weighted.
- **Our gap.** *What if the expert's suboptimality is not scalar but structural - a single, nameable decision gate that the environment does not require?* A scalar quality cannot recover it: the defect is invisible in task-level success statistics (§4.2) and costly only in closed loop. We do not down-weight the expert; we **localize** the gate and correct exactly it.

### 2.2 Policy distillation and model editing

- Knowledge distillation into a student (Hinton, Vinyals & Dean, NIPS 2014 Deep Learning Workshop, arXiv:1503.02531), policy distillation (Rusu, Gomez Colmenarejo, Gulcehre et al., ICLR 2016, arXiv:1511.06295), and Actor-Mimic (Parisotto, Ba & Salakhutdinov, ICLR 2016, arXiv:1511.06342) all **transfer or compress a policy globally**, with fidelity to the teacher as the objective.
- Parameter-efficient adaptation (LoRA; Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, ICLR 2022, arXiv:2106.09685) and model editing (ROME; Meng, Bau, Andonian & Belinkov, NeurIPS 2022, arXiv:2202.05262, pp. 17359-17372) do modify a small subspace while freezing the remainder - but they edit **learned knowledge** and evaluate by task success, not by attributing a residual to one behavioural dimension.
- **Core sentence.** *Existing approaches transfer or modify policies globally; we study localized correction of a single decision dimension, and we use the freezing itself as a causal identification device.* The distinction from distillation is precise: SPC's objective is **not** to approach the teacher but to **selectively depart** from it on one head. Freezing is not an engineering convenience here - it is the identification strategy (§4.4).

### 2.3 Autonomous air combat decision-making

- **Closest neighbours, stated plainly.** Li et al. (Neurocomputing 584:127591, 2024) learn a within-visual-range 6-DOF combat policy whose top layer decides autopilot commands **and missile launch**, with BC and PPO cross-coordinated. Li et al. (ACM TAAS 21(1):6:1-6:20, 2026) learn a "pursuit-lock-launch" policy with TD3 + BC and an adaptive imitation weight in Harfang3D. Both clone a launch decision from demonstrations in a WVR missile-engagement setting.
- **Where we differ - two ways, both load-bearing.** (i) They treat the expert as a **bootstrap to be surpassed**; we treat it as an **object of diagnosis**, and we keep the maneuver provably frozen so the residual is attributed to the launch head rather than to aggregate learning. (ii) Their success criterion is aggregate task success; we add a **conditional** criterion (CLR) that exposes failures aggregate success hides. Notably, Li et al. (2026) present selective firing as a virtue - their policy "fires only when a hit was highly probable rather than spamming launches" - the **opposite prior** to the one this paper tests, and in our regime the measurement goes the other way (§4.3).
- Background only, kept short: approximate dynamic programming for air combat (McGrew, How, Williams & Roy, JGCD 33(5):1641-1654, 2010); hierarchical maximum-entropy RL for an F-16 in within-visual-range combat (Pope et al., ICUAS 2021, pp. 275-284, arXiv:2105.00990); short-range UAV maneuver decisions via deep RL (Yang, Zhang, Shi, Hu & Wu, IEEE Access 8:363-378, 2020). JSBSim is the open-source flight-dynamics model underneath all of these.
- **Constraint.** This subsection stays under half a page. It is scene-setting, not contribution positioning: a reviewer should enter this paper through §2.1/§2.2, not file it as another air-combat RL paper.

> **Citation status (re-checked 2026-09-18 against the live record, not from memory).** Verified: Pomerleau (**NIPS'88**, pp. 305-313; presented 1988, MIT Press 1989 - the year is now confirmed, not inferred), Ross/Gordon/Bagnell (AISTATS 2011), Bojarski (arXiv:1604.07316), Brown et al. (ICML 2019 pp. 783-792), Huo/Wang/Xu (AAAI 2023 pp. 7953-7961), Beliaev/Pedarsani (arXiv:2402.01886), Li Z. et al. (NeurIPS 2023), Parisotto/Ba/Salakhutdinov (ICLR 2016, arXiv:1511.06342), McGrew et al. (JGCD 33(5):1641-1654), Pope et al. (ICUAS 2021 pp. 275-284), Yang et al. (IEEE Access 8:363-378), Li L. et al. (Neurocomputing 584:127591, 2024), Li S. et al. (ACM TAAS 21(1):6, 2026), **Rusu et al. (ICLR 2016 conference track, arXiv:1511.06295)**, **Hinton/Vinyals/Dean (NIPS 2014 Deep Learning Workshop, arXiv:1503.02531 - a workshop paper, so it has no page numbers and none are invented here)**, **LoRA (Hu et al., ICLR 2022, arXiv:2106.09685)**, **ROME (Meng, Bau, Andonian & Belinkov, NeurIPS 2022, Advances in NeurIPS 35, pp. 17359-17372)**. **Nothing on this list is now unconfirmed.**
>
> **Three citation errors were caught across the two passes.** (1) The earlier placeholder "Pang et al. 2024" does not exist; the work actually intended - BC + PPO for WVR 6-DOF air combat - is Li, Zhang, Qian, Zhao & Wang, Neurocomputing 584:127591 (2024). (2) "Actor-Learner Distillation" is a misnomer; the correct title is Actor-Mimic (Parisotto, Ba & Salakhutdinov, ICLR 2016). (3) ROME's author list was given as "Meng, Wang, Pfaff & Yang"; the paper is by **Meng, Bau, Andonian & Belinkov** - both the arXiv record and the NeurIPS 35 proceedings agree. A wrong author list on a cited method is the kind of error a reviewer notices, and it was found only by looking the citation up rather than recalling it.
---

## 3. Method (1.5 pages)

### 3.1 Task and policy
- **Task**: 1v1 **within-visual-range** air combat, JSBSim F-16 dynamics; pursuer carries 4 short-range IR missiles (AIM-9L class); initial range 2–5 km tail-chase, target heading bias 0–60° (default), target speed U(180,240) m/s. Max engagement range 15 km.
- **Observation**: 41-dim flattened = 30-dim state features + 11-dim action-mask bits. The BC encoder consumes the 30-dim state block; the mask block is applied as a separate masked-argmax step, not fed to the network.
- **Action space**: `MultiDiscrete([3, 5, 1, 2])` = (speed Δ, heading Δ, altitude, fire).
- **Episode**: max 1500 steps; termination = target killed / low altitude / fled altitude / target lost / ammo exhausted / timeout.
- **Step duration — state it explicitly, it is a units trap.** One environment step advances the simulation by **0.2 s** (`BaseEnv._acmi_time += 0.2`), i.e. the agent runs at **5 Hz**; 1500 steps ≈ 300 s. This is *not* the JSBSim internal frame rate. Every step→second conversion in this paper uses 0.2 s/step (`§4.3`: 67 steps = 13.4 s, 743.5 steps = 149 s). Any figure or table that appears to use 1/60 s per step is wrong by 12×.
- **Evaluation protocol**: **deterministic masked argmax**, JSBSim physics, `difficulty_level` as noted per experiment.
- **Three policies in play**:
  - **Rule expert** — hand-designed, stateless: every label is a pure function of the current observation (`scripts/generate_shoot_rule_expert.py`). **Provenance, stated plainly: there is no external publication to cite.** It is this project's own rule controller, built from standard BFM heuristics (nose alignment, DLZ depth, closure sign) plus the launch-quality gate of §3.2, and trained against nothing. We do not claim it is a strong or a state-of-the-art expert; we claim only that it is *a* competent teacher (`wez_reach_rate = 1.00`, `hit_rate ≈ 0.997`, `lost_target_rate = 0`) whose behaviour carries a measurable conservative bias. Every headline claim in this paper is about *what imitation inherits from a given teacher*, holding the teacher fixed — so the paper does not depend on where the teacher came from, and inventing a citation for it would be worse than saying this.
  - **BC round1** — MLP encoder (256-256-128) + 4 heads, trained on 200 expert episodes, masked cross-entropy.
  - **SPC** — BC with only the launch head re-trained (§3.3).

### 3.2 C1 — Diagnostic metric and localization
**Conditional Launch Rate (CLR)**:

```
CLR(π) = P(a_fire = 1 | m_fire = 1)
       = (# steps with a_fire=1 and m_fire=1) / (# steps with m_fire=1)
```

- `m_fire` is the **environment's** legality mask (index 10 of the flat mask), which requires: missiles remaining > 0; launch cooldown (30 decision steps) elapsed; target alive; `ATA < 15°`; `MIN_ATTACK_DISTANCE (1500 m) ≤ range ≤ dynamic max range (3–8 km by aspect angle)`; closure < 0 (closing).
- **Critical property: CLR is not a claim that 100% is optimal.** It is a descriptive quantity. The optimal value must be established separately — which is exactly what §4.3 does. (v1 的 "theoretical maximum of 100%" 措辞已删除。)

**Why the expert's CLR is low — the designed gate.** The expert does not simply follow `m_fire`. It requires its own stricter quality predicate:

| Condition | Environment mask | Expert `fire_desired` | Effect |
|---|---|---|---|
| Nose alignment | `ATA < 15°` | `ATA < 10°` | refuses 10–15° shots |
| Closure | `closure < 0` | `closure < −5 m/s` | refuses marginal closure |
| DLZ position | legality only | depth ∈ **[0.25, 0.75]** | refuses deep-in-DLZ (closer) and far-edge shots |
| Output | — | `fire = allowed AND desired` | cooldown/inventory handled by mask |

**Measured — two independent bases, one answer.** The rows are of different kinds and must not be pooled; the first is the paper's authoritative figure and the last two are corroboration.

| Quantity | Value | Basis |
|---|---|---|
| **Expert CLR (fresh env, 400 ep, seeds 20000–20399, U(0,60))** | **6.07%** (1131 / 18639) | `E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json` — same seeds, same protocol, same env lifecycle as every other number in this paper |
| Expert CLR (`action[:,3]` on allowed steps, read off the demonstration dataset) | 5.96% (560 / 9395) | `data/expert/shoot_rule_expert.npz`, 200 episodes |
| BC CLR (launch-head argmax on allowed steps, 200 rollouts) | 7.37% (614 / 8327) | `results/health_check/fire_hesitancy.json` (2026-09-16, so U(0,60)) |
| Expert `fire_desired` on allowed steps | 560 / 9395 — **identical to `action[:,3]`** ⇒ the expert's behaviour on legal steps is fully determined by its own gate | same dataset |
| Expert `fire_desired` on *disallowed* steps | 4.03% (10336 / 256357) — the gate would fire outside the window too; the mask (mostly cooldown) holds it back | same dataset |

> **⚠ Geometry caveat on the corroboration rows (caught 2026-09-17).** `shoot_rule_expert.npz` has mtime **2026-09-13 20:22**, which *predates* commit `fb48155` (2026-09-14 09:06) that changed the heading-bias range from `U(30,60)` to `U(0,60)`. The demonstration dataset is therefore an **old-geometry** artifact, and its 5.96% is *not* the same measurement basis as the paper's U(0,60) figures. The BC 7.37% row comes from `results/health_check/fire_hesitancy.json` generated 2026-09-16, i.e. *after* the change, so the two corroboration rows are **geometry-mixed**. This is why neither is allowed to carry a headline claim: the authoritative expert number is the re-measured **6.07%**, and the older rows are quoted only to show that the low figure is stable across bases, geometries and decade-old seed sets. Do not average them, and do not present 5.96% alongside 7.26% as if they were one matched comparison.

⇒ The conservatism is **entirely attributable to the designed gate**, not to the environment and not to BC's fitting error.

### 3.3 C2 — Surgical Policy Correction (SPC)
- **Input**: frozen BC θ_BC; BC rollouts collected over `R` episodes.
- **Target**: on steps where `m_fire = 1`, command fire. All other steps are irrelevant (the mask forces `a_fire = 0` at inference).
- **Procedure**:
  1. Roll out θ_BC; collect `(obs_t, m_fire,t)`.
  2. Filter to `m_fire,t = 1`.
  3. Train **only** `action_heads[3]` by cross-entropy toward `a_fire = 1`. Encoder, speed head, heading head, altitude head: `requires_grad_(False)`.
  4. Early stop when allowed-window fire accuracy ≥ 0.995.
- **Verification gates** (all must PASS; implemented in `scripts/distill_fire_asap.py`):
  - `G1` heading/speed logits vs θ_BC: `max diff < 1e-9`
  - `G2` heading/speed **action sequences** 100% identical on fixed seeds
  - `G3` **CLR ≥ 99.5%** over allowed windows (`P(a_fire=1 | m_fire=1) ≥ 0.995`) on held-out rollouts
  - `G4` no new `lost_target` / `bad` launch introduced
- **Interpretation**: G1–G2 are the causal-identification device — the maneuver policy is provably untouched, so any performance change is attributable to the launch decision alone.

> **Naming/provenance note (internal):** in code and docs this pipeline is called "ASAP distillation" (`scripts/distill_fire_asap.py`, docs 中的 "P2A"). ASAP = "As Soon As allowable by Policy", i.e. the target policy of §3.3. Paper-facing name = **SPC**. Map explicitly in the README/paper appendix to avoid reviewer confusion.

### 3.4 Why isolating the launch head is the right intervention (not arbitrary)
- The launch decision is a **discrete binary commitment** with a one-step-wide opportunity in the close-range regime (§4.3), so its error is not smoothable by the rest of the policy.
- The maneuver heads are demonstrably near-ceiling: `wez_reach_rate = 1.0`, `hit_rate ≈ 0.997–1.000`, `lost_target_rate = 0` across all evaluated suites. Re-training them would buy little and destroy the clean attribution.
- Contrast with end-to-end PPO fine-tuning: the project's own record (`docs/summary_phase1.md` §4.5) shows that once SPC reproduces the legality policy, the current reward/environment admits no launch-timing trade-off, so PPO adds no increment and carries regression risk. → report as a *documented negative result*, not as a claim about PPO in general.

---

## 4. Experiments (2 pages)

### 4.1 Setup
- Deterministic masked argmax; JSBSim physics; single RTX 3060 Laptop (cu126).
- Distributions: **ID** (2–5 km tail-chase, bias per geometry), **dist2_3k** (hard close-range cell), **target_evasive** (`difficulty_level = 0.3`).
- Wall clock: BC closed-loop ≈ 6.7–13.0 s/episode.

### 4.2 Diagnosis result — CLR

| Policy | Basis (all U(0,60) unless noted) | Allowed steps | Launch commands | **CLR** |
|---|---|---|---|---|
| **Rule expert** | **E3 seed set, 400 ep, d=0, fresh env** | **18639** | **1131** | **6.07%** |
| **Rule expert** | **E3 seed set, 400 ep, d=0.3, fresh env** | **19729** | **1066** | **5.40%** |
| **BC round1** | E7 seed set, 400 ep, d=0, fresh env | 16839 | 1223 | **7.26%** |
| **BC round1** | E5 seed set, 400 ep, d=0.3, fresh env | 17169 | 1194 | **6.95%** |
| **SPC** | E7 seed set, 400 ep, d=0, fresh env | 1560 | 1560 | **100.00%** |
| SPC | E5 seed set, 400 ep, d=0.3, fresh env | 1577 | 1563 | **99.11%** |
| Rule oracle (fire whenever the mask permits) | by construction | — | all allowed | 100% (by construction) |
| *Expert, 200-ep demonstration dataset* | *old geometry U(30,60) — corroboration only* | *9395* | *560* | *5.96%* |
| *BC round1, 200 rollouts* | *health-check diagnostic* | *8327* | *614* | *7.37%* |
| ~~Rule expert 400 ep d=0 / d=0.3~~ | ~~reused env — **excluded**~~ | ~~18521 / 19736~~ | ~~1119 / 1071~~ | ~~6.04% / 5.43%~~ |

**Every un-struck row now comes from one protocol** — fresh env per episode, seeds 20000–20399, geometry U(0,60), deterministic argmax — and the expert and BC rows are produced by the *same run* (`eval_paired_bc_vs_expert.py`, CLR support added 2026-09-17), so their difference is paired by construction. The italic corroboration rows are a different basis and are quoted only where labelled (§3.2). The struck rows come from the reused-env path and are excluded for the reason given in §4.4.

**Cross-check that the new run agrees with the rest of the paper:** the same artifact reproduces the E3 kill numbers bit-for-bit (expert 36.75%, BC 46.50%, paired +9.75 pp, W/L/T 63/24/313) and reproduces BC's CLR to the digit (1223/16839). So adding CLR instrumentation changed nothing measurable about the policy — the new field is additive, and the table is self-consistent.

**The conservatism ordering can now be asserted.** On one protocol: expert **6.07%** < BC **7.26%** ≪ SPC **100.00%**. Earlier drafts deliberately withheld this sentence because its expert leg came from the excluded reused-env path; that reason is gone. Caveat (b) still applies — the denominators differ by construction, so this is an ordering of *tendencies*, not of fractions of one shared opportunity set.

Two readings:
1. **Expert and BC agree closely on every basis available**, so the conservatism is a **stable property**, not sampling noise: 6.07% vs 7.26% on the matched 400-seed set, and 5.96% vs 7.37% on the earlier diagnostics. BC is not more conservative than the teacher in any meaningful sense — it reproduces the gate and adds a little variance. The bias is the teacher's; the imitation is faithful.
2. **Neither figure alone proves suboptimality.** A low CLR diagnoses *conservatism* only; whether that conservatism is *costly* is established separately (§4.3, §4.4).

> **⚠ Two caveats that must appear in the paper (both are reviewer-attack surfaces).**
>
> **(a) CLR(SPC) = 100.00% is true by construction, not an empirical discovery.** SPC's training target is literally "fire on every legal step", and gate G3 tests exactly that. It must not be presented as a result.
>
> **(b) The CLR denominator is policy-dependent, so CLR is not a like-for-like "fraction of opportunities exploited".** BC is legal on 16839 steps (≈42/episode); SPC on only 1560 (≈3.9/episode) — a factor of 10.8. The cause is mechanical: firing triggers the 30-step launch cooldown, which *removes* legality, so a policy that fires collapses its own opportunity set, while a policy that abstains stays "legal" across many consecutive steps. Comparing CLR across policies therefore compares different opportunity sets. This is precisely why the primary identification is the **frozen-trajectory enumeration** (§4.3), where the maneuver is fixed by construction and the opportunity set is held constant, rather than a cross-policy CLR comparison.

### 4.3 C3(a) — Off-policy enumeration on a frozen trajectory (the key identification)

**Design.** Take BC's heading/speed decisions as a *frozen* maneuver trajectory per seed, and enumerate a family of fire policies on top of it. Because the maneuver is fixed by construction, the maneuver policy is controlled, and differences are attributable to the launch decision.

**Cell `dist2_3k`, 60 seeds, geometry U(0,60) — regenerated 2026-09-17 (E1):**

| Fire policy | Kill rate (measured) | *docs, old geom U(30,60)* | Launches/ep | Reach-4-launch |
|---|---|---|---|---|
| **`asap` — fire whenever the launch mask permits** | **86.7%** | *73.3%* | 3.85 | 86.7% |
| `delay_30` (first legal + 30) | 38.3% | *28.3%* | 3.25 | 38.3% |
| `delay_60` | 10.0% | *10.0%* | 2.78 | 10.0% |
| `dlz_mid` (depth 0.4–0.6) | 0.0% | *0.0%* | 1.27 | 0.0% |
| `dlz_deep` (depth ≥ 0.6) | 0.0% | *3.3%* | 1.12 | 0.0% |
| `interval_100` | 0.0% | *0.0%* | 1.95 | 0.0% |
| **frozen BC (inherited)** | **15.0%** | *6.7%* | 2.37 | 15.0% |

> **⚠ Geometry generation matters — never mix these two columns.** The italic column is the 2026-08-06 run recorded in `docs/plan_bc_rule_expert.md`, executed under the *pre-`fb48155`* geometry U(30,60). The measured column is today's default U(0,60). `delay_60` (10.0%) and `dlz_mid` (0.0%) reproduce **exactly**, which shows the script's behaviour is unchanged; `asap` moves 73.3% → 86.7% purely because the benchmark got easier (§4.5), where the same-weights geometry change alone was worth ≈ +4.5 pp. **Cite only the measured column, always with its geometry attached.**

**Reachability audit (asap arm, measured):** median first-legal step **67** (13.4 s); **cumulative legal-window length ≈ 4 steps per episode** (median 4.0) — i.e. roughly one legal step per DLZ transit; median step-1 → step-4 launch **743.5** (149 s); 52/60 killed, 8 timeouts, **8 of which ended with the window still open** (`blockers = {episode_end_in_window: 8}`). The "≈4 legal steps per episode" figure is the load-bearing structural fact and it reproduces the 2026-08-06 audit.

**Reading:**
- The scenario is **not infeasible** — the environment permits a 4-launch salvo in 86.7% of episodes, and `asap` realizes 86.7% kills.
- The bottleneck is the launch policy: the inherited policy uses **2.37 of 3.85** available windows, because most windows have DLZ depth < 0.25 and are rejected by `fire_desired`.
- **Every** selective alternative is worse than the mask-permissive policy, and by a wide margin (best rival `delay_30` at 38.3%). Within this family, on this trajectory class, launching on every mask-permitted step dominates.
- Note what this does and does not show: it establishes suboptimality **conditional on the frozen maneuver**. It does not establish that the mask-permissive policy is optimal for arbitrary maneuvers of arbitrary policies. (§4.4 supplies the within-policy counterpart.)
- **Terminology discipline.** The mask-permissive policy is *not* "always fire": it fires only on steps where the environment-level launch mask is 1. The correction removes a gate the expert added *on top of* the environment's legality condition; it does not relax the environment's condition itself.

> **Artifact status:** verified against the live artifact `results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json` (tracked in git, carries `run_meta` with `cell_config`, geometry, difficulty, seed range and the frozen checkpoint's sha256). The old JSON is genuinely gone; the docs column above is all that survived of it, which is exactly why the regenerated file is now version-controlled.

**Same enumeration under an evading target — matched 2×2 (E6, seeds 20000–20059, n=60).**

`target_evasive` changes difficulty *and* nothing else if we pair it with the default cell: its config is `{difficulty_level: 0.3}` — all other parameters default (2–5 km). Its matched d=0 counterpart is therefore cell `id_bias30_60` (empty config ≡ defaults). Both cells are therefore the same scenario distribution, differing **only** in the target's evasion.

| difficulty | fire policy | Kill | Launches/ep | Legal-window median |
|---|---|---|---|---|
| d=0 | `asap` | **93.3%** | 3.92 | 4.0 |
| d=0 | frozen BC (inherited) | **53.3%** | 3.18 | 42.0 |
| d=0.3 | `asap` | **96.7%** | 3.97 | 4.0 |
| d=0.3 | frozen BC (inherited) | **46.7%** | 3.05 | 42.0 |
| | **`asap` − inherited gap** | **+40.0 pp → +50.0 pp** | | |

Three things follow:

1. **The direction replicates under an independent method.** Here the maneuver is frozen and only the fire policy varies; in §4.4/§4.6 the fire head is re-trained and the maneuver heads are frozen. Two different identification strategies, same answer: **evasion punishes the conservative launch policy and not the aggressive one.** `asap` rises slightly (93.3 → 96.7%) while the inherited policy falls (53.3 → 46.7%), so the gap widens by 10.0 pp — the same sign as the +4.25 pp widening measured by the within-policy intervention.
2. **The opportunity-set asymmetry is a structural constant, not a difficulty effect.** The legal-window median is 4.0 for `asap` and 42.0 for the abstaining policy — *identical at both difficulties*. Firing collapses your own window (cooldown removes legality); abstaining inflates it ~10×. This is the measured basis for the §4.2 caveat, and it is stable across the difficulty change.
3. **Cell and difficulty were separated, not conflated.** An earlier reading of this run compared it against the `dist2_3k` table in §4.3 and would have attributed part of the difference to evasion when it was actually the 2–3 km range restriction. The matched pair above removes that confound.

### 4.3b Figure 2 — the mechanism, made visible

**Figure 2** (`results/shoot_eval/mechanism_seed20007_d00.png`) shows one matched episode (seed 20007, d=0) played by BC and by SPC, with everything else held identical.

- **Panel A** plots the state the decision is taken on — range to target and angle-off (ATA) — with the environment-legal launch windows shaded, every legal step BC declines marked, and both policies' launches marked.
- **Panel B** plots the decisions themselves as three rows: launch mask, BC fire command, SPC fire command.

**What the figure is for.** It is an *attribution* device, not a trajectory showcase. BC uses **3 of 41** environment-legal steps on this seed (CLR 7.32%, close to the 7.26% aggregate) and times out after 1500 steps; SPC uses **4 of 4** and kills the target at step 842. The state trace is **bit-identical** over all 842 common steps (maximum position deviation **0.0e+00 m**) — the maneuver heads are frozen — so the figure contains a *single* state trajectory and the entire difference is confined to Panel B. That is the claim: same geometry, same maneuver, different launch decision.

Two caveats the caption must carry: (i) this is a **single illustrative episode**, chosen to make the mechanism legible — the aggregate evidence is Table 3 (`§4.3`) and Table 4 (`§4.4`); (ii) SPC's legal-step count (4) is smaller than BC's (41) because firing triggers the launch cooldown, which removes legality — the policy-dependent denominator documented in §4.2(b), not a defect of the figure.

### 4.4 C3(b) — SPC intervention result (primary endpoint, E7)

**Setting (all arms identical):** d=0, deterministic masked argmax, geometry U(0,60), **seeds 20000–20399 (n=400)**, one code path (`eval_bc_1v1.py`), identity recorded in each file's `run_meta` (BC `sha256 aad05b45…`, SPC `sha256 36d79bd9…`).

| Arm | Kill rate | Wilson 95% | Launches/ep | CLR | lost | hit |
|---|---|---|---|---|---|---|
| Rule expert (fresh-env path, E3) | 147/400 = **36.75%** | [32.17, 41.58] | 2.83 | — | 0 | 0.9973 |
| **BC round1 (frozen)** | **186/400 = 46.50%** | [41.67, 51.40] | 3.06 | 7.26% | 0 | 0.9984 |
| **SPC (this paper)** | **363/400 = 90.75%** | [87.51, 93.21] | 3.90 | 100.00% | 0 | 1.0000 |

**Paired BC vs expert on the same 400 seeds:** Δkill **+9.75 pp**, win/lose/tie **63 / 24 / 313**, exact McNemar **p = 3.48e-05** (χ²cc p = 4.62e-05); 72% of the 87 discordant seeds favour BC. Δlost = 0.00, Δlaunches = +0.23. This is the first *significance test* attached to the BC-beats-expert claim — the sealed Phase-1 record had only a bootstrap CI ([+3.8, +11.0] pp, 500 seeds, old geometry).

> **⚠ Env-lifecycle confound — read before citing any expert number.** Two scripts evaluate "the expert" and they disagree on identical seeds (32.25% vs 36.75% at n=400; a 20-seed screen shows **different per-seed kill vectors**, not noise). Both import the same rule functions (`hdg_label` / `spd_label` / `fire_desired`) at the same `CMD_SPEED = 280`, and `BaseEnv.__init__` already installs `SafetyInterceptor(PIDFlightController())`, so neither the rule nor the controller explains it. The cause is **environment lifecycle**: `generate_shoot_rule_expert.py --validate` builds **one** `BaseEnv` and reuses it for all episodes, whereas `eval_paired_bc_vs_expert.py`, `eval_bc_1v1.py`, `fire_oracle_audit.py` and `eval_asap_baseline.py` all build a **fresh env per episode**. Every policy figure in this paper comes from the fresh-env path, so the fresh-env expert (**36.75%**) is the only comparable one. The reused-env figures (32.25%, and 25.25% at d=0.3) are **excluded from this paper** — they are not wrong, they are measured under a different episode-construction protocol, and mixing them would repeat exactly the confound this section warns about.

**Paired contingency and test** (`scripts/paired_mcnemar.py`, exact McNemar, no scipy):

| | count |
|---|---|
| both kill | 186 |
| neither kill | 37 |
| **only SPC kills** | **177** |
| **only BC kills** | **0** |
| discordant total | 177 |

- **Paired difference +44.25 pp**; exact McNemar **p = 1.04e-53** (χ² with continuity correction p = 5.97e-40).
- **The direction is unanimous: all 177 discordant seeds favour the correction; there is no seed where BC succeeds and SPC fails.** This is stronger than a small p-value — it is a *dominance* statement over the evaluated scenario set, and it is the cleanest single number in the paper.
- `lost_target = 0` and `launch_quality.bad = 0` in both arms — the corrected policy does **not** buy kills with reckless launches: premium 921 / good 639 / bad 0, i.e. **every** SPC launch still scores "good-or-better" under the expert's own quality function.
- Heading/speed behaviour is bit-identical to BC by construction (G1 `max diff < 1e-9`; G2 identical action sequences), so the entire gain is attributable to the launch head alone.

**Report it bound to its conditions, never as a bare improvement.** Use: *"Under a matched d=0, deterministic-argmax evaluation on 400 paired seeds, replacing the conservative launch decision with SPC raises the kill rate from 46.50% to 90.75% (+44.25 pp, exact McNemar p = 1.04e-53)."* Not: *"SPC improves performance by 44 pp."* The four qualifiers (matched setting, d=0, deterministic argmax, isolated intervention) are load-bearing — they are *why* the attribution is legitimate.

> **Artifact status.** E7 is **verified against live artifacts** (`results/shoot_eval/E7_{bc_round1,spc}_d0_n400_s20000.json`, `E7_paired_bc_vs_spc_d0_n400.json`).
>
> **⚠ Superseded numbers.** The docs-attested Phase-1 pair **43.2% → 91.2% = +48.0 pp** (500 seeds, *old* geometry U(30,60)) is **replaced** by the measured **46.50% → 90.75% = +44.25 pp** (400 seeds, U(0,60)). The old pair was not self-consistent: its BC figure came from a different geometry generation than its CLR figure. E7 puts kill rate, CLR, and geometry on **one** seed set. Cite 46.50 / 90.75 / +44.25 in the paper.
>
> **Both former docs-only numbers have now been replaced by fresh-env measurements.** The rule-expert ID kill rate is **36.75%** (E3 paired, fresh env, seeds 20000–20399) — *not* the docs' 35.8%, and *not* the reused-env 32.25%. The rule-oracle ID kill rate is **90.75%** (E2), measured on the same seed set as SPC and exactly equal to it; that equality is a completeness check, not independent validation (same policy by construction, §4.4). The originals (`paired_bc_vs_expert_500.json`, `asap_baseline.json`) remain lost — the replacements are new measurements, not recoveries, and were taken under U(0,60) rather than the docs' U(30,60).

**SPC vs the rule oracle on one seed set (E2, seeds 20000–20399, n=400).** Resolved: `eval_asap_baseline.py` had no `--start-seed`, so the oracle could only ever run seeds `0..N-1` and could not be matched to a trained policy's evaluation. Patched, then run on the same seed set.

| Arm | Kill | Launches/ep | Hit | lost | Termination reasons |
|---|---|---|---|---|---|
| Rule oracle (`asap`: frozen BC maneuver + fire on every legal step) | **363/400 = 90.75%** | 3.90 | 1.0000 | 0 | `{target_killed: 363, timeout: 37}` |
| **SPC** | **363/400 = 90.75%** | 3.90 | 1.0000 | 0 | `{target_killed: 363, timeout: 37}` |

**Identical on every reported metric.** Identities verified from `run_meta`: the oracle arm loaded `sha256 aad05b45…` (BC round1 — the *frozen maneuver* source, correct), SPC loaded `36d79bd9…`.

> **⚠ Interpret this as a completeness check, not as independent validation.** The two are the *same policy by construction*: the oracle is "frozen BC maneuver + fire on every legal step", and SPC is "bit-identical frozen maneuver heads + a launch head whose measured CLR is 100.00%", i.e. it also fires on every legal step. Identity is therefore *expected*, not evidence of agreement between independent methods. What it does establish is that SPC captures the **entire** oracle gap — 46.50% → 90.75%, i.e. 100% of it — rather than some fraction. The genuinely independent replication in this paper is §4.3(E6) vs §4.4/§4.6, where the identification strategies differ in kind (off-policy enumeration vs within-policy intervention).
>
> Caveat: `eval_asap_baseline.py` writes aggregates only, with no per-episode detail, so the identity above is argued from construction rather than a seed-by-seed match. Emitting per-episode records would make it mechanically checkable.

### 4.5 Robustness A — widened initial-geometry (heading bias)

Compressed to one subsection per scope decision: **this is a benchmark-change + retraining robustness check, NOT a second contribution.**

| Comparison (d=0, new geometry U(0,60)) | n | Kill rate | Paired diff | McNemar |
|---|---|---|---|---|
| SPC base @ U(0,60) | 400 | 362/400 = **90.50%** (Wilson [0.872, 0.930]) | — | — |
| SPC retrained under U(0,60) | 400 | 375/400 = **93.75%** (Wilson [0.909, 0.957]) | **+3.25 pp** | **p = 0.0146** |
| same pair, unpaired Fisher | 400 | — | — | p = 0.1146 (n.s.) |

- **Same weights, geometry-only change** (n=100, s42): 87% → 91% ⇒ widening the bias range makes the benchmark easier by ≈ **+4.5 pp**. This is a *benchmark* effect and must be reported as such.
- The +3.25 pp retraining gain is real and paired-significant (19 vs 6 discordant pairs, 76% favouring the retrained arm), robust across 4/4 bias bins (+1/+5/+5/+2 pp).
- **Caveats to state in the paper:** single seed family (seeds 42–441); d=0 only; Bonferroni α=0.0125 would put p=0.0146 just outside; JSBSim physics is not cross-machine reproducible.
- **Framing rule:** never present 87→95 as the effect of the launch-head correction. The launch-head correction is §4.4.

### 4.6 Robustness B — target evasion (`difficulty_level = 0.3`) — **DONE (E5)**

Target adds S-turn `±30°·d·sin(0.3t)` plus a missile-threat break-turn and a dive to `−800·d` m (floor 2000 m). Identical protocol to §4.4 — d=0 vs d=0.3 differ **only** in `--difficulty`, same seeds 20000–20399, same code path, identity recorded in each file's `run_meta`.

| Arm | d=0 kill | d=0.3 kill | Δ | d=0 CLR | d=0.3 CLR |
|---|---|---|---|---|---|
| Rule expert (fresh env) | **36.75%** (147/400) | **25.75%** (103/400) | **−11.00 pp** | **6.07%** | **5.40%** |
| BC round1 | 46.50% | **42.25%** (169/400) | −4.25 pp | 7.26% | **6.95%** |
| **SPC** | 90.75% | **90.75%** (363/400) | **0.00** | 100.00% | **99.11%** |
| **BC − expert margin** | +9.75 pp | **+16.50 pp** | **+6.75 pp** | — | — |

**All rows now come from one protocol**: fresh env per episode, seeds 20000–20399, geometry U(0,60), deterministic argmax, `difficulty` the only variable. Expert and BC share a run (`eval_paired_bc_vs_expert.py`), so their difference is paired by construction.

**Paired BC-vs-expert at both difficulties** (same seeds, fresh env):

| difficulty | Δkill | W / L / T | discordant | favouring BC | exact McNemar |
|---|---|---|---|---|---|
| d=0 | **+9.75 pp** | 63 / 24 / 313 | 87 | 72% | **3.48e-05** |
| d=0.3 | **+16.50 pp** | 85 / 19 / 296 | 104 | 82% | **3.79e-11** |

Per-seed flips under evasion: expert **+29 gained / −73 lost**, BC **+25 / −42**. Both are loss-dominated, but the expert's net is −44 seeds against BC's −17 — the asymmetry, not just the mean, drives the ordering below.

**All three policies on one seed set** (seeds 20000–20399, U(0,60), deterministic argmax), so the rows are directly comparable. Source table is generated by `scripts/collect_matrix.py` from each file's own `run_meta`, not assembled by hand.

**Paired test at d=0.3:** discordant **194 : 0** (once again unanimous), exact McNemar **p = 7.97e-59**. SPC launches 3.91/ep, hit rate 0.9994, `lost_target = 0`, `launch_quality.bad = 0`.

Four findings:1. **The correction is robust, and its benefit grows under evasion.** SPC's kill rate is *unchanged* (90.75% at both difficulties) while BC loses 4.25 pp, so the SPC−BC gap widens from +44.25 to **+48.50 pp**, and the discordant count rises from 177 to 194. **This widening is independently replicated by a different identification strategy**: in the frozen-trajectory oracle enumeration (§4.3, E6) the `asap`−inherited gap widens from **+40.0 to +50.0 pp** across the same difficulty change. One method re-trains the launch head with the maneuver frozen; the other freezes the maneuver and swaps in rule-based launch policies. Both say the same thing: evasion punishes the conservative launch policy, not the aggressive one.
2. **The conservatism is structural, not scenario-specific — and now measured on both policies.** BC's CLR barely moves (**7.26% → 6.95%**), and the expert's moves just as little (**6.07% → 5.40%**): the designed gate suppresses launches to the same degree whether or not the target manoeuvres. So the teacher's conservatism is not "caution that pays off when the target turns" — it is a fixed property of the predicate, and the imitation inherits it as a fixed property too.
3. **The identical aggregate is not an artefact — it was verified explicitly.** 363 kills at both difficulties looked like a bug, so it was tested: **368/400 episodes change length** (so `difficulty=0.3` is definitely applied) and **382/400 seeds keep the same kill outcome, with 18 flips split perfectly 9 gained / 9 lost**. The match is a genuine near-cancellation, not a no-op. Report this check in the paper — a reviewer will ask.
4. **Evasion cost is ordered by how conservative the launch policy is — now confirmed on one protocol.** Expert **−11.00 pp**, BC **−4.25 pp**, SPC **0.00 pp**. All three legs now come from fresh-env evaluations on the same seeds, so the ordering is no longer an artefact of mixing episode-construction protocols (the earlier estimate used an incomparable reused-env expert leg and suggested −7.00 pp; the corrected figure is larger and the ordering holds). The per-seed flips explain the mechanism directly: the expert loses 73 seeds and gains only 29, while BC loses 42 and gains 25 — a suppressed launch window is unrecoverable once the target turns away. Still a single seed family, so state it as a well-supported ordering rather than a law.
5. **The imitation gap itself widens under evasion.** BC's paired margin over the expert goes from **+9.75 pp** (d=0) to **+16.50 pp** (d=0.3), i.e. it more than doubles. This is a new observation and a useful one: the value of imitating-then-correcting this expert is *larger* in the harder regime, which is the opposite of the usual expectation that a stronger expert is needed as the problem gets harder.

**Expert arm at d=0.3** is running (task `s4Oqks`) as an unpaired reference; it now uses the seed-pairing patch (`run_one(seed=...)`) so it can be aligned to the same seeds.

> **Artifact status:** verified against live artifacts — `results/shoot_eval/E5_{bc_round1,spc}_d03_n400_s20000.json` and `E5_paired_bc_vs_spc_d03_n400.json`.

### 4.7 Ablation ladder (ordered by the decided priority)

| # | Ablation | Status | Purpose |
|---|---|---|---|
| A1 | **Fire-policy oracle enumeration on frozen trajectory** (§4.3) | **DONE** (E1 regenerated, tracked) | establishes that the inherited launch policy is suboptimal, maneuver held fixed |
| A2 | **SPC (learned head) vs ASAP rule oracle** (§4.4) | **DONE** — exact match on one seed set (90.75% both); completeness check, not independent validation | SPC captures 100% of the oracle gap |
| A3 | **`difficulty_level = 0.3`** (§4.6) | **DONE** — +48.50 pp, 194:0, p=7.97e-59 | interactive/reacting opponent |
| A4 | **Heading-shift robustness** (§4.5) | done (n=400) | benchmark sensitivity |
| A5 | Fire head from random init, encoder frozen | not run | is the BC encoder necessary, or is the mask enough? |
| A6 | **Full-network fine-tune on the same objective** (unfreeze all) | **DONE** 2026-09-18 (60-seed screening) | shows that *freezing* is what buys clean attribution, not the objective |

**A6 protocol (state it, because the fairness objection is obvious).** Identical objective, identical data, identical loss and identical initialisation as SPC; the **only** change is that every parameter is trainable instead of the launch head alone. Budget is set **more generously** than the fire-head-only run (`--epochs 40 --lr 3e-4` versus `--epochs 15 --lr 1e-2`), specifically so the ablation cannot be dismissed as under-trained. Two measurements decide the question, and they are measured, not asserted: (i) **maneuver deviation** — the heading/speed logit max-diff vs the frozen BC, which the fire-head-only run holds below `1e-9` and which full-network fine-tuning is expected to move; and (ii) **kill rate and CLR under the paper's primary protocol** (d=0, U(0,60), 400 paired seeds), so the result is directly comparable to the 46.50% / 90.75% pair of §4.4. A smoke run already confirms the instrument reads what it should: fire-head-only gives logit diff `<1e-9` and identical action sequences; full-network at a 6-epoch budget gives diff **0.613**, non-identical sequences, and only 22% window utilisation — the maneuver moved *and* the fire head did not finish learning. The gates therefore report `FAIL` in A6 mode **by construction**; the failure is the measurement.

**A6 result — the maneuver moves, and the correction is still worse than doing nothing clever.** Both pre-registered measurements came back on 2026-09-18:

| Measurement | SPC (fire head only) | A6 (full network) | Reading |
|---|---|---|---|
| heading/speed logit max-diff vs θ_BC | **< 1e-9** | **8.51** | the maneuver is provably untouched under SPC and provably *changed* under A6 |
| heading/speed action sequences identical | **yes** | **no** | same conclusion, measured behaviourally rather than in logit space |
| launch-mask agreement | yes | yes | both reach the target behaviour on the launch head |
| validation allowed-window accuracy | ≥ 0.995 | 0.9967 | neither arm can be dismissed as under-trained |

Kill rate on the **60-seed screening set** used by the ablation script (which evaluates the `id` and `dist2_3k` cells side by side, so all three arms see the same seeds):

| Arm (60 seeds, same seed set) | `id` cell | `dist2_3k` cell | window utilisation (`id`) |
|---|---|---|---|
| BC (inherited launch policy) | 40.0% | 15.0% | 7.2% |
| **A6 — full-network fine-tune** | **81.7%** | **66.7%** | **98.3%** |
| mask-permissive rule oracle | **95.0%** | **86.7%** | 100% |

The load-bearing reading is the middle row against the bottom row, because they share one seed set: **full-network fine-tuning recovers most of the gap but is still beaten by a rule that simply fires whenever the mask permits (81.7% vs 95.0%)**, and it pays for that with an 8.51-logit maneuver change. So A6 is dominated on both axes at once — less effective, and no longer attributable. Freezing is therefore doing real work: it is what makes "the launch decision alone explains the gap" a *statement* rather than a hope.

> **⚠ Two honest limits on this table.** (i) It is a **60-seed screening** result, not the paper's primary protocol: the same 60 seeds give BC 40.0% where n=400 gives 46.50%, and the rule oracle 95.0% where n=400 gives 90.75% — so the screening numbers carry roughly ±10 pp of small-sample spread and must **not** be quoted alongside the 400-seed figures as if they were one comparison. (ii) The 400-seed primary-protocol run of the A6 weights (`eval_bc_1v1.py --episodes 400 --seed 20000`, killing/CLR against the 46.50% / 90.75% pair) is **queued, not run** — deliberately deferred so the d=0.3 expert-CLR measurement gets the machine first. Until it lands, the A6 arm is a screening result and is presented as one.
| A7 | CLR for other binary decisions | conceptual | generality of the metric (future work) |

---

## 5. Discussion + Conclusion (0.5 page)

### 5.1 Why this is a general concern (and why it is an *interactive*-systems concern)
- Rule-based experts are **designed**, not optimized; a locally plausible quality gate can be globally costly in a closed loop, and the cost is invisible in the expert's own statistics.
- BC is a *faithful* propagator: the bias becomes a floor under every downstream method and every evaluation.
- The failure mode is characteristic of interacting-agent environments: the "correct" launch time depends on the trajectory the launch itself induces, so a static quality predicate cannot be validated offline.
- CLR is cheap, needs no oracle, and applies to any policy with a discrete commit-or-not decision (launch, hand-off, emergency stop, lane change).

### 5.2 Limitations
- **1v1 WVR only**; N-vs-N extension is future work (and the thesis's core).
- **Single expert, single designed gate**: we identify one instance of the failure mode, not its prevalence.
- **Suboptimality is established conditionally**: on the frozen BC maneuver trajectory (§4.3) and within the frozen-heads intervention (§4.4). We do not claim the mask-permissive policy is optimal in general.
- **Statistical scope**: the +3.25 pp geometry result is a single seed family and fails a Bonferroni α=0.0125.
- **Evasion is a single interpolated setting** (`difficulty = 0.3`, one scripted evasion family), not a learned or optimised opponent. "Interactive" here means a reacting scripted target, not an adversary trained against the policy.
- SPC requires knowing *which* head to correct; automating that (e.g. per-head CLR + ablation ranking) is open.

### 5.3 Future Work
- Automate head selection from per-head diagnostics.
- N-vs-N: per-agent CLR and per-agent surgical correction; does the bias compound with fleet size?
- Launch *timing* when launch cost/ammo scarcity makes launching on every mask-permitted step genuinely suboptimal — the regime where PPO becomes meaningful (project record documents that the current reward admits no such trade-off).
- Integrate into the hierarchical tactical architecture planned for the thesis.

### 5.4 Claim statement (exact wording to use)
> We do **not** claim that the mask-permissive policy is optimal in general. We claim that **a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime**: with the maneuver policy provably frozen (`max logit diff < 1e-9`, identical action sequences), correcting the launch head alone raises the kill rate from 46.50% to 90.75% on 400 paired seeds (+44.25 pp, exact McNemar p = 1.04e-53, with all 177 discordant seeds favouring the correction and none favouring the original), and off-policy enumeration on a fixed maneuver trajectory shows the inherited launch policy to be dominated by the mask-permissive policy within the enumerated fire-policy family.

---

## 6. Number Provenance & Integrity Ledger

### 6.0 Figure and table plan (final structure — Sean 2026-09-17 拍板)

正文固定为 **4 表 3 图**，每张都对应一个贡献，不设冗余表。

| # | 内容 | 服务贡献 | 数据来源 / 状态 |
|---|---|---|---|
| **Table 1** | Setup：任务、观测/动作空间、三个策略、评测协议（d=0、deterministic argmax、U(0,60)、fresh env/episode） | 全部 | §3.1，**可写** |
| **Table 2** | CLR diagnosis：Expert / BC / SPC × (allowed steps, fire decisions, CLR) | **C1** | §4.2，**完整** —— expert **6.07%**（fresh env, E3 v3 clr）/ BC **7.26%** / SPC **100.00%**，同 seeds、同 env 生命周期 |
| **Table 3** | Causal intervention A：frozen BC trajectory 上的 fire-policy 枚举（inherited vs mask-permissive 及 5 个选择性策略） | **C3** | §4.3，E1 已入库 |
| **Table 4** | SPC performance：BC vs SPC，d=0 与 d=0.3，kill / CLR / 配对检验 | **C2 + C3** | §4.4–4.6，E7/E5 已入库 |
| **Figure 1** | Framework：Expert → BC → CLR diagnosis → SPC | 叙事 | `results/shoot_eval/framework_fig1.{png,pdf}`（**已产出且数值已填齐**：expert 6.07% / BC 7.26%） |
| **Figure 2** | **Mechanism visualization**（不是普通轨迹图）：同一 seed、同一机动轨迹（已实测 max deviation = 0 m），Panel A = 距离/ATA 与决策点，Panel B = mask / BC fire / SPC fire 三行时间轴 | **C3** | `results/shoot_eval/mechanism_seed20007_d00.{png,json}`（**已产出并入库**，seed 20007：BC 3/41 合法步、超时；SPC 4/4、击杀） |
| **Figure 3** | Difficulty robustness：d=0 vs d=0.3 的 kill rate 与 gap | C3 稳健性 | `results/shoot_eval/robustness_fig3.{png,pdf}`（**已产出并入库**：三策略 × 两难度柱状图，带 Wilson 95% CI，并标注配对 gap **+44.25 pp** / **+48.50 pp**） |

> **Figure 1 与 Figure 3 的生成纪律**：两图都由 `scripts/make_paper_figures.py` 从 `results/shoot_eval/` 的活产物**读数**绘制（kill rate 取 `termination_reasons.target_killed` / `kill_rate`，配对 gap 取 `kill_rate.paired_diff_pp`），脚本内**不写任何数字常量** ⇒ 图与文中的数字不可能各自漂移。Figure 3 中 expert / BC 的两条腿来自 E3 与 E7/E5，全部为 fresh env、同 seeds、同 U(0,60)。缺失的专家 CLR 会渲染为 `pending` 并在 stderr 报警，脚本**不会**用其它口径的数字顶替（这正是 §4.4 的教训）。

> **Figure 2 的定位**：它不展示「飞得多漂亮」，只展示 *same engagement geometry, same maneuver trajectory, different launch decision*。因为机动头被冻结，BC 与 SPC 的轨迹**逐位相同**（脚本实测 max position deviation = 0.000e+00 m），所以图里只有**一条**轨迹 —— 这比画两条更能说明问题。选种子时须在 caption 中声明是**示例性单局**，聚合证据在 Table 3/4。

**Verified against live artifacts on 2026-09-16 (re-checkable):**

| Metric | Value | Artifact |
|---|---|---|
| **E7: BC kill rate** | **186/400 = 46.50%** (Wilson [41.67, 51.40]) | `results/shoot_eval/E7_bc_round1_d0_n400_s20000.json` |
| **E7: SPC kill rate** | **363/400 = 90.75%** (Wilson [87.51, 93.21]) | `results/shoot_eval/E7_spc_d0_n400_s20000.json` |
| **E7: paired diff / p / discordant** | **+44.25 pp / 1.04e-53 / 177:0** | `results/shoot_eval/E7_paired_bc_vs_spc_d0_n400.json` |
| **E7: BC CLR** | 7.26% (1223/16839) | same |
| **E7: SPC CLR** | 100.00% (1560/1560) — by construction | same |
| E7 identity | BC `sha256 aad05b45…`, SPC `sha256 36d79bd9…`, geometry U(0,60), d=0, argmax | `run_meta` in each file |
| **E5: BC kill @ d=0.3** | **169/400 = 42.25%** (Wilson [37.51, 47.14]) | `results/shoot_eval/E5_bc_round1_d03_n400_s20000.json` |
| **E5: SPC kill @ d=0.3** | **363/400 = 90.75%** (Wilson [87.51, 93.21]) | `results/shoot_eval/E5_spc_d03_n400_s20000.json` |
| **E5: paired diff / p / discordant @ d=0.3** | **+48.50 pp / 7.97e-59 / 194:0** | `results/shoot_eval/E5_paired_bc_vs_spc_d03_n400.json` |
| **E5: SPC invariance check** | 368/400 episode lengths differ, 382/400 same kill, 18 flips split 9:9 | computed from `E7_spc_*` vs `E5_spc_*` `episodes_detail` |
| ~~Matched expert @ d=0 / d=0.3 (**reused env**)~~ | ~~32.25% / 25.25%~~, ~~CLR 6.04% / 5.43%~~ | **EXCLUDED** — reused-env path; artifacts renamed `EXCLUDED_reused_env_expert_*` (§4.4 note). **Never cite.** |
| **Matched expert @ d=0, fresh env (E3)** | **147/400 = 36.75%**, CLR **6.07%** (1131/18639) | `results/shoot_eval/E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json` — reproduces the v2 kill numbers bit-for-bit and BC's CLR to the digit |
| **Matched expert @ d=0.3, fresh env (E3)** | **103/400 = 25.75%**, CLR **5.40%** (1066/19729) | `results/shoot_eval/E3_paired_bc_vs_expert_d03_n400_s20000_v2_clr.json` — also reproduces the kill side of the d=0.3 pair and BC's CLR (1194/17169) exactly |
| Full 3×2 matrix | auto-generated from `run_meta` | `scripts/collect_matrix.py` → `results/shoot_eval/matrix_E1_E5_E7.md` |
| **E1: fire-oracle enumeration, dist2_3k** | asap **86.7%** / delay_30 38.3% / delay_60 10.0% / dlz_mid 0.0% / dlz_deep 0.0% / interval_100 0.0% / frozen BC **15.0%** | `results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json` |
| **E1: reachability audit** | legal window median **4.0 steps/ep**, first-legal median 67, 1st→4th median 743.5, 8/60 end with window open | same |
| **E2: rule oracle vs SPC, same seeds** | both **363/400 = 90.75%**, launches 3.90, hit 1.0000, lost 0, reasons identical | `results/shoot_eval/E2_asap_oracle_id_n400_s20000.json` vs `E7_spc_*` |
| **E3: paired BC vs rule expert** | d=0: expert **36.75%** vs BC **46.50%**, Δ **+9.75 pp**, W/L/T 63/24/313, exact McNemar **p=3.48e-05** · d=0.3: expert **25.75%** vs BC **42.25%**, Δ **+16.50 pp**, W/L/T 85/19/296, **p=3.79e-11** | `results/shoot_eval/E3_paired_bc_vs_expert_{d0_n400_s20000_v2,d03_n400_s20000}.json` (both have `run_meta` incl. `env_lifecycle`) |
| **Evasion cost by policy** | expert **−11.00 pp** > BC **−4.25 pp** > SPC **0.00 pp** (all fresh-env, one protocol) | derived from the E3 pair + `E7`/`E5` |
| **Env-lifecycle confound (expert)** | reused-env expert **32.25%** vs fresh-env expert **36.75%** on identical seeds; per-seed kill vectors differ | `tests/check_expert_path_consistency.sh` |
| **A6: full-network fine-tune (60-seed screening)** | maneuver logits **8.51** (SPC: <1e-9), action sequences **not** identical, val allowed-acc 0.9967; kill `id` **81.7%** vs mask-permissive rule **95.0%** vs BC **40.0%** on one seed set ⇒ dominated on both axes. ⚠ **60 seeds ≠ the 400-seed primary protocol** — do not quote these alongside §4.4 | `results/shoot_eval/ablation_A6_fullnet_train.json` |
| **E6: oracle envelope, matched 2×2** | d=0: asap **93.3%** vs inherited **53.3%**; d=0.3: **96.7%** vs **46.7%** ⇒ gap **+40.0 → +50.0 pp**; window median 4.0/42.0 at both | `E6_default_d0_s60.json` + `E6_fire_oracle_target_evasive_s60.json` |
| Expert CLR (corroboration only) | 5.96% (560/9395) | recomputable from `data/expert/shoot_rule_expert.npz` — ⚠ **old geometry U(30,60)**: the npz mtime 2026-09-13 20:22 predates `fb48155` (2026-09-14 09:06) |
| BC CLR (corroboration only) | 7.37% (614/8327) | `results/health_check/fire_hesitancy.json` (2026-09-16 ⇒ U(0,60)). **The two corroboration rows are geometry-mixed — never present them as one matched pair** |
| Expert `fire_desired` ≡ `action[:,3]` on allowed steps | 560 = 560 | same |
| SPC @ U(0,60), n=100 s42 | 91/100 | `results/shoot_eval/eval_2x2_base_geoNew_s42.json` |
| SPC @ U(30,60), n=100 s42 | 87/100 | `results/shoot_eval/eval_2x2_base_geoOld_s42.json` (**= `eval_distilled_d0_s42.json`**) |
| geomA-retrained @ U(0,60), n=100 s42 | 95/100 | `results/shoot_eval/eval_geomA_d0_s42.json` |
| n=400 paired, base vs geomA | 362 vs 375, p=0.0146 | `eval_2x2_{base,geomA}_geoNew_n400_s42.json` + `geom2x2_n400_report.md` |
| Porpoising / low-level control | **no sustained porpoising**; script verdict = `PARTIAL` (alt p-p 15.2–16.9 m, alt osc 0.03–0.06 Hz; pitch p-p 3.1–4.6° but pitch std only ≈0.4°) | `results/health_check/porpoise_v2.json`. ⚠ its "1000-step cap < 30 s" parenthetical only holds at 1/60 s per step; at the environment's **0.2 s per step** 1000 steps = 200 s. **Do not cite that parenthetical until the runner behind this JSON is confirmed** — it may have driven JSBSim directly rather than through `BaseEnv`. Residual: revisit turn-pattern pitch during M2. |
| d=0.3 smoke | 2/2 kills | `results/shoot_acmi_d03/manifest.json` |

**Docs-attested only — JSON lost, MUST be regenerated before submission:**

| Metric | Value | Doc source | Regenerating script |
|---|---|---|---|
| ~~Expert ID kill~~ | ~~35.8%~~ → **use 36.75%** (fresh-env, E3 paired) — **not** the reused-env 32.25% | superseded | — |
| ~~BC ID kill~~ | ~~43.2%~~ → **use 46.50%** (E7) | superseded | — |
| ~~ASAP rule oracle ID kill~~ | ~~90.6%~~ → **use 90.75%** (E2, matched) | superseded | — |
| ~~SPC ID kill~~ | ~~91.2%~~ → **use 90.75%** (E7) | superseded | — |
| ~~Fire-policy oracle table (§4.3)~~ | ~~asap 73.3% … BC 6.7%~~ → **superseded by E1** (old geometry U(30,60)) | `docs/plan_bc_rule_expert.md` | — |
| dist2_3k: BC / ASAP / SPC | 6.7% / 74% / 77% | `docs/summary_phase1.md` §2 | `eval_scenario_matrix.py` |

**Not yet measured:** multi-seed-family extrapolation (seeds 42–441 and 20000–20399 are each a single family; see §4.5 caveat 1); a seed-by-seed oracle-vs-SPC match (blocked by E2 writing aggregates only).

---

## 7. Experiment Queue (execution order)

### 7.0 MANDATORY RESULT-FILE SPECIFICATION (introduced 2026-09-16)

**Every evaluation JSON must carry its own identity block.** This exists because a core claim was once read off a *filename* that was assumed to identify a model and did not (`eval_distilled_d0_s42.json` and `eval_2x2_base_geoOld_s42.json` are the same run; the real BC number was 43.2%, not 87%).

Implemented in `scripts/eval_meta.py` (`build_run_meta`) and emitted as a top-level `run_meta` object:

| field | purpose |
|---|---|
| `model_id` | explicit identifier passed on the CLI (`bc_round1` / `spc_distilled` / `rule_expert`) |
| `checkpoint` + `checkpoint_sha256` | kills "same filename, new weights" |
| `seed_range` | makes paired tests verifiable |
| `geometry`, `difficulty`, `action_mode` | guards against comparing across settings |
| `env_lifecycle` | **added 2026-09-17** — whether the simulator was rebuilt per episode or reused. Two runs can match on geometry, difficulty, seeds and weights and still be incomparable if they differ here (see §4.4). |
| `git_head`, `script`, `timestamp_utc` | provenance |

**⚠ Honest coverage status of `env_lifecycle`.** The field was added to the contract on 2026-09-17, so only artifacts produced after that date carry it: `E3_paired_bc_vs_expert_d0_n400_s20000_v2.json` and `E3_paired_bc_vs_expert_d03_n400_s20000.json`. **`E1_*`, `E5_*`, `E6_*`, `E7_*` do not carry it, and were not retro-edited** — back-filling a provenance field into a finished artifact would manufacture evidence rather than record it. For those files the lifecycle is established from the *producing code* instead, which is checkable at the recorded `git_head`: `eval_bc_1v1.py` constructs its `BaseEnv` inside the per-episode loop, and `fire_oracle_audit.py` likewise; `generate_shoot_rule_expert.py --validate` constructs one env *before* the loop and now labels itself accordingly. Any future re-run attaches the field automatically.

`scripts/paired_mcnemar.py` **refuses to compute a p-value** if any of these disagree across arms, or if the two arms share a `model_id`/checkpoint hash. Identity is asserted, not assumed.

### 7.1 Queue

| Step | Command (WSL Ubuntu shell) | Cost | Status |
|---|---|---|---|
| **E7** | `eval_bc_1v1.py --weights <round1> --model-id bc_round1 --episodes 400 --seed 20000 --difficulty 0` and the same with `<asap_distilled>` / `--model-id spc_distilled`, then `paired_mcnemar.py --a ... --b ...` | ≈1.8 h | **DONE** — 46.50% vs 90.75%, **+44.25 pp**, exact McNemar **p=1.04e-53**, discordant **177:0** |
| **E5** | same two commands with `--difficulty 0.3`, plus the expert arm via `eval_paired_bc_vs_expert.py --difficulty 0.3` | ≈3.4 h total | **DONE** — 42.25% vs 90.75%, **+48.50 pp**, p=7.97e-59, discordant **194:0**; expert arm 25.75% (fresh env, in the E3 d=0.3 pair) |
| **E6** | `fire_oracle_audit.py --cell target_evasive --start-seed 20000 --seeds 60 --oracles asap,bc` + matched d=0 cell `id_bias30_60` | ≈50 min | **DONE** — matched 2×2: gap **+40.0 → +50.0 pp**; replicates the §4.6 widening by an independent method |
| **E1** | `fire_oracle_audit.py --cell dist2_3k --seeds 60 --oracles all` | ≈2.2 h | **DONE** — asap **86.7%** vs inherited **15.0%**; all selective policies worse (§4.3). Re-running once more to attach `run_meta` |
| **E2** | `eval_asap_baseline.py --mode id --id-seeds 400 --start-seed 20000` | ≈1 h | **DONE** — rule oracle 90.75%, exactly matching SPC; required adding `--start-seed` (the script previously could only run seeds 0..N-1) |
| **E3** | `eval_paired_bc_vs_expert.py --seeds 400 --start-seed 20000 --difficulty {0,0.3}` | ≈1.7 h per difficulty | **DONE** — d=0 expert **36.75%** vs BC 46.50% (+9.75 pp, p=3.48e-05) · d=0.3 expert **25.75%** vs BC 42.25% (+16.50 pp, p=3.79e-11). This is the only fresh-env expert evaluation, and the only pair carrying `env_lifecycle` |
| **E3-CLR** | the same command with `--model-id paired_bc_vs_expert_d0_clr` after the CLR patch | ≈1.7 h | **DONE** — `E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json`: expert **6.07%** (1131/18639) and BC **7.26%** (1223/16839) at d=0, with the kill side reproducing v2 exactly. Fills the CLR table's expert row and the abstract. d=0.3 queued behind it |
| **E4** | already covered by E7's `spc_distilled` arm | — | folded into E7 |

**E6 interface check — DONE 2026-09-17 (zero GPU spend).** Three findings that change how the command must be written:
1. **`fire_oracle_audit.py` has no `--difficulty` flag.** Verified against `--help`: the real flags are `--cell --seeds --start-seed --oracles --weights --out --device`. Passing `--difficulty 0.3` fails with `error: unrecognized arguments`. Difficulty is supplied **only** through the cell config.
2. **`--cell target_evasive` is valid** and resolves to `{'difficulty_level': 0.3}`. It works because the script does a label lookup over `CELLS` (`eval_scenario_matrix.py`), so any of the 13 labels is accepted.
3. **Cost is much higher than the ≈1 h first assumed.** The script loops oracles × seeds, and all 7 oracles × 400 seeds = 2800 rollouts ≈ 8 h. The paper needs only the reference pair, so scope it: `--oracles asap,bc` → 800 rollouts ≈ 2.2 h at n=400; or `--seeds 60` to match the original `dist2_3k` audit ≈ 20 min.
   Also note the frozen trajectory comes from `--weights` (default BC round1), so an E6 run at d=0.3 freezes **BC's d=0.3** maneuver — which is the correct comparator for §4.3 at d=0.3, but it is *not* SPC's trajectory.

**Scope decision (Sean):** only artifacts that enter the paper or its supplement get regenerated. Scenario-matrix and stress suites are dropped unless a reviewer asks.

**Note:** BC's CLR (7.37%, measured under the *current* U(0,60) geometry) and BC's kill rate (43.2%, from the sealed Phase-1 record under the *old* U(30,60) geometry) came from different geometry generations. **E7 resolved this**: kill rate, CLR, and geometry now all sit on seeds 20000–20399 under U(0,60) — see §4.4. The 43.2%/91.2% pair is **superseded**; use 46.50%/90.75%.


---

## 8. Submission Target

- **Venue**: AAMAS 2027 short paper (main conference)
- **Deadline**: ≈2026-11
- **Length**: 5–7 pages
- **Positioning check** (must pass before writing): the contribution reads as *expert-induced policy bias in interacting-agent environments, with a causal-identification methodology*, not as *a classifier head was retrained*.

---

## 9. TODO Before Submission

**Integrity prerequisites**
- [x] **E7**: paired McNemar BC vs SPC at n=400 — **DONE**, +44.25 pp, p=1.04e-53, discordant 177:0
- [x] **E5**: d=0.3 for BC / SPC on the shared seed set — **DONE**: 42.25% vs 90.75%, +48.50 pp, p=7.97e-59, discordant 194:0
- [x] **Expert arm at d=0.3** — done as the fresh-env paired arm inside `E3_paired_bc_vs_expert_d03_n400_s20000.json` (**25.75%**), which supersedes the reused-env `E5_expert_d03` figure as the citable number
- [x] **d=0 vs d=0.3 comparison write-up (§4.6)** — SPC invariant at 90.75%, gap widens **+44.25 → +48.50 pp**; five findings written up
- [x] Regenerate the paper's evidence artifacts — E1 (oracle, reproducibility-verified), E2 (rule oracle), E3 (paired BC vs expert, both difficulties) all done
- [x] **Produce a fresh-env expert CLR** — **DONE 2026-09-17**: `eval_paired_bc_vs_expert.py` now computes CLR, and `E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json` gives **6.07%** (1131/18639) at d=0 on the same 400 seeds, same protocol and same env lifecycle as every other number. The same artifact reproduces the v2 kill numbers bit-for-bit (expert 36.75%, BC 46.50%, +9.75 pp, W/L/T 63/24/313) and BC's CLR to the digit, so the instrumentation is additive. Filled into the abstract, §4.2, §4.6, §6 and Figure 1. The **conservatism ordering (expert 6.07% < BC 7.26% ≪ SPC 100%) is now assertable**; it was withheld until this leg existed. **Complete at both difficulties.** The d=0.3 leg then returned **5.40%** (1066/19729) versus BC **6.95%** (1194/17169), again reproducing the kill side exactly (25.75% / 42.25%, +16.50 pp, W/L/T 85/19/296) and BC's CLR to the digit. The 3×2 table now has all six cells measured under one protocol, and the ordering — expert **6.07% / 5.40%** < BC **7.26% / 6.95%** ≪ SPC **100% / 99.11%** — holds at both difficulties.
- [ ] Know the geometry of every quoted number — **a 2026-09-17 audit found the two "200-episode diagnostic" rows were labelled `current geometry` and were not**: `shoot_rule_expert.npz` predates `fb48155`, so the expert's 5.96% is U(30,60) while the BC 7.37% beside it is U(0,60). Because the two rows sat in the same table and read as a matched pair, the paper was one edit away from quoting a geometry-mixed comparison. Both rows are now relabelled as corroboration-only, and the authoritative expert figure is the re-measured 6.07%. Apply the same question to any *future* number before it enters a table.
- [x] Decide whether to keep the reused-env path's `--validate` diagnostic at all — **decided 2026-09-17: keep it, but fence it.** It is still the cheapest rule-behaviour check, so it stays; it now (i) prints a five-line lifecycle banner at the top of its output, (ii) warns in its `--help` text, (iii) writes `env_lifecycle='reused across episodes (… NOT comparable to fresh-env arms)'` into `run_meta`, and (iv) its artifacts are renamed `EXCLUDED_reused_env_expert_*` so `collect_matrix.py` skips them. Delegating the exclusion to a filename a human must remember was the original failure mode; the fence is now mechanical.
- [x] Reconcile BC's CLR geometry with BC's kill-rate geometry — E7 does this (both now U(0,60), seeds 20000–20399)
- [x] `run_meta` identity block + `paired_mcnemar.py` identity guards (2026-09-16)
- [x] Expert seed-pairing patch (`run_one(seed=...)`) — previously impossible
- [x] **Re-measure SPC vs the rule oracle on one seed set** — done: exactly equal (90.75% both); framed as a completeness check, not independent validation

**Content**
- [x] **Related Work** — rewritten 2026-09-17 with 13 citations checked against the live record (arXiv IDs, venues, page numbers). **Fully closed 2026-09-18: the three outstanding items are resolved** — Rusu et al. is ICLR 2016 (conference track), Pomerleau is NIPS'88 pp. 305-313, and Hinton is a NIPS 2014 *workshop* paper with no page numbers. **Three errors caught in total**: a nonexistent "Pang et al. 2024"; "Actor-Learner Distillation" (a misnomer for Actor-Mimic); and ROME's author list, which was wrong ("Meng, Wang, Pfaff & Yang" → **Meng, Bau, Andonian & Belinkov**).
- [x] **Mechanism figure** — `scripts/make_mechanism_figure.py` → `results/shoot_eval/mechanism_seed20007_d00.png` (tracked). Seed 20007: BC uses 3/41 legal steps and times out; SPC uses 4/4 and kills. The maneuver is bit-identical over 842 common steps (max deviation 0.0e+00 m). Shipped as a range/ATA-versus-time panel instead of a top view, because with a bit-identical maneuver a top view contains one line and no visible divergence.
- [x] **Appendix A–D** — internal↔paper name map (including the retired `SHD` and the misleading `id_bias30_60` cell), the mask-versus-`fire_desired` constant-by-constant table with the strict-subset proof, the seven-arm oracle family, and the reproduction commands.
- [x] **Figures 1 and 3** — `scripts/make_paper_figures.py` → `results/shoot_eval/framework_fig1.{png,pdf}` and `robustness_fig3.{png,pdf}` (both tracked). Every plotted value is read from the tracked E-numbered artifacts; the script holds no numeric constants, so figure and text cannot drift apart. Figure 3 carries Wilson 95% CIs and the paired +44.25 / +48.50 pp spans.
- [x] Acknowledge the rule expert's provenance honestly (§3.1) — done: the text now states outright that there is no external publication to cite, that it is this project's own rule controller, and that no state-of-the-art claim is made for it (only that it is competent and measurably conservative). The paper's claims are all teacher-conditional, so the teacher's origin is not load-bearing.
- [ ] Ablation A5/A6 (random-init fire head; full-network) if space permits
- [ ] Format to AAMAS template


---

## Appendix A. Internal names versus paper names

Artifact names predate the terminology used here and several of them are actively misleading. This table is the authority: **never read a claim off a filename** (see §7.0 for why that rule exists).

| internal identifier | paper term | note |
|---|---|---|
| `shoot_bc_round1_baseline.pth`, `model_id = bc_round1` | **BC round1** | MLP 256-256-128 + 4 heads, 200 expert episodes, masked cross-entropy |
| `shoot_bc_asap_distilled.pth`, `model_id = spc_distilled` | **SPC** | legacy internal wording ("distilled", "asap"); **the artifact name is not the method** |
| `shoot_bc_asap_geomA.pth` | geomA-retrained | used only in the §4.5 geometry 2×2. **Not an arm of this paper** — its 95/100 is geometry + retrain, not the launch-head correction |
| `asap` oracle | **mask-permissive policy** | fires on every step the environment-level launch mask permits (Appendix C) |
| `fire_desired` (in `generate_shoot_rule_expert.py`) | **the expert's extra gate**, launch-quality gate | the designed restriction strictly tighter than the mask (Appendix B) |
| `difficulty_level` | **d** | scripted target evasion, 0 = straight-and-level |
| `min_heading_bias_deg` | heading-bias geometry knob | `0.0` = today's default U(0,60); `30.0` = the pre-`fb48155` U(30,60) |
| `dist2_3k`, `id_bias30_60`, `target_evasive` | evaluation cells | `fire_oracle_audit.py` cell configurations (2–3 km chase band etc.). ⚠ **`id_bias30_60` is a misleading name with a correct config**: its config is `{}` = task defaults, and since `fb48155` the default is **U(0,60)**, not U(30,60). It means "the current default geometry". Kept because it is already baked into recorded `model_id` values; read `cell_config` in `run_meta`, never the string. |
| `SHD` | **retired — do not use** | acronym from v1 drafts that never named anything measurable |
| `E1 … E7` | evidence artifacts | numbered per §6/§7.1; the number is a label, the `run_meta` is the identity |

## Appendix B. The environment's launch mask versus the expert's extra gate

**The mask.** The flat action mask is `[speed(3), heading(5), altitude(1), fire(2)]`. The launch branch is flat index **10**. `mask[10] = 1` requires *all* of:

| # | condition | constant (`src/environment/singlecombat_shoot_task.py`) |
|---|---|---|
| C1 | missiles remaining > 0 | `NUM_MISSILES = 4` |
| C1b | steps since last launch ≥ `MIN_ATTACK_INTERVAL` | `30` steps = **6 s at 5 Hz** |
| C2 | target alive | — |
| C3 | ATA < `MAX_ATTACK_ANGLE` | `15.0` deg |
| C4 | `MIN_ATTACK_DISTANCE` ≤ range ≤ `3000 + 5000·(AA/180)` | `1500.0` m |
| C5 | closure < 0 (closing on the target) | — |

**The expert's gate.** The rule expert never fires merely because `mask[10] = 1`. It additionally requires its own `fire_desired`, a pure function of the observation:

| axis | environment permits | expert additionally requires | relation |
|---|---|---|---|
| ATA | < 15.0 deg | < `ATA_STRICT` = **10.0** deg | strictly tighter |
| closure | < 0 m/s | < `−CLOSURE_MIN` = **−5.0** m/s | strictly tighter |
| range | 1500 m … `r_max(AA)` | DLZ depth in `[0.25, 0.75]`, where depth = `(range − 1500)/(r_max − 1500)` | strictly tighter **and two-sided** |

**Why "strict subset" is exact, not rhetorical.** With `r_max(AA) = 3000 + 5000·(AA/180)`:

- lower bound: `1500 + 0.25·(r_max − 1500) > 1500` whenever `r_max > 1500`, i.e. for every `AA > 0`;
- upper bound: `1500 + 0.75·(r_max − 1500) = 0.75·r_max + 375 < r_max` whenever `r_max > 1500`.

So on every axis the expert's admissible set is a strict subset of the environment's. This is the mechanical content of the paper's central claim that *the expert carries a conservative engagement preference that the environment does not require* — not a claim about the expert's competence.

**Consequence to keep explicit: the mask, hence CLR's denominator, is policy-dependent.** Firing triggers C1b, which *closes* the window; abstaining leaves it open. A policy that fires therefore collapses its own opportunity set, while a policy that abstains inflates it. Measured on the Figure-2 seed: **41** legal steps for BC versus **4** for SPC (a factor of 10.25; the aggregate in §4.2 is ≈10.8×). CLR is consequently a **descriptive** statistic across policies; the causal claim rests on the frozen-trajectory enumeration of §4.3, where the opportunity set is held fixed by construction.

## Appendix C. The enumerated fire-policy family

All seven arms are evaluated on the *same frozen maneuver trajectory* per seed (the BC heading/speed actions), so only the launch decision varies (`scripts/fire_oracle_audit.py`):

| arm | rule |
|---|---|
| `asap` | fire at the first legal step (the cooldown is already inside the mask), then whenever legal |
| `delay_30` | fire at the first legal step ≥ first-legal + 30 |
| `delay_60` | fire at the first legal step ≥ first-legal + 60 |
| `dlz_mid` | fire only when DLZ depth ∈ [0.4, 0.6] |
| `dlz_deep` | fire only when DLZ depth ≥ 0.6 |
| `interval_100` | fire ASAP, then at most every 100 steps while legal |
| `bc` | the inherited BC launch head (the reference arm) |

`dlz_mid` and `dlz_deep` deliberately bracket the expert's own band `[0.25, 0.75]`, so the family contains a faithful approximation of the expert's gate as well as the mask-permissive extreme.

## Appendix D. Reproduction commands

All commands are for the **WSL Ubuntu shell**, from the repository root, using the project interpreter `/home/sean/miniconda3/envs/marl_env/bin/python`. Trace JSONs, logs and `.pth` checkpoints are excluded from version control, except the `E*_*.json` and `mechanism_*` artifacts, which are tracked.

Primary endpoints (`§4.4`, Table 4):

```
python scripts/eval_bc_1v1.py --episodes 400 --seed 20000 --difficulty 0.0 \
    --weights data/expert/shoot_bc_round1_baseline.pth --model-id bc_round1 \
    --out results/shoot_eval/E7_bc_round1_d0_n400_s20000.json
python scripts/eval_bc_1v1.py --episodes 400 --seed 20000 --difficulty 0.0 \
    --weights data/expert/shoot_bc_asap_distilled.pth --model-id spc_distilled \
    --out results/shoot_eval/E7_spc_d0_n400_s20000.json
python scripts/paired_mcnemar.py \
    --a results/shoot_eval/E7_bc_round1_d0_n400_s20000.json \
    --b results/shoot_eval/E7_spc_d0_n400_s20000.json \
    --label-a bc --label-b spc \
    --out results/shoot_eval/E7_paired_bc_vs_spc_d0_n400.json
```

Frozen-trajectory enumeration (`§4.3`, Table 3):

```
python scripts/fire_oracle_audit.py --cell dist2_3k --seeds 60 --start-seed 0 \
    --oracles all --weights data/expert/shoot_bc_round1_baseline.pth \
    --out results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json
```

Matched BC-versus-expert pair (`§4.4`, fresh env per episode — note this path also computes CLR as of 2026-09-17):

```
python scripts/eval_paired_bc_vs_expert.py --seeds 400 --start-seed 20000 \
    --difficulty 0.0 --weights data/expert/shoot_bc_round1_baseline.pth \
    --model-id paired_bc_vs_expert_d0 \
    --out results/shoot_eval/E3_paired_bc_vs_expert_d0_n400_s20000_v2.json
```

The ε-difficulty arms are the same two commands with `--difficulty 0.3`. The CLR measurement is the same command again with `--model-id paired_bc_vs_expert_d0_clr` and `--out results/shoot_eval/E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json`.

Figures 1–3 and the auto-generated matrix:

```
python scripts/make_mechanism_figure.py --seed 20000 --scan 30
python scripts/make_paper_figures.py            # Figures 1 and 3
python scripts/collect_matrix.py --glob 'results/shoot_eval/E*_*.json' \
    --out results/shoot_eval/matrix_E1_E5_E7.md
python scripts/audit_paper_numbers.py           # recompute every headline number
```

`make_paper_figures.py` reads every plotted value from the tracked `E*_*.json` artifacts and holds **no numeric constants**, so Figures 1 and 3 cannot drift away from the text; if an artifact is missing the figure renders `pending` and warns rather than substituting a value from another basis. `audit_paper_numbers.py` recomputes the headline numbers straight from the artifacts and then checks that the draft still quotes them, including two must-be-absent guards: the abstract's expert-CLR placeholder (once the CLR run completes) and the bogus `2.5e-65` p-value produced by feeding tie counts into `exact_mcnemar`. Both guards are themselves documented in the draft, so the checker ignores occurrences sitting inside a line that is explicitly talking about the trap.

`collect_matrix.py` reads **only** each file's `run_meta` — never filenames — and skips any artifact whose name is marked `EXCLUDED_`, so retired evidence cannot silently re-enter the table.
