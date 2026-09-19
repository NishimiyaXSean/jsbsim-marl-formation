# Small Paper Draft v2 — 2026-09-16

> *(internal, delete before submission)* **v1 → v2 重大更正（2026-09-16）**
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

Behaviour cloning (BC) from rule-based experts is a standard bootstrap for air-combat policies, under the implicit assumption that the expert's decisions are worth copying. On a JSBSim F-16 within-visual-range 1v1 missile-engagement benchmark we test that assumption and find it fails in a specific, measurable way: the hand-designed expert launches on only **6%** of the steps its environment permits, and BC reproduces that restraint faithfully — a restrictive launch preference encoded in the expert rule, not an imitation defect. We introduce the **Conditional Launch Rate (CLR)** to quantify it, and **Surgical Policy Correction (SPC)**, a localized intervention that retrains *only* the launch head while every other parameter — and therefore the entire maneuver policy — stays bit-identical. Correcting that single decision dimension recovers the performance of an oracle that fires whenever the environment permits launch: **46.50% → 90.75%** kill rate on 400 paired seeds (**+44.25 pp**, exact McNemar **p = 1.04e-53**, all 177 discordant seeds favouring the correction), matching the level a frozen-trajectory intervention reaches in the hardest close-range cell (**15.0% → 86.7%**). Under an evading target the benefit grows rather than decays (**+48.50 pp**). The measured failure mode was localized to one discrete decision dimension, and correcting only that dimension recovered oracle-level performance while preserving the original maneuver — the oracle being a reference defined by the environment's launch mask, not a claimed optimum.

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
- **Terminology, fixed once (2026-09-18):** we say **restrictive launch preference** when describing the *mechanism* — the expert's designer would call that gate *precision*, and locally it is; we reserve **expert-induced decision bias** for the *measured, globally costly consequence* of inheriting it. We do not assume the expert rule is erroneous in isolation; we test whether its preference remains beneficial under closed-loop interaction, which is a different question.

### 1.3 Contributions

> *(internal, delete before submission)* 结构按 Sean 2026-09-17 定型。**第一贡献是诊断，不是性能提升**；叙事主线：*专家含有一个隐藏的、局部但灾难性的决策偏差；BC 忠实复制它；一个局部策略修正即可在不改变机动策略的前提下恢复性能。* 措辞纪律：不说 "expert 很差"，说 *the expert contains a restrictive launch preference that is not required by the environment constraints*（环境允许 ≠ 专家愿意 —— 这正是机制所在）。

1. **C1 — Diagnosis: expert-induced decision bias, quantified.** We introduce the **Conditional Launch Rate (CLR)**, `P(a_fire=1 | m_fire=1)` — an oracle-free metric that transfers to any commit-or-not decision (launch, hand-off, emergency stop, lane change) — and find that the rule expert launches on only ~6% of the steps its environment permits, with BC reproducing that preference faithfully (~7%). We localise the cause to a **designed** launch-quality gate that is strictly stricter than the environment's legality mask (§3.2). This is not a case of the expert being incompetent: the expert carries a **restrictive launch preference the environment does not require**, and its closed-loop cost is the decision bias this paper measures.
2. **C2 — Surgical Policy Correction: a localized *intervention* on the decision dimension the diagnosis identifies.** SPC modifies **only the single binary launch head** so that it fires whenever the environment permits, while **every other parameter — and therefore the entire maneuver policy — is provably preserved** (bit-identical heading/speed logits and action sequences, §3.3). The contribution is not the implementation detail of training one head; it is *isolating the behavioural dimension responsible for the failure*: the maneuver is bit-identical, so the residual difference can only be attributed to the launch decision. A5/A6 (§4.7) then show that the benefit comes from **which parameters are allowed to move** — not from how many, and not from where that head started.
   > *(internal, delete before submission)* 措辞纪律：说 **intervention**，不说 fine-tune/training。
3. **C3 — Causal and locality validation: two interventions, one conclusion.** Two complementary causal arguments (§4.3, §4.4), plus the A5/A6 validation that locality is the key factor (§4.7). **The two interventions live at different scales, and each must be cited with its own scope attached**:
   - **Intervention A (cell-specific frozen-trajectory enumeration, `dist2_3k`, 60 seeds)**: 15.0% → 86.7% — *the launch decision alone can explain the performance gap*;
   - **Intervention B (primary 400-seed policy intervention, seeds 20000–20399)**: 46.50% → 90.75% (paired, 177:0, p≈1e-53) — *the correction can be internalized into the policy*.
   - The effect survives and grows under an evading target (+44.25 → +48.50 pp), and is replicated by an independent identification strategy (§4.3 E6 matched 2×2: +40.0 → +50.0 pp).

**Structure of the argument** (the body follows this, one level above a conventional ablation report):

```
Observation: CLR is anomalously low
   -> Where is the failure?     -> frozen-trajectory intervention => the launch decision is the causal factor
   -> Can the policy be repaired? -> SPC                       => the correction can be internalised
```

[[figure: framework_fig1.pdf | wide:0.82 | Framework. A hand-designed rule expert is imitated by behaviour cloning; the inherited launch behaviour is diagnosed with CLR; and the launch head is corrected by SPC with the maneuver held provably frozen. Annotated rates are the primary-protocol values (Table 2); the frozen-trajectory figure is the enumeration of Table 3.]]

---

## 2. Related Work (1 page)

> *(internal, delete before submission)* 结构按 Sean 2026-09-17 拍板：**三个关键词，不堆 RL 文献**。本文不是 air-combat RL 论文，related work 必须服务三个贡献（诊断 / 局部修正 / 因果验证），不要让 §2 变成 MARL 综述。

### 2.1 Imitation learning and expert demonstration

- **BC and its standard pathology.** Cloning a demonstrated policy by supervised learning goes back to ALVINN [@pomerleau1989alvinn] and was scaled to end-to-end driving by Bojarski et al. [@bojarski2016end]. The canonical account of its failure mode is DAgger [@ross2011dagger], which attributes the failure to **distribution shift / compounding error** between the learner's and the expert's state visitation.
- **Core sentence.** *Existing IL methods usually assume the demonstrations are informative and near-optimal; the learner's objective is fidelity to the expert.*
- **The literature that does relax optimality relaxes it as a scalar.** T-REX learns a reward from ranked suboptimal demonstrations [@brown2019trex]; LERP models suboptimality as reward noise [@huo2023lerp]; IRLEED models a demonstrator's suboptimality as *reward bias plus action variance* [@beliaev2025irleed]; IL-from-imperfection reweights supplementary suboptimal data [@li2023imperfect]. In every case suboptimality is a **scalar quality** to be estimated, ranked, or down-weighted.
- **Our gap.** *What if the expert's suboptimality is not scalar but structural - a single, nameable decision gate that the environment does not require?* A scalar quality cannot recover it: the defect is invisible in task-level success statistics (§4.2) and costly only in closed loop. We do not down-weight the expert; we **localize** the gate and correct exactly it.

### 2.2 Policy distillation and model editing

- Knowledge distillation into a student [@hinton2014distilling], policy distillation [@rusu2016policy], and Actor-Mimic [@parisotto2016actormimic] all **transfer or compress a policy globally**, with fidelity to the teacher as the objective.
- Parameter-efficient adaptation (LoRA [@hu2022lora]) and model editing (ROME [@meng2022rome]) do modify a small subspace while freezing the remainder - but they edit **learned knowledge** and evaluate by task success, not by attributing a residual to one behavioural dimension.
- **Core sentence.** *Existing approaches transfer or modify policies globally; we study localized correction of a single decision dimension, and we use the freezing itself as a causal identification device.* The distinction from distillation is precise: SPC's objective is **not** to approach the teacher but to **selectively depart** from it on one head. Freezing is not an engineering convenience here - it is the identification strategy (§4.4).

### 2.3 Autonomous air combat decision-making

- **Closest neighbours, stated plainly.** Li et al. [@li2024cross] learn a within-visual-range 6-DOF combat policy whose top layer decides autopilot commands **and missile launch**, with BC and PPO cross-coordinated. Li et al. [@li2026imitative] learn a "pursuit-lock-launch" policy with TD3 + BC and an adaptive imitation weight in Harfang3D. Both clone a launch decision from demonstrations in a WVR missile-engagement setting.
- **Where we differ - two ways, both load-bearing.** (i) They treat the expert as a **bootstrap to be surpassed**; we treat it as an **object of diagnosis**, and we keep the maneuver provably frozen so the residual is attributed to the launch head rather than to aggregate learning. (ii) Their success criterion is aggregate task success; we add a **conditional** criterion (CLR) that exposes failures aggregate success hides. Notably, Li et al. [@li2026imitative] present selective firing as a virtue - their policy "fires only when a hit was highly probable rather than spamming launches" - the **opposite prior** to the one this paper tests, and in our regime the measurement goes the other way (§4.3).
> *(internal, delete before submission)* **Constraint.** This subsection stays under half a page. It is scene-setting, not contribution positioning: a reviewer should enter this paper through §2.1/§2.2, not file it as another air-combat RL paper. *(2026-09-19: the three background citations — McGrew/Pope/Yang — were cut from the paper under the page budget; they are recorded here so the .bib can still carry them if a related-work sentence is restored.)*

> **Citation status (re-checked 2026-09-18 against the live record, not from memory).** Verified: Pomerleau (**NIPS'88**, pp. 305-313; presented 1988, MIT Press 1989 - the year is now confirmed, not inferred), Ross/Gordon/Bagnell (AISTATS 2011), Bojarski (arXiv:1604.07316), Brown et al. (ICML 2019 pp. 783-792), Huo/Wang/Xu (AAAI 2023 pp. 7953-7961), Beliaev/Pedarsani (arXiv:2402.01886), Li Z. et al. (NeurIPS 2023), Parisotto/Ba/Salakhutdinov (ICLR 2016, arXiv:1511.06342), McGrew et al. (JGCD 33(5):1641-1654), Pope et al. (ICUAS 2021 pp. 275-284), Yang et al. (IEEE Access 8:363-378), Li L. et al. (Neurocomputing 584:127591, 2024), Li S. et al. (ACM TAAS 21(1):6, 2026), **Rusu et al. (ICLR 2016 conference track, arXiv:1511.06295)**, **Hinton/Vinyals/Dean (NIPS 2014 Deep Learning Workshop, arXiv:1503.02531 - a workshop paper, so it has no page numbers and none are invented here)**, **LoRA (Hu et al., ICLR 2022, arXiv:2106.09685)**, **ROME (Meng, Bau, Andonian & Belinkov, NeurIPS 2022, Advances in NeurIPS 35, pp. 17359-17372)**. **Nothing on this list is now unconfirmed.**
>
> *(internal, delete before submission)* **Three citation errors were caught across the two passes.** (1) The earlier placeholder "Pang et al. 2024" does not exist; the work actually intended - BC + PPO for WVR 6-DOF air combat - is Li, Zhang, Qian, Zhao & Wang, Neurocomputing 584:127591 (2024). (2) "Actor-Learner Distillation" is a misnomer; the correct title is Actor-Mimic (Parisotto, Ba & Salakhutdinov, ICLR 2016). (3) ROME's author list was given as "Meng, Wang, Pfaff & Yang"; the paper is by **Meng, Bau, Andonian & Belinkov** - both the arXiv record and the NeurIPS 35 proceedings agree. A wrong author list on a cited method is the kind of error a reviewer notices, and it was found only by looking the citation up rather than recalling it.
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
  - **Rule expert** — hand-designed, stateless: every label is a pure function of the current observation (`scripts/generate_shoot_rule_expert.py`), built from standard BFM heuristics (nose alignment, DLZ depth, closure sign) plus the launch-quality gate of §3.2. **It has no external publication to cite — it is this project's own controller** — and we claim only that it is *a* competent teacher (`wez_reach_rate = 1.00`, `hit_rate ≈ 0.997`, `lost_target_rate = 0`), not a strong or state-of-the-art one. Every headline claim concerns *what imitation inherits from a given teacher*, holding the teacher fixed, so the paper does not depend on where the teacher came from.
  - **BC round1** — MLP encoder (256-256-128) + 4 heads, trained on 200 expert episodes, masked cross-entropy.
  - **SPC** — BC with only the launch head re-trained (§3.3).

### 3.2 C1 — Diagnostic metric and localization
**Conditional Launch Rate (CLR)**:

```
CLR(π) = P(a_fire = 1 | m_fire = 1)
       = (# steps with a_fire=1 and m_fire=1) / (# steps with m_fire=1)
```

- `m_fire` is the **environment's** legality mask (index 10 of the flat mask), which requires: missiles remaining > 0; launch cooldown (30 decision steps) elapsed; target alive; `ATA < 15°`; `MIN_ATTACK_DISTANCE (1500 m) ≤ range ≤ dynamic max range (3–8 km by aspect angle)`; closure < 0 (closing).
- **Critical property: CLR is not a claim that 100% is optimal.** It is a descriptive quantity. The optimal value must be established separately — which is exactly what §4.3 does.

**Why the expert's CLR is low — the designed gate.** The expert does not simply follow `m_fire`. It requires its own stricter quality predicate:

| Condition | Environment mask | Expert `fire_desired` | Effect |
|---|---|---|---|
| Nose alignment | `ATA < 15°` | `ATA < 10°` | refuses 10–15° shots |
| Closure | `closure < 0` | `closure < −5 m/s` | refuses marginal closure |
| DLZ position | legality only | depth ∈ **[0.25, 0.75]** | refuses deep-in-DLZ (closer) and far-edge shots |
| Output | — | `fire = allowed AND desired` | cooldown/inventory handled by mask |

**Measured — the gate fully determines the expert's behaviour on legal steps.** On steps where launch is legal, the expert's `fire_desired` is *identical* to its executed `action[:,3]` (560 / 9395 on the demonstration dataset), so nothing except the gate explains what it does when launch is permitted. Under the evaluated protocol, **the observed restriction is therefore explained by the designed gate** — not by the environment and not by BC's fitting error. The measured CLR itself is Table 2; two further measurements on a different basis, and the geometry caveat that travels with them, are in Appendix E.2.

### 3.3 C2 — Surgical Policy Correction (SPC)
- **Input**: frozen BC θ_BC; BC rollouts collected over `R` episodes.
- **Target**: on steps where `m_fire = 1`, command fire. All other steps are irrelevant (the mask forces `a_fire = 0` at inference).
- **Procedure**:
  1. Roll out θ_BC; collect `(obs_t, m_fire,t)`.
  2. Filter to `m_fire,t = 1`.
  3. Train **only** `action_heads[3]` by cross-entropy toward `a_fire = 1`. Encoder, speed head, heading head, altitude head: `requires_grad_(False)`.
  4. Early stop when allowed-window fire accuracy ≥ 0.995.
- **Verification gates** (all must PASS; implemented in `scripts/distill_fire_asap.py`): **G1** heading/speed logits vs θ_BC (`max diff < 1e-9`); **G2** heading/speed action sequences 100% identical on fixed seeds; **G3** CLR ≥ 99.5% over allowed windows (`P(a_fire=1 | m_fire=1) ≥ 0.995`) on held-out rollouts; **G4** no new `lost_target` / `bad` launch introduced.
- **Interpretation**: G1–G2 are the causal-identification device — the maneuver policy is provably untouched, so any performance change is attributable to the launch decision alone.
- **Naming, stated once:** this pipeline is called *ASAP distillation* in the code (`scripts/distill_fire_asap.py`) and *P2A* in older docs. ASAP = "As Soon As allowable by Policy", i.e. the target behaviour. The paper-facing name is **SPC**, and the mapping is recorded in Appendix A/B.

### 3.4 Why isolating the launch head is the right intervention (not arbitrary)
- The launch decision is a **discrete binary commitment** with a one-step-wide opportunity in the close-range regime (§4.3), so its error is not smoothable by the rest of the policy.
- The maneuver heads are demonstrably near-ceiling: `wez_reach_rate = 1.0`, `hit_rate ≈ 0.997–1.000`, `lost_target_rate = 0` across all evaluated suites. Re-training them would buy little and destroy the clean attribution.
- Contrast with end-to-end PPO fine-tuning: the project's own record (`docs/summary_phase1.md` §4.5) shows that once SPC reproduces the legality policy, the current reward/environment admits no launch-timing trade-off, so PPO adds no increment and carries regression risk. → report as a *documented negative result*, not as a claim about PPO in general.

---

## 4. Experiments (2 pages)

### 4.1 Setup
- JSBSim F-16 physics; 1v1 within-visual-range missile engagement; **deterministic masked argmax**.
- One seed protocol throughout: **seeds 20000–20399, fresh environment per episode, geometry U(0,60)**. Hardware, wall-clock and reproduction commands: Appendix D/E.
- Three evaluation settings: **ID** (2–5 km tail-chase), **dist2_3k** (hard close-range cell), **target_evasive** (`difficulty_level = 0.3`).
- **All reported numbers are verified against version-controlled artifacts**; the per-number provenance record is Appendix E.

### 4.2 Diagnosis: CLR reveals a restrictive launch preference

**The expert fires on only 6.07% of the steps its environment permits (1131/18639), and BC inherits the preference (7.26%).** Under an evading target both barely move (**5.40%** / **6.95%**). The preference is a property of the expert's designed launch gate — strictly stricter than the environment's legality condition (§3.2) — and imitation transmits it faithfully; it is not an imitation defect.

**Table 2 — Conditional Launch Rate by policy and difficulty** (fresh env, seeds 20000–20399, U(0,60), deterministic argmax):

| Policy | CLR (d=0) | CLR (d=0.3) | Legal steps (d=0 / d=0.3) |
|---|---|---|---|
| Rule expert | **6.07%** (1131/18639) | **5.40%** (1066/19729) | 18639 / 19729 |
| BC round1 | 7.26% (1223/16839) | 6.95% (1194/17169) | 16839 / 17169 |
| SPC | 100.00% (1560/1560) | 99.11% (1563/1577) | 1560 / 1577 |

Two caveats travel with this table. **(a)** SPC's 100.00% is true *by construction* — its training target is "fire on every legal step" and gate G3 tests exactly that; it is not a finding. **(b)** The CLR denominator is policy-dependent: firing starts a 30-step cooldown that *removes* legality, so a policy that fires collapses its own opportunity set (SPC ≈3.9 legal steps/episode, BC ≈42). Cross-policy CLR is therefore an ordering of tendencies, not a like-for-like exploitation fraction — which is why the causal claim rests on §4.3, where the opportunity set is held constant by construction.

One cross-check on the instrument itself: adding CLR changed nothing measurable. The d=0 run that first reported CLR reproduces the kill side of the E3 pair bit-for-bit (expert 36.75%, BC 46.50%, paired +9.75 pp, W/L/T 63/24/313) and reproduces BC's CLR to the digit, and the d=0.3 run does the same (25.75%, 42.25%, +16.50 pp, 85/19/296). Expert and BC also agree closely on every basis measured — 6.07% vs 7.26%, 5.40% vs 6.95%, and 5.96% vs 7.37% on the older geometry-mixed diagnostics — so the restriction is a stable property rather than sampling noise, and BC is in no meaningful sense more restrictive than its teacher. Appendix E.3 states the scope limit on what CLR alone can establish.

### 4.3 C3(a) — Off-policy enumeration on a frozen trajectory (the key identification)

**Design.** Take BC's heading/speed decisions as a *frozen* maneuver trajectory per seed, and enumerate a family of fire policies on top of it. Because the maneuver is fixed by construction, the maneuver policy is controlled, and differences are attributable to the launch decision.

**Table 3 — frozen-trajectory enumeration (cell `dist2_3k`).** 60 seeds, geometry U(0,60), deterministic masked argmax, maneuver frozen per seed from the BC round-1 checkpoint (E1, regenerated 2026-09-17):

| Fire policy | Kill rate | Launches/ep | Reach-4-launch |
|---|---|---|---|
| **`asap` — fire whenever the launch mask permits** | **86.7%** | 3.85 | 86.7% |
| `delay_30` (first legal + 30) | 38.3% | 3.25 | 38.3% |
| `delay_60` | 10.0% | 2.78 | 10.0% |
| `dlz_mid` (depth 0.4–0.6) | 0.0% | 1.27 | 0.0% |
| `dlz_deep` (depth ≥ 0.6) | 0.0% | 1.12 | 0.0% |
| `interval_100` | 0.0% | 1.95 | 0.0% |
| **frozen BC (inherited)** | **15.0%** | 2.37 | 15.0% |

A 2026-08-06 audit of the same family on the pre-widening geometry, and what its agreement tells us about the script versus the benchmark, are in Appendix E.10.

**Reachability audit (asap arm, measured):** median first-legal step **67** (13.4 s); **cumulative legal-window length ≈ 4 steps per episode** (median 4.0), i.e. roughly one legal step per DLZ transit; 52/60 killed. The "≈4 legal steps per episode" figure is the load-bearing structural fact: it is what makes window *usage* rather than window *availability* the binding constraint. Full numbers, including the 8 timeouts whose window was still open at episode end, are in Appendix E.10.

**Reading:**
- The scenario is **not infeasible** — the environment permits a 4-launch salvo in 86.7% of episodes, and `asap` realizes 86.7% kills.
- The bottleneck is the launch policy: the inherited policy uses **2.37 of 3.85** available windows, because most windows have DLZ depth < 0.25 and are rejected by `fire_desired`.
- **Every** selective alternative is worse than the mask-permissive policy, and by a wide margin (best rival `delay_30` at 38.3%). Within the enumerated family and on this trajectory class, the mask-permissive policy **achieves the highest kill rate**; the comparison is confined to that family and is not a claim about fire policies in general.
- Note what this does and does not show: it establishes that the inherited launch policy is **dominated within the enumerated family, conditional on the frozen maneuver**. It does not establish that the mask-permissive policy is optimal for arbitrary maneuvers of arbitrary policies. (§4.4 supplies the within-policy counterpart.)
- **Terminology discipline.** The mask-permissive policy is *not* "always fire": it fires only on steps where the environment-level launch mask is 1. The correction removes a gate the expert added *on top of* the environment's legality condition; it does not relax the environment's condition itself.

> **Provenance.** The measured column comes from `results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json`, which is version-controlled and carries `run_meta` with the cell configuration, geometry, difficulty, seed range and the frozen checkpoint's SHA-256. The 2026-08-06 artifact no longer exists; the italic column is all that survived of it, which is why the regenerated file is tracked rather than regenerated on demand.

**Same enumeration under an evading target — matched 2×2 (E6, seeds 20000–20059, n=60, geometry U(0,60), deterministic masked argmax).**

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

[[figure: mechanism_seed20007_d00.png | column:0.82 | Mechanism, on one illustrative episode (seed 20007, d=0). Panel A: range and angle-off, with the environment-legal launch windows shaded and both policies' launches marked. Panel B: the decisions themselves — launch mask, BC fire, SPC fire. The state trace is bit-identical between the two policies here (max deviation 0.0e+00 m), so the whole difference sits in Panel B. Single episode, chosen for legibility; the aggregate evidence is Tables 3 and 4.]]

The figure is an *attribution* device, not a trajectory showcase: same engagement geometry, same maneuver, different launch decision. On this seed BC uses **3 of 41** legal steps (CLR 7.32%, against the 7.26% aggregate) and times out, while SPC uses **4 of 4** and kills the target at step 842.

### 4.4 C3(b) — SPC intervention result (primary endpoint, E7)

**Setting (all arms identical):** d=0, deterministic masked argmax, geometry U(0,60), **seeds 20000–20399 (n=400)**, one code path (`eval_bc_1v1.py`), identity recorded in each file's `run_meta` (BC `sha256 aad05b45…`, SPC `sha256 36d79bd9…`).

| Arm | Kill rate | Wilson 95% | Launches/ep | CLR | lost | hit |
|---|---|---|---|---|---|---|
| Rule expert (fresh-env path, E3) | 147/400 = **36.75%** | [32.17, 41.58] | 2.83 | — | 0 | 0.9973 |
| **BC round1 (frozen)** | **186/400 = 46.50%** | [41.67, 51.40] | 3.06 | 7.26% | 0 | 0.9984 |
| **SPC (this paper)** | **363/400 = 90.75%** | [87.51, 93.21] | 3.90 | 100.00% | 0 | 1.0000 |

**Paired BC vs expert on the same 400 seeds:** Δkill **+9.75 pp**, win/lose/tie **63 / 24 / 313**, exact McNemar **p = 3.48e-05** (χ²cc p = 4.62e-05); 72% of the 87 discordant seeds favour BC. Δlost = 0.00, Δlaunches = +0.23. This is the first *significance test* attached to the BC-beats-expert claim — the sealed Phase-1 record had only a bootstrap CI ([+3.8, +11.0] pp, 500 seeds, old geometry).

> **Expert numbers in this paper are fresh-env measurements.** A second evaluation path that reuses one environment across all episodes disagrees with it on identical seeds — 32.25% versus 36.75% at d=0, and a 20-seed screen shows different per-seed kill vectors rather than noise. Both paths import the same rule functions at the same commanded speed, so neither the rule nor the flight controller explains the gap; the cause is environment lifecycle, since the reused-env path builds a *single* `BaseEnv` while every other script evaluated here builds a fresh one per episode. The reused-env figures (32.25%, 25.25% at d=0.3) are therefore excluded rather than wrong: they are measured under a different episode-construction protocol, and their artifacts are renamed `EXCLUDED_reused_env_expert_*` so no table can pick them up by accident. The consistency gate is `tests/check_expert_path_consistency.sh`; Appendix E.1 lists the affected scripts and artifacts.

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

> *(internal, delete before submission)* **Wording discipline.** Report it bound to its conditions, never as a bare improvement: *"Under a matched d=0, deterministic-argmax evaluation on 400 paired seeds, replacing the restrictive launch decision with SPC raises the kill rate from 46.50% to 90.75% (+44.25 pp, exact McNemar p = 1.04e-53)."* Not: *"SPC improves performance by 44 pp."* The four qualifiers (matched setting, d=0, deterministic argmax, isolated intervention) are load-bearing — they are *why* the attribution is legitimate. The paper-facing version of this rule is §5.4.

> **Artifact status.** E7 is **verified against live artifacts** (`results/shoot_eval/E7_{bc_round1,spc}_d0_n400_s20000.json`, `E7_paired_bc_vs_spc_d0_n400.json`).
>
> *(internal, delete before submission)* **Superseded numbers.** The docs-attested Phase-1 pair **43.2% → 91.2% = +48.0 pp** (500 seeds, *old* geometry U(30,60)) is **replaced** by the measured **46.50% → 90.75% = +44.25 pp** (400 seeds, U(0,60)). The old pair was not self-consistent: its BC figure came from a different geometry generation than its CLR figure. E7 puts kill rate, CLR, and geometry on **one** seed set. Cite 46.50 / 90.75 / +44.25.
>
> *(internal, delete before submission)* **Both former docs-only numbers have now been replaced by fresh-env measurements.** The rule-expert ID kill rate is **36.75%** (E3 paired, fresh env, seeds 20000–20399) — *not* the docs' 35.8%, and *not* the reused-env 32.25%. The rule-oracle ID kill rate is **90.75%** (E2), measured on the same seed set as SPC and exactly equal to it; that equality is a completeness check, not independent validation (§4.4). The originals (`paired_bc_vs_expert_500.json`, `asap_baseline.json`) remain lost — the replacements are new measurements, not recoveries, and were taken under U(0,60) rather than the docs' U(30,60).

**SPC reaches the rule oracle's level on the same seed set (E2, seeds 20000–20399, n=400).** The oracle is "frozen BC maneuver + fire on every legal step", and SPC is "bit-identical frozen maneuver heads + a launch head whose measured CLR is 100.00%"; both therefore fire on every legal step, and their agreement is expected rather than evidence that two independent methods concur. What it does establish is that SPC captures the **entire** oracle gap — 46.50% → 90.75%, i.e. 100% of it — rather than some fraction. The independent replication in this paper is §4.3(E6) against §4.4/§4.6, where the identification strategies differ in kind (off-policy enumeration versus within-policy intervention). The per-arm identities and the aggregate-only limitation are in Appendix E.

### 4.5 Robustness check 1 — widened initial geometry (heading bias) — *a benchmark check, not a contribution*

> *(internal, delete before submission)* Both robustness subsections (4.5, 4.6) are secondary and carry no contribution weight — 2026-09-18 Sean: "NOT a second contribution", 这个判断现在落到版面结构上。

**A benchmark-change check, not a contribution.** Widening the heading-bias range makes the benchmark easier by ≈ **+4.5 pp** at identical weights (87% → 91%, n=100), and retraining SPC under the new geometry adds a further paired **+3.25 pp** (19 vs 6 discordant pairs, robust across 4/4 bias bins) that is *not* significant unpaired.

| Comparison (d=0, new geometry U(0,60)) | n | Kill rate | Paired diff | McNemar |
|---|---|---|---|---|
| SPC base @ U(0,60) | 400 | 362/400 = **90.50%** (Wilson [0.872, 0.930]) | — | — |
| SPC retrained under U(0,60) | 400 | 375/400 = **93.75%** (Wilson [0.909, 0.957]) | **+3.25 pp** | **p = 0.0146** |
| same pair, unpaired Fisher | 400 | — | — | p = 0.1146 (n.s.) |

Caveats: a single seed family (42–441), d=0 only, and Bonferroni α=0.0125 would place p = 0.0146 just outside significance. The 87%→95% shift sometimes quoted from earlier drafts is *not* the launch-head effect; the launch-head correction is §4.4.

### 4.6 Robustness check 2 — target evasion (`difficulty_level = 0.3`)

Target adds S-turn `±30°·d·sin(0.3t)` plus a missile-threat break-turn and a dive to `−800·d` m (floor 2000 m). Identical protocol to §4.4 — d=0 vs d=0.3 differ **only** in `--difficulty`, same seeds 20000–20399, same code path, identity recorded in each file's `run_meta`.

**Table 4 — main results: kill rate and paired tests.** Fresh env per episode, seeds 20000–20399, geometry U(0,60), deterministic masked argmax; `difficulty` is the only variable between the two columns' groups. CLR values live in Table 2 (§4.2) and are not repeated here. Every row is generated by `scripts/collect_matrix.py` from each artifact's own `run_meta` — not assembled by hand.

| Policy | d=0 kill | d=0.3 kill | Δ (evasion cost) |
|---|---|---|---|
| Rule expert (fresh env) | **36.75%** (147/400) | **25.75%** (103/400) | **−11.00 pp** |
| BC round1 (inherited launch policy) | 46.50% | **42.25%** (169/400) | −4.25 pp |
| **SPC (corrected launch head)** | 90.75% | **90.75%** (363/400) | **0.00** |

Read against Table 2's CLR column, the table says three things at once — the teacher is restrictive, imitation transmits it, and evasion punishes it in proportion to how restrictive the policy is (**expert −11.00 > BC −4.25 > SPC 0.00**).

**Paired tests, both against the same 400 seeds** (primary protocol: d=0 and d=0.3, geometry U(0,60), deterministic masked argmax, fresh env per episode). The expert-vs-BC comparison and the SPC-vs-BC comparison share a seed set, so both are paired by construction rather than by assumption:

| Paired comparison | d=0 | d=0.3 | Test |
|---|---|---|---|
| **BC − expert** | **+9.75 pp** (63/24/313, discordant 87, 72% favouring BC) | **+16.50 pp** (85/19/296, discordant 104, 82% favouring BC) | exact McNemar **3.48e-05** / **3.79e-11** |
| **SPC − BC** | **+44.25 pp** (discordant **177 : 0**) | **+48.50 pp** (discordant **194 : 0**) | exact McNemar **1.04e-53** / **7.97e-59** |

Per-seed flips under evasion: expert **+29 gained / −73 lost**, BC **+25 / −42**. Both are loss-dominated, but the expert's net is −44 seeds against BC's −17 — the asymmetry, not just the mean, drives the ordering above.

**Three findings:**
1. **Performance.** SPC's kill rate is *unchanged* under evasion (90.75% at both difficulties) while BC falls from 46.50% to 42.25%, so the SPC−BC gap widens from +44.25 to **+48.50 pp** and the discordant count rises from 177 to 194. The same widening appears under the independent identification of §4.3 (+40.0 → +50.0 pp), so it is not an artefact of the retraining procedure: one method re-trains the launch head with the maneuver frozen, the other freezes the maneuver and swaps in rule-based launch policies.
2. **The mechanism is preserved.** CLR barely moves on either policy — BC **7.26% → 6.95%**, expert **6.07% → 5.40%** — so the designed gate suppresses launches to the same degree whether or not the target manoeuvres. The restriction is a fixed property of the predicate, and the imitation inherits it as one.
3. **Interpretation.** Evasion cost is ordered by how restrictive the launch policy is: expert **−11.00 pp**, BC **−4.25 pp**, SPC **0.00 pp**, all three legs on fresh-env evaluations over the same seeds. Evasion therefore *amplifies the cost of the restrictive launch preference* rather than penalising the correction. The per-seed flips say the same thing: the expert loses 73 seeds and gains 29, BC loses 42 and gains 25 — a suppressed launch window is unrecoverable once the target turns away. This rests on a single seed family, so it is reported as a well-supported ordering rather than a law.

Two secondary items are in Appendix E: the direct check that `difficulty=0.3` actually takes effect (368/400 episodes change length; 382/400 keep the same kill outcome, 18 flips split 9/9, so the identical 363 at both difficulties is a genuine near-cancellation rather than a no-op), and the observation that BC's paired margin over the expert itself widens under evasion (+9.75 → +16.50 pp).

[[figure: robustness_fig3.pdf | column:0.90 | Difficulty robustness. Kill rate by policy at d=0 and d=0.3 (primary protocol, Wilson 95\% intervals). The paired SPC--BC gap widens from +44.25 to +48.50 pp; discordant counts are 177 and 194, all favouring the correction.]]

> **Artifact status:** verified against live artifacts — `results/shoot_eval/E5_{bc_round1,spc}_d03_n400_s20000.json`, `E5_paired_bc_vs_spc_d03_n400.json`, and the two CLR runs `E3_paired_bc_vs_expert_{d0_n400_s20000_v3,d03_n400_s20000_v2}_clr.json`. The expert arm is no longer "running": it is the fresh-env paired arm of E3 at both difficulties, and it is what Table 4 reports.

### 4.7 Ablations: why *surgical*?

> *(internal, delete before submission)* **2026-09-18 Sean：A5/A6 都留正文，但 A6 为主、A5 压缩** —— A6 回答的是"冻结是否只是工程便利"（C2 真正依赖它），A5 只回答"初始化是否有影响"。

> *(internal, delete before submission)* 2026-09-19 Sean：正文只回答两个问题 —— A5「SPC 是否依赖初始化」、A6「locality 是否必要」；A5/A6 的训练细节、tracker、筛选表全部进附录 E。

Two ablations decide whether the *locality* of the correction is doing the work, or whether any correction of the same objective would do. Both run at the paper's primary protocol — 400 seeds, d=0, the same seeds as Table 4 — so they are directly comparable with §4.4 rather than being screening runs.

**Table 5 — ablations at the primary protocol** (fresh env, seeds 20000–20399, geometry U(0,60), deterministic masked argmax):

| Intervention | Result (400 seeds, d=0) | What it tests |
|---|---|---|
| SPC (launch head only) | **90.75%** (363/400), CLR 100.00% | reference |
| **A5** — random-init launch head, encoder and maneuver heads frozen | **90.75%** (363/400), **0 discordant seeds vs SPC**, maneuver deviation **exactly 0.00e+00** | does the correction depend on warm-starting from BC's launch head? → **no** |
| **A6** — full-network fine-tune, every parameter trainable | **77.50%** (310/400), **−13.25 pp** vs SPC, exact McNemar **p = 1.33e-11**, maneuver logits move by **8.51** | is *freezing* merely an engineering convenience? → **no** |

**A5 rules out a warm-start effect.** Random initialisation of the editable head produced behaviourally indistinguishable results under the evaluated metric, so SPC's benefit does not depend on having started near BC's launch head. The supportable statement is exactly that: A5 and SPC are *behaviourally indistinguishable under the evaluated metric*, which is not the same as being proven equivalent — no action-logit comparison is made, and a formal claim would need a pre-registered margin (§E.5).

**A6 shows locality is necessary rather than convenient.** Unfreezing everything makes the correction **worse** (77.50% against SPC's 90.75%, p = 1.33e-11) *and* moves the maneuver by 8.51 logits, so the result is both less effective and no longer attributable. Full-network adaptation therefore reduces attribution and introduces maneuver changes, indicating that unrestricted updates are not equivalent to a localized intervention.

**What the pair establishes.** The observed benefit is attributable to **the choice of editable parameters** — which decision dimension is allowed to change — rather than to the amount of trainable parameters or the initialisation of the launch head. Protocol, budget, identity measurements, the 60-seed screening set and the A5 training log are in Appendix E.


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

> **The sentence the paper exists to support (2026-09-18):** *the failure was localized to a single discrete decision dimension, and correcting only that dimension recovered oracle-level performance while preserving the original maneuver policy.* Everything below is scoping, not hedging — this sentence is the claim, the qualifiers are what keep it honest.
>
> We do **not** claim that the mask-permissive policy is optimal in general. We claim that **a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime**: with the maneuver policy provably frozen (`max logit diff < 1e-9`, identical action sequences), correcting the launch head alone raises the kill rate from 46.50% to 90.75% on 400 paired seeds (+44.25 pp, exact McNemar p = 1.04e-53, with all 177 discordant seeds favouring the correction and none favouring the original), and off-policy enumeration on a fixed maneuver trajectory shows the inherited launch policy to be dominated by the mask-permissive policy within the enumerated fire-policy family.

---

## 6. Number Provenance & Integrity Ledger

### 6.0 Figure and table plan (superseded 2026-09-18 晚 — Sean 压缩 pass：每节一张紧凑表，audit 进附录)

**现行结构（2026-09-18 晚 Sean 定）**：正文按因果链展开（diagnosis → frozen intervention → SPC → robustness → ablations），每节一张**只含本节所需列**的紧凑表；reproducibility 材料全部移 **Appendix E**。此前的「3 表合并」方案被此方案取代 —— 合并解决的是重复，但把 CLR 列与 kill 列绑在一张表里会让 §4.2（诊断）和 §4.5b（稳健性）都要引用一张放在别处的表；按节分表后每个数字在正文仍只出现一次，且每张表都在使用它的位置上。

| # | 内容 | 位置 | 服务贡献 | 状态 |
|---|---|---|---|---|
| **Table 1** | Setup：任务、观测/动作空间、三策略、评测协议 | §3.1/§4.1 | 全部 | 可写 |
| **Table 2** | **CLR 诊断**：三策略 × 两难度的 CLR + 合法步数（不含 kill 列） | §4.2 | **C1** | **已成形** |
| **Table 3** | Causal intervention A：frozen-trajectory 上的 fire-policy 枚举 | §4.3 | **C3** | E1 已入库 |
| **Table 4** | **主结果**：kill rate + Δ（evasion cost）+ 两组配对检验（BC−expert、SPC−BC），不含 CLR 列 | §4.5b | **C2 + C3** | **已成形** |
| **Figure 1** | Framework：Expert → BC → CLR diagnosis → SPC | §1/§3 | 叙事 | `framework_fig1.{png,pdf}` 已入库 |
| **Figure 2** | **Mechanism visualization**：同 seed、同机动轨迹、Panel B = mask/BC/SPC 三行决策 | §4.4 | **C3** | `mechanism_seed20007_d00.{png,json}` 已入库 |
| **Figure 3** | Difficulty robustness：d=0 vs d=0.3 kill rate 与 gap | §4.5b | C3 稳健性 | `robustness_fig3.{png,pdf}` 已入库 |
| **Appendix E** | reproducibility 材料：env-lifecycle confound、佐证 CLR 与几何告警、CLR 交叉核对、算力环境、A5 训练细节 | — | 可信度 | **已建成** |

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
| **A6: full-network fine-tune** | 400 seeds, d=0: **77.50%** (310/400), CLR **99.67%** (1508/1513); paired vs SPC **−13.25 pp**, discordant **67** (60 SPC / 7 A6), exact **p=1.33e-11**; maneuver logits **8.51** vs SPC `<1e-9`, sequences non-identical ⇒ dominated on both axes. Secondary 60-seed screening also carries the mask-permissive arm (A6 81.7% vs rule 95.0%) | `ablation_A6_fullnet_d0_n400_s20000.json` + `ablation_A6_vs_spc_paired_d0_n400.json` + `ablation_A6_fullnet_train.json` |
| **A5: random-init fire head** | 400 seeds, d=0: **90.75%** (363/400), CLR **99.94%** (1560/1561); paired vs SPC **discordant 0** (both 363, neither 37, neither arm misses a seed the other kills); maneuver deviation **0.00e+00**; val allowed-window recall **100.00%** at epoch 1 ⇒ SPC is not a warm-start effect | `ablation_A5_random_fire_head_d0_n400_s20000.json` + `ablation_A5_vs_spc_paired_d0_n400.json` + `ablation_A5_random_fire_head_train.json` |
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

Primary endpoints (`§4.4`, `§4.5b`, Table 4):

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

---

## Appendix E. Reproducibility and artifact provenance

Supporting evidence for claims made in the main text. The reasoning each item supports is stated in the main text; what follows is the raw material needed to re-run or re-check it.

### E.1 The environment-lifecycle confound — affected scripts and artifacts

§4.4 states the confound and why the reused-env numbers are excluded. The specifics: both paths import the same rule functions (`hdg_label` / `spd_label` / `fire_desired`) at `CMD_SPEED = 280`, and `BaseEnv.__init__` already installs `SafetyInterceptor(PIDFlightController())`, so neither the rule nor the controller explains the disagreement. The reused-env path is `generate_shoot_rule_expert.py --validate`, which builds **one** `BaseEnv` and reuses it for every episode; the fresh-env path is used by `eval_paired_bc_vs_expert.py`, `eval_bc_1v1.py`, `fire_oracle_audit.py` and `eval_asap_baseline.py`, each of which builds a new environment per episode. Affected artifacts are renamed `EXCLUDED_reused_env_expert_*.json`, and `collect_matrix.py` skips anything marked `EXCLUDED_`. Gate: `tests/check_expert_path_consistency.sh`.

### E.2 Corroborating CLR: timestamps and the gate on disallowed steps

§3.2 quotes the two corroborating values (expert 5.96%, BC 7.37%) and their geometry caveat; the exact provenance is:

- `data/expert/shoot_rule_expert.npz`, 200 episodes, **mtime 2026-09-13 20:22** — this *predates* commit `fb48155` (2026-09-14 09:06), which changed the heading-bias range from `U(30,60)` to `U(0,60)`. Its 5.96% is therefore an **old-geometry** measurement.
- `results/health_check/fire_hesitancy.json`, 200 rollouts, generated **2026-09-16**, i.e. after the change — so the 7.37% beside it is U(0,60). The two are **geometry-mixed** and are never averaged or presented as a matched pair.

| Quantity | Value | Basis |
|---|---|---|
| Expert `fire_desired` on *disallowed* steps | 4.03% (10336 / 256357) — the gate would fire outside the window too; the mask (mostly cooldown) holds it back | `data/expert/shoot_rule_expert.npz` |

### E.3 Scope reading for the CLR claim

Neither CLR figure alone proves suboptimality: a low CLR diagnoses restriction only, and whether that restriction is *costly* is established separately (§4.3, §4.4). The instrumentation and stability cross-checks are in §4.2.

### E.4 Compute environment (moved from §4.1)

Single RTX 3060 Laptop GPU (6 GB, cu126, driver 528.79). Closed-loop evaluation: BC ≈6.7–13.0 s/episode, SPC ≈10 s/episode, so one n=400 arm pair is ≈1.7 h; rule-expert rollout ≈6.3 s/episode. All runs execute in WSL2 Ubuntu 22.04 with the project interpreter (`/home/sean/miniconda3/envs/marl_env/bin/python`); reproduction commands are in Appendix D.

### E.5 Ablation A5 — full training and screening detail (moved from §4.7)

- **Implementation.** `scripts/distill_fire_asap.py --random-fire-head`: SPC's freeze pattern is kept exactly (encoder and every maneuver head frozen at θ_BC) and only the fire head is discarded and rebuilt from `nn.Linear.reset_parameters()` — the constructor's own initialisation, so it is a genuine random init (weight std 0.049, bias std 0.044) rather than a hand-rolled approximation of one. Mutually exclusive with `--full-network` (A5 and A6 are different ablations).
- **Training.** 300 rollout episodes (392091 transitions, 11976 allowed steps; train/val split 9578/2398 allowed steps). The random fire head reaches fire-CE 0.0413 with train recall 100.00% and validation recall 100.00% **in its first epoch** (of 15) and triggers the ≥99.5% early stop. Identity gates: `hdg/spd logits diff = 0.00e+00` — *exactly* zero, not merely under tolerance — with identical action sequences and `VERDICT: PASS`.
- **Primary protocol.** 90.75% (363/400) kills, CLR 99.94% (1560/1561) — the CLR denominator differs from SPC's by exactly one legal step (1561 vs 1560), the only trace of the different initialisation anywhere in the evaluation.
- **60-seed screening** (same seed set as A6b; ±10 pp small-sample spread applies): A5 **95.0%** (`id`) / **86.7%** (`dist2_3k`) at 100% window utilisation, bit-for-bit equal to the mask-permissive rule oracle on the same seeds; BC 40.0% / 15.0%. The screening set is kept because it carries the third arm.

### E.6 Ablation A6 — protocol, identity measurements and results (moved from §4.7)

**Protocol.** Identical objective, data, loss and initialisation as SPC; the only change is that every parameter is trainable rather than the launch head alone, at a deliberately **more generous** budget (`--epochs 40 --lr 3e-4` versus `--epochs 15 --lr 1e-2`), so that a negative result cannot be attributed to under-training.

**Identity measurements** — the maneuver is untouched under SPC and demonstrably changed under A6:

| Identity measurement | SPC (fire head only) | A6 (full network) |
|---|---|---|
| heading/speed logit max-diff vs θ_BC | **< 1e-9** | **8.51** |
| heading/speed action sequences identical | **yes** | **no** |
| launch-mask agreement | yes | yes |
| validation allowed-window accuracy | ≥ 0.995 | 0.9967 |

**Kill rate and CLR at the primary protocol** (d=0, U(0,60), the same 400 seeds as Table 4):

| Arm (400 seeds, d=0) | Kill rate | CLR | vs SPC (paired) |
|---|---|---|---|
| BC (inherited launch policy) | 46.50% | 7.26% | — |
| **A6 — full-network fine-tune** | **77.50%** (310/400) | **99.67%** (1508/1513) | **−13.25 pp**, discordant 67 (60 SPC / 7 A6), exact McNemar **p = 1.33e-11** |
| SPC — launch head only | 90.75% (363/400) | 100.00% | — |

Full-network fine-tuning recovers most of the gap (46.50% → 77.50%) but remains significantly worse than the localized intervention on identical seeds, and pays for it with an 8.51-logit change to the maneuver.

### E.7 Ablation screening set — the mask-permissive third arm (moved from §4.7)

The ablation script's own 60-seed evaluation (geometry U(0,60), deterministic masked argmax) runs the `id` and `dist2_3k` cells side by side and adds the **mask-permissive rule oracle**, which the 400-seed runs do not:

| Arm (60 seeds, same seed set) | `id` cell | `dist2_3k` cell | window utilisation (`id`) |
|---|---|---|---|
| BC (inherited launch policy) | 40.0% | 15.0% | 7.2% |
| A6 — full-network fine-tune | 81.7% | 66.7% | 98.3% |
| mask-permissive rule oracle | **95.0%** | **86.7%** | 100% |

A6 also loses to a rule that simply fires whenever the mask permits (81.7% vs 95.0%). The set is **60 seeds** and carries roughly ±10 pp of small-sample spread — the same seeds give BC 40.0% where n=400 gives 46.50% — so these figures are quoted only for the third arm and never alongside the 400-seed numbers as one comparison.

### E.8 Ablation tracking table (project bookkeeping)

| # | Ablation | Status | Purpose |
|---|---|---|---|
| A1 | Fire-policy oracle enumeration on frozen trajectory (§4.3) | done (E1 regenerated, tracked) | establishes that the inherited launch policy is suboptimal, maneuver held fixed |
| A2 | SPC (learned head) vs ASAP rule oracle (§4.4) | done — exact match on one seed set (90.75% both); completeness check, not independent validation | SPC captures 100% of the oracle gap |
| A3 | `difficulty_level = 0.3` (§4.6) | done — +48.50 pp, 194:0, p=7.97e-59 | interactive/reacting opponent |
| A4 | Heading-shift robustness (§4.5) | done (n=400) | benchmark sensitivity |
| A5 | Fire head from random init, encoder frozen (§4.7, E.5) | done 2026-09-18 (400-seed primary protocol) | warm-start dependence |
| A6 | Full-network fine-tune on the same objective (§4.7, E.6) | done 2026-09-18 (400-seed primary protocol) | is freezing merely engineering convenience? |
| A6b | 60-seed screening incl. the mask-permissive arm (E.7) | done, secondary | third-arm comparison only |
| A7 | CLR for other binary decisions | conceptual | generality of the metric (future work) |

### E.9 The E2 rule-oracle comparison — identities and limitations (moved from §4.4)

`eval_asap_baseline.py` had no `--start-seed`, so the oracle could only ever run seeds `0..N-1` and could not be matched to a trained policy's evaluation. Patched, then run on the same seed set:

| Arm | Kill | Launches/ep | Hit | lost | Termination reasons |
|---|---|---|---|---|---|
| Rule oracle (`asap`: frozen BC maneuver + fire on every legal step) | **363/400 = 90.75%** | 3.90 | 1.0000 | 0 | `{target_killed: 363, timeout: 37}` |
| SPC | **363/400 = 90.75%** | 3.90 | 1.0000 | 0 | `{target_killed: 363, timeout: 37}` |

Identities verified from `run_meta`: the oracle arm loaded `sha256 aad05b45…` (BC round1 — the *frozen maneuver* source, correct), SPC loaded `36d79bd9…`.

**One limitation.** `eval_asap_baseline.py` writes aggregates only, with no per-episode detail, so the identity above follows from construction rather than from a seed-by-seed match. Emitting per-episode records would make it mechanically checkable.

### E.10 The 2026-08-06 frozen-trajectory audit on the old geometry (moved from §4.3)

| Fire policy | Kill rate (measured, U(0,60)) | *2026-08-06 audit, U(30,60)* |
|---|---|---|
| `asap` | **86.7%** | *73.3%* |
| `delay_30` | 38.3% | *28.3%* |
| `delay_60` | 10.0% | *10.0%* |
| `dlz_mid` | 0.0% | *0.0%* |
| `dlz_deep` | 0.0% | *3.3%* |
| `interval_100` | 0.0% | *0.0%* |
| frozen BC (inherited) | **15.0%** | *6.7%* |

Two of the seven arms reproduce exactly (`delay_60`, `dlz_mid`) while `asap` moves from 73.3% to 86.7%. That pattern indicates the *script's* behaviour is unchanged and the benchmark itself became easier by ≈ +4.5 pp once the heading-bias range was widened (§4.5). Only the measured column is cited in the paper, always with its geometry attached; the paper's Table 3 carries no geometry-mixed column.
