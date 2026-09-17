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

**首选**：
*Diagnosing and Surgically Correcting Conservative Engagement Gating in Imitation-Learned Air Combat Policies*

**备选**：
- *Diagnosing and Surgically Correcting Conservative Engagement Policies in Rule-Based Air Combat Agents*
- *Conditional Launch Rate: Diagnosing Inherited Conservatism in Imitation-Learned Air Combat Policies*

> **为什么从「Hidden Suboptimalities…」改掉（2026-09-16 Sean 拍板）**：真正的机制不是泛泛的「隐藏次优」，而是**conservative engagement gating** —— 一个**可定位、有名字、可量化**的设计门。标题直接说出机制，比说出症状更有力。
>
> **一处必须注意的归属问题**：Sean 给的版本是「…in Rule-Based Air Combat Agents」。但本文**被修正的对象是 BC 策略，不是规则 agent 本身** —— 规则专家只是偏差的**来源**（诊断靶点），修正发生在模仿出来的策略上。若标题落在 "Rule-Based … Agents"，reviewer 会读成「我们改进了规则专家」，与正文不符。故首选版把对象写成 "Imitation-Learned … Policies"，同时保留 "Conservative Engagement Gating" 这个机制词。备选第 1 条即 Sean 原版，若最终采用需在 §1 显式声明修正对象为模仿策略。
>
> 定位（AAMAS）：落到 **interacting-agent environments 中 expert-induced policy bias** 这一层，而不是「给 F-16 分类头打了个补丁」。
> 卖点是：规则专家的**局部**决策规则（一条 DLZ 质量门）在**闭环**对抗中产生**全局**性能损失，且该偏差被模仿学习无损继承 —— 这是 multi-agent / interactive 场景特有的失效模式。

---

## Abstract (~170 words)

Behavior cloning (BC) from rule-based experts is a standard bootstrap for air-combat policies, under the implicit assumption that the expert's decisions are worth imitating. We test that assumption on a JSBSim F-16 within-visual-range (WVR) 1v1 missile-engagement benchmark. We introduce the **Conditional Launch Rate (CLR)** — the probability that a policy commands launch at a step where the environment permits it, `P(a_fire=1 | m_fire=1)` — and show that the hand-designed rule expert fires at only **5.96%** of permitted steps, because it applies a launch-quality gate strictly tighter than the environment's legality mask. BC reproduces that gate faithfully (**7.37%**): the defect lies in the teacher's designed decision boundary, not in the imitation, which is behaving as intended. To isolate the cost we freeze the BC maneuver trajectory and enumerate the fire-policy family; firing whenever legal achieves 73.3% kills where the inherited policy achieves 6.7%, and every range- or delay-selective alternative is worse. We then apply **Surgical Policy Correction (SPC)**, which re-trains *only* the launch head while every other parameter stays bit-identical (heading/speed logits `max diff < 1e-9`). Under a matched d=0, deterministic-argmax evaluation on **400 paired seeds**, replacing the conservative launch decision alone raises the kill rate from **46.5% to 90.8%** (**+44.3 pp**, exact McNemar **p ≈ 1e-53**), and SPC climbs to a CLR of exactly **100.0%**. The pairing is uniform, not merely significant: of 177 discordant seeds, **all 177 favour the correction and none favour the original**. Under an evading target (`difficulty_level = 0.3`) the benefit grows rather than decays: SPC's kill rate is **unchanged at 90.8%** while the baseline falls to 42.3%, widening the gap to **+48.5 pp** with all 194 discordant seeds again favouring the correction. We do not claim that launching whenever legal is generally optimal; we claim that a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime.

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
1. **C1 — Diagnosis.** A cheap, transferable metric: the **Conditional Launch Rate (CLR)**, `P(a_fire=1 | m_fire=1)`. Measured on one matched seed set, the expert sits at **6.04%** (d=0) / **5.43%** (d=0.3) and BC at **7.26%** / **6.95%**, and we localize the cause to a *designed* launch-quality gate strictly tighter than the environment's legality mask (§3.2).
2. **C2 — Intervention.** **Surgical Policy Correction (SPC)**: re-train a single binary action head toward the environment-legality policy while freezing all other parameters, verified bit-identical by logit and action-sequence gates (§3.3).
3. **C3 — Causal evidence.** Two independent identifications that the inherited launch policy is suboptimal *conditional on the maneuver*: (a) off-policy enumeration of the fire-policy family on a **frozen** BC maneuver trajectory (§4.3), and (b) the SPC intervention with provably frozen maneuver heads (§4.4). The effect holds under an evading, reacting target — and grows (+44.3 → +48.5 pp, §4.6).

---

## 2. Related Work (1 page)

### 2.1 Behavior cloning for air combat
- [Pope et al. 2021] hierarchical action decomposition for BFM
- [Yang et al. 2022] BC + self-play for UCAV maneuvering
- [Pang et al. 2024] BC from a rule-based expert + RL fine-tuning
- 待补：BC 在 safety-critical control 中「继承专家缺陷」的既有讨论（若有，必须引用并说明差异）

### 2.2 Expert suboptimality and its diagnosis
- 既有的 imitation-vs-expert gap 工作多聚焦 **distribution shift**（DAgger 系）或 **compounding error**；
- 本文不同：误差**不来自纠缠**，而来自**专家自身的一个可定位的判断门**，且被无损复制。
- 待补 3–5 篇：reward misspecification / expert conservatism / "imitation of suboptimal demonstrations"。

### 2.3 Knowledge distillation & head-only fine-tuning
- Hinton et al. 2015 通用 teacher–student；
- Policy distillation: Rusu et al. 2016 (DQN→DQN), Parisotto et al. 2020 (Actor-Learner Distillation)；
- 本文的 SPC 与 policy distillation 的区别：**目标不是逼近 teacher**，而是**有选择地偏离 teacher 的一个 head**，并以「其余部分逐比特冻结」作为因果识别手段。这是方法学上的关键差异，务必在正文写清。

### 2.4 Benchmarks
- JSBSim 系空战仿真；SMAC / PettingZoo（协同 MARL 基准，作对照说明本文是 1v1 对抗）

---

## 3. Method (1.5 pages)

### 3.1 Task and policy
- **Task**: 1v1 **within-visual-range** air combat, JSBSim F-16 dynamics; pursuer carries 4 short-range IR missiles (AIM-9L class); initial range 2–5 km tail-chase, target heading bias 0–60° (default), target speed U(180,240) m/s. Max engagement range 15 km.
- **Observation**: 41-dim flattened = 30-dim state features + 11-dim action-mask bits. The BC encoder consumes the 30-dim state block; the mask block is applied as a separate masked-argmax step, not fed to the network.
- **Action space**: `MultiDiscrete([3, 5, 1, 2])` = (speed Δ, heading Δ, altitude, fire).
- **Episode**: max 1500 steps; termination = target killed / low altitude / fled altitude / target lost / ammo exhausted / timeout.
- **Evaluation protocol**: **deterministic masked argmax**, JSBSim physics, `difficulty_level` as noted per experiment.
- **Three policies in play**:
  - **Rule expert** — hand-designed, stateless: every label is a pure function of the current observation (`scripts/generate_shoot_rule_expert.py`). No external publication is the source; it is this project's own rule design. → §6 acknowledgement note.
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

**Measured (d=0, deterministic, 200 episodes, seed 20000+, current geometry):**

| Quantity | Value |
|---|---|
| Expert CLR (`action[:,3]` on allowed steps, from `data/expert/shoot_rule_expert.npz`) | **5.96%** (560 / 9395) |
| BC CLR (launch-head argmax on allowed steps, 200 rollouts) | **7.37%** (614 / 8327) |
| Expert `fire_desired` on allowed steps | 560 / 9395 — **identical to `action[:,3]`** ⇒ the expert's behaviour on legal steps is fully determined by its own gate |
| Expert `fire_desired` on *disallowed* steps | 4.03% (10336 / 256357) — the gate would fire outside the window too; the mask (mostly cooldown) holds it back |

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

| Policy | Allowed steps | Launch commands | **CLR** |
|---|---|---|---|
| Rule expert (200 ep diagnostic) | 9395 | 560 | 5.96% |
| ~~Rule expert (400 ep, d=0)~~ | ~~18521~~ | ~~1119~~ | ~~6.04%~~ — **excluded, reused-env path** |
| ~~Rule expert (400 ep, d=0.3)~~ | ~~19736~~ | ~~1071~~ | ~~5.43%~~ — **excluded, reused-env path** |
| BC round1 (200 ep diagnostic) | 8327 | 614 | 7.37% |
| BC round1 (E7 seed set, 400 ep, d=0) | 16839 | 1223 | **7.26%** |
| BC round1 (E5 seed set, 400 ep, d=0.3) | 17169 | 1194 | **6.95%** |
| **SPC (E7 seed set, 400 ep, d=0)** | 1560 | 1560 | **100.00%** |
| SPC (E5 seed set, 400 ep, d=0.3) | 1577 | 1563 | **99.11%** |
| Rule oracle (fire = mask) | — | all allowed | 100% (by construction) |

**BC and SPC rows are safe**: both come from `eval_bc_1v1.py`, which builds a fresh env per episode. **The expert CLR rows are excluded** — they come from the reused-env path (§4.4 note). Producing a comparable expert CLR requires the paired script to compute CLR, which it does not; that is an open gap, not a measurement to substitute. The claim "expert is the most launch-conservative, then BC, then SPC" therefore currently rests on one excluded leg and must be re-derived before it is asserted.

Two readings:
1. Expert and BC agree closely across two independent evaluations (5.96% vs 7.37% / 7.26%), so the conservatism is a **stable property**, not sampling noise. BC is not "more broken" than the expert — it reproduces the gate and adds a little variance.
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
| **`asap` — fire whenever legal** | **86.7%** | *73.3%* | 3.85 | 86.7% |
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
- **Every** selective alternative is worse than fire-whenever-legal, and by a wide margin (best rival `delay_30` at 38.3%). Within this family, on this trajectory class, always-launching dominates.
- Note what this does and does not show: it establishes suboptimality **conditional on the frozen maneuver**. It does not establish that always-launching is optimal for arbitrary maneuvers of arbitrary policies. (§4.4 supplies the within-policy counterpart.)

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

**Report it bound to its conditions, never as a bare improvement.** Use: *"Under a matched d=0, deterministic-argmax evaluation on 400 paired seeds, replacing the conservative launch decision with SPC raises the kill rate from 46.5% to 90.8% (+44.3 pp, exact McNemar p ≈ 1e-53)."* Not: *"SPC improves performance by 44 pp."* The four qualifiers (matched setting, d=0, deterministic argmax, isolated intervention) are load-bearing — they are *why* the attribution is legitimate.

> **Artifact status.** E7 is **verified against live artifacts** (`results/shoot_eval/E7_{bc_round1,spc}_d0_n400_s20000.json`, `E7_paired_bc_vs_spc_d0_n400.json`).
>
> **⚠ Superseded numbers.** The docs-attested Phase-1 pair **43.2% → 91.2% = +48.0 pp** (500 seeds, *old* geometry U(30,60)) is **replaced** by the measured **46.50% → 90.75% = +44.25 pp** (400 seeds, U(0,60)). The old pair was not self-consistent: its BC figure came from a different geometry generation than its CLR figure. E7 puts kill rate, CLR, and geometry on **one** seed set. Cite 46.50 / 90.75 / +44.25 in the paper.
>
> Still **docs-only and to be regenerated if used**: the rule-expert ID kill rate (35.8%) and the ASAP rule-oracle ID kill rate (90.6%) — `paired_bc_vs_expert_500.json` / `asap_baseline.json` are gone. The SPC-vs-oracle comparison (90.75% vs 90.6%) currently spans two evaluations and should be re-measured on one seed set before being asserted in print.

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
| Rule expert (fresh env) | 36.75% | *pending (E3 @ d=0.3 running)* | — | — | — |
| BC round1 | 46.50% | **42.25%** (169/400) | −4.25 pp | 7.26% | **6.95%** |
| **SPC** | 90.75% | **90.75%** (363/400) | **0.00** | 100.00% | **99.11%** |
| **SPC − BC gap** | +44.25 pp | **+48.50 pp** | **+4.25 pp** | — | — |

> **The expert row is measured under a different protocol and is excluded.** The figures 32.25% / 25.25% come from `generate_shoot_rule_expert.py --validate`, which reuses one env across episodes; every other arm here builds a fresh env per episode. The two protocols disagree on identical seeds (§4.4 note). The fresh-env expert is being re-measured at d=0.3 (E3) and the row will be filled from that run, not from the reused-env path.

**All three policies on one seed set** (seeds 20000–20399, U(0,60), deterministic argmax), so the rows are directly comparable. Source table is generated by `scripts/collect_matrix.py` from each file's own `run_meta`, not assembled by hand.

**Paired test at d=0.3:** discordant **194 : 0** (once again unanimous), exact McNemar **p = 7.97e-59**. SPC launches 3.91/ep, hit rate 0.9994, `lost_target = 0`, `launch_quality.bad = 0`.

Four findings:1. **The correction is robust, and its benefit grows under evasion.** SPC's kill rate is *unchanged* (90.75% at both difficulties) while BC loses 4.25 pp, so the SPC−BC gap widens from +44.25 to **+48.50 pp**, and the discordant count rises from 177 to 194. **This widening is independently replicated by a different identification strategy**: in the frozen-trajectory oracle enumeration (§4.3, E6) the `asap`−inherited gap widens from **+40.0 to +50.0 pp** across the same difficulty change. One method re-trains the launch head with the maneuver frozen; the other freezes the maneuver and swaps in rule-based launch policies. Both say the same thing: evasion punishes the conservative launch policy, not the aggressive one.
2. **The conservatism is structural, not scenario-specific.** BC's CLR barely moves (7.26% → 6.95%), i.e. the designed gate suppresses launches to the same degree whether or not the target manoeuvres.
3. **The identical aggregate is not an artefact — it was verified explicitly.** 363 kills at both difficulties looked like a bug, so it was tested: **368/400 episodes change length** (so `difficulty=0.3` is definitely applied) and **382/400 seeds keep the same kill outcome, with 18 flips split perfectly 9 gained / 9 lost**. The match is a genuine near-cancellation, not a no-op. Report this check in the paper — a reviewer will ask.
4. **Evasion cost appears ordered by how conservative the launch policy is — but this finding is NOT yet usable.** The excluded reused-env run suggested expert −7.00 pp > BC −4.25 pp > SPC 0.00 pp, i.e. the more launch-suppressing the policy, the more evasion costs it. The direction is mechanistically plausible (a launch *window* wasted by an over-strict gate cannot be recovered once the target turns away), but the expert leg rests on the incomparable protocol, and n=400 on one seed family is a single draw. **Re-derive the expert leg from the fresh-env E3 d=0.3 run before asserting the ordering at all.** The BC and SPC legs are sound as measured.

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
| A6 | Full-network distillation (unfreeze all) | not run | shows freezing is what buys clean attribution |
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
- **Suboptimality is established conditionally**: on the frozen BC maneuver trajectory (§4.3) and within the frozen-heads intervention (§4.4). We do not claim fire-whenever-legal is optimal in general.
- **Statistical scope**: the +3.25 pp geometry result is a single seed family and fails a Bonferroni α=0.0125.
- **Evasion is a single interpolated setting** (`difficulty = 0.3`, one scripted evasion family), not a learned or optimised opponent. "Interactive" here means a reacting scripted target, not an adversary trained against the policy.
- SPC requires knowing *which* head to correct; automating that (e.g. per-head CLR + ablation ranking) is open.

### 5.3 Future Work
- Automate head selection from per-head diagnostics.
- N-vs-N: per-agent CLR and per-agent surgical correction; does the bias compound with fleet size?
- Launch *timing* when launch cost/ammo scarcity makes always-firing genuinely suboptimal — the regime where PPO becomes meaningful (project record documents that the current reward admits no such trade-off).
- Integrate into the hierarchical tactical architecture planned for the thesis.

### 5.4 Claim statement (exact wording to use)
> We do **not** claim that always-launching is optimal. We claim that **a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime**: with the maneuver policy provably frozen (`max logit diff < 1e-9`, identical action sequences), correcting the launch head alone raises the kill rate from 46.50% to 90.75% on 400 paired seeds (+44.25 pp, exact McNemar p = 1.04e-53, with all 177 discordant seeds favouring the correction and none favouring the original), and off-policy enumeration on a fixed maneuver trajectory shows the inherited launch policy to be dominated by every-launch-when-legal within the enumerated fire-policy family.

---

## 6. Number Provenance & Integrity Ledger

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
| **Matched expert @ d=0** | **129/400 = 32.25%**, CLR 6.04% (1119/18521) | `results/shoot_eval/E5_expert_d0_n400_s20000.json` |
| **Matched expert @ d=0.3** | **101/400 = 25.25%**, CLR 5.43% (1071/19736) | `results/shoot_eval/E5_expert_d03_n400_s20000.json` |
| Full 3×2 matrix | auto-generated from `run_meta` | `scripts/collect_matrix.py` → `results/shoot_eval/matrix_E1_E5_E7.md` |
| **E1: fire-oracle enumeration, dist2_3k** | asap **86.7%** / delay_30 38.3% / delay_60 10.0% / dlz_mid 0.0% / dlz_deep 0.0% / interval_100 0.0% / frozen BC **15.0%** | `results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json` |
| **E1: reachability audit** | legal window median **4.0 steps/ep**, first-legal median 67, 1st→4th median 743.5, 8/60 end with window open | same |
| **E2: rule oracle vs SPC, same seeds** | both **363/400 = 90.75%**, launches 3.90, hit 1.0000, lost 0, reasons identical | `results/shoot_eval/E2_asap_oracle_id_n400_s20000.json` vs `E7_spc_*` |
| **E3: paired BC vs rule expert** | expert **36.75%** (147/400) vs BC **46.50%** (186/400); Δ **+9.75 pp**, W/L/T **63/24/313**, exact McNemar **p = 3.48e-05** | `results/shoot_eval/E3_paired_bc_vs_expert_d0_n400_s20000_v2.json` (has `run_meta` incl. `env_lifecycle`; reproduces the original run bit-for-bit) |
| **Env-lifecycle confound (expert)** | reused-env expert **32.25%** vs fresh-env expert **36.75%** on identical seeds; per-seed kill vectors differ | `tests/check_expert_path_consistency.sh` |
| **E6: oracle envelope, matched 2×2** | d=0: asap **93.3%** vs inherited **53.3%**; d=0.3: **96.7%** vs **46.7%** ⇒ gap **+40.0 → +50.0 pp**; window median 4.0/42.0 at both | `E6_default_d0_s60.json` + `E6_fire_oracle_target_evasive_s60.json` |
| Expert CLR | 5.96% (560/9395) | `results/health_check/fire_hesitancy.json`; recomputable from `data/expert/shoot_rule_expert.npz` |
| BC CLR | 7.37% (614/8327) | same |
| Expert `fire_desired` ≡ `action[:,3]` on allowed steps | 560 = 560 | same |
| SPC @ U(0,60), n=100 s42 | 91/100 | `results/shoot_eval/eval_2x2_base_geoNew_s42.json` |
| SPC @ U(30,60), n=100 s42 | 87/100 | `results/shoot_eval/eval_2x2_base_geoOld_s42.json` (**= `eval_distilled_d0_s42.json`**) |
| geomA-retrained @ U(0,60), n=100 s42 | 95/100 | `results/shoot_eval/eval_geomA_d0_s42.json` |
| n=400 paired, base vs geomA | 362 vs 375, p=0.0146 | `eval_2x2_{base,geomA}_geoNew_n400_s42.json` + `geom2x2_n400_report.md` |
| Porpoising / low-level control | **no sustained porpoising**; script verdict = `PARTIAL` (1000-step cap < 30 s target, not instability) | `results/health_check/porpoise_v2.json` — alt p-p 15.2–16.9 m, alt osc 0.03–0.06 Hz; pitch p-p 3.1–4.6° but pitch std only ≈0.4° (small-amplitude, larger on turn patterns). Residual: revisit turn-pattern pitch during M2. |
| d=0.3 smoke | 2/2 kills | `results/shoot_acmi_d03/manifest.json` |

**Docs-attested only — JSON lost, MUST be regenerated before submission:**

| Metric | Value | Doc source | Regenerating script |
|---|---|---|---|
| ~~Expert ID kill~~ | ~~35.8%~~ → **use 32.25%** (matched, E5) | superseded | — |
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
| `git_head`, `script`, `timestamp_utc` | provenance |

`scripts/paired_mcnemar.py` **refuses to compute a p-value** if any of these disagree across arms, or if the two arms share a `model_id`/checkpoint hash. Identity is asserted, not assumed.

### 7.1 Queue

| Step | Command (WSL Ubuntu shell) | Cost | Status |
|---|---|---|---|
| **E7** | `eval_bc_1v1.py --weights <round1> --model-id bc_round1 --episodes 400 --seed 20000 --difficulty 0` and the same with `<asap_distilled>` / `--model-id spc_distilled`, then `paired_mcnemar.py --a ... --b ...` | ≈1.8 h | **DONE** — 46.50% vs 90.75%, **+44.25 pp**, exact McNemar **p=1.04e-53**, discordant **177:0** |
| **E5** | same two commands with `--difficulty 0.3`, plus `generate_shoot_rule_expert.py --validate --difficulty 0.3` | ≈3.4 h total | **BC+SPC DONE** — 42.25% vs 90.75%, **+48.50 pp**, p=7.97e-59, discordant **194:0**; expert arm running |
| **E6** | `fire_oracle_audit.py --cell target_evasive --start-seed 20000 --seeds 60 --oracles asap,bc` + matched d=0 cell `id_bias30_60` | ≈50 min | **DONE** — matched 2×2: gap **+40.0 → +50.0 pp**; replicates the §4.6 widening by an independent method |
| **E1** | `fire_oracle_audit.py --cell dist2_3k --seeds 60 --oracles all` | ≈2.2 h | **DONE** — asap **86.7%** vs inherited **15.0%**; all selective policies worse (§4.3). Re-running once more to attach `run_meta` |
| **E2** | `eval_asap_baseline.py --mode id --id-seeds 400 --start-seed 20000` | ≈1 h | **DONE** — rule oracle 90.75%, exactly matching SPC; required adding `--start-seed` (the script previously could only run seeds 0..N-1) |
| **E3** | `eval_paired_bc_vs_expert.py --seeds 500` | ≈1–2 h | pending |
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
- [ ] E5 expert arm at d=0.3 (running) — unpaired reference only
- [ ] d=0 vs d=0.3 comparison write-up (§4.6) — SPC invariant, gap widens
- [x] Regenerate the paper's evidence artifacts — E1 (oracle, reproducibility-verified) and E2 (rule oracle) done; E3 running
- [x] Reconcile BC's CLR geometry with BC's kill-rate geometry — E7 does this (both now U(0,60), seeds 20000–20399)
- [x] `run_meta` identity block + `paired_mcnemar.py` identity guards (2026-09-16)
- [x] Expert seed-pairing patch (`run_one(seed=...)`) — previously impossible
- [x] **Re-measure SPC vs the rule oracle on one seed set** — done: exactly equal (90.75% both); framed as a completeness check, not independent validation

**Content**
- [ ] **Related Work §2.2 is a stub** — needs 5–10 *real* citations on imitation from suboptimal demonstrations / expert conservatism. Do not submit with placeholders.
- [ ] ACMI trajectory figure: BC vs SPC on one seed, marking the missed legal window (the figure that makes the mechanism legible)
- [ ] Appendix: internal-name ↔ paper-name mapping (`ASAP distillation` ↔ SPC), oracle definitions, the fire-mask vs `fire_desired` contrast table
- [ ] Acknowledge the rule expert's provenance honestly (§3.1) — it is this project's own hand-designed rule; there is no external paper to cite, and inventing one would be worse than saying so
- [ ] Ablation A5/A6 (random-init fire head; full-network) if space permits
- [ ] Format to AAMAS template

