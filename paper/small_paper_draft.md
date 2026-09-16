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

Behavior cloning (BC) from rule-based experts is a standard bootstrap for air-combat policies, under the implicit assumption that the expert's decisions are worth imitating. We test that assumption on a JSBSim F-16 within-visual-range (WVR) 1v1 missile-engagement benchmark. We introduce the **Conditional Launch Rate (CLR)** — the probability that a policy commands launch at a step where the environment permits it, `P(a_fire=1 | m_fire=1)` — and show that the hand-designed rule expert fires at only **5.96%** of permitted steps, because it applies a launch-quality gate strictly tighter than the environment's legality mask. BC reproduces that gate faithfully (**7.37%**): the defect lies in the teacher's designed decision boundary, not in the imitation, which is behaving as intended. To isolate the cost we freeze the BC maneuver trajectory and enumerate the fire-policy family; firing whenever legal achieves 73.3% kills where the inherited policy achieves 6.7%, and every range- or delay-selective alternative is worse. We then apply **Surgical Policy Correction (SPC)**, which re-trains *only* the launch head while every other parameter stays bit-identical (heading/speed logits `max diff < 1e-9`). Under a matched d=0, deterministic-argmax evaluation on paired seeds, replacing the conservative launch decision raises the kill rate from **43.2% to 91.2%**, matching the rule oracle (90.6%). We do not claim that launching whenever legal is generally optimal; we claim that a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime.

> **措辞纪律（Sean 2026-09-16 要求）**：不得写 "SPC improves performance by 48 pp" 这类泛化句式。必须始终绑定四个限定：**matched setting / d=0 / deterministic argmax / isolated intervention**。否则 reviewer 的第一反应是「为什么只改一个 head 能提升这么多？」—— 答案正是「因为轨迹冻结，所以差异只能来自这个 head」，但这个因果必须自己讲出来，不能被追问。
>
> **待 E7 落地后回填**：paired seed 数与 p 值（E7 正在跑，n=400，seeds 20000–20399）。

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
1. **C1 — Diagnosis.** A cheap, transferable metric: the **Conditional Launch Rate (CLR)**, `P(a_fire=1 | m_fire=1)`. We show the expert sits at 5.96% and BC at 7.37%, and we localize the cause to a *designed* launch-quality gate strictly tighter than the environment's legality mask (§3.2).
2. **C2 — Intervention.** **Surgical Policy Correction (SPC)**: re-train a single binary action head toward the environment-legality policy while freezing all other parameters, verified bit-identical by logit and action-sequence gates (§3.3).
3. **C3 — Causal evidence.** Two independent identifications that the inherited launch policy is suboptimal *conditional on the maneuver*: (a) off-policy enumeration of the fire-policy family on a **frozen** BC maneuver trajectory (§4.3), and (b) the SPC intervention with provably frozen maneuver heads (§4.4).

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
| Rule expert | 9395 | 560 | **5.96%** |
| BC round1 (frozen) | 8327 | 614 | **7.37%** |
| SPC | — | — | **≥ 99.5%** (gate G3) |
| Rule oracle (fire = mask) | — | all allowed | 100% (by construction) |

Two observations: (i) BC is *not* more conservative than the expert in aggregate — it inherits the gate and adds noise; (ii) both sit ~an order of magnitude below the oracle. **Neither figure alone proves suboptimality** — that requires §4.3.

### 4.3 C3(a) — Off-policy enumeration on a frozen trajectory (the key identification)

**Design.** Take BC's heading/speed decisions as a *frozen* maneuver trajectory per seed, and enumerate a family of fire policies on top of it. Because the maneuver is fixed by construction, the maneuver policy is controlled, and differences are attributable to the launch decision.

**Cell `dist2_3k`, 60 seeds:**

| Fire policy | Kill rate | Launches/ep | Reach-4-launch |
|---|---|---|---|
| **`asap` — fire whenever legal** | **73.3%** | 3.68 | 73.3% |
| `delay_30` (first legal + 30) | 28.3% | 2.97 | 28.3% |
| `delay_60` | 10.0% | 2.60 | 10.0% |
| `dlz_mid` (depth 0.4–0.6) | 0.0% | 1.08 | 0.0% |
| `dlz_deep` (depth ≥ 0.6) | 3.3% | 1.12 | 3.3% |
| `interval_100` | 0.0% | 1.92 | 0.0% |
| **frozen BC (inherited)** | **6.7%** | 2.07 | 6.7% |

**Reachability audit (asap arm):** median first-legal step 75 (15 s); **cumulative legal-window length ≈ 4 steps per episode (~1 step per DLZ transit)**; median step-1 → step-4 launch = 783 (157 s); 16/60 seeds never reach 4 launches (15 episodes end with the window still open, 1 never reopens).

**Reading:**
- The scenario is **not infeasible** — the environment permits a 4-launch salvo in 73% of episodes, and `asap` realizes 73.3% kills.
- The bottleneck is the launch policy: the inherited policy uses **2.07 of 3.68** available windows, because most windows have DLZ depth < 0.25 and are rejected by `fire_desired`.
- **Every** selective alternative is worse than fire-whenever-legal. Within this family, on this trajectory class, always-launching dominates.
- Note what this does and does not show: it establishes suboptimality **conditional on the frozen maneuver**. It does not establish that always-launching is optimal for arbitrary maneuvers of arbitrary policies. (§4.4 supplies the within-policy counterpart.)

> **Artifact status:** the JSON (`results/shoot_eval/fire_oracle_dist2_3k_60.json`) is **GONE** — `results/shoot_eval/*.json` is gitignored and the file no longer exists on disk. The table above is transcribed from `docs/plan_bc_rule_expert.md` §"dist2_3k fire oracle" and `docs/summary_phase1.md` §4.2. **Must be regenerated before submission** — `scripts/fire_oracle_audit.py` survives.

### 4.4 C3(b) — SPC intervention result

| Policy | ID kill rate (500 paired seeds) | Launches/ep | Notes |
|---|---|---|---|
| v19 PPO (historical baseline) | 10% | 0.48 | old training artifact |
| Rule expert | 35.8% | 2.89 | 500 paired seeds |
| **BC round1 (frozen)** | **43.2%** | 3.01 | Δ vs expert **+7.4 pp**, bootstrap 95% CI [+3.8, +11.0] |
| ASAP rule override (oracle) | 90.6% | 3.89 | fire = mask, maneuver frozen |
| **SPC (this paper)** | **91.2%** | 3.90 | gates G1–G4 PASS; hit 99.9% |

- **Fire-only intervention effect: 43.2% → 91.2% = +48.0 pp**, with `lost = 0` and `bad = 0` in all suites.
- **Report it bound to its conditions, never as a bare improvement.** The sentence to use is: *"Under a matched d=0, deterministic-argmax evaluation, replacing the conservative launch decision with SPC raises the kill rate from 43.2% to 91.2%."* Not: *"SPC improves performance by 48 pp."* The four qualifiers (matched setting, d=0, deterministic argmax, isolated intervention) are load-bearing: they are the reason the attribution is legitimate.
- **Significance test**: the +48 pp figure is currently a point estimate. `scripts/paired_mcnemar.py` produces the paired exact McNemar test on shared seeds (E7). The abstract must not state +48 pp without the accompanying n and p.
- SPC lands within **0.6 pp of the rule oracle** (91.2% vs 90.6%) without being told the rule — the learned head reproduces the ceiling.
- Heading/speed behaviour is bit-identical to BC by construction (G1: `max diff < 1e-9`; G2: sequences identical), so the entire gain is attributable to the launch head.
- Single-seed detail: `eval_distilled_d0_s42.json` (n=100, s42) → kill 87/100, hit 0.9974, launches 3.85, `lost_after_wez = 0`, launch quality `premium 218 / good 167 / bad 0` (i.e. the corrected policy does **not** trade shot quality for quantity — every launch still scores good-or-better under the expert's own quality function).

> **Artifact status:** `paired_bc_vs_expert_500.json` and `asap_baseline.json` are **GONE** (gitignored). Table transcribed from `docs/summary_phase1.md` §2/§4. **Must be regenerated.** `scripts/eval_paired_bc_vs_expert.py` and `scripts/eval_asap_baseline.py` survive.

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

### 4.6 Robustness B — target evasion (`difficulty_level = 0.3`) — **REQUIRED, NOT YET RUN**

- Target adds S-turn `±30°·d·sin(0.3t)` plus a missile-threat break-turn and a dive to `−800·d` m (floor 2000 m).
- Status: only a 2-seed ACMI render exists (`results/shoot_acmi_d03/manifest.json`, both seeds killed, `wez_to_fire_latency` 1 and 18 steps). **No statistical evaluation.**
- **Unified protocol (all three arms, identical seeds, one code path):**
  - BC and SPC: `eval_bc_1v1.py --difficulty 0.3 --episodes 400 --seed 20000` (same command as E7, only `--difficulty` changes — so d=0 and d=0.3 share the seed framework and are directly comparable)
  - Expert: `generate_shoot_rule_expert.py --validate --difficulty 0.3 --episodes 400 --seed 20000 --out-json ...`
- **Expert seed-pairing was previously impossible**: `run_one()` called `env.reset()` with no seed, so the expert's episodes could not be aligned to BC/SPC by seed and no paired test involving the expert was possible. Patched 2026-09-16: `run_one(..., seed=...)` plus `--out-json` emitting `run_meta` and CLR. The data-generation path is deliberately left unseeded so the existing `shoot_rule_expert.npz` remains byte-reproducible.
- **Smoke result (3 episodes, seeds 20000–20002, d=0.3, not evidential):** expert CLR **6.94%** (10/144 allowed) — statistically indistinguishable from the d=0 value of 5.96%. Early indication that the designed gate is **insensitive to evasion**, i.e. the conservatism is structural rather than scenario-specific. Needs the full n=400 run before being asserted.
- Why it matters: it is the only test of whether the diagnosis+correction survives a *non-stationary, reacting* opponent — i.e. the interactive claim. Priority: **highest among remaining experiments.**

### 4.7 Ablation ladder (ordered by the decided priority)

| # | Ablation | Status | Purpose |
|---|---|---|---|
| A1 | **Fire-policy oracle enumeration on frozen trajectory** (§4.3) | done, JSON lost → re-run | establishes that the inherited launch policy is suboptimal, maneuver held fixed |
| A2 | **SPC (learned head) vs ASAP rule oracle** (§4.4) | done, JSON lost → re-run | learned head matches the ceiling ⇒ not a hand-coded hack |
| A3 | **`difficulty_level = 0.3`** (§4.6) | **not run** | interactive/reacting opponent |
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
- **d=0.3 unverified** as of this draft (§4.6).
- SPC requires knowing *which* head to correct; automating that (e.g. per-head CLR + ablation ranking) is open.

### 5.3 Future Work
- Automate head selection from per-head diagnostics.
- N-vs-N: per-agent CLR and per-agent surgical correction; does the bias compound with fleet size?
- Launch *timing* when launch cost/ammo scarcity makes always-firing genuinely suboptimal — the regime where PPO becomes meaningful (project record documents that the current reward admits no such trade-off).
- Integrate into the hierarchical tactical architecture planned for the thesis.

### 5.4 Claim statement (exact wording to use)
> We do **not** claim that always-launching is optimal. We claim that **a behaviorally isolated intervention reveals the inherited launch policy to be suboptimal in the studied regime**: with the maneuver policy provably frozen (`max logit diff < 1e-9`, identical action sequences), correcting the launch head alone raises the kill rate from 43.2% to 91.2% on 500 paired seeds, and off-policy enumeration on a fixed maneuver trajectory shows the inherited launch policy to be dominated by every-launch-when-legal within the enumerated fire-policy family.

---

## 6. Number Provenance & Integrity Ledger

**Verified against live artifacts on 2026-09-16 (re-checkable):**

| Metric | Value | Artifact |
|---|---|---|
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
| Expert ID kill | 35.8% | `docs/summary_phase1.md` §2 | `eval_paired_bc_vs_expert.py` |
| BC ID kill | 43.2% (Δ+7.4 pp, CI [+3.8,+11.0]) | same | same |
| ASAP rule oracle ID kill | 90.6% | same | `eval_asap_baseline.py` |
| SPC ID kill | 91.2% | same | `eval_bc_1v1.py` |
| Fire-policy oracle table (§4.3) | asap 73.3% … BC 6.7% | `docs/plan_bc_rule_expert.md` §"dist2_3k fire oracle" | `fire_oracle_audit.py` |
| dist2_3k: BC / ASAP / SPC | 6.7% / 74% / 77% | `docs/summary_phase1.md` §2 | `eval_scenario_matrix.py` |

**Not yet measured:** d=0.3 statistics for any policy; paired BC-vs-SPC McNemar at n≥400; multi-seed-family extrapolation.

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
| **E7** | `eval_bc_1v1.py --weights <round1> --model-id bc_round1 --episodes 400 --seed 20000 --difficulty 0` and the same with `<asap_distilled>` / `--model-id spc_distilled`, then `paired_mcnemar.py --a ... --b ...` | ≈2–3 h | **RUNNING** (n=400, seeds 20000–20399) |
| **E5** | same two commands with `--difficulty 0.3`, plus `generate_shoot_rule_expert.py --validate --difficulty 0.3` | ≈2–3 h | pending E7 |
| **E6** | `fire_oracle_audit.py --cell target_evasive ...` | ≈1 h | **interface unverified** — Sean's call: dry-run/接口确认 only, do not spend GPU time yet |
| **E1** | `fire_oracle_audit.py --cell dist2_3k --seeds 60 --oracles all` | ≈1 h | pending |
| **E2** | `eval_asap_baseline.py --mode all` | ≈30 min | pending |
| **E3** | `eval_paired_bc_vs_expert.py --seeds 500` | ≈1–2 h | pending |
| **E4** | already covered by E7's `spc_distilled` arm | — | folded into E7 |

**Scope decision (Sean):** only artifacts that enter the paper or its supplement get regenerated. Scenario-matrix and stress suites are dropped unless a reviewer asks.

**Note:** BC's CLR (7.37%, measured under the *current* U(0,60) geometry) and BC's kill rate (43.2%, from the sealed Phase-1 record under the *old* U(30,60) geometry) came from different geometry generations. **E7 resolves this** by putting kill rate, CLR, and geometry on one seed set (seeds 20000–20399, U(0,60)). Treat the 43.2%/91.2% pair as *provisional* until E7 reports.


---

## 8. Submission Target

- **Venue**: AAMAS 2027 short paper (main conference)
- **Deadline**: ≈2026-11
- **Length**: 5–7 pages
- **Positioning check** (must pass before writing): the contribution reads as *expert-induced policy bias in interacting-agent environments, with a causal-identification methodology*, not as *a classifier head was retrained*.

---

## 9. TODO Before Submission

**Integrity prerequisites**
- [ ] **E7**: paired McNemar BC vs SPC at n=400 — *running*; the headline +48 pp currently has no significance test
- [ ] **E5**: d=0.3 for expert / BC / SPC on the shared seed set
- [ ] Regenerate the lost JSON artifacts that enter the paper (E1 oracle, E2 rule-oracle, E3 paired baseline)
- [ ] Reconcile BC's CLR geometry (U(0,60)) with BC's kill-rate geometry (U(30,60)) — E7 does this
- [x] `run_meta` identity block + `paired_mcnemar.py` identity guards (2026-09-16)
- [x] Expert seed-pairing patch (`run_one(seed=...)`) — previously impossible

**Content**
- [ ] **Related Work §2.2 is a stub** — needs 5–10 *real* citations on imitation from suboptimal demonstrations / expert conservatism. Do not submit with placeholders.
- [ ] ACMI trajectory figure: BC vs SPC on one seed, marking the missed legal window (the figure that makes the mechanism legible)
- [ ] Appendix: internal-name ↔ paper-name mapping (`ASAP distillation` ↔ SPC), oracle definitions, the fire-mask vs `fire_desired` contrast table
- [ ] Acknowledge the rule expert's provenance honestly (§3.1) — it is this project's own hand-designed rule; there is no external paper to cite, and inventing one would be worse than saying so
- [ ] Ablation A5/A6 (random-init fire head; full-network) if space permits
- [ ] Format to AAMAS template

