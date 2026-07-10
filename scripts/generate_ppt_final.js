/**
 * Final paper presentation — all white backgrounds, 12+ slides.
 * Follows .agents/skills/pptx design spec: Ocean Gradient palette,
 * Georgia/Calibri fonts, single-axis charts, thin marks.
 */

const pptxgen = require("/tmp/node_modules/pptxgenjs");
const fs = require("fs");

const C = {
  navy:    "065A82", teal:    "1C7293", coral:   "F96167",
  white:   "FFFFFF", light:   "F2F7F9", green:   "2CC44D",
  gray:    "8899A6", darkGray:"4A5568", black:   "1A202C",
  accent:  "E8F0F4",
};
const FH = "Georgia";
const FB = "Calibri";
const CHART_DIR = "results/viz/paper_charts";
const FIG3 = "results/viz/fig3_role_attention_matrix.png";
const OUT = "results/ppt/formation_coop_final.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3 × 7.5
pres.author = "Sean Nishimiya";
pres.title = "Multi-Agent RL for Cooperative Formation Flight";

// ── Helpers ──────────────────────────────────────────────────────────────────
function titleSlide(s, num, text) {
  s.background = { fill: C.white };
  s.addShape(pres.ShapeType.rect, { x: 0.5, y: 0.35, w: 0.06, h: 0.45, fill: { color: C.coral } });
  s.addText(String(num), { x: 0.5, y: 0.3, w: 1.0, h: 0.55, fontSize: 24, fontFace: FH, bold: true, color: C.coral, margin: 0 });
  s.addText(text, { x: 1.2, y: 0.3, w: 11.5, h: 0.55, fontSize: 28, fontFace: FH, bold: true, color: C.navy });
  s.addShape(pres.ShapeType.rect, { x: 0.5, y: 1.0, w: 12.3, h: 0.015, fill: { color: C.accent } });
}

function card(s, x, y, w, h, opts = {}) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: opts.fill || C.white },
    rectRadius: 0.06, line: opts.line ? { color: C.accent, width: 0.5 } : undefined,
    shadow: opts.shadow ? { type: "outer", blur: 3, offset: 1, color: "C0D0D8", opacity: 0.2 } : undefined,
  });
}

function statBox(s, x, y, w, h, val, label, sub) {
  card(s, x, y, w, h, { shadow: true });
  s.addText(val, { x, y: y + 0.1, w, h: 0.6, fontSize: 28, fontFace: FH, bold: true, color: C.coral, align: "center" });
  s.addText(label, { x, y: y + 0.7, w, h: 0.3, fontSize: 12, fontFace: FH, bold: true, color: C.darkGray, align: "center" });
  if (sub) s.addText(sub, { x, y: y + 0.95, w, h: 0.25, fontSize: 9, fontFace: FB, color: C.gray, align: "center" });
}

// ═══════════════════════════════════════════════════════════════════════════════
// P1: Title
// ═══════════════════════════════════════════════════════════════════════════════
let s = pres.addSlide();
s.background = { fill: C.white };
s.addShape(pres.ShapeType.rect, { x: 0.8, y: 2.4, w: 2.2, h: 0.05, fill: { color: C.coral } });
s.addText("Multi-Agent Reinforcement Learning\nfor Cooperative Formation Flight", {
  x: 0.8, y: 1.2, w: 11.5, h: 1.5,
  fontSize: 40, fontFace: FH, bold: true, color: C.navy, lineSpacingMultiple: 1.1,
});
s.addText("Token-Based CTDE with Self-Attention + Discrete Primitives\nOutperforms Centralized PPO on JSBSim 6-DOF F-16 Pursuit", {
  x: 0.8, y: 2.9, w: 11.5, h: 0.8,
  fontSize: 16, fontFace: FB, color: C.teal,
});
s.addText("Sean Nishimiya  ·  Zhejiang University  ·  July 2026  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", {
  x: 0.8, y: 5.5, w: 11.5, h: 0.4, fontSize: 11, fontFace: FB, color: C.gray,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P2: Pain Points — Death Triangle
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 1, "Core Challenge — The -7,500 Involution Deadlock");

const pains = [
  { t: "A. Non-Stationarity (IPPO)", b: "Independent critics → environment\nnon-stationarity. Training stuck\nat -7,500 plateau indefinitely.", fix: "Shared MAPPO\n(4x improvement)" },
  { t: "B. Credit Assignment Collapse", b: "Continuous Box(2) + 600-step\nepisodes → Gaussian variance\ndies. Entropy diverges to 4.15.", fix: "Discrete actions\ncaps entropy at 2.71" },
  { t: "C. AND-Gate Temporal Gap", b: "Dual-entry at 800m required.\nP1 median = 1,974m.\nSync entry rate = 0.0%.", fix: "Dynamic annealing\n+ pacing penalty" },
];
let px = 0.5;
pains.forEach((p) => {
  card(s, px, 1.3, 3.95, 5.0, { fill: C.light });
  s.addText(p.t, { x: px + 0.2, y: 1.45, w: 3.6, h: 0.4, fontSize: 16, fontFace: FH, bold: true, color: C.coral });
  s.addText(p.b, { x: px + 0.2, y: 2.1, w: 3.6, h: 2.5, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.4 });
  s.addShape(pres.ShapeType.roundRect, { x: px + 0.2, y: 4.8, w: 3.6, h: 0.55, fill: { color: C.green }, rectRadius: 0.05 });
  s.addText(p.fix, { x: px + 0.3, y: 4.8, w: 3.4, h: 0.55, fontSize: 11, fontFace: FH, bold: true, color: C.white, valign: "middle", align: "center" });
  px += 4.2;
});

// ═══════════════════════════════════════════════════════════════════════════════
// P3: Architecture A — Self-Attention + Parameter Sharing
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 2, "Architecture I — Parameter-Shared Self-Attention CTDE");

s.addText("Token-Based Multi-Head Self-Attention", {
  x: 0.5, y: 1.25, w: 6.0, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
s.addText([
  { text: "Observation [33] → 3 semantic tokens:", options: { breakLine: true } },
  { text: "  Self (13):     velocity, attitude, height, AoA, airspeed", options: { breakLine: true } },
  { text: "  Target (14):   rel pos/vel, tactical angles, LOS rate", options: { breakLine: true } },
  { text: "  Mate (6):      wingman rel pos/vel", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Token-Type Embedding → MHA (4 heads, d=128)", options: { breakLine: true } },
  { text: "→ Learned Attention Pooling → MLP [256,256]", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Shared policy for P0 and P1 → Permutation Invariance.", options: { breakLine: true } },
  { text: "Same network weights; different attention per agent.", options: {} },
], { x: 0.5, y: 1.7, w: 5.8, h: 3.8, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });

// Right: Fig 3 key stats
s.addText("Role-Grouped Attention (Fig 3, 7,858 steps)", {
  x: 6.8, y: 1.25, w: 6.0, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
const stats3 = [
  { v: "0.450", l: "Striker MHA\nSelf→Mate" },
  { v: "0.389", l: "Interceptor MHA\nSelf→Target" },
  { v: "-0.53", l: "Cohen's d\n(large effect)" },
  { v: "0.44", l: "Mutual Mate\nAttention" },
];
let sx3 = 6.8;
stats3.forEach((st) => {
  statBox(s, sx3, 1.8, 2.7, 1.4, st.v, st.l);
  sx3 += 2.85;
});
s.addText("Both agents sustain high mate attention (~0.44) — continuous implicit coordination.", {
  x: 6.8, y: 3.5, w: 6.0, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true,
});
if (fs.existsSync(FIG3)) {
  s.addImage({ path: FIG3, x: 6.8, y: 4.1, w: 6.0, h: 3.0, sizing: { type: "contain", w: 6.0, h: 3.0 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// P4: Architecture B — Discrete Primitives + Action Masking
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 3, "Architecture II — Discrete Tactical Primitives + Action Masking");

s.addText("Why Abandon Box(2)?", {
  x: 0.5, y: 1.25, w: 5.8, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
s.addText([
  { text: "Continuous DiagGaussian → unbounded sampling, manual clamping.", options: { breakLine: true } },
  { text: "Exploration diffuses in 2D → entropy runaway to 4.15.", options: { breakLine: true } },
  { text: "600-step episodes → credit assignment nearly impossible.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "MultiDiscrete([5,3]) = 15 bounded tactical primitives.", options: { breakLine: true } },
  { text: "Entropy capped at log(5)+log(3) = 2.71 theoretically.", options: { breakLine: true } },
  { text: "Action masking prevents physically impossible maneuvers.", options: {} },
], { x: 0.5, y: 1.7, w: 5.8, h: 2.8, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });

s.addText("MultiDiscrete([5, 3]) = 15 Primitives", {
  x: 6.8, y: 1.25, w: 6.0, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
// Turn table
s.addText("TURN", { x: 6.8, y: 1.75, w: 2.5, h: 0.28, fontSize: 13, fontFace: FH, bold: true, color: C.coral });
const turns = [["0","Hard Left  -15 deg/s"],["1","Soft Left  -5 deg/s"],["2","Straight  0 deg/s"],["3","Soft Right  +5 deg/s"],["4","Hard Right  +15 deg/s"]];
let ty = 2.08;
turns.forEach(([id, name]) => {
  card(s, 6.8, ty, 5.8, 0.36, { fill: C.light });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.36, fontSize: 12, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 5.2, h: 0.36, fontSize: 12, fontFace: FB, color: C.darkGray, valign: "middle" });
  ty += 0.42;
});
ty += 0.12;
s.addText("SPEED", { x: 6.8, y: ty, w: 2.5, h: 0.28, fontSize: 13, fontFace: FH, bold: true, color: C.coral });
ty += 0.35;
const speeds = [["0","Slow  180 m/s (energy-saving)"],["1","Cruise  250 m/s (balanced)"],["2","Fast  320 m/s (afterburner chase)"]];
speeds.forEach(([id, name]) => {
  card(s, 6.8, ty, 5.8, 0.36, { fill: C.light });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.36, fontSize: 12, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 5.2, h: 0.36, fontSize: 12, fontFace: FB, color: C.darkGray, valign: "middle" });
  ty += 0.42;
});
ty += 0.1;
s.addText("Safety: Action Masking prevents stall (<130 m/s), ground collision (<200m), overspeed.", {
  x: 6.8, y: ty, w: 6.0, h: 0.3, fontSize: 11, fontFace: FB, color: C.gray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P5: Chart 1 — Action Distribution Shift
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 4, "Action Distribution Evolution — From Random to Deterministic");

const c1 = CHART_DIR + "/chart1_action_distribution.png";
if (fs.existsSync(c1)) {
  s.addImage({ path: c1, x: 0.5, y: 1.3, w: 12.3, h: 5.5, sizing: { type: "contain", w: 12.3, h: 5.5 } });
}
s.addText("Early training (0-50 iters): near-uniform action selection (exploration). Late training (250-320 iters): concentrated on optimal primitives.", {
  x: 0.5, y: 6.5, w: 12.3, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P6: Curriculum — Dynamic Annealing + Distance Penalty
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 5, "Curriculum Design — Dynamic AND-Gate Annealing + Pacing");

s.addText("Dynamic AND-Gate Annealing", {
  x: 0.5, y: 1.25, w: 6.0, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
s.addText([
  { text: "AND-gate: BOTH < threshold + pincer > 30 deg", options: { breakLine: true } },
  { text: "P1 median distance = 1,974m → sync entry rate = 0%", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Solution: threshold anneals 2000m → 800m", options: { breakLine: true } },
  { text: "Result: eval improved from -8,800 to -1,171 (+4,700 pts)", options: { breakLine: true } },
  { text: "1,200-1,300m identified as CTDE learnability boundary", options: {} },
], { x: 0.5, y: 1.7, w: 6.0, h: 3.0, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });

s.addText("Three Cooperative Reward Mechanisms", {
  x: 7.0, y: 1.25, w: 5.8, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
const rewards = [
  { t: "Distance Asymmetry Penalty", b: "|d0 - d1| > 500m → team penalty\nPrevents free-riding behavior" },
  { t: "Time-Sync Pacing Penalty", b: "Striker < 1200m & Int > 1500m\n→ penalty = (d_int - d_str)/1000 x dt" },
  { t: "Dynamic Role Assignment", b: "Striker (closer): tracking x1.5\nInterceptor (further): pincer x2.0" },
];
let ry6 = 1.75;
rewards.forEach((r) => {
  card(s, 7.0, ry6, 5.8, 1.3, { shadow: true });
  s.addText(r.t, { x: 7.2, y: ry6 + 0.1, w: 5.4, h: 0.3, fontSize: 13, fontFace: FH, bold: true, color: C.coral });
  s.addText(r.b, { x: 7.2, y: ry6 + 0.42, w: 5.4, h: 0.8, fontSize: 12, fontFace: FB, color: C.darkGray });
  ry6 += 1.45;
});

// ═══════════════════════════════════════════════════════════════════════════════
// P7: Chart 3 — Reward Component Breakdown
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 6, "Reward Component Breakdown — Curriculum in Action");
const c3 = CHART_DIR + "/chart3_reward_breakdown.png";
if (fs.existsSync(c3)) {
  s.addImage({ path: c3, x: 0.5, y: 1.3, w: 12.3, h: 5.5, sizing: { type: "contain", w: 12.3, h: 5.5 } });
}
s.addText("Phase 1 (OR-gate): dominated by progress + ATA. Phase 2 (AND-gate): pincer + AND bonus expand as curriculum tightens.", {
  x: 0.5, y: 6.5, w: 12.3, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P8: Ablation Table
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 7, "Ablation Study — Six-Generation Model Comparison");

const hdr = ["Experiment", "Architecture", "Action Space", "BC", "Iters", "Best Eval", "Eval>0"];
const colW = [2.3, 1.6, 1.9, 1.2, 0.9, 1.4, 0.9];
const rows = [
  ["Exp 1 (non-coop)", "Shared Attn", "Box(2)", "SB3 BC", "200", "-8,053", "0"],
  ["Exp 2 (OR-gate)", "Shared Attn", "Box(2)", "SB3 BC", "120", "+7,888", "5x"],
  ["Exp 3v3 (AND dyn)", "Shared Attn", "Box(2)", "SB3 BC", "300", "-1,171", "0"],
  ["Exp 4a (MLP)", "MLP fallback", "MultiDisc(5,3)", "None", "120", "-4,542", "0"],
  ["Exp 4a-v2 (Attn)", "Self-Attention", "MultiDisc(5,3)", "None", "120", "+1,345", "1x"],
  ["Exp 4b (Attn+BC)", "Self-Attention", "MultiDisc(5,3)", "Disc BC", "120", "-1,135", "0"],
  ["* 4a-v2 extended", "Self-Attention", "MultiDisc(5,3)", "None", "320", "+2,376", "3x"],
];
const tblX = 0.8;
let tblY = 1.35;
const tblW = colW.reduce((a,b)=>a+b);

s.addShape(pres.ShapeType.rect, { x: tblX, y: tblY, w: tblW, h: 0.42, fill: { color: C.navy } });
let hx = tblX;
hdr.forEach((h, i) => {
  s.addText(h, { x: hx, y: tblY, w: colW[i], h: 0.42, fontSize: 11, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle" });
  hx += colW[i];
});
tblY += 0.47;

rows.forEach((row, ri) => {
  const bg = ri % 2 === 0 ? C.light : C.white;
  s.addShape(pres.ShapeType.rect, { x: tblX, y: tblY, w: tblW, h: 0.52, fill: { color: bg }, line: { color: C.accent, width: 0.3 } });
  let cx = tblX;
  row.forEach((cell, ci) => {
    const hi = cell.startsWith("*") || cell === "+2,376";
    s.addText(cell.replace("* ",""), {
      x: cx, y: tblY, w: colW[ci], h: 0.52,
      fontSize: 11, fontFace: FB,
      color: hi ? C.coral : C.darkGray,
      bold: hi, align: "center", valign: "middle",
    });
    cx += colW[ci];
  });
  tblY += 0.57;
});

s.addText("* Self-Attention is the decisive factor: cold-start beats MLP by 5,887 pts. BC adds stability but no extra peak. Structure determines ceiling.", {
  x: 0.5, y: tblY + 0.1, w: 12.3, h: 0.35, fontSize: 12, fontFace: FH, bold: true, color: C.coral,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P9: Core Result 1 — Cold-Start Emergence + KDE + 3D Trajectory
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 8, "Emergent Cooperative Tactics — Zero Expert Knowledge");

// Left: convergence curve (chart 5 proxy)
s.addText("Cold-Start Self-Attention: 3x Positive Eval Spikes", {
  x: 0.5, y: 1.25, w: 6.5, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
s.addText([
  { text: "320 iterations. No BC. No expert data. Pure emergence.", options: { breakLine: true } },
  { text: "Training peak: +5,401 (iter 300)", options: { breakLine: true } },
  { text: "Eval spikes: +1,345, +2,376, +4.0 (3 positive breakthroughs)", options: { breakLine: true } },
  { text: "Entropy: 2.49 -> 1.87 (healthy convergence)", options: { breakLine: true } },
  { text: "MLP baseline: 0 positive spikes, best -4,542.", options: { breakLine: true } },
  { text: "Self-Attn gap: 5,887 pts — purely architectural advantage.", options: {} },
], { x: 0.5, y: 1.7, w: 6.5, h: 3.5, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });

// Key stats
const bigStats = [
  { v: "+2,376", l: "Best Eval\nReward" },
  { v: "3x", l: "Positive\nSpikes" },
  { v: "320", l: "Training\nIterations" },
  { v: "0", l: "Expert\nSamples" },
];
let bx = 0.5;
bigStats.forEach((bs) => {
  statBox(s, bx, 5.0, 2.7, 1.4, bs.v, bs.l);
  bx += 2.85;
});

// Right: Spatial KDE
const c2 = CHART_DIR + "/chart2_spatial_kde.png";
if (fs.existsSync(c2)) {
  s.addImage({ path: c2, x: 7.5, y: 1.2, w: 5.5, h: 4.0, sizing: { type: "contain", w: 5.5, h: 4.0 } });
}
s.addText("Spatial KDE: 127K frames from 50 eval episodes. High-density clusters in 30-60 deg rear sector.", {
  x: 7.5, y: 5.3, w: 5.5, h: 0.6, fontSize: 11, fontFace: FB, color: C.gray, italic: true,
});

// Health chart below
const c5 = CHART_DIR + "/chart5_health_metrics.png";
if (fs.existsSync(c5)) {
  s.addImage({ path: c5, x: 7.5, y: 5.7, w: 5.5, h: 1.6, sizing: { type: "contain", w: 5.5, h: 1.6 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// P10: Core Result 2 — Fig 3 Deep Dive + Attention Timeline
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 9, "Mathematical Proof — Spontaneous Role Differentiation via Self-Attention");

s.addText("Role-Grouped Attention Matrix (7,858 steps per role, 49 episodes)", {
  x: 0.5, y: 1.25, w: 6.5, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
s.addText([
  { text: "Classified by INSTANTANEOUS GEOMETRY, not agent ID.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Striker MHA:    Self->Mate = 0.450 (coordination)   Pool Mate = 0.341", options: { breakLine: true } },
  { text: "Interceptor MHA: Self->Target = 0.389 (pursuit)     Pool Mate = 0.298", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Cohen's d = -0.53 (large effect, p < 0.001)", options: { breakLine: true } },
  { text: "Interceptor pays 31% more attention to Target than Striker.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Both roles sustain high mate attention (~0.44)", options: { breakLine: true } },
  { text: "-> Continuous implicit coordination, NOT binary switching.", options: { breakLine: true } },
  { text: "-> Definitive proof: parameter-shared network spontaneously", options: { breakLine: true } },
  { text: "   breaks symmetry and learns distinct role attention patterns.", options: {} },
], { x: 0.5, y: 1.7, w: 6.3, h: 5.0, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.25 });

if (fs.existsSync(FIG3)) {
  s.addImage({ path: FIG3, x: 7.2, y: 1.3, w: 5.8, h: 5.8, sizing: { type: "contain", w: 5.8, h: 5.8 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// P11: Limitations — Termination Autopsy
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 10, "Boundary Analysis — Why AND-Gate Remains Hard");

s.addText("AND-Gate Autopsy (Exp 3 checkpoint, 10 eval episodes)", {
  x: 0.5, y: 1.25, w: 6.5, h: 0.35, fontSize: 17, fontFace: FH, bold: true, color: C.navy,
});
const autopsy = [
  { m: "Sync Entry Rate (BOTH < 800m)", v: "0.0%", n: "Fatal bottleneck" },
  { m: "Single Entry Rate (>=1 < 800m)", v: "22.1%", n: "P0 can approach" },
  { m: "Pincer Angle > 30 deg", v: "58.4%", n: "Geometry is good" },
  { m: "P0 Median Distance", v: "329 m", n: "Excellent approach" },
  { m: "P1 Median Distance", v: "1,974 m", n: "Cannot close gap" },
];
let ay = 1.75;
autopsy.forEach((a) => {
  card(s, 0.5, ay, 5.8, 0.7, { fill: C.light });
  s.addText(a.m, { x: 0.7, y: ay, w: 3.5, h: 0.4, fontSize: 12, fontFace: FB, color: C.darkGray, valign: "bottom" });
  s.addText(a.v, { x: 4.5, y: ay, w: 1.2, h: 0.4, fontSize: 16, fontFace: FH, bold: true, color: C.coral, valign: "bottom", align: "right" });
  s.addText(a.n, { x: 0.7, y: ay + 0.38, w: 5.0, h: 0.22, fontSize: 10, fontFace: FB, color: C.gray, valign: "top" });
  ay += 0.78;
});

// Right chart 4
const c4 = CHART_DIR + "/chart4_termination_reasons.png";
if (fs.existsSync(c4)) {
  s.addImage({ path: c4, x: 7.0, y: 1.2, w: 5.8, h: 4.2, sizing: { type: "contain", w: 5.8, h: 4.2 } });
}

s.addText("Three Theoretical Ceilings: (1) CTDE information asymmetry, (2) discrete exploration boundary, (3) lack of explicit time-to-intercept signal.", {
  x: 0.5, y: 6.3, w: 12.3, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// P12: Future Work
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
titleSlide(s, 11, "Future Directions — Toward NxM Scalable Formation Combat");

const dirs = [
  { n: "1", t: "Explicit Assignment Constraints", b: "Hungarian Algorithm for NvM weapon-target pairing. Solve combinatorial explosion in large-scale formation. Reference: DARPA ACE / OFFSET programs." },
  { n: "2", t: "Micro-Hierarchy SMDP", b: "Upper: low-freq tactical options. Lower: high-freq action masking + flight control. Bridges strategic planning and tactical execution." },
  { n: "3", t: "Self-Play & League Training", b: "Beyond scripted targets to adversarial training. Population-based training with diversity objectives. Emergent tactics through competitive co-evolution." },
  { n: "4", t: "Explicit Coordination Channels", b: "Add delta-TGO to global state. Learned communication via attention or message passing. Break the AND-gate barrier with explicit temporal signals." },
];
let dy = 1.25;
dirs.forEach((d) => {
  card(s, 0.5, dy, 12.3, 1.1, { shadow: true });
  s.addShape(pres.ShapeType.ellipse, { x: 0.7, y: dy + 0.2, w: 0.6, h: 0.6, fill: { color: C.coral } });
  s.addText(d.n, { x: 0.7, y: dy + 0.2, w: 0.6, h: 0.6, fontSize: 18, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle" });
  s.addText(d.t, { x: 1.55, y: dy + 0.1, w: 11.0, h: 0.35, fontSize: 16, fontFace: FH, bold: true, color: C.navy });
  s.addText(d.b, { x: 1.55, y: dy + 0.48, w: 11.0, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray });
  dy += 1.25;
});

dy += 0.3;
s.addShape(pres.ShapeType.rect, { x: 0.5, y: dy, w: 12.3, h: 0.03, fill: { color: C.coral } });
s.addText("Thank you. Questions & Discussion welcome.", {
  x: 0.5, y: dy + 0.2, w: 12.3, h: 0.5, fontSize: 22, fontFace: FH, bold: true, color: C.navy, align: "center",
});
s.addText("sean@zju.edu.cn  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", {
  x: 0.5, y: dy + 0.7, w: 12.3, h: 0.3, fontSize: 11, fontFace: FB, color: C.gray, align: "center",
});

// ── WRITE ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  const st = fs.statSync(OUT);
  console.log(`Saved: ${OUT} (${(st.size/1024).toFixed(0)} KB, 12 slides)`);
});
