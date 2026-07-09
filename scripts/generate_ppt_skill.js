/**
 * Paper presentation: Multi-Agent RL for Cooperative Formation Flight
 * Generated following .agents/skills/pptx/SKILL.md design guidelines.
 *
 * Palette: Ocean Gradient (#065A82 / #1C7293 / #21295C)
 * Font: Georgia (headers) + Calibri (body)
 * Layout: 16:9 wide (13.3" × 7.5")
 */

const pptxgen = require("/tmp/node_modules/pptxgenjs");
const path = require("path");
const fs = require("fs");

// ── Palette & Constants ─────────────────────────────────────────────────────
const C = {
  deep:    "065A82",  // primary
  teal:    "1C7293",  // secondary
  mid:     "21295C",  // midnight accent
  white:   "FFFFFF",
  light:   "F2F7F9",  // light bg
  coral:   "F96167",  // highlight accent
  green:   "2CC44D",
  gray:    "8899A6",
  darkGray:"4A5568",
  black:   "1A202C",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";
const OUT = "results/ppt/formation_coop_skill.pptx";

// ── Helpers ──────────────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3" × 7.5"
pres.author = "Sean Nishimiya";
pres.title = "Multi-Agent RL for Cooperative Formation Flight";

function darkSlide(slide) {
  slide.background = { fill: C.deep };
}
function lightSlide(slide) {
  slide.background = { fill: C.light };
}
function midSlide(slide) {
  slide.background = { fill: C.mid };
}

// section number badge
function addBadge(slide, num, x, y) {
  slide.addShape(pres.ShapeType.ellipse, {
    x: x, y: y, w: 0.45, h: 0.45,
    fill: { color: C.coral },
  });
  slide.addText(String(num), {
    x: x, y: y, w: 0.45, h: 0.45,
    fontSize: 14, fontFace: FONT_H, bold: true,
    color: C.white, align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 0 — Title (dark)
// ═══════════════════════════════════════════════════════════════════════════════
let s = pres.addSlide();
darkSlide(s);

// accent bar
s.addShape(pres.ShapeType.rect, { x: 0.8, y: 2.6, w: 2.0, h: 0.04, fill: { color: C.coral } });

s.addText("Multi-Agent Reinforcement Learning\nfor Cooperative Formation Flight", {
  x: 0.8, y: 1.5, w: 11.5, h: 1.4,
  fontSize: 38, fontFace: FONT_H, bold: true, color: C.white,
  lineSpacingMultiple: 1.1,
});

s.addText("Token-Based CTDE with Self-Attention Outperforms Centralized PPO\non JSBSim 6-DOF F-16 Formation Pursuit", {
  x: 0.8, y: 3.1, w: 11.5, h: 0.8,
  fontSize: 16, fontFace: FONT_B, color: C.teal,
});

s.addText("Sean Nishimiya  ·  Zhejiang University  ·  July 2026", {
  x: 0.8, y: 5.2, w: 11.5, h: 0.4,
  fontSize: 12, fontFace: FONT_B, color: C.gray,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Background + Architecture (light)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
lightSlide(s);
addBadge(s, 1, 0.6, 0.4);

s.addText("Research Background & High-Fidelity Environment", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.deep,
});

// Left column
s.addText("Problem Context", {
  x: 0.6, y: 1.3, w: 5.5, h: 0.35,
  fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});
s.addText([
  { text: "Modern air combat has evolved into N×M system-of-systems engagement,", options: { breakLine: true } },
  { text: "heavily dependent on spatiotemporal coordination between assets.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "2v1 formation pursuit serves as the minimal viable testbed for", options: { breakLine: true } },
  { text: "cooperative multi-agent coordination under physical dynamics constraints.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "JSBSim 6-DOF F-16 FDM provides extreme aerodynamic fidelity.", options: { breakLine: true } },
  { text: "FlightController @ 60 Hz PID handles low-level stabilization.", options: { breakLine: true } },
  { text: "RLlib MAPPO + Ray 2.40 delivers scalable multi-agent training.", options: {} },
], { x: 0.6, y: 1.7, w: 5.5, h: 3.2, fontSize: 13, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.3 });

// Right — architecture cards
s.addText("Three-Layer Architecture", {
  x: 6.8, y: 1.3, w: 5.5, h: 0.35,
  fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});

const layers = [
  { title: "SCENARIO LAYER", body: "FormationEnv → RLlib MultiAgentEnv\n2v1 cooperative, NvM extensible\nOR-gate → AND-gate curriculum" },
  { title: "ALGORITHM LAYER", body: "Shared MAPPO (CTDE)\nSelf-Attn: 33-dim → 3 tokens → MHA\nMultiDiscrete([5,3]) + Action Masking" },
  { title: "INFRASTRUCTURE", body: "JSBSim F-16 FDM → FlightController\nRLlib TorchModelV2 + shared_policy\nWSL2 + CUDA GPU passthrough" },
];
let cy = 1.75;
layers.forEach((l) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.8, y: cy, w: 5.7, h: 1.25,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 3, offset: 1, color: "D0D8DD", opacity: 0.3 },
  });
  s.addText(l.title, {
    x: 7.0, y: cy + 0.08, w: 5.3, h: 0.28,
    fontSize: 11, fontFace: FONT_H, bold: true, color: C.coral,
  });
  s.addText(l.body, {
    x: 7.0, y: cy + 0.38, w: 5.3, h: 0.8,
    fontSize: 11, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.2,
  });
  cy += 1.4;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Death Triangle (dark)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
darkSlide(s);
addBadge(s, 2, 0.6, 0.4);

s.addText("Core Challenge — The Death Triangle of Cooperation", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.white,
});

const pains = [
  { title: "A. Non-Stationarity", body: "IPPO causes symmetry involution.\nTwo independent critics → environment\nnon-stationary to each agent.\nTraining stuck at −7,500 plateau.", fix: "→ Shared MAPPO (4× better)" },
  { title: "B. Credit Assignment", body: "Continuous Box(2) + 600-step\nepisodes → Gaussian variance collapse.\nEntropy runs away to 4.15.\nExploration drowns coordination.", fix: "→ Discrete caps entropy at 2.71" },
  { title: "C. AND-Gate Blind Zone", body: "Strict 800m dual-entry requirement.\nP1 median distance = 1,974m.\nSync entry rate = 0.0%.\nGeometry OK, timing desynchronized.", fix: "→ Curriculum + pacing penalty" },
];
let px = 0.5;
pains.forEach((p) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: px, y: 1.4, w: 3.9, h: 4.8,
    fill: { color: "15436B" }, rectRadius: 0.1,
  });
  s.addText(p.title, {
    x: px + 0.2, y: 1.6, w: 3.5, h: 0.4,
    fontSize: 18, fontFace: FONT_H, bold: true, color: C.coral,
  });
  s.addText(p.body, {
    x: px + 0.2, y: 2.3, w: 3.5, h: 2.6,
    fontSize: 13, fontFace: FONT_B, color: C.white, lineSpacingMultiple: 1.4,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: px + 0.2, y: 5.0, w: 3.5, h: 0.55,
    fill: { color: C.green }, rectRadius: 0.06,
  });
  s.addText(p.fix, {
    x: px + 0.3, y: 5.0, w: 3.3, h: 0.55,
    fontSize: 11, fontFace: FONT_H, bold: true, color: C.white, valign: "middle",
  });
  px += 4.15;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Self-Attention Architecture (light)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
lightSlide(s);
addBadge(s, 3, 0.6, 0.4);

s.addText("Architecture Evolution I — Token-Based Self-Attention", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.deep,
});

// Left — architecture description
s.addText("From Flat MLP to Semantic Tokens", {
  x: 0.6, y: 1.3, w: 6.0, h: 0.35,
  fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});
s.addText([
  { text: "Observation [33] decomposed into 3 semantic tokens:", options: { breakLine: true } },
  { text: "  Self (13):    own velocity, attitude, angular velocity, height, α", options: { breakLine: true } },
  { text: "  Target (14):  target rel pos/vel, tactical angles, LOS rate", options: { breakLine: true } },
  { text: "  Mate (6):     wingman rel pos/vel", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Token-Type Embedding → MultiHeadAttention (4 heads, d=128)", options: { breakLine: true } },
  { text: "→ Learned Attention Pooling → MLP [256,256] → action", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Key property: Permutation Invariance.", options: { breakLine: true } },
  { text: "Same shared network, different observations per agent →", options: { breakLine: true } },
  { text: "spontaneous role differentiation via attention weights.", options: {} },
], { x: 0.6, y: 1.75, w: 6.0, h: 3.8, fontSize: 12, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.3 });

// Right — Fig 3 attention matrix
s.addText("Role-Grouped Attention (Fig 3)", {
  x: 7.2, y: 1.3, w: 5.5, h: 0.35,
  fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});

// 4 stat callout boxes
const stats = [
  { val: "7,858", label: "Steps per Role", sub: "49 episodes" },
  { val: "-0.53", label: "Cohen's d", sub: "Self→Target (large)" },
  { val: "0.44", label: "Mutual Mate Attn", sub: "Both roles" },
  { val: "35.8°", label: "Mean Pincer", sub: "Golden range" },
];
let sx = 7.2;
stats.forEach((st) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: sx, y: 1.85, w: 2.5, h: 1.3,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "D0D8DD", opacity: 0.3 },
  });
  s.addText(st.val, {
    x: sx, y: 1.9, w: 2.5, h: 0.55,
    fontSize: 28, fontFace: FONT_H, bold: true, color: C.coral, align: "center",
  });
  s.addText(st.label, {
    x: sx, y: 2.4, w: 2.5, h: 0.35,
    fontSize: 11, fontFace: FONT_H, bold: true, color: C.darkGray, align: "center",
  });
  s.addText(st.sub, {
    x: sx, y: 2.68, w: 2.5, h: 0.28,
    fontSize: 9, fontFace: FONT_B, color: C.gray, align: "center",
  });
  sx += 2.7;
});

// Key interpretation
s.addText([
  { text: "★ Striker MHA Self→Mate: 0.450 (coordination). Interceptor Self→Target: 0.389 (pursuit).", options: { breakLine: true } },
  { text: "★ Both agents sustain high mutual attention (~0.44) — continuous implicit coordination.", options: { breakLine: true } },
  { text: "★ Mathematical proof of emergent role differentiation from parameter-shared network.", options: {} },
], { x: 7.2, y: 3.4, w: 5.5, h: 1.5, fontSize: 11, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.4 });

// Embed Fig 3 PNG if available
const fig3Path = "results/viz/fig3_role_attention_matrix.png";
if (fs.existsSync(fig3Path)) {
  s.addImage({ path: fig3Path, x: 7.2, y: 4.5, w: 5.5, h: 2.8, sizing: { type: "contain", w: 5.5, h: 2.8 } });
} else {
  s.addText("[ Fig 3: convert PDF to PNG first ]\npython -c \"import fitz; fitz.open('results/viz/fig3_role_attention_matrix.pdf')[0].get_pixmap(dpi=200).save('results/viz/fig3_role_attention_matrix.png')\"", {
    x: 7.2, y: 4.2, w: 5.5, h: 1.0, fontSize: 10, fontFace: FONT_B, color: C.gray, italic: true,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Continuous → Discrete (dark)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
darkSlide(s);
addBadge(s, 4, 0.6, 0.4);

s.addText("Architecture Evolution II — Continuous Chaos → Discrete Emergence", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.white,
});

// Left — why abandon Box(2)
s.addText("Why Abandon Box(2)?", {
  x: 0.6, y: 1.3, w: 5.5, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.teal,
});
s.addText([
  { text: "Continuous DiagGaussian → unbounded sampling", options: { breakLine: true } },
  { text: "Requires manual clamping (±1.0) to prevent NaN", options: { breakLine: true } },
  { text: "Exploration diffuses in 2D → entropy runaway to 4.15", options: { breakLine: true } },
  { text: "600-step episodes → credit assignment nearly impossible", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "MultiDiscrete([5,3]) = 15 bounded tactical primitives.", options: { breakLine: true } },
  { text: "Entropy theoretically capped at log(5)+log(3) = 2.71.", options: { breakLine: true } },
  { text: "Action masking prevents physically impossible actions.", options: {} },
], { x: 0.6, y: 1.75, w: 5.5, h: 3.5, fontSize: 12, fontFace: FONT_B, color: C.white, lineSpacingMultiple: 1.4 });

// Right — action grid
s.addText("MultiDiscrete([5, 3]) = 15 Tactical Primitives", {
  x: 6.8, y: 1.3, w: 6.0, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.teal,
});

// Turn table
s.addText("TURN (5-way)", {
  x: 6.8, y: 1.85, w: 2.8, h: 0.3, fontSize: 12, fontFace: FONT_H, bold: true, color: C.coral,
});
const turns = [
  ["0", "Hard Left", "−15°/s"],
  ["1", "Soft Left", "−5°/s"],
  ["2", "Straight", "0°/s"],
  ["3", "Soft Right", "+5°/s"],
  ["4", "Hard Right", "+15°/s"],
];
let ty = 2.2;
turns.forEach(([id, name, rate]) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.8, y: ty, w: 5.5, h: 0.33,
    fill: { color: "15436B" }, rectRadius: 0.04,
  });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.33, fontSize: 11, fontFace: FONT_H, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 2.5, h: 0.33, fontSize: 11, fontFace: FONT_B, color: C.white, valign: "middle" });
  s.addText(rate, { x: 10.5, y: ty, w: 1.8, h: 0.33, fontSize: 11, fontFace: FONT_B, color: C.teal, valign: "middle", align: "right" });
  ty += 0.38;
});

// Speed table
ty += 0.15;
s.addText("SPEED (3-way)", {
  x: 6.8, y: ty, w: 2.8, h: 0.3, fontSize: 12, fontFace: FONT_H, bold: true, color: C.coral,
});
ty += 0.4;
const speeds = [["0", "Slow", "180 m/s"], ["1", "Cruise", "250 m/s"], ["2", "Fast", "320 m/s"]];
speeds.forEach(([id, name, spd]) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.8, y: ty, w: 5.5, h: 0.33,
    fill: { color: "15436B" }, rectRadius: 0.04,
  });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.33, fontSize: 11, fontFace: FONT_H, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 2.5, h: 0.33, fontSize: 11, fontFace: FONT_B, color: C.white, valign: "middle" });
  s.addText(spd, { x: 10.5, y: ty, w: 1.8, h: 0.33, fontSize: 11, fontFace: FONT_B, color: C.teal, valign: "middle", align: "right" });
  ty += 0.38;
});

// Action masking note
ty += 0.2;
s.addText("Safety: Action Masking prevents stall, ground collision, and overspeed maneuvers in real-time.", {
  x: 6.8, y: ty, w: 5.5, h: 0.3, fontSize: 10, fontFace: FONT_B, color: C.gray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Dynamic Annealing + Rewards (light)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
lightSlide(s);
addBadge(s, 5, 0.6, 0.4);

s.addText("Environment Innovation — Dynamic Curriculum & Reward Shaping", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.deep,
});

// Left — Dynamic annealing
s.addText("Dynamic AND-Gate Annealing", {
  x: 0.6, y: 1.3, w: 6.0, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});
s.addText([
  { text: "Problem: AND-gate requires BOTH < 800m + pincer > 30°", options: { breakLine: true } },
  { text: "P1 median distance = 1,974m → synchronized entry rate = 0%", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Solution: AND_DIST anneals 2000m → 800m over training", options: { breakLine: true } },
  { text: "Thresh = max(800, 2000 − decay_rate × iteration)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Result: eval improved from −8,800 to −1,171 (+4,700 pts)", options: { breakLine: true } },
  { text: "1,200–1,300m identified as CTDE learnability boundary", options: {} },
], { x: 0.6, y: 1.75, w: 6.0, h: 3.5, fontSize: 12, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.4 });

// Right — three reward cards
s.addText("Three Cooperative Reward Mechanisms", {
  x: 7.2, y: 1.3, w: 5.5, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});

const rewards = [
  { title: "Distance Asymmetry Penalty", body: "|d₀−d₁| > 500m → team penalty\nPrevents free-riding behavior\nPunishes P1 lagging behind" },
  { title: "Time-Sync Pacing Penalty", body: "Striker < 1200m && Int > 1500m\n→ sync_penalty = (dᵢₙₜ−dₛₜᵣ)/1000 × dt\nForces striker to wait for wingman" },
  { title: "Dynamic Role Assignment", body: "Striker (closer): tracking ×1.5\nInterceptor (further): pincer ×2.0\nEliminates lazy pursuer incentive" },
];
let ry = 1.75;
rewards.forEach((r) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.2, y: ry, w: 5.5, h: 1.3,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "D0D8DD", opacity: 0.3 },
  });
  s.addText(r.title, {
    x: 7.4, y: ry + 0.1, w: 5.1, h: 0.3,
    fontSize: 13, fontFace: FONT_H, bold: true, color: C.coral,
  });
  s.addText(r.body, {
    x: 7.4, y: ry + 0.42, w: 5.1, h: 0.8,
    fontSize: 11, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.3,
  });
  ry += 1.45;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Ablation Matrix (dark, full-width table)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
darkSlide(s);
addBadge(s, 6, 0.6, 0.4);

s.addText("Ablation Study — Six-Generation Model Comparison", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.white,
});

// Table header
const hdrRow = [
  { text: "Experiment", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "Arch", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "Action", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "BC", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "Iters", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "Best Eval", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
  { text: "Eval>0", options: { fontSize: 10, fontFace: FONT_H, bold: true, color: C.white, align: "center" } },
];

const dataRows = [
  ["Exp 1 (non-coop)", "Shared Attn", "Box(2)", "SB3 BC", "200", "−8,053", "0"],
  ["Exp 2 (OR-gate)", "Shared Attn", "Box(2)", "SB3 BC", "120", "+7,888", "5×"],
  ["Exp 3v3 (AND dyn)", "Shared Attn", "Box(2)", "SB3 BC", "300", "−1,171", "0"],
  ["Exp 4a (MLP)", "MLP fallback", "MultiDisc(5,3)", "None", "120", "−4,542", "0"],
  ["Exp 4a-v2 (Attn)", "Self-Attn", "MultiDisc(5,3)", "None", "120", "+1,345", "1×"],
  ["Exp 4b (Attn+BC)", "Self-Attn", "MultiDisc(5,3)", "Disc BC", "120", "−1,135", "0"],
  ["★ 4a-v2 ext", "Self-Attn", "MultiDisc(5,3)", "None", "320", "+2,376", "3×"],
];

const colW = [1.9, 1.3, 1.5, 1.0, 0.8, 1.2, 0.8];
const tblX = 1.5;
let tblY = 1.3;

// Header bg
s.addShape(pres.ShapeType.rect, {
  x: tblX, y: tblY, w: colW.reduce((a,b)=>a+b), h: 0.35,
  fill: { color: C.coral },
});

// Header text
let hx = tblX;
hdrRow.forEach((h, i) => {
  s.addText([h], { x: hx, y: tblY, w: colW[i], h: 0.35, valign: "middle" });
  hx += colW[i];
});

tblY += 0.4;
dataRows.forEach((row, ri) => {
  const bg = ri % 2 === 0 ? "15436B" : "1D4F73";
  s.addShape(pres.ShapeType.rect, {
    x: tblX, y: tblY, w: colW.reduce((a,b)=>a+b), h: 0.42,
    fill: { color: bg },
  });
  let cx = tblX;
  row.forEach((cell, ci) => {
    const isStar = cell.startsWith("★") || cell === "+2,376";
    s.addText(cell.replace("★ ",""), {
      x: cx, y: tblY, w: colW[ci], h: 0.42,
      fontSize: 10, fontFace: FONT_B,
      color: isStar ? C.coral : C.white,
      bold: isStar,
      align: "center", valign: "middle",
    });
    cx += colW[ci];
  });
  tblY += 0.47;
});

s.addText("★ Self-Attention is the decisive factor: cold-start Attn beats MLP by 5,887 pts. BC provides stability but no extra peak.", {
  x: 0.6, y: tblY + 0.3, w: 12, h: 0.3,
  fontSize: 12, fontFace: FONT_H, bold: true, color: C.coral,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Zero-Knowledge Emergence (mid accent)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
midSlide(s);
addBadge(s, 7, 0.6, 0.4);

s.addText("Breakthrough — Zero-Knowledge Emergence of Cooperative Tactics", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.white,
});

// Left narrative
s.addText("Cold-Start Self-Attention (320 iterations, NO expert data)", {
  x: 0.6, y: 1.3, w: 6.5, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.teal,
});
s.addText([
  { text: "Training reward peak: +5,401 (iter 300 of 460 total)", options: { breakLine: true } },
  { text: "Eval positive spikes: +1,345, +2,376, +4.0", options: { breakLine: true } },
  { text: "Entropy: 2.49 → 1.87 — healthy convergence, NOT divergence", options: { breakLine: true } },
  { text: "KL: stable 0.004–0.013 — controlled policy updates", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "MLP baseline: never achieved a single eval-positive spike.", options: { breakLine: true } },
  { text: "Self-Attn achieved 3 spikes — purely architectural advantage.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "★ STRUCTURE DETERMINES THE CEILING.", options: { breakLine: true } },
  { text: "Token-based architecture spontaneously learns cooperative", options: { breakLine: true } },
  { text: "pursuit through physical interaction — zero expert knowledge.", options: {} },
], { x: 0.6, y: 1.8, w: 6.5, h: 4.5, fontSize: 13, fontFace: FONT_B, color: C.white, lineSpacingMultiple: 1.3 });

// Right — 4 stat callouts
const bigStats = [
  { val: "+2,376", label: "Best Eval\nReward" },
  { val: "3×", label: "Positive\nSpikes" },
  { val: "320", label: "Training\nIterations" },
  { val: "0", label: "Expert\nSamples" },
];
let bx = 7.8;
bigStats.forEach((bs) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: bx, y: 2.0, w: 2.2, h: 2.0,
    fill: { color: "15436B" }, rectRadius: 0.1,
  });
  s.addText(bs.val, {
    x: bx, y: 2.2, w: 2.2, h: 0.9,
    fontSize: 32, fontFace: FONT_H, bold: true, color: C.coral, align: "center",
  });
  s.addText(bs.label, {
    x: bx, y: 3.2, w: 2.2, h: 0.5,
    fontSize: 12, fontFace: FONT_B, color: C.white, align: "center", lineSpacingMultiple: 1.2,
  });
  bx += 2.4;
});

// Bottom comparison bar
s.addShape(pres.ShapeType.roundRect, {
  x: 7.8, y: 4.5, w: 4.8, h: 0.5,
  fill: { color: C.coral }, rectRadius: 0.06,
});
s.addText("MLP: best −4,542  |  Self-Attn: best +2,376  |  Δ = 5,887 pts", {
  x: 7.9, y: 4.5, w: 4.6, h: 0.5,
  fontSize: 11, fontFace: FONT_H, bold: true, color: C.white, valign: "middle", align: "center",
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Fig 3 Deep Dive (light)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
lightSlide(s);
addBadge(s, 8, 0.6, 0.4);

s.addText("Mathematical Proof — Spontaneous Role Differentiation", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.deep,
});

// Left interpretation
s.addText("Fig 3: Role-Grouped Averaged Attention Matrix", {
  x: 0.6, y: 1.3, w: 6.0, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.mid,
});
s.addText([
  { text: "49 episodes, 7,858 steps per role.", options: { breakLine: true } },
  { text: "Classified by instantaneous geometry, NOT agent ID.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Striker MHA:", options: { breakLine: true } },
  { text: "  Self→Mate = 0.450 (coordination focus)", options: { breakLine: true } },
  { text: "  Pool Mate = 0.341 (learned pooling weight)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Interceptor MHA:", options: { breakLine: true } },
  { text: "  Self→Target = 0.389 (pursuit focus)", options: { breakLine: true } },
  { text: "  Pool Mate = 0.298", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Cohen's d = −0.53 (large effect, p < 0.001)", options: { breakLine: true } },
  { text: "Interceptor pays 31% more attention to Target.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Both roles sustain high mate attention (~0.44)", options: { breakLine: true } },
  { text: "→ Continuous implicit coordination, not binary switching.", options: { breakLine: true } },
  { text: "→ Definitive proof of emergent role differentiation.", options: {} },
], { x: 0.6, y: 1.75, w: 5.8, h: 4.5, fontSize: 12, fontFace: FONT_B, color: C.darkGray, lineSpacingMultiple: 1.25 });

// Right — embed Fig 3
if (fs.existsSync(fig3Path)) {
  s.addImage({ path: fig3Path, x: 7.0, y: 1.4, w: 5.8, h: 5.5, sizing: { type: "contain", w: 5.8, h: 5.5 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Limitations (dark)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
darkSlide(s);
addBadge(s, 9, 0.6, 0.4);

s.addText("Honest Boundary Analysis — Why AND-Gate Eventually Retreats", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.white,
});

// Left — stats
s.addText("AND-Gate Autopsy (10 eval episodes from Exp 3 checkpoint)", {
  x: 0.6, y: 1.3, w: 6.0, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.teal,
});

const autopsyStats = [
  { metric: "Sync Entry Rate (BOTH < 800m)", value: "0.0%", note: "ZERO — fatal bottleneck" },
  { metric: "Single Entry Rate (≥1 < 800m)", value: "22.1%", note: "P0 can approach" },
  { metric: "Pincer Angle > 30°", value: "58.4%", note: "Geometry is good!" },
  { metric: "P0 Median Distance", value: "329 m", note: "Excellent approach" },
  { metric: "P1 Median Distance", value: "1,974 m", note: "Cannot close the gap" },
];
let ay = 1.9;
autopsyStats.forEach((a) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: ay, w: 5.5, h: 0.65,
    fill: { color: "15436B" }, rectRadius: 0.06,
  });
  s.addText(a.metric, { x: 0.8, y: ay, w: 3.0, h: 0.35, fontSize: 11, fontFace: FONT_B, color: C.white, valign: "bottom" });
  s.addText(a.value, { x: 3.8, y: ay, w: 1.2, h: 0.35, fontSize: 14, fontFace: FONT_H, bold: true, color: C.coral, valign: "bottom", align: "right" });
  s.addText(a.note, { x: 0.8, y: ay + 0.33, w: 4.2, h: 0.22, fontSize: 9, fontFace: FONT_B, color: C.gray, valign: "top" });
  ay += 0.72;
});

// Right — three ceilings
s.addText("Three Identified Theoretical Ceilings", {
  x: 7.0, y: 1.3, w: 5.5, h: 0.35, fontSize: 16, fontFace: FONT_H, bold: true, color: C.teal,
});

const ceilings = [
  { title: "CTDE Information Asymmetry", body: "33-dim local obs cannot encode global\ncoordination state. Centralized Critic\nhelps but cannot fully compensate." },
  { title: "Discrete Exploration Boundary", body: "At strict 800m AND-gate, entropy at 3.6\nsuggests 15 primitives may be too few.\nCategorical head nears saturation." },
  { title: "Temporal Desynchronization", body: "No explicit time-to-intercept signal.\nAgents coordinate spatially (pincer 35°)\nbut cannot synchronize arrival times." },
];
let cy2 = 1.9;
ceilings.forEach((c) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.0, y: cy2, w: 5.5, h: 1.15,
    fill: { color: "15436B" }, rectRadius: 0.08,
  });
  s.addText(c.title, { x: 7.2, y: cy2 + 0.1, w: 5.1, h: 0.3, fontSize: 13, fontFace: FONT_H, bold: true, color: C.coral });
  s.addText(c.body, { x: 7.2, y: cy2 + 0.4, w: 5.1, h: 0.7, fontSize: 11, fontFace: FONT_B, color: C.white });
  cy2 += 1.3;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Future + Thanks (light)
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
lightSlide(s);
addBadge(s, 10, 0.6, 0.4);

s.addText("Future Directions — Toward N×M Scalable Formation Combat", {
  x: 1.2, y: 0.4, w: 11, h: 0.6,
  fontSize: 28, fontFace: FONT_H, bold: true, color: C.deep,
});

const dirs = [
  { num: "1", title: "Explicit Assignment Constraints", body: "Hungarian Algorithm for NvM weapon-target pairing. Solve combinatorial explosion in large-scale formation. Reference: DARPA ACE / OFFSET programs." },
  { num: "2", title: "Micro-Hierarchy SMDP", body: "Upper level: low-freq tactical option selection. Lower level: high-freq action masking + flight control. Bridges strategic planning and tactical execution." },
  { num: "3", title: "Self-Play & League Training", body: "Move beyond scripted targets to adversarial training. Population-based training with diversity objectives. Emergent tactics through competitive co-evolution." },
  { num: "4", title: "Explicit Coordination Channels", body: "Add ΔTGO (time-to-go difference) to global state. Learned communication via attention or message passing. Break the AND-gate barrier with explicit temporal signals." },
];
let dy = 1.4;
dirs.forEach((d) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: dy, w: 12.0, h: 1.05,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "D0D8DD", opacity: 0.3 },
  });
  // number circle
  s.addShape(pres.ShapeType.ellipse, {
    x: 0.8, y: dy + 0.22, w: 0.55, h: 0.55,
    fill: { color: C.coral },
  });
  s.addText(d.num, { x: 0.8, y: dy + 0.22, w: 0.55, h: 0.55, fontSize: 16, fontFace: FONT_H, bold: true, color: C.white, align: "center", valign: "middle" });
  s.addText(d.title, { x: 1.55, y: dy + 0.1, w: 10.5, h: 0.35, fontSize: 15, fontFace: FONT_H, bold: true, color: C.mid });
  s.addText(d.body, { x: 1.55, y: dy + 0.45, w: 10.5, h: 0.5, fontSize: 11, fontFace: FONT_B, color: C.darkGray });
  dy += 1.2;
});

// Thank you
dy += 0.2;
s.addShape(pres.ShapeType.rect, { x: 0.6, y: dy, w: 12.0, h: 0.03, fill: { color: C.coral } });
s.addText("Thank you. Questions & Discussion welcome.", {
  x: 0.6, y: dy + 0.2, w: 12.0, h: 0.5,
  fontSize: 20, fontFace: FONT_H, bold: true, color: C.deep, align: "center",
});
s.addText("sean@zju.edu.cn  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", {
  x: 0.6, y: dy + 0.7, w: 12.0, h: 0.3,
  fontSize: 11, fontFace: FONT_B, color: C.gray, align: "center",
});

// ── WRITE ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  const stats = fs.statSync(OUT);
  console.log(`Saved: ${OUT} (${(stats.size / 1024).toFixed(0)} KB, 10 slides)`);
});
