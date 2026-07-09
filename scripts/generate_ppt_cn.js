/**
 * Paper presentation: 基于多智能体强化学习的协同编队飞行
 * 按 .agents/skills/pptx/SKILL.md 规范生成。
 * 中文正文 + 英文术语，加大字号，减少留白。
 */

const pptxgen = require("/tmp/node_modules/pptxgenjs");
const fs = require("fs");

const C = {
  deep:    "065A82", teal:    "1C7293", mid:     "21295C",
  white:   "FFFFFF", light:   "EEF3F7",
  coral:   "F96167", green:   "2CC44D",
  gray:    "8899A6", darkGray:"4A5568", black:   "1A202C",
};
const FH = "Georgia";
const FB = "Calibri";
const OUT = "results/ppt/formation_coop_cn.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3 × 7.5
pres.author = "Sean Nishimiya";
pres.title = "多智能体强化学习协同编队飞行";

function dark(s) { s.background = { fill: C.deep }; }
function light(s) { s.background = { fill: C.light }; }
function mid(s) { s.background = { fill: C.mid }; }

function badge(s, n, x, y) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.5, h: 0.5, fill: { color: C.coral } });
  s.addText(String(n), { x, y, w: 0.5, h: 0.5, fontSize: 16, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 0 — 封面
// ═══════════════════════════════════════════════════════════════════════════════
let s = pres.addSlide();
dark(s);
s.addShape(pres.ShapeType.rect, { x: 0.8, y: 2.5, w: 2.2, h: 0.05, fill: { color: C.coral } });
s.addText("基于多智能体强化学习的\n协同编队飞行决策与规划", {
  x: 0.8, y: 1.2, w: 11.5, h: 1.6,
  fontSize: 42, fontFace: FH, bold: true, color: C.white, lineSpacingMultiple: 1.15,
});
s.addText("Token-Based CTDE with Self-Attention\n在 JSBSim 6-DOF F-16 编队追猎任务中超越集中式 PPO 上限", {
  x: 0.8, y: 3.0, w: 11.5, h: 0.9,
  fontSize: 18, fontFace: FB, color: C.teal, lineSpacingMultiple: 1.2,
});
s.addText("Sean Nishimiya  ·  Zhejiang University  ·  July 2026", {
  x: 0.8, y: 5.5, w: 11.5, h: 0.4,
  fontSize: 13, fontFace: FB, color: C.gray,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — 研究背景与架构
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
light(s);
badge(s, 1, 0.5, 0.3);
s.addText("研究背景与高保真仿真环境", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.deep,
});

// 左栏
s.addText("问题语境", {
  x: 0.5, y: 1.1, w: 6.0, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
s.addText([
  { text: "现代空战已演变为 N×M 体系对抗，高度依赖时空协同。", options: { breakLine: true } },
  { text: "2v1 编队追猎是验证多智能体协同的最小可行测试平台。", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 7 } },
  { text: "基础平台：JSBSim 6-DOF F-16 飞行动力学模型 (FDM)", options: { breakLine: true } },
  { text: "飞控层：FlightController @ 60 Hz PID 稳定控制", options: { breakLine: true } },
  { text: "训练框架：RLlib MAPPO + Ray 2.40 多智能体训练", options: { breakLine: true } },
  { text: "可视化：Tacview ACMI 导出 + TensorBoard + Matplotlib", options: {} },
], { x: 0.5, y: 1.55, w: 6.0, h: 3.5, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });

// 右栏：三层架构
s.addText("三层技术架构", {
  x: 7.0, y: 1.1, w: 5.8, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
const layers = [
  { t: "场景层", b: "FormationEnv → RLlib MultiAgentEnv\n2v1 协同追猎，NvM 可扩展\nOR-gate → AND-gate 课程学习\n规避机动：螺旋/李萨如/蛇形" },
  { t: "算法层", b: "参数共享 MAPPO (CTDE)\nSelf-Attention: 33-dim → 3 Token → MHA\nMultiDiscrete([5,3]) + Action Masking" },
  { t: "基础设施层", b: "JSBSim F-16 FDM → FlightController\nRLlib TorchModelV2 + shared_policy\nWSL2 + CUDA GPU 透传" },
];
let ly = 1.6;
layers.forEach((l) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.0, y: ly, w: 5.8, h: 1.4,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 3, offset: 1, color: "C0D0D8", opacity: 0.25 },
  });
  s.addText(l.t, { x: 7.2, y: ly + 0.08, w: 5.4, h: 0.3, fontSize: 13, fontFace: FH, bold: true, color: C.coral });
  s.addText(l.b, { x: 7.2, y: ly + 0.42, w: 5.4, h: 0.9, fontSize: 12, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });
  ly += 1.55;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — 死亡三角
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
dark(s);
badge(s, 2, 0.5, 0.3);
s.addText("核心痛点 — 协同的[死亡三角]", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.white,
});

const pains = [
  { t: "A. 非平稳性 Non-Stationarity", b: "IPPO 导致对称性内卷，两个独立 Critic\n使环境对每个 Agent 呈现非平稳。\n训练永久卡在 −7,500 奖励平台。", fix: "→ 参数共享 MAPPO（提升 4×）" },
  { t: "B. 时序信用分配灾难", b: "连续 Box(2) + 600 步长剧集 →\nDiagGaussian 方差崩溃。\n策略熵失控发散至 4.15。\n探索噪声完全淹没协同信号。", fix: "→ 离散动作，熵理论上限 2.71" },
  { t: "C. AND-Gate 维度盲区", b: "要求双机同时逼近 800m 并维持 30° 夹击。\nP1 中位距离 = 1,974m。\n双机同步入线率 = 0.0%。\n空间几何合格（夹击 35°）但时序脱节。", fix: "→ 动态退火课程 + 配速惩罚" },
];
let px = 0.4;
pains.forEach((p) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: px, y: 1.2, w: 4.05, h: 5.3,
    fill: { color: "15436B" }, rectRadius: 0.1,
  });
  s.addText(p.t, { x: px+0.2, y: 1.35, w: 3.7, h: 0.45, fontSize: 18, fontFace: FH, bold: true, color: C.coral });
  s.addText(p.b, { x: px+0.2, y: 2.0, w: 3.7, h: 3.0, fontSize: 14, fontFace: FB, color: C.white, lineSpacingMultiple: 1.45 });
  s.addShape(pres.ShapeType.roundRect, {
    x: px+0.2, y: 5.2, w: 3.7, h: 0.6,
    fill: { color: C.green }, rectRadius: 0.06,
  });
  s.addText(p.fix, { x: px+0.3, y: 5.2, w: 3.5, h: 0.6, fontSize: 12, fontFace: FH, bold: true, color: C.white, valign: "middle" });
  px += 4.3;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Self-Attention 架构
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
light(s);
badge(s, 3, 0.5, 0.3);
s.addText("架构演进 I — Token-Based Self-Attention 破除内卷", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.deep,
});

s.addText("从扁平 MLP 到语义 Token 分解", {
  x: 0.5, y: 1.1, w: 6.0, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
s.addText([
  { text: "观测 [33] 分解为 3 个语义 Token：", options: { breakLine: true } },
  { text: "  Self (13):   自身速度、姿态、角速度、高度、攻角、空速", options: { breakLine: true } },
  { text: "  Target (14): 目标相对位置/速度、战术几何角、LOS率", options: { breakLine: true } },
  { text: "  Mate (6):    友机相对位置/速度", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "Token-Type 嵌入 → MultiHeadAttention (4 heads, d=128)", options: { breakLine: true } },
  { text: "→ Learned Attention Pooling → MLP [256,256] → 动作输出", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "核心性质：置换不变性 (Permutation Invariance)。", options: { breakLine: true } },
  { text: "同一网络，不同观测 → 注意力权重自发分化角色。", options: {} },
], { x: 0.5, y: 1.55, w: 6.0, h: 4.2, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });

// 右栏：Fig 3 关键统计
s.addText("角色分组注意力矩阵 (Fig 3)", {
  x: 7.0, y: 1.1, w: 5.8, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});

const stats = [
  { v: "7,858", l: "每角色步数", sub: "49 集 × 2 角色" },
  { v: "−0.53", l: "Cohen's d", sub: "Self→Target (大效应量)" },
  { v: "0.44", l: "双向 Mate 关注", sub: "两角色均持续" },
  { v: "35.8°", l: "平均夹击角", sub: "落入黄金区间" },
];
let sx = 7.0;
stats.forEach((st) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: sx, y: 1.65, w: 2.7, h: 1.55,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "C0D0D8", opacity: 0.25 },
  });
  s.addText(st.v, { x: sx, y: 1.75, w: 2.7, h: 0.65, fontSize: 32, fontFace: FH, bold: true, color: C.coral, align: "center" });
  s.addText(st.l, { x: sx, y: 2.35, w: 2.7, h: 0.35, fontSize: 13, fontFace: FH, bold: true, color: C.darkGray, align: "center" });
  s.addText(st.sub, { x: sx, y: 2.65, w: 2.7, h: 0.3, fontSize: 10, fontFace: FB, color: C.gray, align: "center" });
  sx += 2.85;
});

s.addText([
  { text: "★ Striker MHA Self→Mate: 0.450（协调优先）。Interceptor Self→Target: 0.389（追击优先）。", options: { breakLine: true } },
  { text: "★ 双机均维持高互相关注 (~0.44) —— 持续性隐式协调，非二元切换。", options: { breakLine: true } },
  { text: "★ 参数共享网络自发涌现角色分化的数学铁证。", options: {} },
], { x: 7.0, y: 3.5, w: 5.8, h: 1.5, fontSize: 12, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.4 });

const fig3 = "results/viz/fig3_role_attention_matrix.png";
if (fs.existsSync(fig3)) {
  s.addImage({ path: fig3, x: 7.0, y: 4.8, w: 5.8, h: 2.4, sizing: { type: "contain", w: 5.8, h: 2.4 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — 连续→离散
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
dark(s);
badge(s, 4, 0.5, 0.3);
s.addText("架构演进 II — 从连续混沌到离散涌现的[降维打击]", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.white,
});

s.addText("为何抛弃 Box(2) 连续空间？", {
  x: 0.5, y: 1.1, w: 5.8, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.teal,
});
s.addText([
  { text: "连续 DiagGaussian → 采样无界，需手动 clamp (±1.0)", options: { breakLine: true } },
  { text: "探索在 2D 流形上扩散 → 熵失控至 4.15", options: { breakLine: true } },
  { text: "600 步长剧集 → 信用分配几乎不可能", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "MultiDiscrete([5,3]) = 15 个有界战术基元。", options: { breakLine: true } },
  { text: "熵理论上限 log(5)+log(3) = 2.71。", options: { breakLine: true } },
  { text: "Action Masking 实时剪枝不可行动作。", options: {} },
], { x: 0.5, y: 1.55, w: 5.8, h: 3.5, fontSize: 14, fontFace: FB, color: C.white, lineSpacingMultiple: 1.4 });

s.addText("MultiDiscrete([5, 3]) = 15 战术基元", {
  x: 6.8, y: 1.1, w: 6.0, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.teal,
});

s.addText("TURN 航向 (5档)", { x: 6.8, y: 1.65, w: 2.8, h: 0.3, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
const turns = [["0","急左转 −15°/s"],["1","缓左转 −5°/s"],["2","直飞 0°/s"],["3","缓右转 +5°/s"],["4","急右转 +15°/s"]];
let ty = 2.0;
turns.forEach(([id, name]) => {
  s.addShape(pres.ShapeType.roundRect, { x: 6.8, y: ty, w: 5.8, h: 0.38, fill: { color: "15436B" }, rectRadius: 0.04 });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.38, fontSize: 13, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 5.2, h: 0.38, fontSize: 13, fontFace: FB, color: C.white, valign: "middle" });
  ty += 0.44;
});

ty += 0.1;
s.addText("SPEED 速度 (3档)", { x: 6.8, y: ty, w: 2.8, h: 0.3, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
ty += 0.4;
const speeds = [["0","慢速 180 m/s (节能)"],["1","巡航 250 m/s (均衡)"],["2","快速 320 m/s (加力追击)"]];
speeds.forEach(([id, name]) => {
  s.addShape(pres.ShapeType.roundRect, { x: 6.8, y: ty, w: 5.8, h: 0.38, fill: { color: "15436B" }, rectRadius: 0.04 });
  s.addText(id, { x: 6.9, y: ty, w: 0.3, h: 0.38, fontSize: 13, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(name, { x: 7.3, y: ty, w: 5.2, h: 0.38, fontSize: 13, fontFace: FB, color: C.white, valign: "middle" });
  ty += 0.44;
});

ty += 0.15;
s.addText("安全防护：Action Masking 实时拦截失速、撞地、超速等危险动作", {
  x: 6.8, y: ty, w: 5.8, h: 0.35, fontSize: 12, fontFace: FB, color: C.gray, italic: true,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — 动态退火 + 奖励设计
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
light(s);
badge(s, 5, 0.5, 0.3);
s.addText("环境侧创新 — 动态退火课程与非对称奖励塑造", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.deep,
});

s.addText("动态 AND-Gate 距离退火", {
  x: 0.5, y: 1.1, w: 6.2, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
s.addText([
  { text: "问题：AND-gate 要求 BOTH < 800m + pincer > 30°", options: { breakLine: true } },
  { text: "P1 中位距离 = 1,974m → 同步入线率 = 0%", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "方案：AND_DIST 从 2000m 线性退火至 800m", options: { breakLine: true } },
  { text: "Thresh = max(800, 2000 − decay_rate × iteration)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "结果：Eval 从 −8,800 改善至 −1,171 (+4,700 分)", options: { breakLine: true } },
  { text: "1,200–1,300m 被识别为 CTDE 可学习边界", options: {} },
], { x: 0.5, y: 1.55, w: 6.2, h: 3.5, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.4 });

s.addText("三类协同奖励机制", {
  x: 7.2, y: 1.1, w: 5.6, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
const rewards = [
  { t: "距离不对称惩罚", b: "|d₀−d₁| > 500m → 团队惩罚\n阻止搭便车效应\n逼迫 P1 保持跟进" },
  { t: "时间同步配速惩罚", b: "Striker < 1200m && Int > 1500m\n→ penalty = (dᵢₙₜ−dₛₜᵣ)/1000 × dt\n强制长机等待僚机同步" },
  { t: "动态角色分配", b: "Striker (近距离): 追击奖励 ×1.5\nInterceptor (远距离): 夹击奖励 ×2.0\n消除懒惰 Agent 动机" },
];
let ry = 1.6;
rewards.forEach((r) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.2, y: ry, w: 5.6, h: 1.45,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "C0D0D8", opacity: 0.25 },
  });
  s.addText(r.t, { x: 7.4, y: ry + 0.1, w: 5.2, h: 0.3, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
  s.addText(r.b, { x: 7.4, y: ry + 0.45, w: 5.2, h: 0.9, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });
  ry += 1.6;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — 消融实验矩阵
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
dark(s);
badge(s, 6, 0.5, 0.3);
s.addText("五代模型消融大盘点 — The Definitive Ablation", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.white,
});

const hdr = ["实验", "架构", "动作空间", "BC", "轮数", "最佳Eval", "Eval>0"];
const colW = [2.2, 1.6, 1.8, 1.2, 0.9, 1.4, 0.9];
const rows = [
  ["Exp 1 (非协同)", "Shared Attn", "Box(2)", "SB3 BC", "200", "−8,053", "0"],
  ["Exp 2 (OR-gate)", "Shared Attn", "Box(2)", "SB3 BC", "120", "+7,888", "5×"],
  ["Exp 3v3 (AND退火)", "Shared Attn", "Box(2)", "SB3 BC", "300", "−1,171", "0"],
  ["Exp 4a (MLP离散)", "MLP 降级", "MultiDisc(5,3)", "无", "120", "−4,542", "0"],
  ["Exp 4a-v2 (Attn)", "Self-Attn", "MultiDisc(5,3)", "无", "120", "+1,345", "1×"],
  ["Exp 4b (Attn+BC)", "Self-Attn", "MultiDisc(5,3)", "离散BC", "120", "−1,135", "0"],
  ["★ 4a-v2 续训", "Self-Attn", "MultiDisc(5,3)", "无", "320", "+2,376", "3×"],
];

const tblX = 1.1;
let tblY = 1.2;
const tblW = colW.reduce((a,b)=>a+b);

s.addShape(pres.ShapeType.rect, { x: tblX, y: tblY, w: tblW, h: 0.4, fill: { color: C.coral } });
let hx = tblX;
hdr.forEach((h, i) => {
  s.addText(h, { x: hx, y: tblY, w: colW[i], h: 0.4, fontSize: 11, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle" });
  hx += colW[i];
});
tblY += 0.45;

rows.forEach((row, ri) => {
  const bg = ri % 2 === 0 ? "15436B" : "1D4F73";
  s.addShape(pres.ShapeType.rect, { x: tblX, y: tblY, w: tblW, h: 0.5, fill: { color: bg } });
  let cx = tblX;
  row.forEach((cell, ci) => {
    const hi = cell.startsWith("★") || cell === "+2,376";
    s.addText(cell.replace("★ ",""), {
      x: cx, y: tblY, w: colW[ci], h: 0.5,
      fontSize: 11, fontFace: FB,
      color: hi ? C.coral : C.white,
      bold: hi, align: "center", valign: "middle",
    });
    cx += colW[ci];
  });
  tblY += 0.55;
});

s.addText("★ Self-Attention 是决定性因素：冷启动 Attn 超越 MLP 达 5,887 分。离散 BC 提供稳定性但无额外峰值。结构决定上限。", {
  x: 0.5, y: tblY + 0.15, w: 12.3, h: 0.35,
  fontSize: 13, fontFace: FH, bold: true, color: C.coral,
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — 从零涌现
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
mid(s);
badge(s, 7, 0.5, 0.3);
s.addText("黄金成果 — 从零涌现的协同战术 (Zero-Knowledge Emergence)", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.white,
});

s.addText("冷启动 Self-Attention 320 轮训练 — 零专家数据，零 BC 预训练", {
  x: 0.5, y: 1.1, w: 7.0, h: 0.35,
  fontSize: 17, fontFace: FH, bold: true, color: C.teal,
});
s.addText([
  { text: "训练峰值：+5,401（总第 300 轮）", options: { breakLine: true } },
  { text: "Eval 正向突破：+1,345, +2,376, +4.0 三次转正", options: { breakLine: true } },
  { text: "策略熵：2.49 → 1.87（健康收敛，非发散）", options: { breakLine: true } },
  { text: "KL 散度：0.004–0.013（策略更新平滑可控）", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 7 } },
  { text: "纯 MLP 基线：0 次正向突破，最佳 −4,542。", options: { breakLine: true } },
  { text: "Self-Attention：3 次正向突破，最佳 +2,376。", options: { breakLine: true } },
  { text: "差距 5,887 分 —— 纯架构优势。", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 7 } },
  { text: "★ 结构决定上限。Token 架构仅凭物理交互，无需任何", options: { breakLine: true } },
  { text: "  专家知识，即可自发涌现高级协同追猎行为。", options: {} },
], { x: 0.5, y: 1.6, w: 7.0, h: 5.0, fontSize: 14, fontFace: FB, color: C.white, lineSpacingMultiple: 1.3 });

const bstats = [
  { v: "+2,376", l: "最佳 Eval" },
  { v: "3×", l: "正向突破" },
  { v: "320", l: "训练轮数" },
  { v: "0", l: "专家样本" },
];
let bx = 8.2;
bstats.forEach((bs) => {
  s.addShape(pres.ShapeType.roundRect, { x: bx, y: 1.8, w: 2.2, h: 2.4, fill: { color: "15436B" }, rectRadius: 0.1 });
  s.addText(bs.v, { x: bx, y: 2.0, w: 2.2, h: 1.0, fontSize: 34, fontFace: FH, bold: true, color: C.coral, align: "center" });
  s.addText(bs.l, { x: bx, y: 3.1, w: 2.2, h: 0.5, fontSize: 14, fontFace: FB, color: C.white, align: "center" });
  bx += 2.4;
});

s.addShape(pres.ShapeType.roundRect, { x: 8.2, y: 4.8, w: 4.6, h: 0.6, fill: { color: C.coral }, rectRadius: 0.06 });
s.addText("MLP: −4,542   vs   Self-Attn: +2,376   (Δ = 5,887)", {
  x: 8.3, y: 4.8, w: 4.4, h: 0.6, fontSize: 13, fontFace: FH, bold: true, color: C.white, valign: "middle", align: "center",
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Fig 3 深度解读
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
light(s);
badge(s, 8, 0.5, 0.3);
s.addText("数学铁证 — 参数共享网络的自发角色分化", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.deep,
});

s.addText("Fig 3: 角色分组均值注意力矩阵 (49集, 7,858步/角色)", {
  x: 0.5, y: 1.1, w: 6.2, h: 0.35,
  fontSize: 18, fontFace: FH, bold: true, color: C.mid,
});
s.addText([
  { text: "按瞬时几何分角色，不按 Agent ID 分 —— 位置无关验证。", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Striker (主攻手) MHA:", options: { breakLine: true } },
  { text: "  Self→Mate = 0.450（关注队友，维持包抄）", options: { breakLine: true } },
  { text: "  Pool Mate  = 0.341（Learned Pooling 权重）", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Interceptor (封锁手) MHA:", options: { breakLine: true } },
  { text: "  Self→Target = 0.389（专注追击，缩小距离）", options: { breakLine: true } },
  { text: "  Pool Mate = 0.298", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Cohen's d = −0.53 (大效应量, p < 0.001)", options: { breakLine: true } },
  { text: "Interceptor 对 Target 的关注比 Striker 高 31%。", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "双机 MHA Self→Mate 均维持 ~0.44 高水平。", options: { breakLine: true } },
  { text: "→ 持续性隐式协调，非二元切换。", options: { breakLine: true } },
  { text: "→ 参数共享 Self-Attention 自发打破对称性的终极证据。", options: {} },
], { x: 0.5, y: 1.55, w: 6.0, h: 5.0, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.25 });

if (fs.existsSync(fig3)) {
  s.addImage({ path: fig3, x: 7.0, y: 1.3, w: 5.8, h: 5.8, sizing: { type: "contain", w: 5.8, h: 5.8 } });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — 边界分析
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
dark(s);
badge(s, 9, 0.5, 0.3);
s.addText("诚实边界分析 — AND-Gate 为何最终回落", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.white,
});

s.addText("AND-Gate 死亡解剖 (Exp 3 checkpoint, 10 集 Eval)", {
  x: 0.5, y: 1.1, w: 6.5, h: 0.35,
  fontSize: 17, fontFace: FH, bold: true, color: C.teal,
});

const autopsy = [
  { m: "双机同步入线率 (BOTH < 800m)", v: "0.0%", n: "致命瓶颈" },
  { m: "单机入线率 (≥1 < 800m)", v: "22.1%", n: "P0 可以接近" },
  { m: "夹击角 > 30°", v: "58.4%", n: "空间几何合格！" },
  { m: "P0 中位距离", v: "329 m", n: "逼近出色" },
  { m: "P1 中位距离", v: "1,974 m", n: "无法缩小差距" },
];
let ay = 1.65;
autopsy.forEach((a) => {
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: ay, w: 5.8, h: 0.72, fill: { color: "15436B" }, rectRadius: 0.06 });
  s.addText(a.m, { x: 0.7, y: ay, w: 3.3, h: 0.4, fontSize: 12, fontFace: FB, color: C.white, valign: "bottom" });
  s.addText(a.v, { x: 4.2, y: ay, w: 1.3, h: 0.4, fontSize: 16, fontFace: FH, bold: true, color: C.coral, valign: "bottom", align: "right" });
  s.addText(a.n, { x: 0.7, y: ay + 0.38, w: 4.8, h: 0.24, fontSize: 10, fontFace: FB, color: C.gray, valign: "top" });
  ay += 0.8;
});

s.addText("三大理论天花板", {
  x: 7.0, y: 1.1, w: 5.8, h: 0.35,
  fontSize: 17, fontFace: FH, bold: true, color: C.teal,
});
const ceilings = [
  { t: "CTDE 信息不对称", b: "33-dim 局部观测无法编码全局协调状态。\nCentralized Critic 可缓解但无法完全补偿。\nAgent 无法从局部观测推断\"我比队友超前还是落后\"。" },
  { t: "离散探索边界", b: "在严格 800m AND-gate 下熵升至 3.6，\n提示 15 个基元可能不足。\nCategorical 头接近饱和极限。" },
  { t: "时序同步缺失", b: "缺乏显式 TGO 预期到达时间差信号。\nAgent 在空间上协调良好 (夹击 35°)，\n但无法同步到达时间。" },
];
let cy = 1.65;
ceilings.forEach((c) => {
  s.addShape(pres.ShapeType.roundRect, { x: 7.0, y: cy, w: 5.8, h: 1.35, fill: { color: "15436B" }, rectRadius: 0.08 });
  s.addText(c.t, { x: 7.2, y: cy + 0.1, w: 5.4, h: 0.3, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
  s.addText(c.b, { x: 7.2, y: cy + 0.42, w: 5.4, h: 0.85, fontSize: 12, fontFace: FB, color: C.white, lineSpacingMultiple: 1.3 });
  cy += 1.45;
});

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — 未来 + 致谢
// ═══════════════════════════════════════════════════════════════════════════════
s = pres.addSlide();
light(s);
badge(s, 10, 0.5, 0.3);
s.addText("未来探索方向 — 迈向 N×M 规模化编队对抗", {
  x: 1.15, y: 0.3, w: 11, h: 0.55,
  fontSize: 30, fontFace: FH, bold: true, color: C.deep,
});

const dirs = [
  { n: "1", t: "显式运筹约束 (Hungarian Algorithm)", b: "引入匈牙利算法解决 NvM 大规模编队的火力-目标分配组合爆炸。参考 DARPA ACE / OFFSET 项目框架。" },
  { n: "2", t: "微层级半马尔可夫 SMDP", b: "上层：低频战术选项选择。下层：高频 Action Masking + 飞控执行。桥接战略规划与战术执行。" },
  { n: "3", t: "Self-Play 自我对局与联赛训练", b: "超越脚本化目标，进入对抗性训练。基于种群多样性的 League Training。通过竞争性共同进化涌现高阶战术。" },
  { n: "4", t: "显式协同通信信道", b: "在全局状态中引入 ΔTGO（预期到达时间差）。通过 Attention 或 Message Passing 学习通信。用显式时序信号突破 AND-Gate。" },
];
let dy = 1.2;
dirs.forEach((d) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: dy, w: 12.3, h: 1.15,
    fill: { color: C.white }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 2, offset: 1, color: "C0D0D8", opacity: 0.25 },
  });
  s.addShape(pres.ShapeType.ellipse, { x: 0.7, y: dy + 0.25, w: 0.6, h: 0.6, fill: { color: C.coral } });
  s.addText(d.n, { x: 0.7, y: dy + 0.25, w: 0.6, h: 0.6, fontSize: 18, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle" });
  s.addText(d.t, { x: 1.55, y: dy + 0.1, w: 11.0, h: 0.35, fontSize: 16, fontFace: FH, bold: true, color: C.mid });
  s.addText(d.b, { x: 1.55, y: dy + 0.48, w: 11.0, h: 0.55, fontSize: 12, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.25 });
  dy += 1.3;
});

dy += 0.25;
s.addShape(pres.ShapeType.rect, { x: 0.5, y: dy, w: 12.3, h: 0.03, fill: { color: C.coral } });
s.addText("感谢聆听 · 欢迎提问与讨论", {
  x: 0.5, y: dy + 0.15, w: 12.3, h: 0.5,
  fontSize: 22, fontFace: FH, bold: true, color: C.deep, align: "center",
});
s.addText("sean@zju.edu.cn  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", {
  x: 0.5, y: dy + 0.65, w: 12.3, h: 0.3,
  fontSize: 12, fontFace: FB, color: C.gray, align: "center",
});

// ── WRITE ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  const st = fs.statSync(OUT);
  console.log(`Saved: ${OUT} (${(st.size/1024).toFixed(0)} KB, 10 slides)`);
});
