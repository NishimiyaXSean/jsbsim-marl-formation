/**
 * Final paper PPT — Chinese + English terms, large fonts, compact layout.
 * All white backgrounds. Ocean Gradient palette.
 * pptxgenjs just writes font names into XML — Chinese renders fine on viewer's PowerPoint.
 */

const pptxgen = require("/tmp/node_modules/pptxgenjs");
const fs = require("fs");

const C = { navy: "065A82", teal: "1C7293", coral: "F96167", white: "FFFFFF",
  light: "F2F7F9", green: "2CC44D", gray: "8899A6", darkGray: "4A5568",
  accent: "E8F0F4", black: "1A202C" };
const FH = "Georgia"; const FB = "Calibri";
const CH = "results/viz/paper_charts"; const F3 = "results/viz/fig3_role_attention_matrix.png";
const OUT = "results/ppt/formation_coop_final_v4.pptx";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; pres.author = "Sean Nishimiya";
pres.title = "基于MARL的协同编队飞行决策";

// helpers
function titleBar(s, num, text) {
  s.background = { fill: C.white };
  s.addShape(pres.ShapeType.rect, { x: 0.4, y: 0.25, w: 0.06, h: 0.5, fill: { color: C.coral } });
  s.addText(String(num), { x: 0.4, y: 0.2, w: 0.8, h: 0.55, fontSize: 26, fontFace: FH, bold: true, color: C.coral, margin: 0 });
  s.addText(text, { x: 1.15, y: 0.2, w: 11.5, h: 0.55, fontSize: 30, fontFace: FH, bold: true, color: C.navy });
  s.addShape(pres.ShapeType.rect, { x: 0.4, y: 0.9, w: 12.5, h: 0.012, fill: { color: C.accent } });
}
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: o.fill || C.white },
    rectRadius: 0.06, line: o.line ? { color: C.accent, width: 0.5 } : undefined,
    shadow: o.shadow ? { type: "outer", blur: 3, offset: 1, color: "C0D0D8", opacity: 0.18 } : undefined });
}
function statBox(s, x, y, w, h, val, label) {
  card(s, x, y, w, h, { shadow: true });
  s.addText(val, { x, y: y + 0.08, w, h: 0.6, fontSize: 30, fontFace: FH, bold: true, color: C.coral, align: "center" });
  s.addText(label, { x, y: y + 0.7, w, h: 0.35, fontSize: 13, fontFace: FB, bold: true, color: C.darkGray, align: "center", lineSpacingMultiple: 1.1 });
}
function img(s, path, x, y, w, h) {
  if (fs.existsSync(path)) s.addImage({ path, x, y, w, h, sizing: { type: "contain", w, h } });
}

// ═══════════════════════════ P1: Cover ═══════════════════════════
let s = pres.addSlide();
s.background = { fill: C.white };
s.addShape(pres.ShapeType.rect, { x: 0.8, y: 2.2, w: 2.0, h: 0.05, fill: { color: C.coral } });
s.addText("基于多智能体强化学习的\n协同编队飞行决策与规划", {
  x: 0.8, y: 1.0, w: 11.5, h: 1.6,
  fontSize: 42, fontFace: FH, bold: true, color: C.navy, lineSpacingMultiple: 1.1,
});
s.addText("Token-Based CTDE with Self-Attention + Discrete Primitives\n在 JSBSim 6-DOF F-16 编队追猎中超越集中式 PPO 上限", {
  x: 0.8, y: 2.7, w: 11.5, h: 0.8, fontSize: 17, fontFace: FB, color: C.teal, lineSpacingMultiple: 1.2,
});
s.addText("Sean Nishimiya  ·  Zhejiang University  ·  July 2026  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", {
  x: 0.8, y: 5.6, w: 11.5, h: 0.35, fontSize: 11, fontFace: FB, color: C.gray,
});

// ═══════════════════════════ P2: Pain Points ═══════════════════════════
s = pres.addSlide(); titleBar(s, 1, "核心挑战 — 协同的三大死锁困境");
const pains = [
  { t: "A. 非平稳性 Non-Stationarity", b: "IPPO 导致对称性内卷。两个独立 Critic\n使环境对每个 Agent 呈现非平稳。\n训练永久卡在 -7,500 奖励平台。", fix: "参数共享 MAPPO (提升 4x)" },
  { t: "B. 时序信用分配灾难", b: "连续 Box(2) + 600 步剧集\n→ DiagGaussian 方差崩溃。\n策略熵失控发散至 4.15。\n探索噪声淹没协同信号。", fix: "离散动作: 熵理论上限 2.71" },
  { t: "C. AND-Gate 时空盲区", b: "要求双机同时逼近 800m + 夹击 >30°。\nP1 中位距离 = 1,974m。\n双机同步入线率 = 0.0%。\n空间几何 OK (夹击 35°) 但时序脱节。", fix: "动态退火 + 配速惩罚" },
];
let px3 = 0.4;
pains.forEach(p => {
  card(s, px3, 1.15, 4.1, 5.5, { fill: C.light });
  s.addText(p.t, { x: px3+0.2, y: 1.3, w: 3.7, h: 0.4, fontSize: 17, fontFace: FH, bold: true, color: C.coral });
  s.addText(p.b, { x: px3+0.2, y: 1.9, w: 3.7, h: 3.0, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.45 });
  s.addShape(pres.ShapeType.roundRect, { x: px3+0.2, y: 5.15, w: 3.7, h: 0.6, fill: { color: C.green }, rectRadius: 0.05 });
  s.addText(p.fix, { x: px3+0.3, y: 5.15, w: 3.5, h: 0.6, fontSize: 12, fontFace: FH, bold: true, color: C.white, valign: "middle", align: "center" });
  px3 += 4.35;
});

// ═══════════════════════════ P3: Self-Attention Architecture ═══════════════════════════
s = pres.addSlide(); titleBar(s, 2, "架构重构 A: 参数共享 Self-Attention CTDE");
s.addText("Token-Based Multi-Head Self-Attention 网络架构", { x: 0.4, y: 1.15, w: 6.0, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText([
  { text: "观测 [33] 分解为 3 个语义 Token:", options: { breakLine: true } },
  { text: "  Self (13):    自身速度、姿态、角速度、高度、攻角、空速", options: { breakLine: true } },
  { text: "  Target (14):  目标相对位置/速度、战术几何角、LOS 率", options: { breakLine: true } },
  { text: "  Mate (6):     友机相对位置/速度", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Token-Type 嵌入 → MultiHeadAttention (4 heads, d=128)", options: { breakLine: true } },
  { text: "→ Learned Attention Pooling → MLP [256,256] → 动作输出", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "共享策略 (shared_policy) → 置换不变性。同一网络，不同观测 → 注意力自发分化角色。", options: {} },
], { x: 0.4, y: 1.6, w: 6.0, h: 3.5, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });

s.addText("角色分组注意力矩阵 (Fig 3, 7,858 步/角色)", { x: 6.8, y: 1.15, w: 6.0, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
const s3 = [{ v:"0.450",l:"Striker MHA\nSelf->Mate"},{ v:"0.389",l:"Interceptor MHA\nSelf->Target"},{ v:"-0.53",l:"Cohen's d\n(大效应量)"},{ v:"0.44",l:"双向 Mate\n持续关注"}];
let sx = 6.8; s3.forEach(st => { statBox(s, sx, 1.75, 2.7, 1.3, st.v, st.l); sx += 2.85; });
s.addText("双机 MHA Self->Mate 均 ~0.44 — 持续性隐式协调, 非二元切换 — 参数共享网络自发打破对称性的数学铁证。", { x: 6.8, y: 3.3, w: 6.0, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true });
img(s, F3, 6.8, 3.9, 6.0, 3.2);

// ═══════════════════════════ P4: Discrete + Masking ═══════════════════════════
s = pres.addSlide(); titleBar(s, 3, "架构重构 B: 离散战术基元 + Action Masking");
s.addText("为何抛弃 Box(2) 连续空间?", { x: 0.4, y: 1.15, w: 5.8, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText([
  { text: "连续 DiagGaussian → 采样无界, 需手动 clamp (±1.0)", options: { breakLine: true } },
  { text: "探索在 2D 流形扩散 → 熵失控至 4.15", options: { breakLine: true } },
  { text: "600 步剧集 → 信用分配几乎不可能", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "MultiDiscrete([5,3]) = 15 个有界战术基元", options: { breakLine: true } },
  { text: "熵理论上限 ln(5)+ln(3) = 2.71 — 硬性约束探索", options: { breakLine: true } },
  { text: "Action Masking: 失速禁慢速, 近地禁转向, 超速禁加力", options: {} },
], { x: 0.4, y: 1.6, w: 5.8, h: 3.2, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });

s.addText("MultiDiscrete([5, 3]) = 15 战术基元", { x: 6.8, y: 1.15, w: 6.0, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText("TURN (航向, 5 档)", { x: 6.8, y: 1.7, w: 3.0, h: 0.28, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
[["0","急左转 -15 deg/s"],["1","缓左转  -5 deg/s"],["2","直飞     0 deg/s"],["3","缓右转  +5 deg/s"],["4","急右转 +15 deg/s"]].forEach(([id,nm],i) => {
  card(s, 6.8, 2.05 + i*0.42, 5.8, 0.38, { fill: C.light });
  s.addText(id, { x: 6.9, y: 2.05+i*0.42, w: 0.3, h: 0.38, fontSize: 13, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(nm, { x: 7.3, y: 2.05+i*0.42, w: 5.2, h: 0.38, fontSize: 13, fontFace: FB, color: C.darkGray, valign: "middle" });
});
let ty = 2.05 + 5*0.42 + 0.2;
s.addText("SPEED (速度, 3 档)", { x: 6.8, y: ty, w: 3.0, h: 0.28, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
ty += 0.35;
[["0","慢速 180 m/s (节能巡逻)"],["1","巡航 250 m/s (均衡追击)"],["2","快速 320 m/s (加力冲刺)"]].forEach(([id,nm],i) => {
  card(s, 6.8, ty + i*0.42, 5.8, 0.38, { fill: C.light });
  s.addText(id, { x: 6.9, y: ty+i*0.42, w: 0.3, h: 0.38, fontSize: 13, fontFace: FH, bold: true, color: C.coral, valign: "middle" });
  s.addText(nm, { x: 7.3, y: ty+i*0.42, w: 5.2, h: 0.38, fontSize: 13, fontFace: FB, color: C.darkGray, valign: "middle" });
});

// ═══════════════════════════ P5: Chart 1 ═══════════════════════════
s = pres.addSlide(); titleBar(s, 4, "动作分布演变 — 从随机探索到确定性战术");
img(s, CH+"/chart1_action_distribution.png", 0.25, 1.15, 12.8, 6.0);
s.addText("早期 (0-50 轮): 动作接近均匀分布 (探索)。后期 (250-320 轮): 策略高度集中于最优基元 (确定性战术)。", { x: 0.4, y: 6.8, w: 12.5, h: 0.4, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true });

// ═══════════════════════════ P6: Curriculum ═══════════════════════════
s = pres.addSlide(); titleBar(s, 5, "课程设计 — 动态 AND-Gate 退火 + 配速惩罚");
s.addText("动态 AND-Gate 距离退火", { x: 0.4, y: 1.15, w: 6.0, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText([
  { text: "问题: AND-gate 要求 BOTH < 800m + pincer > 30°", options: { breakLine: true } },
  { text: "P1 中位距离 1,974m → 同步入线率 = 0%", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "方案: AND_DIST 从 2000m 线性退火至 800m:", options: { breakLine: true } },
  { text: "  Thresh = max(800, 2000 - decay_rate x iter)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "结果: Eval 从 -8,800 改善至 -1,171 (+4,700 分)", options: { breakLine: true } },
  { text: "1,200-1,300m 被识别为 CTDE 可学习边界", options: {} },
], { x: 0.4, y: 1.65, w: 6.0, h: 3.0, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });

s.addText("三类协同奖励机制", { x: 7.0, y: 1.15, w: 5.8, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
[{ t: "距离不对称惩罚", b: "|d0-d1| > 500m -> 团队受罚\n阻止搭便车行为, 逼迫 P1 跟进" },
 { t: "时间同步配速惩罚", b: "Striker < 1200m & Int > 1500m\n-> penalty = (d_int-d_str)/1000 x dt\n强制长机等僚机, 防止独狼冲锋" },
 { t: "动态角色分配", b: "Striker (近距离): 追击奖励 x1.5\nInterceptor (远距离): 夹击奖励 x2.0\n消除懒惰 Agent 动机" }].forEach((r,i) => {
  card(s, 7.0, 1.7 + i*1.5, 5.8, 1.35, { shadow: true });
  s.addText(r.t, { x: 7.2, y: 1.8+i*1.5, w: 5.4, h: 0.3, fontSize: 14, fontFace: FH, bold: true, color: C.coral });
  s.addText(r.b, { x: 7.2, y: 2.15+i*1.5, w: 5.4, h: 0.8, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });
});

// ═══════════════════════════ P7: Chart 3 ═══════════════════════════
s = pres.addSlide(); titleBar(s, 6, "奖励成分解耦 — 课程学习效果验证");
img(s, CH+"/chart3_reward_breakdown.png", 0.25, 1.15, 12.8, 6.0);
s.addText("Phase 1 (OR-gate): 奖励由 Progress + ATA 主导。Phase 2 (AND-gate): Pincer + AND Bonus 随课程收紧逐步扩张, 未出现 Reward Hacking。", { x: 0.4, y: 6.8, w: 12.5, h: 0.4, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true });

// ═══════════════════════════ P8: Ablation ═══════════════════════════
s = pres.addSlide(); titleBar(s, 7, "消融实验大盘点 — 六代模型全矩阵");
const hdr = ["实验", "架构", "动作空间", "BC", "轮数", "最佳 Eval", "Eval>0"];
const colW8 = [2.3, 1.7, 1.9, 1.2, 0.9, 1.4, 0.9];
const rows8 = [
  ["Exp 1 (非协同)", "Shared Attn", "Box(2)", "SB3 BC", "200", "-8,053", "0"],
  ["Exp 2 (OR-gate)", "Shared Attn", "Box(2)", "SB3 BC", "120", "+7,888", "5x"],
  ["Exp 3v3 (AND退火)", "Shared Attn", "Box(2)", "SB3 BC", "300", "-1,171", "0"],
  ["Exp 4a (MLP 降级)", "MLP fallback", "MultiDisc(5,3)", "无", "120", "-4,542", "0"],
  ["Exp 4a-v2 (Attn)", "Self-Attn", "MultiDisc(5,3)", "无", "120", "+1,345", "1x"],
  ["Exp 4b (Attn+BC)", "Self-Attn", "MultiDisc(5,3)", "离散BC", "120", "-1,135", "0"],
  ["* 4a-v2 续训320", "Self-Attn", "MultiDisc(5,3)", "无", "320", "+2,376", "3x"],
];
const tx = 0.5, tw = colW8.reduce((a,b)=>a+b); let ty8 = 1.25;
s.addShape(pres.ShapeType.rect, { x: tx, y: ty8, w: tw, h: 0.45, fill: { color: C.navy } });
let hx8 = tx; hdr.forEach((h,i) => { s.addText(h,{ x:hx8, y:ty8, w:colW8[i], h:0.45, fontSize:12, fontFace:FH, bold:true, color:C.white, align:"center", valign:"middle" }); hx8+=colW8[i]; });
ty8 += 0.5;
rows8.forEach((row,ri) => {
  s.addShape(pres.ShapeType.rect, { x: tx, y: ty8, w: tw, h: 0.58, fill: { color: ri%2===0?C.light:C.white }, line: { color: C.accent, width: 0.3 } });
  let cx8 = tx;
  row.forEach((cell,ci) => {
    const hi = cell.startsWith("*") || cell === "+2,376";
    s.addText(cell.replace("* ",""), { x: cx8, y: ty8, w: colW8[ci], h: 0.58, fontSize: 11, fontFace: FB, color: hi?C.coral:C.darkGray, bold: hi, align: "center", valign: "middle" });
    cx8 += colW8[ci];
  });
  ty8 += 0.63;
});
s.addText("* Self-Attention 是决定性因素: 冷启动 Attn 超越 MLP 达 5,887 分。BC 提供稳定性但无额外峰值。结构决定上限。", { x: 0.4, y: ty8+0.1, w: 12.5, h: 0.35, fontSize: 13, fontFace: FH, bold: true, color: C.coral });

// ═══════════════════════════ P9: Emergence ═══════════════════════════
s = pres.addSlide(); titleBar(s, 8, "黄金成果: 从零涌现的协同战术 (Zero-Knowledge Emergence)");
s.addText("冷启动 Self-Attention — 320 轮, 零专家数据, 零 BC", { x: 0.4, y: 1.15, w: 7.0, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText([
  { text: "训练峰值: +5,401 (第 300 轮)    |    策略熵: 2.49 -> 1.87 (健康收敛)", options: { breakLine: true } },
  { text: "+1,345,  +2,376,  +4.0 — 三次 Eval 正向突破!    |    KL: 0.004-0.013 (平稳可控)", options: { breakLine: true } },
  { text: "纯 MLP 基线: 0 次正向突破, 最佳 -4,542.    Self-Attn: 3 次突破, 差距 5,887 分.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "结构决定上限. Token 架构仅凭物理交互, 零专家知识, 自发涌现协同追猎行为.", options: {} },
], { x: 0.4, y: 1.65, w: 7.0, h: 2.5, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.3 });
[{ v:"+2,376",l:"最佳 Eval"}, { v:"3x",l:"正向突破"}, { v:"320",l:"训练轮数"}, { v:"0",l:"专家样本"}].forEach((bs,i) => statBox(s, 0.4+i*2.85, 4.2, 2.65, 1.35, bs.v, bs.l));
img(s, CH+"/chart2_spatial_kde.png", 7.6, 1.1, 5.5, 3.6);
s.addText("Spatial KDE: 50 集 Eval, 127K 帧. 目标尾部 30-60 deg 扇区高密度聚集 — 统计级合围几何收敛.", { x: 7.6, y: 4.8, w: 5.5, h: 0.5, fontSize: 11, fontFace: FB, color: C.gray, italic: true });
img(s, CH+"/chart5_health_metrics.png", 7.6, 5.15, 5.5, 2.1);

// ═══════════════════════════ P10: Fig 3 ═══════════════════════════
s = pres.addSlide(); titleBar(s, 9, "数学铁证: 参数共享网络的自发角色分化");
s.addText("角色分组均值注意力矩阵 (49 集, 7,858 步/角色, 按瞬时几何分角色)", { x: 0.4, y: 1.15, w: 6.5, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
s.addText([
  { text: "Striker MHA:  Self->Mate = 0.450 (协调)    Pool Mate = 0.341", options: { breakLine: true } },
  { text: "Interceptor MHA: Self->Target = 0.389 (追击)  Pool Mate = 0.298", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "Cohen's d = -0.53 (大效应量, p < 0.001)", options: { breakLine: true } },
  { text: "Interceptor 对 Target 的关注比 Striker 高 31%.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 4 } },
  { text: "双机 MHA Self->Mate 均维持 ~0.44 高水平.", options: { breakLine: true } },
  { text: "持续性隐式协调 — 非二元切换 — 参数共享网络自发打破对称性的终极证据.", options: {} },
], { x: 0.4, y: 1.65, w: 6.2, h: 3.5, fontSize: 14, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.35 });
img(s, F3, 7.0, 1.15, 5.8, 6.0);

// ═══════════════════════════ P11: Limitations ═══════════════════════════
s = pres.addSlide(); titleBar(s, 10, "边界分析: AND-Gate 为何最终回落");
s.addText("AND-Gate 死亡解剖 (Exp 3 checkpoint, 10 集 Eval)", { x: 0.4, y: 1.15, w: 6.5, h: 0.35, fontSize: 18, fontFace: FH, bold: true, color: C.navy });
[{ m:"双机同步入线率 (BOTH < 800m)",v:"0.0%",n:"致命瓶颈 — 零次!" },
 { m:"单机入线率 (>=1 < 800m)",v:"22.1%",n:"P0 可以接近" },
 { m:"夹击角 > 30 deg",v:"58.4%",n:"空间几何合格!" },
 { m:"P0 中位距离",v:"329 m",n:"逼近出色" },
 { m:"P1 中位距离",v:"1,974 m",n:"无法缩小差距" }].forEach((a,i) => {
  card(s, 0.4, 1.7+i*0.74, 5.6, 0.68, { fill: C.light });
  s.addText(a.m,{ x:0.6, y:1.7+i*0.74, w:3.6, h:0.38, fontSize:12, fontFace:FB, color:C.darkGray, valign:"bottom" });
  s.addText(a.v,{ x:4.4, y:1.7+i*0.74, w:1.2, h:0.38, fontSize:16, fontFace:FH, bold:true, color:C.coral, valign:"bottom", align:"right" });
  s.addText(a.n,{ x:0.6, y:2.08+i*0.74, w:5.0, h:0.22, fontSize:10, fontFace:FB, color:C.gray, valign:"top" });
});
img(s, CH+"/chart4_termination_reasons.png", 6.5, 1.15, 6.5, 4.0);
s.addText("三大理论天花板: (1) CTDE 信息不对称 — 33-dim 局部观测无法编码全局协调状态. (2) 离散探索边界 — 15 基元在严格 800m 下接近饱和. (3) 缺乏显式预期到达时间差 ΔTGO 信号.", { x: 0.4, y: 6.3, w: 12.5, h: 0.5, fontSize: 12, fontFace: FB, color: C.darkGray, italic: true });

// ═══════════════════════════ P12: Future ═══════════════════════════
s = pres.addSlide(); titleBar(s, 11, "未来探索方向 — 迈向 NxM 规模化编队对抗");
[{ n:"1",t:"显式运筹约束 (Hungarian Algorithm)",b:"引入匈牙利算法解决 NvM 编队的火力-目标分配组合爆炸. 参考 DARPA ACE / OFFSET 项目框架." },
 { n:"2",t:"微层级半马尔可夫 SMDP",b:"上层: 低频战术选项选择. 下层: 高频 Action Masking + 飞控执行. 桥接战略规划与战术执行." },
 { n:"3",t:"Self-Play 自我对局与联赛训练",b:"超越脚本目标, 进入对抗性训练. 基于种群多样性的 League Training. 竞争性共同进化涌现高阶战术." },
 { n:"4",t:"显式协同通信信道",b:"引入 ΔTGO 预期到达时间差. 通过 Attention 或 Message Passing 学习通信. 用显式时序信号突破 AND-Gate." }].forEach((d,i) => {
  card(s, 0.4, 1.2 + i*1.3, 12.5, 1.15, { shadow: true });
  s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: 1.4+i*1.3, w: 0.6, h: 0.6, fill: { color: C.coral } });
  s.addText(d.n, { x: 0.6, y: 1.4+i*1.3, w: 0.6, h: 0.6, fontSize: 18, fontFace: FH, bold: true, color: C.white, align: "center", valign: "middle" });
  s.addText(d.t, { x: 1.45, y: 1.3+i*1.3, w: 11.2, h: 0.35, fontSize: 16, fontFace: FH, bold: true, color: C.navy });
  s.addText(d.b, { x: 1.45, y: 1.68+i*1.3, w: 11.2, h: 0.5, fontSize: 13, fontFace: FB, color: C.darkGray, lineSpacingMultiple: 1.2 });
});
let dy12 = 1.2 + 4*1.3 + 0.15;
s.addShape(pres.ShapeType.rect, { x: 0.4, y: dy12, w: 12.5, h: 0.03, fill: { color: C.coral } });
s.addText("感谢聆听 · 欢迎提问与讨论", { x: 0.4, y: dy12+0.15, w: 12.5, h: 0.5, fontSize: 24, fontFace: FH, bold: true, color: C.navy, align: "center" });
s.addText("sean@zju.edu.cn  ·  github.com/NishimiyaXSean/jsbsim-marl-formation", { x: 0.4, y: dy12+0.65, w: 12.5, h: 0.3, fontSize: 12, fontFace: FB, color: C.gray, align: "center" });

pres.writeFile({ fileName: OUT }).then(() => console.log(`Saved: ${OUT} (${(fs.statSync(OUT).size/1024).toFixed(0)} KB, 12 slides)`));
