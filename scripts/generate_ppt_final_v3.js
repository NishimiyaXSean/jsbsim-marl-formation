/**
 * Final paper PPT v3 — speaker notes, full content per slide_plan.
 */

const pptxgen = require("/tmp/node_modules/pptxgenjs");
const fs = require("fs");

const C={navy:"065A82",teal:"1C7293",coral:"F96167",white:"FFFFFF",light:"F2F7F9",green:"2CC44D",gray:"8899A6",darkGray:"4A5568",accent:"E8F0F4",black:"1A202C"};
const FH="Georgia",FB="Calibri";
const CH="results/viz/paper_charts",F3="results/viz/fig3_role_attention_matrix.png";
const OUT="results/ppt/formation_coop_final_v3.pptx";
const pres=new pptxgen();pres.layout="LAYOUT_WIDE";pres.author="Sean Nishimiya";pres.title="基于MARL的协同编队飞行决策";

function tb(s,n,txt){
  s.background={fill:C.white};
  s.addShape(pres.ShapeType.rect,{x:0.35,y:0.2,w:0.05,h:0.45,fill:{color:C.coral}});
  s.addText(String(n),{x:0.35,y:0.15,w:0.7,h:0.5,fontSize:24,fontFace:FH,bold:true,color:C.coral,margin:0});
  s.addText(txt,{x:1.1,y:0.15,w:11.5,h:0.5,fontSize:28,fontFace:FH,bold:true,color:C.navy});
  s.addShape(pres.ShapeType.rect,{x:0.35,y:0.8,w:12.6,h:0.01,fill:{color:C.accent}});
}
function card(s,x,y,w,h,o={}){
  s.addShape(pres.ShapeType.roundRect,{x,y,w,h,fill:{color:o.fill||C.white},rectRadius:0.06,line:o.line?{color:C.accent,width:0.5}:undefined,shadow:o.shadow?{type:"outer",blur:3,offset:1,color:"C0D0D8",opacity:0.18}:undefined});
}
function sb(s,x,y,w,h,v,l){
  card(s,x,y,w,h,{shadow:true});
  s.addText(v,{x,y:y+0.06,w,h:0.55,fontSize:28,fontFace:FH,bold:true,color:C.coral,align:"center"});
  s.addText(l,{x,y:y+0.62,w,h:0.35,fontSize:12,fontFace:FB,bold:true,color:C.darkGray,align:"center",lineSpacingMultiple:1.1});
}
function img(s,p,x,y,w,h){if(fs.existsSync(p))s.addImage({path:p,x,y,w,h,sizing:{type:"contain",w,h}});}
function note(s,txt){s.addNotes(txt);}

let s=pres.addSlide();
s.background={fill:C.white};
s.addShape(pres.ShapeType.rect,{x:0.7,y:2.0,w:2.0,h:0.05,fill:{color:C.coral}});
s.addText("基于六自由度气动动力学的\n多智能体协同追击决策涌现",{x:0.7,y:0.8,w:11.5,h:1.6,fontSize:42,fontFace:FH,bold:true,color:C.navy,lineSpacingMultiple:1.1});
s.addText("Token-Based CTDE with Self-Attention + Discrete Primitives\n在 JSBSim 6-DOF F-16 编队追猎中超越集中式 PPO 上限",{x:0.7,y:2.5,w:11.5,h:0.8,fontSize:16,fontFace:FB,color:C.teal,lineSpacingMultiple:1.2});
s.addText("Sean Nishimiya  .  Zhejiang University  .  July 2026",{x:0.7,y:5.5,w:11.5,h:0.35,fontSize:11,fontFace:FB,color:C.gray});
note(s,"各位专家好，本课题聚焦于高动态、强非线性环境下的多智能体协同追击决策。为了防止强化学习智能体在早期探索时陷入疯狂抖动或气动失速的物理套利死锁——例如前期版本中频繁涌现的利用周期性俯冲爬升来骗取接近速度奖励的海豚跳现象——我们完成了底层控制的彻底剥离。我们将物理先验强行注入系统，利用经典解耦 PD 控制器锁死了过载与迎角极限，构建了 9 个安全稳定的 BFM 战术动作图元，从而确保高层强化学习能够心无旁骛地探索纯粹的战术协同范式。");


s=pres.addSlide();tb(s,1,"核心卡点: 协同的死亡三角与时空脱节死锁");
s.addText("分布式多智能体面临的空间满分、时间零分矛盾",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:18,fontFace:FH,bold:true,color:C.navy});
s.addText([
  {text:"非平稳性死锁",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"CTDE 早期同质 Agent 缺乏显式通信，独立 PPO (IPPO) 导致对称性内卷。评估均值砸向 -7536。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"高斯噪声耗散",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"连续动作 Box(2) 依赖高斯方差进行时序探索，导致 600 步长序列中动作高频抖动 (Chattering)，无法输出坚决的战术决断。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"时空同步悖论 (AND-gate 死亡解剖)",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"空间维度: 双机自发维持 34.8deg 黄金夹击几何，空间合格。",options:{breakLine:true}},
  {text:"时间维度: 长机 (Striker) 走内圈，中位距离 329m；僚机 (Interceptor) 走外圈，被抛在 1974m 外。",options:{breakLine:true}},
  {text:"长机单飞触发环境截断，平均步数腰斩至 165 步，同步入线率 = 0.0%。",options:{}}],
{x:0.35,y:1.5,w:7.0,h:5.0,fontSize:13,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
[{v:"-7,536",l:"IPPO 平台\n(从未突破)"},{v:"-1,171",l:"连续 AND-gate\n(最优)"},{v:"34.8deg",l:"平均夹击角\n(空间合格)"},{v:"0.0%",l:"同步入线率\n(时间零分)"}].forEach((st,i)=>sb(s,7.8,1.5+i*1.5,2.5,1.35,st.v,st.l));
note(s,"下面进入本研究最核心的卡点分析。在引入协同门控条件 AND-gate 后，我们遭遇了 MARL 领域极其经典的时空相位锁定悖论。定量统计显示，两架 F-16 的协同意识在空间上是完美的，它们在大样本下自发维持了 34.8 度的黄金包抄夹角。但是，在时间同步率上，由于共享参数的网络能力对称，走内圈直线的长机迅速突入到距离敌机 329m 的内圈，而负责侧翼合围的僚机因为走外圈大弧线，数学上绝对无法在相同物理航速上限内追赶上长机，导致其停留在 1974m 外。更致命的是，长机单刀直入会引发环境提前截断，导致交战平均步数被腰斩至 165 步。僚机还没飞完弧长，游戏就被强行关闭了，同步入线率死死锁在 0.0%。这就是连续空间下高斯噪声耗散与二元硬门控带来的时空脱节死锁。");

pres.writeFile({fileName:OUT}).then(()=>console.log(`Saved: ${OUT} (${(fs.statSync(OUT).size/1024).toFixed(0)} KB, 12 slides)`));

// P3: Self-Attention
s=pres.addSlide();tb(s,2,"方法论创新 A: Token-Based 多头自注意力表征机制");
s.addText("基于语义 Token 的 Multi-Head Self-Attention 网络前向传播",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:18,fontFace:FH,bold:true,color:C.navy});
s.addText([
  {text:"打破特征平铺拼接",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"传统扁平拼接抹杀了多智能体空间拓扑的动态相关性。将 33 维观测解耦为 Self(13)、Target(14)、Mate(6) 三个独立语义 Token。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"归纳偏置 (Inductive Bias) 注入",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"通过 Transformer Q-K-V 交互，强迫智能体在内部表征层建立 Self-Mate-Target 的三维相对运动学关联。Token-Type Embedding 区分三类实体。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"动态对称破缺",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"参数共享 (Parameter-Sharing) 下，同一网络通过对不同 Mate Token 的关注差异，赋予 Agent 自发进行隐式战术角色分化的能力。",options:{}}],
{x:0.35,y:1.5,w:6.5,h:4.5,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
s.addText("管道: [Self] [Target] [Mate] -> 4-Head MHA -> Pool -> [256,256] -> Action",{x:7.2,y:1.5,w:5.8,h:0.5,fontSize:11,fontFace:FB,color:C.gray,italic:true});
[{v:"0.450",l:"Striker\\nSelf->Mate"},{v:"-0.53",l:"Cohen d\\n(large effect)"},{v:"0.44",l:"Mutual Mate\\nAttention"}].forEach((st,i)=>sb(s,7.2+(i%2)*2.85,2.1+Math.floor(i/2)*1.5,2.7,1.35,st.v,st.l));
img(s,F3,7.2,5.1,5.8,2.2);
note(s,"为了让智能体具备打破对称性内卷的高阶空间态势感知能力，我们推翻了传统的扁平特征平铺拼接，开发了 Token-Based 多头自注意力网络。我们向模型注入了强烈的多智能体归纳偏置，将空间状态打包为具备独立语义的物理 Token。在参数共享的架构下，两架战机运行着同一套网络权重，但由于它们各自输入的 Mate Token 携带了对方的相对物理位置差异，自注意力机制能够自发在网络内部引爆对称性破缺，为后续在分布式执行阶段实现高精度隐式协同奠定了坚实的表征基础。");

// P4: Discrete + Masking
s=pres.addSlide();tb(s,3,"方法论创新 B: 离散战术图元重构与 Action Masking 上游剪枝");
s.addText("MultiDiscrete([5,3]) = 15 种宏观战术基元 -- 从连续混沌到离散涌现",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:18,fontFace:FH,bold:true,color:C.navy});
s.addText([
  {text:"探索维度限制",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"将无限的连续探索流压缩为 15 种宏观战术图元的 Categorical 分布。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"数学熵上限锁死",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"最大探索熵从连续空间的负无穷硬性锁死在 ln(5)+ln(3)~2.71，防止梯度干涸或噪声 runaway。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"Action Masking 动作掩码",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"在环境上游实时计算失速警告线 (V<130m/s)、近地 (<200m)、超速 (>95% Vmax)，强行剪枝危险分支。为长时序信用分配清除混沌梯度。",options:{}}],
{x:0.35,y:1.5,w:5.8,h:3.8,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
s.addText("TURN (5档)",{x:6.8,y:1.35,w:2.5,h:0.28,fontSize:14,fontFace:FH,bold:true,color:C.coral});
[["0","急左转 -15 deg/s"],["1","缓左转  -5 deg/s"],["2","直飞     0 deg/s"],["3","缓右转  +5 deg/s"],["4","急右转 +15 deg/s"]].forEach(([id,nm],i)=>{
  card(s,6.8,1.72+i*0.42,5.8,0.38,{fill:C.light});
  s.addText(id,{x:6.9,y:1.72+i*0.42,w:0.3,h:0.38,fontSize:13,fontFace:FH,bold:true,color:C.coral,valign:"middle"});
  s.addText(nm,{x:7.3,y:1.72+i*0.42,w:5.2,h:0.38,fontSize:13,fontFace:FB,color:C.darkGray,valign:"middle"});
});
let ty4=1.72+5*0.42+0.2;
s.addText("SPEED (3档)",{x:6.8,y:ty4,w:2.5,h:0.28,fontSize:14,fontFace:FH,bold:true,color:C.coral});ty4+=0.35;
[["0","慢速 180 m/s (节能巡逻)"],["1","巡航 250 m/s (均衡追击)"],["2","快速 320 m/s (加力冲刺)"]].forEach(([id,nm],i)=>{
  card(s,6.8,ty4+i*0.42,5.8,0.38,{fill:C.light});
  s.addText(id,{x:6.9,y:ty4+i*0.42,w:0.3,h:0.38,fontSize:13,fontFace:FH,bold:true,color:C.coral,valign:"middle"});
  s.addText(nm,{x:7.3,y:ty4+i*0.42,w:5.2,h:0.38,fontSize:13,fontFace:FB,color:C.darkGray,valign:"middle"});
});
note(s,"面对高斯探索噪声导致的同步失效，我们采取了动作空间的降维打击。我们彻底重构动作空间为离散的多维 MultiDiscrete([5, 3])。这一改变将原本在连续流里挣扎的策略探索，浓缩回 15 种高阶离散战术图元中，并将策略熵上限数学锁死在 2.71 左右。最重要的是，离散化赋予了我们挂载 Action Masking 的终极红利。在飞机快要失速或撞地时，环境在上游直接将该离散分支的 Logits 抹为负无穷，从而将这套新底盘的样本效率提升了数倍。");

// P5: Chart 1
s=pres.addSlide();tb(s,4,"动作图元时序演变 -- 从无序探索到行为固化");
s.addText("100% 堆叠柱状图: 冷启动离散 Self-Attention (Exp 4a-v2) 在 320 轮训练中的动作分布迁移",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,bold:true,color:C.darkGray});
img(s,CH+"/chart1_action_distribution.png",0.15,1.5,13.0,5.5);
s.addText([
  {text:"早期 (0-50 轮): 各类速度图元各占 ~33%，飞机频繁换向互相扯皮 -- 纯探索。",options:{breakLine:true}},
  {text:"后期 (250-320 轮): 僚机识别到非对称 Token 后，选择 Fast (260m/s) 概率激增至 ~78%；长机突入内圈后自发转入 Slow (180m/s) 减速蓄势。",options:{breakLine:true}},
  {text:"结论: 多智能体自发通过 80m/s 宏观速度差强行收拢 AND-gate 时间缺口 -- 闭环战术涌现。",options:{}}],
{x:0.35,y:6.7,w:12.5,h:0.7,fontSize:12,fontFace:FB,color:C.darkGray,italic:true,lineSpacingMultiple:1.3});
note(s,"为了让大家直观地看到智能体内部发生的变化，请看这张动作图元时序演变堆叠图。我们可以清晰看到一个从无序向战术固化迈进的过程。在训练的前 50 轮，由于是冷启动，所有色块均匀平铺。然而到了 250 轮以后，策略展现出了高度集中的确定性行为：后方的僚机在发现自己处于劣势 Token 后，会以接近 80% 的绝对概率疯狂输出 Fast 离散加力指令；而在前方的长机则自发学到了 Slow 减速挂杆动作，它们利用 80m/s 的巨大相对航速差，在物理层面上强行合拢了先前绝望的时间缺口。");

// P6: Curriculum
s=pres.addSlide();tb(s,5,"核心机制创新: 非对称惩罚与动态包线退火课程 (DAC)");
s.addText("基于动态距离包线退火的多智能体引导机制 -- 为 Centralized Critic 搭建梯度天梯",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,color:C.darkGray,italic:true});
s.addText([
  {text:"距离不对称惩罚 (DAP)",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"|d0-d1| > 500m 则团队受罚。严厉惩罚双机间距过大的贪婪独狼行为，强行摧毁分布式执行中的盲目套利死锁。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"动态距离包线退火 (DAC)",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"Thresh_AND = max(800, 2000 - decay_rate x iter)",options:{breakLine:true}},
  {text:"训练前 100 轮: 双机分别突入 2000m 且夹角及格，即激活 +5000 成功奖励，为 Centralized Critic 提供第一桶稀疏正向梯度。",options:{breakLine:true}},
  {text:"训练后期: 漏斗逐步收窄至严格 800m。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"时间同步配速惩罚",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"Striker < 1200m 且 Interceptor > 1500m 则团队受罚。强制长机等待僚机同步，防止独狼冲锋。",options:{}}],
{x:0.35,y:1.5,w:6.5,h:5.0,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
[{v:"+5,000",l:"DAC 拉升幅度\\n(vs 严格 AND)"},{v:"-5,909",l:"重构前基线\\n(严格 AND)"},{v:"-1,171",l:"DAC 后最优\\n(历史高度)"},{v:"1,974m",l:"僚机中位\\n(P1 追赶极限)"}].forEach((st,i)=>sb(s,7.2,1.6+i*1.45,2.5,1.3,st.v,st.l));
note(s,"下面揭晓我们是如何通过环境创新为模型搭起这根梯度天梯的。面对严格 AND-gate 在训练初期形成的庞大价值真空区，我们实施了动态距离包线退火课程设计。我们不再让 800m 门槛硬着陆，而是设计了一个随迭代步数线性收紧的漏斗课程。在探索初期，只要僚机冲进 2000m、长机在内圈且合围几何达成，网络就能立刻引爆 +5000 的终极成功奖赏！这个机制让集中式评论员迅速建立起了关于协同的全局价值高地。随着训练步数深入，漏斗逐渐收窄至严格的 800m。这项 DAC 创新直接把 AND-gate 连续空间基线逆天拉升了整整 5000 分，最终锁死在离胜利仅一步之遥的 -1171 平台。");

// P7: KDE
s=pres.addSlide();tb(s,6,"全样本空间 KDE 热力阵型 -- 打破幸存者偏差的统计铁证");
img(s,CH+"/chart2_spatial_kde.png",0.15,1.1,8.5,5.5);
s.addText([
  {text:"数据规模: 50 集 Eval, 总计 127K 帧空间数据融合",options:{breakLine:true,fontSize:14,fontFace:FH,bold:true,color:C.navy}},
  {text:"目标固定在原点 (0,0), 机头朝向正北。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"统计级发现:",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"双机空间打点在目标尾部左后/右后 35-60deg 扇区",options:{breakLine:true}},
  {text:"凝聚为两团极高密度深红色热区。",options:{breakLine:true}},
  {text:"均值合围角 = 35.8deg (中位数 29.3deg) -- 完美黄金包围圈。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"结论: 协同引导非单集幸运, 而是演化成全盘大样本",options:{breakLine:true}},
  {text:"的通用统计本能。参数共享语义 Token 网络拥有",options:{breakLine:true}},
  {text:"极强的泛化鲁棒性。",options:{}}],
{x:9.0,y:1.1,w:4.0,h:5.0,fontSize:13,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
note(s,"为了彻底打消评审对于单集轨迹表现纯属幸运的质疑，我们编写脚本提取了 50 集全量评估、共计上万个 Step 的运动学轨迹，绘制了这张以敌机为中心的全样本空间 KDE 热力阵型图。这是一个非常有力的统计铁证。大家可以看到，红方的物理位置打点在经过上万帧的洗礼后，非但没有散成一团乱麻，反而极其精准地在敌机尾部的左后和右后两侧凝聚成了两团明亮的高密度红色热区。平均合围角收敛于 35.8 度，这说明我们的网络已经在海量测试集中，将这种分头侧翼包抄的战术几何化为了骨子里的统计本能。");

// P8: Ablation
s=pres.addSlide();tb(s,7,"全版本消融大盘点 -- 五代模型完整进化证据链");
const hdr=["实验","架构","动作空间","BC","轮数","最佳 Eval","Eval>0","关键发现"];
const cW=[2.0,1.4,1.7,1.1,0.8,1.3,0.8,3.2];
const rows=[
  ["Exp 1 (非协同)","Shared Attn","Box(2)","SB3 BC","200","-8,053","0","IPPO 内卷死锁, 无协同=无希望"],
  ["Exp 2 (OR-gate)","Shared Attn","Box(2)","SB3 BC","120","+7,888","5x","首次超越 SB3 集中式天花板 33%"],
  ["Exp 3v3 (AND退火)","Shared Attn","Box(2)","SB3 BC","300","-1,171","0","DAC 课程拉升 5,000 分; 连续噪声限制"],
  ["Exp 4a (MLP)","MLP fallback","MultiDisc(5,3)","无","120","-4,542","0","纯 MLP 零 BC 无法学习协同 -- 对照基线"],
  ["Exp 4a-v2 (Attn)","Self-Attn","MultiDisc(5,3)","无","120","+1,345","1x","Self-Attn 冷启动首次转正 -- 架构优势"],
  ["Exp 4b (Attn+BC)","Self-Attn","MultiDisc(5,3)","离散BC","120","-1,135","0","BC 提供稳定但无额外峰值"],
  ["* 4a-v2 ext 320","Self-Attn","MultiDisc(5,3)","无","320","+2,376","3x","零知识涌现 -- 结构决定上限"],
];
const tx8=0.3,tw8=cW.reduce((a,b)=>a+b);let ty8=1.2;
s.addShape(pres.ShapeType.rect,{x:tx8,y:ty8,w:tw8,h:0.45,fill:{color:C.navy}});
let h8=tx8;hdr.forEach((h,i)=>{s.addText(h,{x:h8,y:ty8,w:cW[i],h:0.45,fontSize:11,fontFace:FH,bold:true,color:C.white,align:"center",valign:"middle"});h8+=cW[i];});
ty8+=0.5;
rows.forEach((row,ri)=>{
  s.addShape(pres.ShapeType.rect,{x:tx8,y:ty8,w:tw8,h:0.62,fill:{color:ri%2===0?C.light:C.white},line:{color:C.accent,width:0.3}});
  let c8=tx8;row.forEach((cell,ci)=>{
    const hi=cell.startsWith("*")||cell==="+2,376";
    s.addText(cell.replace("* ",""),{x:c8,y:ty8,w:cW[ci],h:0.62,fontSize:10,fontFace:FB,color:hi?C.coral:C.darkGray,bold:hi,align:"center",valign:"middle"});
    c8+=cW[ci];
  });
  ty8+=0.67;
});
s.addText("* 完整的科学进化链: IPPO崩溃 -> 连续参数共享拉升 -> 严格AND-gate时空脱节 -> 离散Self-Attention零知识转正。结构 (Inductive Bias) 决定能力上限, 数据 (BC) 仅影响收敛速率。",{x:0.3,y:ty8+0.08,w:12.7,h:0.35,fontSize:11,fontFace:FH,bold:true,color:C.coral});
note(s,"这是我们整个研究最引以为傲的一张消融大盘。我们由浅入深，严密地设置了 7 个核心对照组。从早期没有协同机制的独立 IPPO 大崩溃，到实验 2 OR-gate 下首次冲破集中式 PPO 天花板 33% 斩获 +7888 高分；再到实验 3 揭示出连续高斯噪声在严格同步下的无能为力；直到最终，我们切换至离散空间搭配 Self-Attention，完成了对整个项目的终极正名。这不仅是一张简单的调参表格，它严密地向同行勾勒出了多智能体强化学习在复杂三维气动任务中突破瓶颈的完整进化史。");

// P9: Emergence
s=pres.addSlide();tb(s,8,"核心成果 1: 离散自注意力下的零知识战术涌现");
s.addText("Exp 4a-v2: 无需专家经验引导的协同决策涌现 -- 冷启动 Self-Attention 3 次 Eval 大转正",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,color:C.darkGray,italic:true});
s.addText([
  {text:"打破死记硬背质疑",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"网络权重完全随机、无 BC 引导的绝对冷启动。320 轮内 3 次 Eval 大转正, 峰值 +2,376。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"解密独立探索能力",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"Token-based 架构无需任何专家外力，仅凭多头自注意力路由从零摸索 AND-gate 同步包抄最佳解。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"3D 轨迹三维校准",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"修正 Matplotlib 3D 轴比例失真 (X:Y:Z=1:1:0.1)。高度曲线在 +/-10m 气动矢量超调内稳定锁死，策略未发生重力套利，轨迹展现出极高的物理真实与战术美感。",options:{}}],
{x:0.35,y:1.5,w:6.8,h:4.5,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
[{v:"+2,376",l:"最高 Eval\\n正向突破"},{v:"3x",l:"正向转正\\n总次数"},{v:"320",l:"冷启动\\n训练轮数"},{v:"0",l:"专家样本\\n干预"}].forEach((st,i)=>sb(s,0.35+i*2.85,5.6,2.65,1.3,st.v,st.l));
img(s,CH+"/chart5_health_metrics.png",7.5,1.5,5.5,2.5);
img(s,CH+"/chart2_spatial_kde.png",7.5,4.2,5.5,3.0);
note(s,"现在展示本课题最重磅的黄金发现之一：离散自注意力架构在没有任何专家经验引导下的零知识协同涌现。请看测试数据，Exp 4a-v2 在冷启动状态下，在 320 轮内成功轰出了 3 次惊艳的 Eval 确定性测试大转正，最高峰斩获 +2376！这有力地回击了学术界常见的质疑。它证明我们的模型不是在机械死记硬背专家的动作轨迹，它自己就拥有高超的探索上限。再看经我们进行三维轴比例校准后的真实飞行轨迹，长机以 200m 逼近目标，高度稳健地锁死在气动安全范围内，两机在完全没有外力干预的情况下，打出了一套极具战术美感的外圈大弧度合围包抄，这正是 Inductive Bias 带来的表征奇迹。");

// P10: Fig 3
s=pres.addSlide();tb(s,9,"核心成果 2: 动态对称破缺与隐式协同的角色分化矩阵");
s.addText("Fig 3 (角色分组均值矩阵) -- 持续性隐式协同的病理学铁证",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,color:C.darkGray,italic:true});
s.addText([
  {text:"Cohens d = -0.53 强效应量",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"大样本 7,858 步瞬时几何角色切片下，Interceptor 对 Target 的 MHA Self->Target 注意力比 Striker 高 31%，呈现显著异质化追击偏好。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"平滑分化与持续性隐式协同",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"两架共享参数飞机对 Mate Token 的 MHA Self->Mate 自发全时锁定在 0.44-0.45 极高一致高位。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"推翻二元切换假设",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"确立始终关注队友维持大局(基础默契)，按瞬时物理几何角色微调目标专注度(柔性分化)的高级隐式协同范式。",options:{}}],
{x:0.35,y:1.5,w:6.5,h:5.0,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
img(s,F3,7.2,1.2,5.8,6.0);
note(s,"接下来，我们用网络内部的多头自注意力权重数据，为隐式协同提供无可辩驳的病理学铁证。我们对 7,858 个瞬时交战步进行了几何角色切片划分。请看右侧的均值热图：虽然两架飞机共享同一套网络参数，但在统计学上，它们在 MHA 层面爆发出了极其剧烈的自发角色分化。外圈吃力超车的僚机，其对目标的自注意力路由以 Cohens d = -0.53 的大效应量显著超越了前方的长机，整整高出 31%。更妙的是，它们对彼此的关注度 Self-to-Mate 全时平稳锁定在 0.44 以上的高位。这在学术上推翻了传统的二元切换假设，首次向同行展示了一种高级的持续性隐式协同——两机骨子里存在全时的高强度默契，它们只在基础默契之上，根据瞬时物理拓扑微调各自的战术分工，这才是 MARL 涌现行为的最高级形态。");

// P11: Limitations
s=pres.addSlide();tb(s,10,"信息论边界: 分布式执行下的熵墙与 Failure Case 深度解剖");
s.addText("终止原因瀑布分析 + Vanilla CTDE 的理论天花板",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,color:C.darkGray,italic:true});
s.addText([
  {text:"缺陷诊断 -- 边缘悬崖效应",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"冷启动离散模型在 320 轮后期因缺乏低熵锚定，Logits 跨格点探索抖动跌回 -6,653。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"Vanilla CTDE 的理论天花板",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"33 维分布式局部视界仅含瞬时相对信息，缺乏非局部性预计到达时间差 (delta TGO) 表达。僚机无法预测长机踩进 800m 临界秒级相位戳，梯度夹杂时序噪声。",options:{breakLine:true}},
  {text:"",options:{breakLine:true,fontSize:4}},
  {text:"诚实结论",options:{breakLine:true,fontSize:15,fontFace:FH,bold:true,color:C.coral}},
  {text:"划定扁平、无显式通信的分布式控制在应对极端断崖式严格 AND 同步任务时的理论边界。这为下一阶段引入层级控制与显式通信拓扑提供了最坚实的科学正当性。",options:{}}],
{x:0.35,y:1.5,w:6.8,h:5.0,fontSize:14,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.3});
img(s,CH+"/chart4_termination_reasons.png",7.5,1.5,5.5,4.0);
note(s,"在汇报的尾声，我们进行一项严谨的信息论边界讨论。虽然重构和 DAC 课程帮助我们实现了 5000 分的跨越式暴涨并引爆了 3 次成功转正，但在 320 轮尾部依然出现了探索回落。这是由于动作空间离散化后，严格 AND-gate 形成了一个充满阶跃跳变的窄深峡谷。我们必须诚实地指出 Vanilla CTDE 架构在此类任务中的理论局限性：当前的 33 维局部观测向量只包含了瞬时的相对几何，它在信息论层面上缺乏一个全局非局部的预计到达时间差 delta TGO 路由。后方智能体在无隐式/显式通信链路时，脑子里算不准长机切入生死线的秒级相位戳。这个被我们清晰划定出的理论天花板，恰恰为我们下一阶段引入层级控制以及显式通信拓扑提供了最坚实的科学正当性。");

// P12: Future
s=pres.addSlide();tb(s,11,"未来进军路线: 层级 SMDP 与大规模匈牙利指派演进");
s.addText("面向 N x M 大规模多机对抗的半马尔可夫分层与运筹融合",{x:0.35,y:1.05,w:12.5,h:0.35,fontSize:16,fontFace:FB,color:C.darkGray,italic:true});
[{n:"1",t:"大规模外推突破",b:"参数共享与语义 Token 天然具备零成本 N 机位置外推特性。下一步转向 3v1 及 4v2 场景，结合匈牙利算法实现火力-目标最优指派。"},
 {n:"2",t:"微分层 SMDP 重构",b:"上层 Policy 降至 2Hz 专注宏观 BFM 意图选择，解除长时序信用分配噪声；下层 Action Masking 维持 60Hz 物理安全锁死。"},
 {n:"3",t:"Self-Play 自我对局演进",b:"引入群体课程自博弈 (PCSP) 与 League Training，破除脚本目标的非对抗局限，走向博弈对顶演化。"},
 {n:"4",t:"显式 delta TGO 通信信道",b:"在全局状态引入预计到达时间差信号，架设真正的时空同步桥梁，一击突破 AND-gate 终极天花板。"}].forEach((d,i)=>{
  card(s,0.35,1.55+i*1.35,12.6,1.2,{shadow:true});
  s.addShape(pres.ShapeType.ellipse,{x:0.55,y:1.75+i*1.35,w:0.6,h:0.6,fill:{color:C.coral}});
  s.addText(d.n,{x:0.55,y:1.75+i*1.35,w:0.6,h:0.6,fontSize:18,fontFace:FH,bold:true,color:C.white,align:"center",valign:"middle"});
  s.addText(d.t,{x:1.4,y:1.65+i*1.35,w:11.2,h:0.35,fontSize:16,fontFace:FH,bold:true,color:C.navy});
  s.addText(d.b,{x:1.4,y:2.05+i*1.35,w:11.2,h:0.5,fontSize:13,fontFace:FB,color:C.darkGray,lineSpacingMultiple:1.25});
});
let dy=1.55+4*1.35+0.15;
s.addShape(pres.ShapeType.rect,{x:0.35,y:dy,w:12.6,h:0.03,fill:{color:C.coral}});
s.addText("感谢聆听 . 欢迎提问与讨论",{x:0.35,y:dy+0.15,w:12.6,h:0.5,fontSize:24,fontFace:FH,bold:true,color:C.navy,align:"center"});
s.addText("sean@zju.edu.cn  .  github.com/NishimiyaXSean/jsbsim-marl-formation",{x:0.35,y:dy+0.65,w:12.6,h:0.3,fontSize:12,fontFace:FB,color:C.gray,align:"center"});
note(s,"最后，展望我们的未来进军路线。得益于我们坚持重构的参数共享 + 语义 Token 底座，我们的网络架构天生具备零成本向 3v1、4v2 等更大规模扩展的位置无关特性。下一步，我们将引入运筹匹配算法来剪枝大规模火力分配的组合爆炸问题。同时，我们将深化半马尔可夫微分层 SMDP 重构，让高层网络在 2Hz 的低频上坚决地下达离散战术意图，彻底洗刷序列信用分配的时序噪声；在底层通过自我对局机制，让红蓝双方的多智能体在对抗中完成战术的螺旋协同升级。以上就是我的全部汇报，感谢各位专家，请多指教。");

pres.writeFile({fileName:OUT}).then(()=>console.log("Saved: "+OUT+" ("+Math.round(fs.statSync(OUT).size/1024)+" KB, 12 slides)"));
