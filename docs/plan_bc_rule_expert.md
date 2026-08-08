# BC 模仿规则专家 — 修订实施计划 (2026-08-06)

## 背景

- 导弹系统最佳基线 = **v19**(发射 0.48/局、命中 100%、击杀 10%、lost_target 32%,d=0.0)。
- 唯一瓶颈 = **策略接近行为**:奖励调参(v14–v23)与绝对扇区接口(×2 失败)均已探底。
- 本计划:用开环规则专家(30/30 稳定捕获)解决"不会稳定接近",再让 PPO 优化收益。

## 核心设计决策(评审修订)

### 1. 发射专家:两层判定(fire_allowed 与 fire_desired 分离)

```
fire_allowed = 动作掩码允许 (ATA<15, DLZ, closure>0, 冷却, 弹药)
fire_desired = 专家窗口质量判断:
  1) 掩码连续打开 >= 2~3 帧(防边缘抖动)
  2) ATA < 8~10 度(严于硬掩码)
  3) closure > 阈值(如 5 m/s)
  4) 位于 DLZ 中部/premium 区(如 2~5km)
  5) 长时间无 premium 窗口时, 用普通合格窗口兜底
fire = fire_allowed AND fire_desired
```

保存连续 `launch_quality_score`(w1*DLZ_depth + w2*closure_quality - w3*ATA - w4*heading_rate),
以及 `fire_allowed / fire_desired / premium_window / continuous_target_hdg / continuous_target_spd`,
区分"不能发 / 可发但不该发 / 应该发",后续调发射策略无需重跑轨迹。

### 2. 数据量按 transition 计 + 整局划分 + 分层采样

- 目标:第一版 20 万~50 万 transition(非"局数")。
- 训练/验证/测试按**整局划分**(防相邻帧泄漏导致验证虚高)。
- 几何覆盖:偏置 0~90 + 部分 90~120 度;距离覆盖 DLZ 外/刚进/中部/近距穿越;
  左右对称;不同初始 ATA、closure 正负;相对高度/俯仰;少量目标速度扰动。
- 按阶段分层采样:远距建立接近 / closure 由负转正 / ATA 收敛 / 首次进入 WEZ /
  发射窗口 / 近距交叉 / 即将丢失 / 丢失前恢复。

### 3. BC → 轻量 DAgger(优先于增加专家局数)

1. 规则轨迹训练 BC;
2. BC 闭环跑 100~200 局;
3. 对 BC 实际访问的状态用规则重标注(保留即将丢目标/ATA 发散/closure 转负样本);
4. 新数据与原始数据按 30:70~50:50 混合;
5. 重训 1~2 轮。

### 4. 连续→离散标签:基于参考指令 + deadband/迟滞 + 单步前瞻

- 基于 `cmd_hdg_err`(bearing - ref_hdg, wrap-around),而非实际航向;
- deadband + 迟滞,避免 -5/0/5 高频切换;
- 标签 = argmin_a[ w_hdg*next_hdg_err(a)^2 + w_spd*next_spd_err(a)^2 + w_rate*cmd_change(a)^2 ],
  对每个候选离散动作估计下一步参考指令误差;
- 速度同样基于参考速度与控制器响应;
- 可选软标签(目标在 +5/+10 之间时按距离给概率)。

### 5. 掩码感知损失

```python
masked_logits = logits.masked_fill(~valid_action_mask, very_negative)
loss = cross_entropy(masked_logits, expert_action)
```

- 专家标签必须落在有效动作上;否则报错并检查数据;
- fire=0 是有效标签,必须正常训练(不跳过整样本);
- fire 不平衡:发射窗口样本采样保证 minibatch 比例 + 类加权 CE + fire 权重 2~5x;
- 观察 fire precision/recall,不能只看总准确率;
- alt 单维(head 大小为 1)跳过该 head 的 CE 与统计。

### 6. 模型兼容性

- BC 与 RLlib 推理单元测试:同一 obs+mask → BC logits 与 policy logits 各 head 数值接近 0;
- 只加载 policy trunk + action heads;价值网络不加载 BC 权重;
- 若 policy/value 共享 backbone:PPO 初期降低 value loss 影响,或独立网络,或加 BC/KL 正则;
- 保证 obs 字段顺序、归一化、mask 拼接、head 顺序、state-dict key 完全一致。

### 7. BC 验证两层

- 快速门禁:20 局(发现不转向/head 顺序错/一直 fire 或从不 fire/obs 归一化错/迅速 lost);
- 正式验证:100~200 局,与规则、v19 使用**相同 seeds**;
- 新增指标:首次 closure>0 时间、closure>0 时间占比、首次 WEZ 时间、ATA p50/p90、
  WEZ 内停留时间、进入 WEZ 后丢失率、发射窗口数量、premium/normal/bad 比例、首次发射时间。
- 判断标准:BC 的 time_to_WEZ、ATA p90、closure_positive_ratio 接近专家,即接近行为学到。

### 8. PPO 两阶段微调 + 行为锚定

- 阶段 A(行为保持):lr 1e-5~3e-5、clip ~0.1、entropy 0.003~0.01 逐步降到 0.001;
  每 5~10 迭代固定评测;lost_target 显著恶化立即停止;
- 阶段 B(奖励引入):确认 BC 行为未破坏后,逐步打开 kill+8000 / ATA 门控 /
  反悬停 / premium,一次一个,记录每个奖励分量 episode sum;
- 行为锚定:PPO loss + lambda*BC loss,或 + beta*KL(current || initial_BC),
  前 50~100 迭代保持,之后衰减到 0;
- 不要用 v22/v23 奖励对 BC 策略自由训练 500 轮。

### 9. batch 与训练稳定性

- 第一轮保留 batch 2048 作基准;后续测试 8192~32768(并行 env / 梯度累积);
- 关注每批完整 episode 数、发射/击杀事件数、advantage 是否被少数 +8000 主导、
  reward/advantage 标准化。

### 10. "训练与评测一致"重定义

- 每个 checkpoint 在固定 seed + 确定性动作 + 固定环境配置下跑同一评测套件;
- 记录 train_episode_reward / exploration-off eval reward / 固定场景 eval reward /
  train-eval 环境配置哈希 / 各奖励分量;
- 据此判断背离来自策略、探索、随机场景还是配置不一致。

## 实施顺序

```text
1. 规则专家生成 (两层fire+质量分, 分层覆盖, 20万-50万transition)
2. 分层/平衡数据集 (按局划分 train/val/test)
3. 掩码CE的BC (deadband/迟滞/单步前瞻标签)
4. 20局 smoke test
5. 100-200局正式验证 (与规则/v19同seed)
6. 1-2轮 DAgger
7. 重新验证
8. 带BC/KL锚定的低LR PPO (阶段A行为保持)
9. 奖励逐步引入 (阶段B)
10. 固定seed持续回归评测 (自动 vs v19)
```

## 成功标准(相对 v19 基线)

- 发射/局 >= 0.48, 击杀 >= 10%, lost_target <= 32%;
- BC/PPO 后训练与评测(固定套件)不出现方向性背离;
- 发射质量保持 premium/good 为主。

## 门禁执行记录 (2026-08-06)

### 门禁 1: 离散化规则闭环验证 — ✅ 通过

- 关键发现: signed-ATA 标签导致 ref 过度旋转(lost 97%);改为 **ref 误差单步前瞻 + 死区** 后修复。
- 训练几何(偏置 30-60°): lost 0%, WEZ 到达 100%, 首次WEZ中位15.1s, 发射2.57/局, 击杀24%。
- 宽几何(偏置 0-120°): lost 0%, WEZ 100%, 发射2.78/局, 击杀31%。
- 结论: 离散专家稳健, BC 上限(lost 0%, 击杀24-31%)远高于 v19(lost 32%, 击杀10%)。

### 门禁 2: 专家标签无状态化 — ✅ 通过

- 航向标签 = f(ref误差 obs[21]) 单步前瞻 + 死区;
- 速度标签 = f(空速, ATA, 参考速度);
- 发射判定 = f(当前 ATA, closure, DLZ深度) — 去掉掩码连续帧/长时间等待/上一动作等历史依赖;
- 冷却移入环境侧开火掩码(策略只见静态许可窗口);
- 连续 target_hdg 仍保存(供软标签/后续分析), 但不用于标签。

### 门禁 3: fire head 损失定义(BC 训练时执行)

- 仅在 fire_allowed=true 的窗口样本上做 B/C 分类平衡(1:1~3:1), 不计 A 类(掩码关闭)样本;
- fire loss 权重从 2 起, 观察 precision/recall;
- 软标签用 masked log-softmax soft-CE/KL。

### 门禁 4: PPO 前 critic warm-up(微调时执行)

- A0: 冻结 policy, 用 BC policy 收集轨迹只训 value head;
- A1: 解冻, 低 LR + 锚定(PPO loss + λ·CE(BC action, current), 或 forward KL(BC || current));
- A2: 锚定衰减; B: 奖励逐项引入。

### 数据量/评测(修订)

- 数据: 300 局宽几何(约35万 transition), 按整局划分 train/val/永久test;
- 开发评测 100-200 局同 seed; 最终 500-1000 配对 seed + 95% bootstrap CI;
- 成功标准(最低): 发射/局≥v19, 击杀≥v19, premium不退化, lost_target显著低于32%;
  目标: lost≤20-25%, 发射≥0.50/局, 击杀≥15%, bad≈0。

### 补充审计与几何勘误 (2026-08-06)

#### 补充审计 1: 无状态标签确定性 — PASS

- 方法: 从 403,544 条 transition 中随机抽 30,000 行, 用纯函数标签器复算两次, stored / run1 / run2 三方一致。
- 修复: spd 复算需从保存的 target_spd 反推生成时的 cmd_speed (ata>=15° 时 target_spd+60), 不能直接用 target_spd 作为 spd_label 参数。
- 结果: 30,000 行全部一致; hdg/fire/fire_allowed/fire_desired/launch_quality 也全部一致。

#### 补充审计 2: split 内 C 分布与几何覆盖 — PASS

- 按整局 80/10/10 (rng=0) 划分: train C=673 (>=600), val C=80 (>=60), test C=85 (>=60), 全部 PASS。
- C 覆盖: 非空角度桶/DLZ/closure/ATA 桶均有 C; top-5 局 C 集中度 2.4% (838 C 分散在 296/300 局)。

#### 几何勘误 (重要)

- 发现: generate_shoot_rule_expert.set_geometry 实际被 env.reset() -> SingleCombatShootTask.reset 覆盖, 未生效。
- 真实初始几何: 尾追 2-5km、横向 ±500m、初始偏置角 30-60° (max_heading_bias_deg=60 默认), 目标 180-240 m/s, 追击机 240-300 m/s, 1s 暖机。
- 因此: 门禁 1 记录中的"宽几何(0-120°) 击杀31%"与"训练几何 24%"实为同一几何下不同随机种子; 数据集中不存在 90-120° 初始偏置。
- 对计划的影响: 首轮 BC 学习 30-60° 尾追接近; 宽几何覆盖留到 DAgger 或定向补数据阶段处理。
- 离散专家在真实几何下 (100 局): lost 0%, WEZ 100%, 发射 2.68/局, 命中 100%, 击杀 23%, 仍是强基线。


### BC 首轮训练与闭环评测 (2026-08-06)

#### 训练配置

- 40 epochs, batch 512, Adam lr=1e-3 cosine, grad clip 1.0;
- minibatch: 20% fire窗口(B:C=2:1) + 80% 航向平衡(heading-0 约55%, 非零均匀);
- fire loss 权重 2.0; speed 用温和类别权重; alt head(dim=1)不参与 CE;
- 模型: 与 ShootMaskModel 同构(encoder 256-256-128 + 4 action heads [3,5,1,2]), 108k 参数, GPU 训练约 3 分钟。

#### 离线指标 (val / test, best epoch=26)

- heading: acc 99.0%, macro-F1 0.979, 转向方向 99%, 五类 recall [0.997,0.975,0.99,0.992,0.994] — 无类别坍缩;
- speed acc 98.2%;
- fire (仅 fire_allowed 窗口): val precision 0.71 / recall 0.89 / F1 0.789; test precision 0.73 / recall 0.80;
- RLlib 兼容: BC 与 ShootMaskModel logits 逐 head max diff = 0 (只加载 policy trunk + action heads)。

#### 闭环评测 (100 局, difficulty=0, 确定性 argmax)

| 指标 | BC 首轮 | v19 基线 | 离散专家 |
| --- | --- | --- | --- |
| lost_target | 0% | 32% | 0% |
| WEZ 到达 | 100% | - | 100% |
| 发射/局 | 2.90 | 0.48 | 2.68 |
| 命中率 | 99.7% | 100% | 100% |
| 击杀率 | 40% | 10% | 23% |
| premium+good | 100% (203+87, bad=0) | - | - |
| time_to_WEZ | 13.7s | - | ~15s |

#### 结论

- 首轮 BC 闭环 lost 0%、击杀率 40%, 接近行为瓶颈已解决; 是否真正"超过"离散专家(23%)需 500 配对 seed 验证(2026-08-06 进行中);
- fire B/C 判别健康 (recall 0.8-0.9, precision 0.7+), 发射质量 100% 合格, 无需定向补数据;
- 下一步: 1-2 轮 DAgger(专家接管生成偏差-恢复状态) → 冻结 policy 的 critic warm-up → 低 LR + BC 锚定的 PPO 微调。



### 几何修复、基线归档与配对评测 (2026-08-06)

#### 基线归档

- 冻结 BC 首轮权重: data/expert/shoot_bc_round1_baseline.pth (sha256=07119a0571a46843edfecad68e746e61be8cd05fe08c5a934d53d9d4bc92aba0), 不可覆盖;
- 对应 100 局评测 JSON: results/shoot_eval/eval_bc_round1_100ep_baseline.json。

#### 几何修复 (可 seed 复现 + 场景参数化)

- BaseEnv.reset 保存 _reset_seed; SingleCombatShootTask.reset 改用 env seed 的 rng 生成几何 — 同一 seed 可精确复现同一初始状态(配对评测前提);
- 场景配置键: max_heading_bias_deg, chase_dist_min/max, lateral_max_m, lateral_sign, alt_diff_m, p_speed_range, t_speed_range, difficulty_level(目标行为);
- 移除生成脚本中无效的 set_geometry(其设置一直被 task.reset 覆盖); run_one 新增返回 bias_est(首帧估计), 审计 E 部分改用它分桶;
- 单元测试 10/10 通过 (tests/test_environment/test_geometry_seeding.py): 同 seed 复现(同 env/新 env)、不同 seed 差异、偏置 30/60/120、距离 5-8km、高度差 300、分离/接近 closure。

#### 配对评测 (进行中)

- scripts/eval_paired_bc_vs_expert.py: 500 个配对 seed, 专家与 BC 共用同一初始状态;
- 新增指标: 第一发命中率、第一发命中后击杀率、每杀导弹数、每发伤害、首发时间、WEZ→首发延迟、逐 seed BC−专家差值(kill/lost/launches);
- scripts/eval_scenario_matrix.py: 13 格 ID/OOD 矩阵(偏置 0-30/30-60/60-90/90-120、距离 2-3/5-8km、横向左/右/中心、高度差、closure 中性/分离、目标规避), 每格固定 seed 配对跑专家验收 + BC 零样本, 输出 expert-fail(不用于训练)/bc-fail(DAgger 目标)/ok。


### 500 配对 seed 评测结果 (2026-08-06)

同 seed 下专家与 BC 共享同一初始几何(difficulty=0, ID 分布: 偏置 30-60°、距离 2-5km)。

| 指标 | 离散专家 | BC | 配对差值 (BC−专家) |
| --- | --- | --- | --- |
| lost_target | 0% | 0% | 0 (500/500 平) |
| 击杀率 | 35.8% | 43.2% | **+7.4pp, 95% CI [+3.8, +11.0]pp** (bootstrap 20k) |
| 发射/局 | 2.89 | 3.01 | +0.12 (CI [+0.05, +0.20]) |
| 命中率 | 99.8% | 99.9% | — |
| 第一发命中率 | 98.0% | 98.0% | 0 |
| 第一发命中后击杀率 | 36.5% | 44.1% | +7.6pp |
| 每杀导弹数 | 4.0 | 4.0 | 0 |
| 首发时间 / WEZ→首发 | 114 / 45 steps | 113 / 44 steps | ≈0 |
| 发射质量 | bad 0, good 385, premium 1058 | bad 0, good 497, premium 1006 | 100% 合格 |

结论:
- BC 在 500 配对 seed 上击杀率显著高于专家(差值 CI 不含 0), 胜/负/平 = 62/25/413; lost 恒为 0;
- 之前 "40% vs 23%" 的非配对对比被专家 100 局低样本(23%)放大; 500 局下专家真实击杀率为 35.8%, BC 优势依然成立但幅度为 +7.4pp;
- 两者每杀均消耗全部 4 发(目标 4 HP), 发射质量 100% premium+good, 弹药利用率维度两策略相同。


### 场景矩阵首轮结果 (2026-08-06, 每格 12 局)

| 格子 | 专家 lost/kill/WEZ | BC lost/kill/WEZ | 备注 |
| --- | --- | --- | --- |
| id_bias30_60 | 0% / 50% / 100% | 0% / 42% / 100% | ID 基线 |
| bias0_30 | 0% / 42% / 100% | 0% / 50% / 100% | |
| bias60_90 | 0% / 50% / 100% | 0% / 42% / 100% | |
| bias90_120 | 0% / 42% / 100% | 0% / 50% / 100% | 宽偏置首次真实验证 |
| dist2_3k | 0% / 0% / 100% | 0% / 8% / 100% | 最近距开局, 两者击杀都低 |
| dist5_8k | 0% / 67% / 100% | 0% / 100% / 100% | BC 全杀 |
| alt_diff300 | 0% / 8% / 100% | 0% / 42% / 100% | BC 优势明显 |
| closure_neutral | 0% / 58% / 100% | 0% / 50% / 100% | |
| closure_separating | 0% / 50% / 100% | 0% / 67% / 100% | |
| target_evasive | 0% / 25% / 100% | 0% / 42% / 100% | difficulty=0.3 |
| lateral_left/right/center | 0% / 33% / 100% | 0% / 50-58% / 100% | |

- 13/13 格 verdict=ok (expert-fail=0, bc-fail=0): 所有格的 lost ≤15% 且 WEZ ≥80% 门禁通过;
- BC 在大多数格击杀率不低于专家, 在 dist5_8k / alt_diff300 / target_evasive / 横向格更高;
- dist2_3k 是最弱格(近距离开局留给转向-发射的窗口小), 但两者均未丢失目标;
- 样本量 12 局/格, 击杀率波动约 ±15pp — 仅作分诊; 最弱格(dist2_3k/alt_diff300/target_evasive)后续用更多 seed 复核并作为 stress-DAgger 重点。


### 场景矩阵扩展评测 (2026-08-06, 关键格100 seed / 其余50 seed, 与专家同seed配对)

| 格子 | 专家 lost/kill/WEZ | BC lost/kill/WEZ |
| --- | --- | --- |
| id_bias30_60 | 0% / 26% / 100% | 0% / 36% / 100% |
| bias0_30 | 0% / 22% / 100% | 0% / 26% / 100% |
| bias60_90 | 0% / 34% / 100% | 0% / 38% / 100% |
| bias90_120 | 0% / 26% / 100% | 0% / 38% / 100% |
| dist2_3k | 0% / 1% / 100% | 0% / 4% / 100% |
| dist5_8k | 0% / 58% / 100% | 0% / 84% / 100% |
| alt_diff300 | 0% / 16% / 100% | 0% / 35% / 99% |
| closure_neutral | 0% / 42% / 100% | 0% / 42% / 100% |
| closure_separating | 0% / 56% / 100% | 0% / 74% / 100% |
| target_evasive | 0% / 25% / 100% | 0% / 33% / 100% |
| lateral_left/right/center | 0% / 24-28% / 100% | 0% / 38-42% / 100% |

- 13/13 格 ok: 大样本下 BC lost 全为 0% (alt_diff300 有 1 局未达 WEZ 但未 lost), 无 OOD 退化;
- BC 击杀率在大多数格 ≥ 专家, 显著更高: dist5_8k(84 vs 58)、alt_diff300(35 vs 16)、closure_separating(74 vs 56)、bias90_120(38 vs 26)、lateral_left(42 vs 28);
- dist2_3k 确认属于交战窗口问题而非接近失败: WEZ 100% 但击杀仅 1-4%, 交由 fire 策略/PPO 分析, 不作为 recovery DAgger 目标。


### 纯 BC 压力评测报告 (2026-08-06, 无接管)

方法: 冻结 BC 与离散专家在同 seed 下分别注入扰动(注入窗口 step 150-525, 接近阶段), 危险=dist>12km 持续3步, 恢复=回到10km 持续10步; 扰动不污染 action mask(噪声仅 obs[:30])。

| 等级 | 扰动 | 专家 lost/kill | BC lost/kill | BC danger |
| --- | --- | --- | --- | --- |
| L0 | 无扰动 (100 seeds) | 0% / 30% | 0% / 38% | 0% |
| L1 (100) | hdg_k1 | 0% / 32% | 0% / 38% | 0% |
| L1 | delay_k1 | 0% / 31% | 0% / 39% | 0% |
| L1 | noise_small | 0% / 31% | 0% / 38% | 0% |
| L1 | spd_k1 | 0% / 31% | 0% / 38% | 0% |
| L2 (80) | hdg_k3 | 0% / 30% | 0% / 37.5% | 0% |
| L2 | delay_k3 | 0% / 31.2% | 0% / 37.5% | 0% |
| L2 | noise_med | 0% / 33.8% | 0% / 38.8% | 0% |
| L2 | target_turn30 | 0% / 31.2% | 0% / 33.8% | 0% |
| L3 (60) | combo_l3 | 0% / 26.7% | 0% / 31.7% | 0% |
| L3 | delay_k6 | 0% / 25% | 0% / 35% | 0% |
| L3 | target_turn45 | 0% / 21.7% | 0% / 36.7% | 0% |

结论:
- 12 种扰动下双方 lost 全为 0% (Wilson 95% 上界: L1 3.7% / L2 4.6% / L3 6.0%), danger=0%, BC 击杀率在所有扰动下 ≥ 专家;
- 按评审原则, 当前瞬态扰动(1-6步错误动作/延迟、30-60步小噪声、3-5s目标转向)被策略与底层控制器完全吸收, 未暴露真实失效模式 — 这些状态不是 DAgger 的有效目标;
- 下一步: 强化压力定位失效边界 — WEZ 时刻/对准关键期注入、更长扰动窗口(10-20步)、在弱格(dist2_3k/alt_diff300/target_evasive)上叠加扰动;
- dist2_3k 的击杀率问题明确归类为发射/交战窗口, 不进入 recovery DAgger。


### 强化压力确认评测 (2026-08-06, 60 seeds/扰动, L4 + WEZ时刻注入)

| 扰动 (60 seeds) | 专家 lost/kill | BC lost/kill | BC danger / 恢复 |
| --- | --- | --- | --- |
| hdg_k10 (接近期) | 0% / 21.7% | 0% / 30% | 0% |
| hdg_k20 (接近期) | 0% / 26.7% | 0% / 38.3% | **5% / 100%恢复** |
| delay_k10 | 0% / 23.3% | 0% / 33.3% | 0% |
| target_turn90 (40步45°) | 0% / 26.7% | 0% / 35% | 0% |
| combo_l4 | 0% / 26.7% | 0% / 33.3% | 0% |
| hdg_k10 @WEZ | 0% / 26.7% | 0% / 33.3% | 0% |
| hdg_k20 @WEZ | 0% / 25% | 0% / 28.3% | 0% |
| combo_l4 @WEZ | 0% / 13.3% | 0% / 18.3% | 0% |

结论:
- 全压力谱(20+ 扰动条件)下双方 lost 恒为 0% (Wilson 95% 上界 6.0%); BC 击杀在所有条件下 ≥ 专家;
- 唯一危险事件: hdg_k20 下 BC 3/60 局逼近 12km 边界, 全部自行恢复(恢复率 100%);
- 按评审停止准则("接管率和 stress lost 不再明显下降就应停止"), 当前扰动族不构成有效的 DAgger 数据源 — 2000+ 受扰局中可采集的"偏差→恢复"样本仅约 3 局;
- **建议: 跳过 recovery-DAgger, 进入 PPO 阶段**(分 head 锚定 + 极低熵 + 奖励逐级引入); 剩余问题(dist2_3k 交战窗口、击杀上限、发射时机)属于 fire 策略/奖励优化, 正是 PPO 的目标;
- 压力套件(L0-L4 + ID/OOD 矩阵)转为 PPO 阶段的永久回归集。


### dist2_3k fire oracle / 可达性审计 (2026-08-06, 60 seeds)

方法: 固定 BC 的 heading/speed 轨迹, 枚举发射时序策略(asap 合法即发 / delay_30/60 / dlz_mid/deep / interval_100), 与 BC 自身 fire 决策对比。

| 发射策略 | 击杀率 | 发射/局 | 达4发率 |
| --- | --- | --- | --- |
| **asap (合法即发)** | **73.3%** | 3.68 | 73.3% |
| delay_30 | 28.3% | 2.97 | 28.3% |
| delay_60 | 10.0% | 2.60 | 10.0% |
| dlz_mid (0.4-0.6) | 0.0% | 1.08 | 0.0% |
| dlz_deep (>=0.6) | 3.3% | 1.12 | 3.3% |
| interval_100 | 0.0% | 1.92 | 0.0% |
| 冻结 BC | 6.7% | 2.07 | 6.7% |

可达性审计 (asap): 首发 allowed 中位 75 步(15s); 全剧合法窗口累计仅约 4 步(每次穿过 DLZ 带约 1 步); 首发→第四发中位 783 步(157s); 16/60 未达 4 发 — 15 局 episode 结束时窗口仍开、1 局窗口不再开。

结论:
- **dist2_3k 不是不可达场景** — 环境允许 73% 的局完成 4 发齐射, asap oracle 击杀 73.3%;
- 瓶颈是 BC fire 策略过于保守: 只打 2.07/3.68 个合法窗口(大量窗口 DLZ 深度 <0.25 被 fire_desired 判为不值得发), 而近距交战窗口极短(约 1 步), 等待 premium 窗口=错过;
- delay/DLZ 选择性策略全部更差 → 近距离最优策略就是"合法即发";
- **PPO 设计确认**: fire-only 优化即可解锁 dist2_3k (目标行为≈asap), 无需解冻 heading/speed; fire 锚定应弱(λ_fire 0.05-0.1 按置信度缩放), 奖励课程用伤害增量+击杀即可把 fire 头推向早发。


### P0: RLlib 加载一致性门禁 (2026-08-06, 500 seeds) — PASS

- BC 权重装入真实 RLlib PPO policy(ShootMaskModel), 零梯度更新;
- 逐 head logits vs 独立 BC 模型: max diff < 1e-4 PASS;
- 500 seeds 闭环(explore=False): kills 43.2% / lost 0% / 3.01 发每局 / 命中 99.9%, 与归档基线配对评测完全一致;
- 逐 seed 匹配率: kill 100% / lost 100% / launches 100% — RLlib 加载路径无预处理/优化器/探索配置引入的隐性变化;
- RLlib checkpoint 已存档: marl_runs/shoot_bc_gate_s42/checkpoints/best (供 P1/P2 直接续训)。

下一步: P1 独立 critic warm-up(不向 BC encoder 回传梯度) → P2 fire-only PPO(冻结 encoder/heading/speed, entropy=0, 伤害增量+小击杀奖励, lr 1e-5-3e-5, clip 0.05, 每2-5轮快速回归)。


### P1: 独立 critic warm-up — PASS (2026-08-06)

- 实现: ShootFireOnlyModel(冻结BC actor + 独立 critic encoder/value); actor 全部 requires_grad_(False) 且不在 critic optimizer 中;
- rollout 混合场景: ID 65% / dist2_3k 20% / 其他OOD 15%, 400 局, 奖励=P2 阶段1(伤害+250/HP, kill+1000, bad-100, premium无加成, 发射成本0);
- 停止条件 6 项全部通过:
  1) val loss 平台(早停, 最优 9.53e3);
  2) explained variance 0.736(稳定为正);
  3) 价值排序 kill(85.3) > nokill(30.4);
  4) actor state-dict sha256 完全一致;
  5) 固定 obs 集 actor logits max diff = 0.00e+00;
  6) 100 固定 seeds 逐动作序列 + 击杀/lost/发射数完全一致;
- critic 存档: data/expert/shoot_critic_p1.pth (critic_encoder.* / critic_value.*, P2 直接加载)。

### ASAP fire 对照组 (2026-08-06) — 全局上限参考

heading/speed = 冻结 BC, fire = 环境 mask 合法即发:

| 场景 | BC 击杀 | ASAP 击杀 | ASAP 发射/局 | bad |
| --- | --- | --- | --- | --- |
| ID 500 seeds | 43.2% | **90.6%** | 3.89 | 0 |
| dist2_3k | 6.7% | **74.0%** | 3.68 | 0 |
| dist5_8k | 84% | **100%** | 4.00 | 0 |
| bias90_120 | 38% | **100%** | 4.00 | 0 |
| alt_diff300 | 35% | **76%** | 3.68 | 0 |
| target_evasive | 33% | **84%** | 3.84 | 0 |
| 其余 8 格 | 22-74% | **80-98%** | 3.76-3.98 | 0 |
| 压力 L1-L4 | 30-38% | **81.7-85%** | 3.82-3.83 | 0 |

- 全部场景 lost=0%、bad=0%、命中≈100%; ASAP 击杀率普遍比 BC 高 40-60pp;
- 结论: fire 策略是全局瓶颈, PPO 目标=逼近 ASAP("合法即发"); ASAP 作为 P2 规则基线(上限参考);
- P2 成功判据(与用户设定一致): dist2_3k 发射/局 2.07→3.5+, 达4发率 6.7%→上升, 击杀向 73-74% 靠近, 同时 ID/OOD/stress lost 保持 0, ID 击杀不显著低于 43.2%。


### P2A: fire-only ASAP 蒸馏 — PASS (2026-08-07)

实现: 冻结 BC encoder/heading/speed/alt, 仅训练 fire head; 训练样本=冻结BC轨迹中的 allowed 窗口(target fire=1), 推理时 mask 强制 disallowed=0; lr 1e-2(经验依据: BC fire head 的 no-fire logit 约 -10, 正样本 CE 在 1e-4 下每 epoch 仅移动 ~0.1 logit; mask 保证推理正确, 蒸馏目标=翻转为"全发"=ASAP)。

内置对比 (60 seeds, ID + dist2_3k):

| 策略 | ID 击杀 | ID 窗口利用率 | dist2_3k 击杀 | dist2_3k 窗口利用率 | bad |
| --- | --- | --- | --- | --- | --- |
| 冻结 BC | 33.3% | 7% | 6.7% | 6% | 0 |
| rule (ASAP) | 85.0% | 100% | 73.3% | 100% | 0 |
| **distilled** | **85.0%** | **100%** | **73.3%** | **100%** | 0 |

门禁全部通过: hdg/spd logits max diff=0.00e+00; 共享前缀动作序列一致; fire-agreement=True (窗口利用率 100%); lost=0; bad=0 → VERDICT PASS。

- 蒸馏模型已精确复现 ASAP 规则(ID/dist2_3k 击杀与 rule 完全一致), 存档 data/expert/shoot_bc_asap_distilled.pth;
- 完整回归(ID 500 / 13格 / L0-L4)进行中;
- 若蒸馏通过完整回归, 即完成当前阶段主要目标, 无需 PPO (PPO 仅在未来出现发射成本/窗口取舍/轨迹改变需求时启用)。


### P2A 蒸馏完整回归 (2026-08-07) — PASS

蒸馏模型 data/expert/shoot_bc_asap_distilled.pth 全套件结果:

| 套件 | lost | 击杀率 | 发射/局 | 备注 |
| --- | --- | --- | --- | --- |
| ID 500 seeds | 0% | **91.2%** | 3.90 | 命中 99.9%, bad=0, premium+good 100% (rule 90.6%, BC 43.2%) |
| 13 格 OOD (key 100/other 50) | 0% | **77-100%** | 3.71-4.00 | dist2_3k 77%, bias90_120/dist5_8k/separating 100% |
| 压力 L0-L4 | 0% | **81.7-91%** | — | L4 hdg_k20 危险 5% 全部自行恢复; BC 基线仅 30-38% |

- 蒸馏模型在所有套件 lost=0%、bad=0%, 击杀率与 ASAP 规则基本一致(ID 91.2% vs 90.6%, dist2_3k 77% vs 74-77%), 全面显著高于冻结 BC;
- **P2A 目标达成: 蒸馏模型稳定复现 ASAP 规则, 当前阶段无需 PPO**;
- 归档双基线: BC + rule fire override (results/shoot_eval/asap_baseline.json) 与 BC + distilled ASAP head (data/expert/shoot_bc_asap_distilled.pth), 已做同 seed 对比;
- P2B(PPO)仅在未来出现发射成本/弹药稀缺/窗口取舍/轨迹改变需求时启用, 届时从蒸馏权重初始化 + P1b critic 刷新。


### 失败局归因 (2026-08-07) — 假设确认: 3-hit timeout + 缺第四窗口

方法: 蒸馏模型 ID 500 seeds + dist2_3k 200 seeds, 逐局记录命中数/窗口/阻断条件; 修复冷却 off-by-one(发射步计数为 env._step_counter 自增后值)后归因完全一致。

| 指标 | ID (500) | dist2_3k (200) |
| --- | --- | --- |
| 击杀率 | 90.6% | 79.0% |
| 非击杀局 | 47 (9.4%) | 42 (21%) |
| 非击杀命中分布 | 3hit=40(85%), 2hit=6, 1hit=1 | 3hit=33(79%), 2hit=7, 1hit=2 |
| 末次命中→结束间隔 | 中位 672 步 (134s) | 中位 678 步 (136s) |
| 第四合法窗口出现率 | 4.3% (2/47) | 0% (0/42) |
| 窗口数分布 | 3窗=38, 2窗=6, 1窗=1 | 3窗=34, 2窗=7, 1窗=1 |
| 缺窗口阻断 | cooldown_timing 45/47 (96%) | cooldown_timing 42/42 (100%) |

冷却结束(≥30步)后的主导几何阻断(逐步计数):
- ID: ATA 17,708 / range_far 17,406 / closure 17,031 / range_close 8,455 — 主导 ATA, 但三者常同时失效; static_ok 后冷却 = 0
- dist2_3k: range_far 16,539 / ATA 16,158 / closure 15,754 — 主导 range_far; static_ok 后冷却 = 0

结论:
- 假设成立: 失败局几乎全部是 3-hit timeout, 第四合法窗口从未出现(或仅在冷却内短暂满足后消失);
- 根因不是 fire(ASAP 已 100% 利用窗口), 而是第三发之后追击机无法重新建立"ATA<15 & closure<0 & DLZ内"的交战几何 — 航向/速度/接近保持问题, 目标在交战后脱离交战几何且不再回到可发状态;
- **下一步: 不优化 fire; 针对 3-hit 失败局研究 range/ATA 保持策略, 更快产生第四窗口** — 这决定是否值得解冻 speed/heading(P2B 路线)。产物: results/shoot_eval/fail_attribution.json(+records), results/ctrl_viz/fail_attribution_*.png。


### hit3 几何崩溃点分析 (2026-08-08) — 100% 近距过冲, speed-first

方法: 收集 20 个 3-hit timeout 失败局, 对齐第三发命中时刻 t_hit3, 找 ATA/closure/range 首个持续失效顺序; 并从 t_hit3 / t_fire3(第三发发射) 做 heading/speed 反事实 oracle。

分型 (20/20 局):
- **T1_near_pass_overshoot 100%** (T2/T3/T4 = 0);
- 第一因时序: range 条件在 t_hit3+1 步失效(近距最小 7-36m, 追击机直接穿过目标) → ATA >90° 于 +35~44 步 → closure 于 +38~44 步恶化; 与"速度过高→近距过冲→掉头重来"完全吻合。

反事实 oracle (10 个 T1 seed, 从 t_fire3 干预):
| 干预 | 恢复第四窗口并击杀 |
| --- | --- |
| 原策略 | 0/10 |
| speed 降为 240/220/200 (任选) | **6/10** (seed 6,23,26,30,82,103; 其中 103 仅需 240) |
| heading 死区减半 (tight) | 0/10 |
| tight + speed 220/200 | 与 speed 单独相同, 无额外增益 |
| speed_hold / 240 | 部分, 但 200 是主要恢复档 |

结论与决策:
- 失败机制 = 第三发后高速近距过冲(100% 分型 + range 先失效); 修复方向 = **speed head(降低交战末段速度)**, heading 单独无效;
- speed 单独恢复 60%, 剩余 40% 即使 200 m/s 仍失败 — 需进一步试更早干预(第二发后)或更慢(180)或过冲后恢复航向;
- 按决策门禁: 以 speed-first 为下一步主线 — 优先解冻/重训 speed head(或先做"第三发后减速"规则蒸馏), heading 保持冻结; 若 40% 顽固失败在后续暴露 heading 需求再单独处理。
- 对齐面板: results/ctrl_viz/hit3_collapse/hit3_seed*.png; 数据: results/shoot_eval/hit3_collapse.json / hit3_counterfactual*.json。


### anti-overshoot 规则门禁与 sweep (2026-08-08)

可观测性审计 (41 维 obs):
- obs = 14 self + 10 target(含 closure/los_rate) + 6 missile-threat(本场景恒0) + 11 mask;
- **无命中次数/目标HP/剩余弹药/已发射数** → "第三发后"不可作为无状态规则条件(会产生冲突监督);
- 但过冲 precursor 完全可观测: distance, delta_speed(追击机-目标), ATA, closure, LOS rate → 几何触发规则可行;
- 结论: 规则用"近距 + 速度明显高于目标 + 强烈接近 + ATA仍小"触发减速, 不用 hit_count。

扩展反事实 sweep (运行中): 20 失败(3-hit timeout) + 12 成功(4-hit kill, control), 干预起点 {fire2, hit2, fire3, hit3, fire3-20/40/80} × 目标速度 {260,240,220,200,180}; 指标 P(kill|start,speed)、第四窗口恢复时间、新 lost、episode 长度; 成功组看退化(kill 保持率、time_to_kill p50/p90)。


### anti-overshoot sweep 结果 (2026-08-08) — 提前减速是核心, 240@fire3-80 最优

全量 20 失败(3-hit timeout) + 12 成功(4-hit kill, control) × {fire2,hit2,fire3,hit3,fire3-20/40/80} × {260..180}:

失败组 P(kill %) 关键行:
| 起点 | 260 | 240 | 220 | 200 | 180 |
| --- | --- | --- | --- | --- | --- |
| hit2 | 15 | 20 | 35 | 10 | 5 |
| fire3 | 5 | 5 | 30 | 20 | 5 |
| fire3-40 | 5 | 10 | 25 | 60 | 75 |
| **fire3-80** | 15 | **95** | **95** | **100** | **100** |

成功组 kill 保留率 (fire3-80): 260→100%, 240→100%, 220/200/180→91.7%; lost 全 0。

结论:
- **时机主导一切**: fire3-80(提前16s) >> fire3-40 >> fire3-20/hit3/fire3; 后期干预仅恢复 20-35%(之前 6/10 为挑选乐观值);
- **fire3-80@240 为最优**: 失败恢复 95%, 成功保留 100%, 无 lost, 击杀中位时间 ~236 步(原 ~900 步, 反而更快);
- 机制: 提前减速给机体足够响应时间, 第三发通过时避免近距过冲 → 冷却结束时几何仍成立 → 第四窗口出现;
- 注意: "fire3-80"是事件触发(不可观测), 最终规则需转成几何 precursor(近距+速度差+ATA+closure+los_rate 阈值), 下一轮用失败/成功轨迹网格搜索阈值;
- 决策: 规则可做(先 rule override 全回归, 通过后再蒸馏 speed head), 不支持 heading 解冻, 也不需 PPO。


### precursor 规则搜索与闭环验证 (2026-08-08) — 几何减速规则有效, 进入speed蒸馏

离线搜索: 正样本=20个T1失败局在fire3-120..-40时间带的precursor(range/delta_speed/closure/ata), 负样本=80个成功局同阶段+失败局早期; 分位数网格(6^4=1296候选), 严格过滤(recall>=90%/fpr_succ<=10%)0通过, top候选recall 70%/fpr_succ 8-18% — 固定阈值离线分离不足, 但闭环才是最终标准。

闭环 rule override (触发→act[0]=-20 if p_spd>240 else 0, 迟滞退出 speed<=245/range离开/closure离开; heading/fire冻结):

| 规则 | 失败救回 | 成功保留 | 盲ID击杀 | 盲d2击杀 | 3-hit(ID/d2) | lost/bad | duty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 (3470,118,117,5) | 75% | 100% | 98% | 92% | 2/3 | 0/0 | 3-4.4% |
| **r2 (4000,90,100,10)** | **90%** | **100%** | **99%** | **90%** | **0/3** | **0/0** | 11.5-17.8% |
| r3 (4400,100,110,8) | 20% | 100% | 93% | 74% | 6/10 | 0/0 | 7.8-16% |

结论与决策:
- **r2 满足门禁**: 救回90%(>=80%), 成功保留100%(>=97%), lost/bad=0, 3-hit timeout ID 9.4%->0% / d2 21%->6%, ttk p50 ~175s不恶化, 总体击杀提升(ID 91.2%->99%盲集, d2 79%->90%盲集);
- 规则本质= "高速尾追下近距anti-overshoot控制律": range<4000 & 追击机快90+ & 强接近(closure<-100) & ATA<10 → 减速至240;
- duty 12-18%为中等触发率(不是重写策略); r1更局部(4%)但救回仅75%;
- **决策: 停止规则搜索, 进入 speed-head 蒸馏**(监督目标=r2式减速; 冻结encoder/heading/fire, 仅speed head, 10-20% anti-overshoot样本, lr 1e-4-5e-4);
- 不需要heading解冻, 不需要PPO。
