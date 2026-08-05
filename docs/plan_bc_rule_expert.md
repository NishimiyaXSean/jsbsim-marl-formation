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