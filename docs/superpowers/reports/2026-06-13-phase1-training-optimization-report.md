# Phase 1 训练优化 — 阶段性实施报告

**日期**: 2026-06-13  
**分支**: `worktree-phase1-training-optimization` → `master`  
**基准提交**: `fb31ed7` (Add Phase 1 training optimization design spec)  
**设计文档**: `docs/superpowers/specs/2026-06-13-phase1-training-optimization-design.md`  
**实施计划**: `docs/superpowers/plans/2026-06-13-phase1-training-optimization.md`

---

## 一、总体进度

| 阶段 | 任务 | 状态 | 提交数 |
|------|------|------|--------|
| Phase 1a | Task 1: PN Guidance Expert in ResidualExpertWrapper | ✅ 完成 | 2 |
| Phase 1a | Task 2: Reward Function Tuning | ✅ 完成 | 1 |
| Phase 1a | Task 3: 5-Stage Float Curriculum in Environment | ✅ 完成 | 2 |
| Phase 1a | Task 5: Phase 1a Validation — Quick Training Run | ⏳ 待执行 | — |
| Phase 1b | Task 4: Update Training Config and CurriculumCallback | ✅ 完成 | 2 |
| Phase 1b | Task 6: Hyperparameter Sweep Runner | ✅ 完成 | 1 |
| Phase 1b | Task 7: Multi-Seed Evaluation Enhancement | ✅ 完成 | 1 |
| Phase 1b | Task 8: End-to-End Integration Test | ✅ 完成 | 1 |

**代码变更**: 5 个文件, +2013 / −96 行, 10 次提交

---

## 二、提交历史

```
4e93ad6 feat: multi-seed evaluation with Wilson CI, CSV export, --stage flag
ca10ad6 feat: hyperparameter sweep runner (8 configs + top-2 x 5 seeds)
a6bac6e fix: add fallback when current_stage not found in CURRICULUM_STAGES
2f835f5 feat: updated training config, 5-stage curriculum callback, train_with_config export
d2ba2f3 fix: heading rate ranges match spec limits for stages 1.5 and 2.0
185d3f8 feat: 5-stage float curriculum — 1.0/1.5/2.0/2.5/3.0 with progressive difficulty
3e4e891 feat: reward function tuning — stronger progress/ATA, terminal boost, time pressure
16c63de fix: eval loop uses wrapper.step() instead of duplicating combination logic
1fc2d72 feat: PN guidance expert in ResidualExpertWrapper
e97ffee Create 2026-06-13-phase1-training-optimization.md (implementation plan)
```

---

## 三、代码变更详情

### Phase 1a: PN Expert + Reward Tuning

#### Task 1: PN Guidance Expert

**文件**: `scripts/train_single_pursuit.py`

- `_compute_expert()` 改用比例导航制导 (Proportional Navigation)
- 通过 `_base_env` 直接读取 `pursuer.position_ned`, `pursuer.velocity_ned`, `target_ac.position_ned`, `target_ac.velocity_ned`
- 调用 `compute_pn_heading()` with `dt=0.5, nav_constant=3.0, max_turn_rate_dps=15.0`
- P-controller: `heading_error * 0.05`, clipped to `[-0.3, 0.3]`
- 返回值: `np.array([aileron, 0.0, 1.0], dtype=np.float32)` — 仅控制横滚, 满油门
- 修复: 评估循环改用 `wrapper.step(action)` 替代手动组合专家+残差

#### Task 2: Reward Function Tuning

**文件**: `src/environment/single_pursuit_env.py`

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `REWARD_PROGRESS` | 2.0 | **5.0** | 更强的接近信号 |
| `REWARD_ATA` | 3.0 | **5.0** | 更强的指向激励 |
| Terminal boost (NEW) | — | **progress × 3** (dist < 500m) | 鼓励末端激进追击 |
| Time pressure (NEW) | — | **−0.5 × (t/120) × dt** | 惩罚悠闲接近 |

未改动: `REWARD_SUCCESS=500`, `REWARD_CRASH=-200`, `REWARD_LOST_TARGET=-200`, `REWARD_GROUND_WARNING=2.0`

#### Task 3: 5-Stage Float Curriculum

**文件**: `src/environment/single_pursuit_env.py`

`curriculum_stage` 从 `int` 改为 `float`, 支持 5 个阶段:

| Stage | Bearing | Init Dist (m) | Target Speed (m/s) | Heading Rate (°/s) | Alt Rate (m/s) |
|-------|---------|---------------|---------------------|---------------------|----------------|
| 1.0 | 0° | 800–1800 | 130 | 0 | 0 |
| 1.5 | ±7° | 900–2000 | 145 | ±3 | ±1.5 |
| 2.0 | ±15° | 1000–2500 | 160 | ±10 | ±3 |
| 2.5 | ±30° | 1200–2700 | 170 | ±15 | ±5 |
| 3.0 | ±45° | 1500–3000 | 180 | ±20 | ±8 |

- 使用 `np.isclose()` 进行浮点阶段匹配
- 修复: Stage 1.5 / 2.0 航向速率范围符合规范限制

### Phase 1b: Curriculum + 评估工具

#### Task 4: Training Config and CurriculumCallback

**文件**: `scripts/train_single_pursuit.py`

| 参数 | 旧值 | 新值 |
|------|------|------|
| `TOTAL_TIMESTEPS` | 200,000 | **500,000** |
| `CURRICULUM_STAGES` | [1, 2, 3] | **[1.0, 1.5, 2.0, 2.5, 3.0]** |
| `EVAL_EPISODES` | 20 | **30** |
| `EVAL_FREQ` | 10,000 | **15,000** |
| Threshold | 40% uniform | **40% (Stage 1–2) / 50% (Stage 2–3)** |
| PPO `batch_size` | 128 | **256** |
| PPO `net_arch` | [64, 64] | **[128, 128]** |

- `CurriculumCallback` 记录 `intercept_times` 和 `_eval_metrics` 列表
- 训练后自动保存 `eval_metrics.csv` (timesteps, stage, capture_rate, avg_min_dist, avg_intercept_time)
- 新增 `train_with_config()` 导出函数 (接受所有超参数, 供 sweep runner 调用)
- 修复: `next()` 在阶段未找到时的 StopIteration 回退

#### Task 6: Hyperparameter Sweep Runner

**文件**: `scripts/run_hyperparam_sweep.py` (新建, 301 行)

正交超参数搜索 — 4 参数 × 2 水平, 分数因子设计 = **8 configs**:

| 参数 | Low (−1) | High (+1) |
|------|----------|-----------|
| `lr` | 1e-4 | 3e-4 |
| `ent_coef` | 0.005 | 0.01 |
| `net_arch_pi` | [128, 128] | [256, 128] |
| `n_steps` | 2048 | 4096 |

固定参数: `batch_size=256`, `gamma=0.99`, `clip_range=0.2`, `total_timesteps=500k`, `vf_coef=0.5`, `max_grad_norm=0.5`

**流程**:
1. 8 configs × seed=0, 500k steps
2. 按 Stage 3 capture rate 排序, 选 top-2
3. top-2 × 5 seeds (0–4), 500k steps
4. 输出 `report.csv` + `summary.txt`

#### Task 7: Multi-Seed Evaluation Enhancement

**文件**: `scripts/evaluate_and_visualize.py` (+173 / −3 行)

新增功能:
- `--multi-seed`: 跨多个模型目录聚合统计
- `--stage`: 指定评估课程阶段 (默认 1.0)
- `--model-dirs`: 手动指定模型目录列表
- `--csv`: 导出 CSV 文件
- `wilson_ci()`: Wilson Score 置信区间计算
- `run_multi_seed_evaluation()`: 多种子评估核心逻辑
- `print_multi_seed_summary()`: 格式化汇总表格

CSV 导出列: model_dir, capture_rate, ci_low, ci_high, avg_min_dist, std_min_dist, avg_intercept_time

---

## 四、代码质量审查结果

| 审查维度 | 项目数 | 结果 |
|----------|--------|------|
| PN expert 规范符合度 | 8 项 | ✅ 全部通过 |
| Reward tuning 规范符合度 | 5 项 | ✅ 全部通过 |
| 5-stage curriculum 规范符合度 | 20+ 项 | ✅ 通过 (修复 2 处 heading rate 超限) |
| Training config 规范符合度 | 13 项 | ✅ 全部通过 |
| Sweep runner 规范符合度 | 9 项 | ✅ 全部通过 |
| Multi-seed eval 规范符合度 | 10 项 | ✅ 全部通过 |

已修复的代码质量问题:
1. 评估循环中 expert + residual 组合逻辑重复 → 改用 `wrapper.step(action)`
2. `next()` 无回退值 → 添加 `None` 回退 + 警告
3. Stage 1.5/2.0 heading rate 超出规范限制 → 修正为 `uniform(-3,3)` / `uniform(-10,10)`

---

## 五、集成测试

```
Test 1: All imports OK
Test 2: Config constants OK (500k steps, 5 stages, EVAL_EP=30, REWARD_PROGRESS=5.0)
Test 3: Sweep generates 8 configs OK
Test 4: Wilson CI OK (p=0.5, lo<p<hi verified)
Test 5: 5 stage env construction OK (1.0, 1.5, 2.0, 2.5, 3.0)
Test 6: train_with_config 1000 steps → best_model.zip + eval_metrics.csv ✅
ALL 6 INTEGRATION TESTS PASSED
```

**1000-step 集成训练输出** (`eval_metrics.csv`):

```csv
timesteps,stage,capture_rate,avg_min_dist,avg_intercept_time
256,1.0,0.40,376m,82.6s
512,1.5,0.00,621m,120.0s
768,1.5,0.13,501m,112.4s
1024,1.5,0.07,573m,115.3s
```

- 1 次评估即完成 Stage 1.0→1.5 进阶 (capture 40% ≥ 40% threshold)
- PN expert + reward tuning 在 Stage 1.0 上工作正常

---

## 六、待执行训练任务

### Phase 1a 验证训练

```bash
conda activate jsbsim_rl
JSBSIM_DEBUG=0 python scripts/train_single_pursuit.py --seed 0 --steps 200000
```

**目标**: Stage 1 capture rate ≥ 50%, 无 JSBSim NaN 崩溃  
**预计耗时**: ~30 分钟

### Phase 1b 完整超参数搜索

```bash
JSBSIM_DEBUG=0 python scripts/run_hyperparam_sweep.py
```

**目标**: Top config Stage 3 capture rate ≥ 90% (5-seed mean)  
**预计耗时**: ~12 小时 (18 configs × 500k steps)

### 多种子评估

```bash
python scripts/evaluate_and_visualize.py --multi-seed --stage 3.0 --episodes 30
```

**输出**: Wilson CI 置信区间 + CSV 报告

---

## 七、成功标准

| # | 标准 | 目标 | 状态 |
|---|------|------|------|
| 1 | Phase 1a 快速检查: Stage 1 capture ≥ 50% (PN expert) | ≥ 50% | ⏳ 待训练 |
| 2 | Phase 1b 完成: Top config Stage 3 capture ≥ 90% (5-seed) | ≥ 90% | ⏳ 待 sweep |
| 3 | 评估报告: CSV + Wilson CI for top-2 configs | 完成 | ⏳ 待 sweep |

---

## 八、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| PN expert 导致 Stage 1 退步 | 降低 nav_constant 至 2.5, 增大 P-gain 阻尼 |
| 500k steps 不足以达到 Stage 3 | 可延长至 750k–1M steps |
| JSBSim NaN 在激进 PN 转弯时崩溃 | aileron 已限制 ±0.3, 监控飞行包线 |
| 5-stage 拖慢训练速度 | EVAL_FREQ 调高至 15k, 阶段自然门控进度 |

---

## 九、文件清单

| 文件 | 变更类型 | 行数变化 |
|------|----------|----------|
| `scripts/train_single_pursuit.py` | 修改 | +236 |
| `src/environment/single_pursuit_env.py` | 修改 | +104 |
| `scripts/run_hyperparam_sweep.py` | **新建** | +301 |
| `scripts/evaluate_and_visualize.py` | 修改 | +173 / −3 |
| `docs/superpowers/plans/2026-06-13-phase1-training-optimization.md` | 新建 | +1295 |
