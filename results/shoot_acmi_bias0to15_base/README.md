# shoot_acmi_bias0to15_base

初始角偏置 `|bias| ∈ [0,15°)` 箱内的**最边缘**一集：seed **441**，实测
`init_heading_bias_deg = +0.0321°`。新几何 `U(0,60)`，`difficulty_level = 0.0`，
权重 `data/expert/shoot_bc_asap_distilled.pth`（**冻结基线**，非 geomA），
`MAX_STEPS 1500`。

结果：`target_killed`，798 步，4 发 4 中，min_dist 66.4 m，`wez_first_step = 0`
（偏置≈0 时开局即在发射包线内，属该区段的固有特征）。

**配对文件**：`../shoot_acmi_bias0to15_geomA/` —— 同一 seed、同一场景，只把权重换成
`shoot_bc_asap_geomA.pth`（813 步，同样 KILL）。两者构成 matched pair。

溯源与逐字段核验见 `results/shoot_eval/acmi_inventory_report.md` §6。
用 Tacview 打开 `shoot_bc_s441_d00.acmi`。
