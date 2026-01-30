# DWTS项目提交包

## 目录结构

```
submission/
├── code/           # 源代码
├── data/           # 输入数据
├── results/        # 输出结果（CSV文件）
├── figures/        # 可视化图表（PNG文件）
├── docs/           # 文档
├── run_all.py      # 主运行脚本
└── README.md       # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r docs/requirements.txt
```

### 2. 运行完整分析

```bash
python run_all.py
```

这将依次执行所有分析阶段，生成所有结果和图表。

### 3. 单独运行各阶段

```bash
# Phase 1: 数据预处理
python code/preprocessing_pipeline.py

# Phase 2: SMC观众投票反演
python code/smc_fan_vote_estimator.py

# Phase 3: 平行宇宙仿真
python code/multiverse_simulator.py

# Phase 4: 特征归因分析
python code/feature_attribution.py

# Phase 5: 终极赛制设计
python code/ultimate_system_design.py

# Phase 6: 模型验证
python code/model_validation.py

# Phase 7: 参数灵敏度分析
python code/sensitivity_analysis.py

# Phase 8: 生成可视化
python code/create_paper_visualizations.py
```

## 输出文件说明

### Results (CSV文件)
- `Processed_DWTS_Long_Format.csv` - 预处理后的数据
- `Q1_Estimated_Fan_Votes.csv` - Q1&Q2答案（观众投票反演）
- `Simulation_Results_Q3_Q4.csv` - Q3&Q4答案（平行宇宙仿真）
- `Q5_Feature_Importance.csv` - Q5答案（特征归因）
- `Q6_New_System_Simulation.csv` - Q6答案（终极赛制）
- `Sensitivity_Grid_Search.csv` - 参数灵敏度分析结果

### Figures (PNG图表)
- 数据预处理：`judge_score_distribution.png`
- SMC分析：`smc_uncertainty_analysis.png`, `season1_trajectories.png`, `q1_particle_cloud.png`
- 平行宇宙：`multiverse_analysis.png`, `q3_sankey_chaos.png`
- 特征归因：`q5_feature_importance.png`, `q5_partner_influence.png`, `q5_tornado_plot.png`
- 终极赛制：`q6_rank_distribution.png`, `q6_injustice_comparison.png`, `q6_case_study.png`
- 模型验证：`model_validation_cv.png`, `model_validation_residuals.png`, `model_validation_robustness.png`
- 参数分析：`sensitivity_heatmap.png`, `sensitivity_3d.png`
- 综合仪表板：`q5_q6_dashboard.png`

## 文档

### 核心文档（docs/目录）
- `FINAL_PROJECT_SUMMARY.md` - 项目总结
- `Q3_Q4_ANSWERS.md` - Q3&Q4详细答案
- `Q5_Q6_ANSWERS.md` - Q5&Q6详细答案
- `模型检验模块.md` - 模型验证文档
- `参数灵敏度分析报告.md` - 参数分析报告
- `核心结论速查表.md` - 关键发现速查

### 战略文档（submission/根目录）
- `F_O奖战略总目录.md` - 战略文档总索引 ⭐⭐⭐⭐⭐
- `一页纸作战地图.md` - 快速参考版（打印贴墙上）⭐⭐⭐⭐⭐
- `O奖战略指南.md` - 核心战略，反驳M奖评估
- `论文摘要模板_直接可用.md` - 可直接复制的摘要
- `可视化展示策略.md` - 图表优化方案
- `获奖潜力评估.md` - 原始评估（已被战略文档升级）

## 系统要求

- Python 3.8+
- 依赖包：pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, tqdm

## 联系方式

如有问题，请参考文档或联系项目团队。

---

**项目完成日期：** 2026年1月30日  
**数据来源：** DWTS Season 1-34  
**状态：** ✅ 全部完成
