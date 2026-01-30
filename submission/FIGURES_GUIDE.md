# 图表完全说明指南 | Complete Figures Guide

## 📊 图表总览

本项目共生成 **20张** 核心图表，涵盖数据预处理、SMC分析、平行宇宙仿真、特征归因、终极赛制设计、模型验证和参数分析。

---

## 1️⃣ 数据预处理图表

### `judge_score_distribution.png`
**裁判分数分布直方图**

**用途**: 展示所有赛季裁判分数的分布情况

**关键信息**:
- X轴: 裁判平均分（Judge_Avg_Score）
- Y轴: 频次
- 分布: 近似正态分布，中心在25-30分
- 范围: 10-30分

**论文使用**: 
- 数据描述部分
- 说明数据质量和分布特征

---

## 2️⃣ SMC分析图表（Q1 & Q2）

### `smc_uncertainty_analysis.png`
**SMC不确定性分析**

**用途**: 展示观众投票估计的不确定性分布

**关键信息**:
- 平均不确定性: 8.5%
- 范围: 0.4% - 21.6%
- 分布: 大部分周次不确定性在5-10%之间

**论文使用**:
- Q2答案的可视化证据
- 说明估计的可靠性

### `season1_trajectories.png`
**Season 1选手轨迹图**

**用途**: 展示Season 1所有选手的观众投票随时间变化

**关键信息**:
- X轴: 周次（Week）
- Y轴: 观众投票比例
- 每条线代表一位选手
- 可以看到选手人气的起伏

**论文使用**:
- 案例研究
- 展示SMC算法的输出

### `q1_particle_cloud.png`
**粒子云可视化**

**用途**: 展示SMC算法中粒子的分布

**关键信息**:
- 每个点代表一个粒子
- 颜色深浅表示权重
- 展示算法的工作原理

**论文使用**:
- 方法论部分
- 说明SMC算法的可视化

### `season_statistics.png`
**赛季统计图**

**用途**: 展示各赛季的统计信息

**关键信息**:
- 选手数量
- 周次数量
- 平均分数

**论文使用**:
- 数据描述部分

---

## 3️⃣ 平行宇宙图表（Q3 & Q4）

### `multiverse_analysis.png`
**平行宇宙分析**

**用途**: 对比三个平行宇宙（排名制、百分比制、拯救机制）的淘汰结果

**关键信息**:
- 逆转率: 100%（排名制 vs 百分比制）
- 拯救率: 23.11%（拯救机制改变结果的比例）
- 冤案率: 94.70%（被淘汰者不是裁判最低分的比例）

**论文使用**:
- Q3&Q4答案的核心证据
- 说明赛制的不公平性

### `q3_sankey_chaos.png`
**淘汰流向图（Sankey图）**

**用途**: 展示不同规则下淘汰结果的流向

**关键信息**:
- 左侧: 排名制淘汰的选手
- 右侧: 百分比制淘汰的选手
- 流向: 展示有多少选手在两种规则下结果不同

**论文使用**:
- Q3答案的可视化
- 直观展示100%逆转率

---

## 4️⃣ 特征归因图表（Q5）

### `q5_feature_importance.png`
**特征重要性排名**

**用途**: 展示影响裁判分数和观众投票的关键因素

**关键信息**:
- **裁判分数模型**:
  - 年龄: -0.494（最大影响）
  - Derek Hough: +0.187（金牌舞伴）
  - R²: 28.28%
- **观众投票模型**:
  - 年龄: -0.007
  - R²: 11.04%（更难预测）

**论文使用**:
- Q5答案的核心证据
- 说明年龄和舞伴的重要性

### `q5_partner_influence.png`
**舞伴影响分析**

**用途**: 展示不同舞伴对裁判分数的影响

**关键信息**:
- Derek Hough: +0.187（最高）
- 其他舞伴的影响系数
- 横轴: 影响系数
- 纵轴: 舞伴名字

**论文使用**:
- Q5答案的补充证据
- 说明舞伴选择的重要性

### `q5_tornado_plot.png`
**龙卷风图（Tornado Plot）**

**用途**: 展示各特征对结果的影响范围

**关键信息**:
- 横轴: 影响范围
- 纵轴: 特征名称
- 条形长度: 影响的大小

**论文使用**:
- Q5答案的可视化
- 直观展示特征重要性

---

## 5️⃣ 终极赛制图表（Q6）

### `q6_rank_distribution.png`
**排名分布对比**

**用途**: 对比旧系统和新系统下被淘汰者的裁判排名分布

**关键信息**:
- **旧系统**: 平均裁判排名 2.59
- **新系统**: 平均裁判排名 8.21
- **改善**: +5.62（越高越好，说明被淘汰的是技术更差的选手）

**论文使用**:
- Q6答案的核心证据
- 说明新系统的改善效果

### `q6_injustice_comparison.png`
**冤案率对比**

**用途**: 对比旧系统和新系统的冤案率

**关键信息**:
- **旧系统**: 94.70%
- **新系统**: 93.43%
- **改善**: -1.26%

**论文使用**:
- Q6答案的核心证据
- 说明新系统降低了冤案率

### `q6_case_study.png`
**案例研究**

**用途**: 展示具体案例，说明新系统如何改变淘汰结果

**关键信息**:
- 选择几个典型案例
- 对比旧系统和新系统的淘汰决策
- 说明新系统的优势

**论文使用**:
- Q6答案的案例证据
- 增强说服力

### `q6_eliminated_profile.png`
**被淘汰者画像**

**用途**: 展示被淘汰者的特征分布

**关键信息**:
- 年龄分布
- 裁判排名分布
- 观众投票排名分布

**论文使用**:
- Q6答案的补充分析
- 说明被淘汰者的特征

---

## 6️⃣ 模型验证图表

### `model_validation_cv.png`
**交叉验证结果**

**用途**: 展示模型的交叉验证性能

**关键信息**:
- 裁判分数模型R²: 28.28%
- 观众投票模型R²: 11.04%
- 5折交叉验证

**论文使用**:
- 方法论部分
- 说明模型的可靠性

### `model_validation_residuals.png`
**残差分析**

**用途**: 展示模型预测的残差分布

**关键信息**:
- 残差应该接近正态分布
- 无明显模式说明模型合理

**论文使用**:
- 方法论部分
- 验证模型假设

### `model_validation_robustness.png`
**鲁棒性分析**

**用途**: 展示模型在不同数据子集上的性能

**关键信息**:
- 不同赛季的性能
- 不同选手类型的性能

**论文使用**:
- 方法论部分
- 说明模型的泛化能力

---

## 7️⃣ 参数分析图表

### `sensitivity_heatmap.png`
**参数灵敏度热力图**

**用途**: 展示不同参数组合对结果的影响

**关键信息**:
- X轴: 裁判权重
- Y轴: 观众权重
- 颜色: 冤案率
- 最优点: 70/30权重

**论文使用**:
- Q6答案的补充分析
- 说明参数选择的合理性

### `sensitivity_3d.png`
**3D参数空间**

**用途**: 3D展示参数对结果的影响

**关键信息**:
- X轴: 裁判权重
- Y轴: 观众权重
- Z轴: 冤案率
- 可以看到参数空间的全貌

**论文使用**:
- Q6答案的可视化
- 增强理解

---

## 8️⃣ 综合图表

### `q5_q6_dashboard.png`
**Q5&Q6综合仪表板**

**用途**: 一张图展示Q5和Q6的所有关键信息

**包含内容**:
1. 特征重要性排名
2. 舞伴影响分析
3. 排名分布对比
4. 冤案率对比

**论文使用**:
- 结果部分的总结图
- 一张图说明所有关键发现

---

## 📝 图表使用建议

### 论文结构建议

**1. Introduction**
- 使用 `judge_score_distribution.png` 介绍数据

**2. Methodology**
- 使用 `q1_particle_cloud.png` 说明SMC算法
- 使用 `model_validation_cv.png` 说明模型验证

**3. Results - Q1&Q2**
- 使用 `smc_uncertainty_analysis.png` 展示不确定性
- 使用 `season1_trajectories.png` 展示案例

**4. Results - Q3&Q4**
- 使用 `multiverse_analysis.png` 展示核心发现
- 使用 `q3_sankey_chaos.png` 展示流向

**5. Results - Q5**
- 使用 `q5_feature_importance.png` 展示特征重要性
- 使用 `q5_partner_influence.png` 展示舞伴影响

**6. Results - Q6**
- 使用 `q6_rank_distribution.png` 展示改善效果
- 使用 `q6_injustice_comparison.png` 展示冤案率降低
- 使用 `sensitivity_heatmap.png` 说明参数选择

**7. Discussion**
- 使用 `q5_q6_dashboard.png` 总结关键发现

### 图表质量

所有图表均为：
- **分辨率**: 300 DPI（适合论文打印）
- **格式**: PNG（无损压缩）
- **尺寸**: 10x6英寸（标准论文图表尺寸）
- **字体**: 清晰可读
- **颜色**: 色盲友好配色

### 图表说明模板

**Figure X: [图表标题]**

**Description**: [简短描述图表内容]

**Key Findings**: 
- [关键发现1]
- [关键发现2]
- [关键发现3]

**Interpretation**: [解释图表的含义和重要性]

---

## 🎯 核心图表优先级

### 必须使用（⭐⭐⭐⭐⭐）
1. `multiverse_analysis.png` - Q3&Q4核心证据
2. `q5_feature_importance.png` - Q5核心证据
3. `q6_rank_distribution.png` - Q6核心证据
4. `q6_injustice_comparison.png` - Q6核心证据
5. `smc_uncertainty_analysis.png` - Q2核心证据

### 强烈推荐（⭐⭐⭐⭐）
6. `q5_q6_dashboard.png` - 综合总结
7. `q3_sankey_chaos.png` - Q3可视化
8. `q5_partner_influence.png` - Q5补充证据
9. `sensitivity_heatmap.png` - Q6参数分析
10. `season1_trajectories.png` - Q1案例

### 可选使用（⭐⭐⭐）
11. `judge_score_distribution.png` - 数据描述
12. `model_validation_cv.png` - 方法验证
13. `q6_case_study.png` - Q6案例
14. `q1_particle_cloud.png` - 方法说明
15. 其他图表

---

## 📊 图表数据来源

所有图表的数据来源：
- **原始数据**: `data/2026 MCM Problem C Data.csv`
- **处理后数据**: `results/Processed_DWTS_Long_Format.csv`
- **Q1&Q2数据**: `results/Q1_Estimated_Fan_Votes.csv`
- **Q3&Q4数据**: `results/Simulation_Results_Q3_Q4.csv`
- **Q5数据**: `results/Q5_Feature_Importance.csv`
- **Q6数据**: `results/Q6_New_System_Simulation.csv`

---

## 🔧 重新生成图表

如需重新生成图表：

```bash
# 生成所有图表
python code/create_paper_visualizations.py

# 生成特定图表
python code/visualize_q5_q6.py
```

---

## 📞 技术支持

如有图表相关问题，请参考：
- **PROJECT_GUIDE.md** - 项目快速指南
- **docs/FINAL_PROJECT_SUMMARY.md** - 完整项目总结
- **code/create_paper_visualizations.py** - 图表生成代码

---

**图表生成日期**: 2026年1月30日  
**图表总数**: 20张  
**状态**: ✅ 全部完成
