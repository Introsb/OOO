# 论文写作快速指南 | Paper Writing Guide

## 📝 文档结构总览

项目已整理完毕，现在只保留核心文档。以下是论文写作时的文档使用指南。

---

## 🎯 核心文档（必读）

### 1. **PROJECT_GUIDE.md** ⭐⭐⭐⭐⭐
**用途**: 项目快速指南，了解全局  
**包含内容**:
- 项目概览和快速开始
- 6个核心问题的答案总结
- 项目结构和文件说明
- 核心发现和技术亮点

**论文写作用途**:
- Introduction部分：项目背景和目标
- Abstract：核心发现总结
- Conclusion：主要成果

### 2. **FIGURES_GUIDE.md** ⭐⭐⭐⭐⭐
**用途**: 所有图表的完整说明  
**包含内容**:
- 20张图表的详细说明
- 每张图表的用途和关键信息
- 图表使用优先级
- 论文中如何使用图表

**论文写作用途**:
- Results部分：选择合适的图表
- Figure captions：参考图表说明
- Discussion：解释图表含义

### 3. **docs/FINAL_PROJECT_SUMMARY.md** ⭐⭐⭐⭐⭐
**用途**: 完整的项目总结  
**包含内容**:
- 详细的方法论
- 完整的结果分析
- 理论贡献和实践启示
- 性能指标和未来改进

**论文写作用途**:
- Methodology部分：详细方法
- Results部分：完整结果
- Discussion部分：理论贡献

---

## 📊 问题答案文档

### 4. **docs/Q3_Q4_ANSWERS.md** ⭐⭐⭐⭐
**用途**: Q3和Q4的详细答案  
**包含内容**:
- Q3：赛制是否公平？（100%逆转率）
- Q4：拯救机制是否有效？（23.11%拯救率）
- 详细的数据分析和证据

**论文写作用途**:
- Results - Q3&Q4部分
- 提供详细的数据支持

### 5. **docs/Q5_Q6_ANSWERS.md** ⭐⭐⭐⭐
**用途**: Q5和Q6的详细答案  
**包含内容**:
- Q5：影响成败的关键因素（年龄、舞伴）
- Q6：如何设计更公平的赛制（70/30权重）
- 详细的特征分析和赛制设计

**论文写作用途**:
- Results - Q5&Q6部分
- 提供详细的分析和设计方案

---

## 🔬 技术文档

### 6. **docs/SMC_README.md** ⭐⭐⭐
**用途**: SMC算法的详细说明  
**包含内容**:
- SMC算法原理
- 实现细节
- 验证结果

**论文写作用途**:
- Methodology - SMC部分
- 说明算法的创新点

### 7. **docs/模型检验模块.md** ⭐⭐⭐
**用途**: 模型验证的详细说明  
**包含内容**:
- 交叉验证
- 残差分析
- 鲁棒性测试

**论文写作用途**:
- Methodology - Validation部分
- 证明模型的可靠性

### 8. **docs/参数灵敏度分析报告.md** ⭐⭐⭐
**用途**: 参数分析的详细报告  
**包含内容**:
- 参数空间搜索
- 最优参数选择
- 灵敏度分析

**论文写作用途**:
- Results - Q6部分
- 说明参数选择的合理性

### 9. **docs/核心结论速查表.md** ⭐⭐⭐⭐
**用途**: 关键数据快速查找  
**包含内容**:
- 所有关键数字和结论
- 快速参考表格

**论文写作用途**:
- 写作时快速查找数据
- 确保数字准确性

---

## 📂 数据文件

### 输入数据
- `data/2026 MCM Problem C Data.csv` - 原始数据

### 输出数据
- `results/Processed_DWTS_Long_Format.csv` - 预处理后的数据
- `results/Q1_Estimated_Fan_Votes.csv` - Q1&Q2答案
- `results/Simulation_Results_Q3_Q4.csv` - Q3&Q4答案
- `results/Q5_Feature_Importance.csv` - Q5答案
- `results/Q6_New_System_Simulation.csv` - Q6答案

### 图表文件
- `figures/*.png` - 所有可视化图表（20张）

---

## ✍️ 论文写作流程

### Step 1: 了解全局（30分钟）
1. 阅读 `PROJECT_GUIDE.md` - 了解项目全貌
2. 浏览 `FIGURES_GUIDE.md` - 了解所有图表
3. 快速浏览 `docs/FINAL_PROJECT_SUMMARY.md` - 了解详细内容

### Step 2: 写Abstract（30分钟）
**参考文档**: `PROJECT_GUIDE.md` 的核心发现部分

**关键内容**:
- 问题：分析DWTS比赛的公平性
- 方法：SMC + 贝叶斯回归 + 平行宇宙仿真
- 结果：
  - 100%逆转率证明赛制不公平
  - 年龄和舞伴是最重要因素
  - 新系统降低冤案率1.26%
- 结论：验证了Arrow不可能定理

### Step 3: 写Introduction（1小时）
**参考文档**: 
- `PROJECT_GUIDE.md` - 项目背景
- `docs/FINAL_PROJECT_SUMMARY.md` - 详细背景

**关键内容**:
- 背景：DWTS比赛的流行和争议
- 问题：赛制是否公平？如何改进？
- 目标：回答6个核心问题
- 贡献：方法创新、理论验证、实践价值

**图表**: `judge_score_distribution.png` - 数据分布

### Step 4: 写Methodology（2-3小时）
**参考文档**:
- `docs/FINAL_PROJECT_SUMMARY.md` - 方法总览
- `docs/SMC_README.md` - SMC算法详细说明
- `docs/模型检验模块.md` - 模型验证

**关键内容**:
- 数据预处理：宽表转长表，特征工程
- SMC算法：5000个粒子，软约束策略
- 平行宇宙仿真：三个宇宙对比
- 贝叶斯岭回归：81个特征分析
- 新赛制设计：70/30权重 + Sigmoid抑制

**图表**: 
- `q1_particle_cloud.png` - SMC算法可视化
- `model_validation_cv.png` - 模型验证

### Step 5: 写Results（3-4小时）

#### Q1&Q2: 观众投票反演
**参考文档**: `docs/FINAL_PROJECT_SUMMARY.md`

**关键内容**:
- 方法：SMC粒子滤波
- 结果：平均不确定性8.5%
- 验证：投票总和100%归一化

**图表**:
- `smc_uncertainty_analysis.png` ⭐⭐⭐⭐⭐
- `season1_trajectories.png` ⭐⭐⭐⭐

#### Q3&Q4: 赛制公平性
**参考文档**: `docs/Q3_Q4_ANSWERS.md`

**关键内容**:
- Q3：100%逆转率，赛制不公平
- Q4：23.11%拯救率，机制有效但有限
- 冤案率：94.70%

**图表**:
- `multiverse_analysis.png` ⭐⭐⭐⭐⭐
- `q3_sankey_chaos.png` ⭐⭐⭐⭐

#### Q5: 关键因素
**参考文档**: `docs/Q5_Q6_ANSWERS.md`

**关键内容**:
- 年龄影响：-0.494（最大）
- Derek Hough：+0.187（金牌舞伴）
- 裁判分数R²：28.28%
- 观众投票R²：11.04%

**图表**:
- `q5_feature_importance.png` ⭐⭐⭐⭐⭐
- `q5_partner_influence.png` ⭐⭐⭐⭐
- `q5_tornado_plot.png` ⭐⭐⭐

#### Q6: 新赛制设计
**参考文档**: `docs/Q5_Q6_ANSWERS.md`

**关键内容**:
- 新赛制：70/30权重 + Sigmoid抑制
- 冤案率：94.70% → 93.43%（-1.26%）
- 裁判排名：2.59 → 8.21（+5.62）

**图表**:
- `q6_rank_distribution.png` ⭐⭐⭐⭐⭐
- `q6_injustice_comparison.png` ⭐⭐⭐⭐⭐
- `sensitivity_heatmap.png` ⭐⭐⭐⭐

### Step 6: 写Discussion（2小时）
**参考文档**: `docs/FINAL_PROJECT_SUMMARY.md` 的理论贡献部分

**关键内容**:
- 理论贡献：验证Arrow不可能定理
- 实践启示：赛制设计建议
- 局限性：模型假设和数据限制
- 未来工作：改进方向

**图表**: `q5_q6_dashboard.png` ⭐⭐⭐⭐⭐

### Step 7: 写Conclusion（30分钟）
**参考文档**: `PROJECT_GUIDE.md` 的核心发现部分

**关键内容**:
- 总结6个问题的答案
- 强调核心发现
- 实践价值
- 未来展望

---

## 📊 图表使用优先级

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

---

## 🔢 关键数字速查

### 数据规模
- 赛季数：34
- 选手数：421
- 有效记录：2777条
- 淘汰周次：264周

### Q1&Q2
- 粒子数：5000
- 平均不确定性：8.5%
- 不确定性范围：0.4% - 21.6%

### Q3&Q4
- 逆转率：100%
- 拯救率：23.11%
- 冤案率：94.70%

### Q5
- 年龄影响：-0.494
- Derek Hough影响：+0.187
- 裁判分数R²：28.28%
- 观众投票R²：11.04%

### Q6
- 新系统权重：70/30
- 冤案率改善：-1.26%
- 裁判排名改善：+5.62

---

## ✅ 写作检查清单

### Abstract
- [ ] 问题陈述清晰
- [ ] 方法简要说明
- [ ] 核心结果突出
- [ ] 结论有力

### Introduction
- [ ] 背景充分
- [ ] 问题明确
- [ ] 贡献清晰
- [ ] 有数据分布图

### Methodology
- [ ] 数据预处理说明
- [ ] SMC算法详细
- [ ] 模型验证充分
- [ ] 有方法可视化

### Results
- [ ] Q1-Q6都有答案
- [ ] 每个问题有图表支持
- [ ] 数字准确
- [ ] 图表说明清晰

### Discussion
- [ ] 理论贡献明确
- [ ] 实践价值清晰
- [ ] 局限性诚实
- [ ] 未来工作合理

### Conclusion
- [ ] 总结全面
- [ ] 核心发现突出
- [ ] 实践价值强调
- [ ] 未来展望积极

---

## 📞 快速参考

### 核心发现（用于Abstract和Conclusion）
1. 100%逆转率证明赛制不公平
2. 年龄是最重要的因素（-0.494）
3. Derek Hough是金牌舞伴（+0.187）
4. 94.70%的被淘汰者是"冤案"
5. 新系统降低冤案率1.26%
6. 验证了Arrow不可能定理

### 技术亮点（用于Methodology）
1. 手写SMC算法（5000个粒子）
2. 软约束策略 + 急救机制
3. 平行宇宙仿真（三个宇宙）
4. 贝叶斯岭回归（81个特征）
5. 新赛制设计（70/30 + Sigmoid）

### 实践价值（用于Discussion）
1. 为赛制设计提供数据支持
2. 为参赛者提供策略建议
3. 为观众提供透明度
4. 验证社会选择理论

---

## 🎯 论文写作时间估算

- Abstract: 30分钟
- Introduction: 1小时
- Methodology: 2-3小时
- Results: 3-4小时
- Discussion: 2小时
- Conclusion: 30分钟
- 图表整理: 1小时
- 格式调整: 1小时

**总计**: 约11-13小时

---

## 📚 文档阅读顺序

### 第一遍（快速了解，1小时）
1. PROJECT_GUIDE.md
2. FIGURES_GUIDE.md
3. docs/核心结论速查表.md

### 第二遍（详细理解，2-3小时）
4. docs/FINAL_PROJECT_SUMMARY.md
5. docs/Q3_Q4_ANSWERS.md
6. docs/Q5_Q6_ANSWERS.md

### 第三遍（深入细节，按需查阅）
7. docs/SMC_README.md
8. docs/模型检验模块.md
9. docs/参数灵敏度分析报告.md

---

**文档整理完成日期**: 2026年1月30日  
**状态**: ✅ 已删除22个冗余文档，保留9个核心文档  
**论文写作准备**: ✅ 完全就绪
