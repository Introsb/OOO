# DWTS项目快速指南

## 📋 项目概览

**Dancing with the Stars (与星共舞)** 数据分析项目 - 完整的数据分析、建模和赛制优化系统

- **数据规模**: 34个赛季，421位选手，2777条记录
- **核心方法**: SMC粒子滤波 + 贝叶斯回归 + 平行宇宙仿真
- **主要成果**: 回答6个核心问题（Q1-Q6），设计更公平的赛制

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r docs/requirements.txt
```

### 2. 运行完整分析
```bash
python run_all.py
```

### 3. 查看结果
- **数据**: `results/` 文件夹中的CSV文件
- **图表**: `figures/` 文件夹中的PNG文件
- **文档**: `docs/` 文件夹中的详细说明

---

## 📊 核心问题与答案

### Q1: 观众投票是多少？
**答案文件**: `results/Q1_Estimated_Fan_Votes.csv`  
**方法**: SMC粒子滤波（5000个粒子）  
**字段**: `Estimated_Fan_Vote` (投票比例)

### Q2: 估计的不确定性有多大？
**答案文件**: `results/Q1_Estimated_Fan_Votes.csv`  
**字段**: `Uncertainty_Std` (标准差)  
**统计**: 平均8.5%，范围0.4%-21.6%

### Q3: 赛制是否公平？
**答案**: 不公平，逆转率100%  
**证据**: 排名制和百分比制在所有264周都淘汰不同的人  
**详细分析**: 见 `docs/Q3_Q4_ANSWERS.md`

### Q4: 裁判拯救机制是否有效？
**答案**: 有效但有限，拯救率23.11%  
**证据**: 在61/264周改变了淘汰结果  
**详细分析**: 见 `docs/Q3_Q4_ANSWERS.md`

### Q5: 影响成败的关键因素？
**答案文件**: `results/Q5_Feature_Importance.csv`  
**核心发现**:
- 年龄影响最大（-0.494）
- Derek Hough是金牌舞伴（+0.187）
- 观众投票更难预测（R²=11%）
**详细分析**: 见 `docs/Q5_Q6_ANSWERS.md`

### Q6: 如何设计更公平的赛制？
**答案文件**: `results/Q6_New_System_Simulation.csv`  
**新赛制**: 70/30权重 + Sigmoid抑制  
**效果**: 冤案率降低1.26%，被淘汰者裁判排名提升5.62  
**详细分析**: 见 `docs/Q5_Q6_ANSWERS.md`

---

## 📁 项目结构

```
submission/
├── code/              # 源代码
│   ├── preprocessing_pipeline.py      # 数据预处理
│   ├── smc_fan_vote_estimator.py     # SMC观众投票反演
│   ├── multiverse_simulator.py       # 平行宇宙仿真
│   ├── feature_attribution.py        # 特征归因分析
│   └── ultimate_system_design.py     # 终极赛制设计
├── data/              # 原始数据
│   └── 2026 MCM Problem C Data.csv
├── results/           # 输出结果（CSV）
│   ├── Q1_Estimated_Fan_Votes.csv    # Q1&Q2答案
│   ├── Simulation_Results_Q3_Q4.csv  # Q3&Q4答案
│   ├── Q5_Feature_Importance.csv     # Q5答案
│   └── Q6_New_System_Simulation.csv  # Q6答案
├── figures/           # 可视化图表（PNG）
├── docs/              # 详细文档
│   ├── FINAL_PROJECT_SUMMARY.md      # 项目总结
│   ├── Q3_Q4_ANSWERS.md              # Q3&Q4详细答案
│   ├── Q5_Q6_ANSWERS.md              # Q5&Q6详细答案
│   └── 核心结论速查表.md             # 关键发现速查
├── PROJECT_GUIDE.md   # 本文件（快速指南）
├── FIGURES_GUIDE.md   # 图表说明指南
└── run_all.py         # 主运行脚本
```

---

## 📈 关键图表说明

### 数据预处理
- `judge_score_distribution.png` - 裁判分数分布

### SMC分析（Q1&Q2）
- `smc_uncertainty_analysis.png` - 不确定性分析
- `season1_trajectories.png` - Season 1选手轨迹
- `q1_particle_cloud.png` - 粒子云可视化

### 平行宇宙（Q3&Q4）
- `multiverse_analysis.png` - 三个宇宙对比
- `q3_sankey_chaos.png` - 淘汰流向图

### 特征归因（Q5）
- `q5_feature_importance.png` - 特征重要性排名
- `q5_partner_influence.png` - 舞伴影响分析
- `q5_tornado_plot.png` - 龙卷风图

### 终极赛制（Q6）
- `q6_rank_distribution.png` - 排名分布对比
- `q6_injustice_comparison.png` - 冤案率对比
- `q6_case_study.png` - 案例研究
- `q6_eliminated_profile.png` - 被淘汰者画像

### 综合分析
- `q5_q6_dashboard.png` - Q5&Q6综合仪表板
- `model_validation_cv.png` - 交叉验证结果
- `sensitivity_heatmap.png` - 参数灵敏度热力图

**详细图表说明**: 见 `FIGURES_GUIDE.md`

---

## 🔬 核心发现

### 1. 年龄是最重要的因素
- 对裁判分数影响：-0.494（标准化系数）
- **结论**: 年轻选手有显著优势

### 2. Derek Hough是"金牌舞伴"
- 对裁判分数影响：+0.187
- **结论**: 与Derek搭档是最大的优势

### 3. 冤案现象普遍存在
- 旧系统冤案率：94.70%
- 新系统冤案率：93.43%
- **结论**: 冤案是淘汰机制的固有特性

### 4. 规则设计的重要性
- 100%逆转率证明规则决定结果
- **结论**: 验证了Arrow不可能定理

### 5. 观众投票的双刃剑
- 29.6%的被淘汰者是观众最喜欢的
- **结论**: 需要在专业性和娱乐性之间平衡

---

## 🎯 技术亮点

### 1. 手写SMC算法
- 5000个粒子，约136万次更新
- 软约束策略 + 急救机制
- 规则自适应识别

### 2. 贝叶斯岭回归
- 81个特征分析
- 提供参数置信度
- 适合解释性分析

### 3. 新赛制设计
- Min-Max归一化保留差距
- Sigmoid抑制防止垄断
- 70/30权重平衡专业性和娱乐性

### 4. 全面的测试
- 21个测试全部通过
- 属性测试 + 单元测试 + 集成测试

---

## 📚 详细文档

### 核心文档
- **PROJECT_GUIDE.md** (本文件) - 快速指南
- **FIGURES_GUIDE.md** - 图表说明指南
- **docs/FINAL_PROJECT_SUMMARY.md** - 完整项目总结
- **docs/核心结论速查表.md** - 关键发现速查

### 问题答案
- **docs/Q3_Q4_ANSWERS.md** - Q3&Q4详细答案
- **docs/Q5_Q6_ANSWERS.md** - Q5&Q6详细答案

### 技术文档
- **docs/README.md** - 数据预处理系统
- **docs/SMC_README.md** - SMC系统详细说明
- **docs/模型检验模块.md** - 模型验证文档
- **docs/参数灵敏度分析报告.md** - 参数分析报告

---

## 💡 使用建议

### 对于论文写作
1. 先阅读 `PROJECT_GUIDE.md`（本文件）了解全局
2. 查看 `FIGURES_GUIDE.md` 理解所有图表
3. 参考 `docs/FINAL_PROJECT_SUMMARY.md` 获取详细内容
4. 使用 `docs/核心结论速查表.md` 快速查找关键数据

### 对于代码理解
1. 从 `run_all.py` 开始，了解整体流程
2. 阅读 `code/` 文件夹中的各个模块
3. 查看 `tests/` 文件夹了解测试覆盖

### 对于结果验证
1. 检查 `results/` 文件夹中的CSV文件
2. 查看 `figures/` 文件夹中的可视化图表
3. 对照文档验证结果的正确性

---

## ⚡ 性能指标

- **运行时间**: 总计约2分钟
  - 数据预处理: < 5秒
  - SMC反演: 30-60秒
  - 其他分析: < 30秒
- **内存占用**: < 500MB
- **准确性**: 投票归一化100%准确

---

## 📞 技术支持

如有问题，请按以下顺序查阅：
1. **PROJECT_GUIDE.md** (本文件) - 快速指南
2. **FIGURES_GUIDE.md** - 图表说明
3. **docs/FINAL_PROJECT_SUMMARY.md** - 完整总结
4. **docs/** 文件夹中的其他文档

---

**项目完成日期**: 2026年1月30日  
**状态**: ✅ 全部完成（Q1-Q6）  
**数据来源**: DWTS Season 1-34
