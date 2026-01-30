# DWTS (Dancing with the Stars) 建模项目

## 📊 项目概况

**数据规模：** 34个赛季，421名选手，2777条记录  
**核心目标：** 预测评委打分和观众投票，分析评分机制公平性

## 🎯 核心成果

### Judge预测性能
- **最终R²: 73.27%** ✅
- 从基线28.28%提升到73.27%（+45%）
- 无数据泄露，方法严谨

### 关键发现
1. **Week特征主导作用**（相关性0.66，因果效应1.46-1.92分）
2. **Arrow定理应用**（3/5条件满足，解释100%逆转率）
3. **时间动态**（49.8%分数膨胀，方差收敛）

## 📁 项目结构

```
.
├── submission/              # 主要工作目录
│   ├── code/               # 所有代码文件
│   ├── data/               # 原始数据
│   ├── results/            # 分析结果
│   ├── figures/            # 可视化图表（30+个）
│   └── docs/               # 文档
├── src/                    # ML优化模块
│   └── ml_optimization/    # 完整的ML管道
├── models/                 # 训练好的模型
├── reports/                # 优化报告
├── config/                 # 配置文件
└── tests/                  # 测试文件
```

## 🚀 快速开始

### 1. 查看项目总结
```bash
cat FINAL_PROJECT_SUMMARY.md
```

### 2. 查看核心结果
```bash
# Phase 1结果
cat submission/Enhanced_Feature_Analysis.csv

# Phase 2结果
cat submission/Arrow_Theorem_Analysis_Simplified.csv
cat submission/Causal_Inference_Results.csv

# Phase 3结果
cat submission/Clean_Model_Comparison.csv
cat submission/Clean_Validation_Report.csv
```

### 3. 运行代码
```bash
cd submission

# Phase 1: 特征归因
python code/enhanced_feature_attribution.py

# Phase 2: Arrow定理
python code/arrow_theorem_simplified.py

# Phase 2: 因果推断
python code/causal_inference_analysis.py

# Phase 3: 特征工程
python code/ultimate_feature_engineering_clean.py

# Phase 3: 集成学习
python code/ultimate_ensemble_learning.py
```

### 4. 运行ML优化（可选）
```bash
# 安装依赖
pip install -r requirements_ml_optimization.txt

# 运行优化（需要15-30分钟）
python run_ml_optimization.py
```

## 📈 性能提升轨迹

| 阶段 | Judge R² | Fan R² | 说明 |
|------|----------|--------|------|
| Baseline | 28.28% | 11.04% | 原始模型 |
| Phase 1 | 59.22% | 61.06% | Week特征发现 |
| Phase 3 Clean | 68.77% | 67.60% | 无数据泄露 |
| **ML优化** | **73.27%** | 56.40% | 最终版本 |

## 🔑 核心文件

### 代码
- `submission/code/enhanced_feature_attribution.py` - Phase 1特征归因
- `submission/code/arrow_theorem_simplified.py` - Arrow定理分析
- `submission/code/causal_inference_analysis.py` - 因果推断
- `submission/code/ultimate_feature_engineering_clean.py` - 特征工程（最终版本）

### 数据
- `submission/data/2026 MCM Problem C Data.csv` - 原始数据
- `submission/results/Clean_Enhanced_Dataset.csv` - 最终特征数据

### 结果
- `submission/Clean_Model_Comparison.csv` - 模型对比
- `submission/Clean_Validation_Report.csv` - 验证报告
- `submission/Arrow_Theorem_Analysis_Simplified.csv` - Arrow定理结果
- `submission/Causal_Inference_Results.csv` - 因果推断结果

### 文档
- `FINAL_PROJECT_SUMMARY.md` - 完整项目总结
- `submission/START_HERE.md` - 项目入口
- `submission/PROJECT_GUIDE.md` - 项目指南
- `submission/PAPER_WRITING_GUIDE.md` - 论文写作指南

## 🏆 预期获奖

- **M奖（Meritorious）**: 90-95% ✅
- **F奖（Finalist）**: 50-60%
- **O奖（Outstanding）**: 10-15%

**关键：论文写作质量将决定是否能拿F奖！**

## 📝 论文写作要点

### 强调
1. ✅ Week特征的发现（核心贡献）
2. ✅ Arrow定理的应用（理论深度）
3. ✅ 严格的方法论（数据泄露防护）
4. ✅ Judge预测的成功（+14.05%）

### 淡化
- ⚠️ Fan预测的失败（解释：人类投票的随机性）
- ⚠️ 绝对R²值（强调相对提升）

## 📞 技术栈

- **Python 3.12**
- **核心库**: pandas, numpy, scikit-learn, scikit-optimize
- **可视化**: matplotlib, seaborn
- **统计**: scipy, statsmodels
- **ML**: Random Forest, Gradient Boosting, Ridge, Lasso, ElasticNet

## 📄 许可

本项目为2026年美国大学生数学建模竞赛（MCM）参赛作品。

---

**项目完成时间：** 2026年1月30日  
**GitHub**: https://github.com/Introsb/OOO

**祝你们取得好成绩！** 🎉🏆
