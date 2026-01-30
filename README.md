# DWTS (Dancing With The Stars) Analysis Project

## 项目简介 / Project Overview

这是一个针对《与星共舞》(Dancing With The Stars) 节目的数据分析项目，使用统计建模和机器学习方法来分析比赛结果、预测选手表现，并评估评分系统的公平性。

This is a data analysis project for the TV show "Dancing With The Stars", using statistical modeling and machine learning methods to analyze competition results, predict contestant performance, and evaluate the fairness of the scoring system.

## 主要功能 / Key Features

- 数据预处理和特征工程 / Data preprocessing and feature engineering
- 粉丝投票估计 (SMC方法) / Fan vote estimation (SMC method)
- 多元宇宙模拟分析 / Multiverse simulation analysis
- 特征归因分析 / Feature attribution analysis
- 评分系统设计与优化 / Scoring system design and optimization

## 项目结构 / Project Structure

```
.
├── src/                          # 核心源代码 / Core source code
│   ├── preprocessing_pipeline.py # 数据预处理管道
│   ├── smc_fan_vote_estimator.py # SMC粉丝投票估计器
│   ├── feature_attribution.py    # 特征归因分析
│   ├── multiverse_simulator.py   # 多元宇宙模拟器
│   └── ultimate_system_design.py # 终极系统设计
├── submission/                   # 提交文件夹
│   ├── code/                     # 分析代码
│   ├── data/                     # 数据文件
│   ├── docs/                     # 文档
│   ├── figures/                  # 可视化图表
│   └── results/                  # 分析结果
├── tests/                        # 测试文件
└── requirements.txt              # Python依赖
```

## 安装 / Installation

```bash
# 克隆仓库
git clone <your-repo-url>
cd MCM

# 安装依赖
pip install -r requirements.txt
```

## 使用方法 / Usage

```bash
# 运行完整分析流程
python submission/run_all.py

# 运行特定分析模块
python submission/code/main.py
```

## 技术栈 / Tech Stack

- Python 3.x
- NumPy, Pandas - 数据处理
- Scikit-learn - 机器学习
- Matplotlib, Seaborn - 数据可视化
- SciPy - 科学计算

## 许可证 / License

MIT License

## 作者 / Author

MCM Competition Team
