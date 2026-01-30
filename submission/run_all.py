#!/usr/bin/env python3
"""
DWTS项目主运行脚本
Dancing with the Stars - Complete Analysis Pipeline

使用方法:
    python run_all.py

输出:
    - results/: 所有CSV结果文件
    - figures/: 所有可视化图表
"""

import sys
import os

# 添加code目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

print("="*80)
print("DWTS Complete Analysis Pipeline")
print("="*80)

# Phase 1: 数据预处理
print("\n[Phase 1] 数据预处理...")
os.system("python code/preprocessing_pipeline.py")

# Phase 2: SMC观众投票反演
print("\n[Phase 2] SMC观众投票反演...")
os.system("python code/smc_fan_vote_estimator.py")

# Phase 3: 平行宇宙仿真
print("\n[Phase 3] 平行宇宙仿真...")
os.system("python code/multiverse_simulator.py")

# Phase 4: 特征归因分析
print("\n[Phase 4] 特征归因分析...")
os.system("python code/feature_attribution.py")

# Phase 5: 终极赛制设计
print("\n[Phase 5] 终极赛制设计...")
os.system("python code/ultimate_system_design.py")

# Phase 6: 模型验证
print("\n[Phase 6] 模型验证...")
os.system("python code/model_validation.py")

# Phase 7: 参数灵敏度分析
print("\n[Phase 7] 参数灵敏度分析...")
os.system("python code/sensitivity_analysis.py")

# Phase 8: 生成所有可视化
print("\n[Phase 8] 生成可视化...")
os.system("python code/visualize_results.py")
os.system("python code/visualize_q5_q6.py")
os.system("python code/visualize_model_validation.py")
os.system("python code/create_paper_visualizations.py")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n结果文件:")
print("  - results/: CSV数据文件")
print("  - figures/: 可视化图表")
print("\n请查看 docs/ 目录获取详细文档")
