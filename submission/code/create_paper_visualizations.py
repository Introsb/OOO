"""
生产"论文级"图表 (Paper-Quality Visualizations)
创建美赛论文所需的高质量可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

print("="*80)
print("生产论文级图表 (Paper-Quality Visualizations)")
print("="*80)


# ============================================================================
# 图1: Q3 桑基图 (Sankey Diagram) - The Chaos of Rules
# ============================================================================
print("\n[1/4] 生成Q3桑基图 (Sankey Diagram)...")

def create_sankey_diagram():
    """创建桑基图展示规则混乱"""
    df_sim = pd.read_csv('Simulation_Results_Q3_Q4.csv')
    
    # 统计排名制和百分比制的差异
    different_weeks = df_sim[df_sim['Simulated_Elim_Rank'] != df_sim['Simulated_Elim_Percent']]
    
    print(f"  发现 {len(different_weeks)} 周的淘汰结果不同（逆转率: {len(different_weeks)/len(df_sim)*100:.1f}%）")
    
    # 创建简化的桑基图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 由于matplotlib的Sankey功能有限，我们使用条形图来展示逆转现象
    # 统计每个赛季的逆转情况
    season_reversals = different_weeks.groupby('Season').size()
    total_weeks_per_season = df_sim.groupby('Season').size()
    reversal_rate = (season_reversals / total_weeks_per_season * 100).fillna(0)
    
    # 绘制条形图
    seasons = reversal_rate.index
    rates = reversal_rate.values
    
    colors = plt.cm.RdYlGn_r(rates / 100)
    bars = ax.bar(seasons, rates, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 添加100%参考线
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Reversal')
    
    # 标注
    ax.set_xlabel('Season', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reversal Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('The Chaos of Rules: 100% Reversal Rate Across All Seasons\n(Rank System vs Percentage System)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加统计信息
    textstr = f'Total Weeks: {len(df_sim)}\nReversed Weeks: {len(different_weeks)}\nReversal Rate: {len(different_weeks)/len(df_sim)*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('q3_sankey_chaos.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: q3_sankey_chaos.png")
    plt.close()


create_sankey_diagram()


# ============================================================================
# 图2: Q5 龙卷风图 (Tornado Plot) - Feature Importance
# ============================================================================
print("\n[2/4] 生成Q5龙卷风图 (Tornado Plot)...")

def create_tornado_plot():
    """创建龙卷风图展示特征重要性"""
    df_q5 = pd.read_csv('Q5_Feature_Importance.csv')
    
    # 选择TOP 15特征（按绝对值）
    df_q5['Abs_Coef_Judge'] = df_q5['Coef_Judge'].abs()
    top_features = df_q5.nlargest(15, 'Abs_Coef_Judge').sort_values('Coef_Judge')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 准备数据
    features = [f.replace('Partner_', '').replace('Industry_', 'Ind_') for f in top_features['Feature']]
    coefficients = top_features['Coef_Judge'].values
    
    # 颜色：负系数红色，正系数绿色
    colors = ['#e74c3c' if c < 0 else '#27ae60' for c in coefficients]
    
    # 特殊标记Age
    for i, feat in enumerate(top_features['Feature']):
        if feat == 'Age':
            colors[i] = '#c0392b'  # 深红色
    
    # 绘制水平条形图
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, coefficients, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 设置标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Coefficient (Impact on Judge Score)', fontsize=13, fontweight='bold')
    ax.set_title('Tornado Plot: TOP 15 Features Influencing Judge Scores\n(Red=Negative, Green=Positive)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        width = bar.get_width()
        label_x = width + (0.01 if width > 0 else -0.01)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{coef:.3f}',
                ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # 添加图例
    negative_patch = mpatches.Patch(color='#e74c3c', label='Negative Impact')
    positive_patch = mpatches.Patch(color='#27ae60', label='Positive Impact')
    age_patch = mpatches.Patch(color='#c0392b', label='Age (Strongest Negative)')
    ax.legend(handles=[negative_patch, positive_patch, age_patch], 
              loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('q5_tornado_plot.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: q5_tornado_plot.png")
    plt.close()


create_tornado_plot()


# ============================================================================
# 图3: Q1 粒子云图 (Particle Cloud) - Uncertainty Convergence
# ============================================================================
print("\n[3/4] 生成Q1粒子云图 (Particle Cloud)...")

def create_particle_cloud():
    """创建粒子云图展示不确定性收敛"""
    df_fan = pd.read_csv('Q1_Estimated_Fan_Votes.csv')
    
    # 选择一个明星选手（Season 1的Kelly Monaco）
    season1 = df_fan[df_fan['Season'] == 1]
    
    # 找到参与周次最多的选手
    contestant_weeks = season1.groupby('Name').size()
    top_contestant = contestant_weeks.idxmax()
    
    contestant_data = season1[season1['Name'] == top_contestant].sort_values('Week')
    
    print(f"  选择选手: {top_contestant} (参与 {len(contestant_data)} 周)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1：观众支持率随时间变化（带误差带）
    weeks = contestant_data['Week'].values
    votes = contestant_data['Estimated_Fan_Vote'].values
    uncertainties = contestant_data['Uncertainty_Std'].values
    
    # 绘制主线
    ax1.plot(weeks, votes, marker='o', linewidth=3, markersize=8, 
             color='steelblue', label='Estimated Fan Vote')
    
    # 绘制误差带（95%置信区间）
    ax1.fill_between(weeks, 
                     votes - 1.96 * uncertainties, 
                     votes + 1.96 * uncertainties,
                     alpha=0.3, color='steelblue', label='95% Confidence Interval')
    
    ax1.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Fan Vote Share', fontsize=13, fontweight='bold')
    ax1.set_title(f'Particle Cloud: {top_contestant} (Season 1)\nFan Support Trajectory with Uncertainty', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(votes + 1.96 * uncertainties) * 1.1])
    
    # 子图2：不确定性收敛
    ax2.plot(weeks, uncertainties, marker='s', linewidth=3, markersize=8, 
             color='coral', label='Uncertainty (Std)')
    ax2.fill_between(weeks, 0, uncertainties, alpha=0.3, color='coral')
    
    ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Uncertainty (Std)', fontsize=13, fontweight='bold')
    ax2.set_title('Uncertainty Convergence Over Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(weeks, uncertainties, 2)
    p = np.poly1d(z)
    ax2.plot(weeks, p(weeks), "--", color='red', linewidth=2, alpha=0.7, label='Trend')
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig('q1_particle_cloud.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: q1_particle_cloud.png")
    plt.close()


create_particle_cloud()


# ============================================================================
# 图4: Q6 对比柱状图 - Case Study (Jerry Rice / Bobby Bones)
# ============================================================================
print("\n[4/4] 生成Q6案例对比图 (Case Study)...")

def create_case_study():
    """创建案例对比图"""
    df_q6 = pd.read_csv('Q6_New_System_Simulation.csv')
    df_processed = pd.read_csv('Processed_DWTS_Long_Format.csv')
    df_fan = pd.read_csv('Q1_Estimated_Fan_Votes.csv')
    
    # 合并数据
    df_merged = df_processed.merge(
        df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
        on=['Season', 'Week', 'Name'],
        how='inner'
    )
    
    # 找到一个有代表性的案例（高人气但低分数）
    # 计算每个选手的平均裁判分数和平均观众投票
    contestant_stats = df_merged.groupby('Name').agg({
        'Judge_Avg_Score': 'mean',
        'Estimated_Fan_Vote': 'mean'
    }).reset_index()
    
    # 找到高人气低分数的选手
    contestant_stats['Score_Rank'] = contestant_stats['Judge_Avg_Score'].rank(ascending=False)
    contestant_stats['Fan_Rank'] = contestant_stats['Estimated_Fan_Vote'].rank(ascending=False)
    contestant_stats['Discrepancy'] = contestant_stats['Fan_Rank'] - contestant_stats['Score_Rank']
    
    # 选择差异最大的选手
    case_study = contestant_stats.nlargest(5, 'Discrepancy')
    
    print(f"  找到 {len(case_study)} 个高人气低分数的案例")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：裁判分数 vs 观众投票
    ax1 = axes[0]
    scatter = ax1.scatter(contestant_stats['Judge_Avg_Score'], 
                         contestant_stats['Estimated_Fan_Vote'],
                         c=contestant_stats['Discrepancy'], 
                         cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    
    # 标注案例
    for _, row in case_study.iterrows():
        ax1.annotate(row['Name'], 
                    xy=(row['Judge_Avg_Score'], row['Estimated_Fan_Vote']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Average Judge Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Fan Vote Share', fontsize=13, fontweight='bold')
    ax1.set_title('Case Study: High Popularity, Low Score Contestants', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Discrepancy (Fan Rank - Score Rank)', fontsize=10)
    
    # 子图2：TOP 5案例的对比
    ax2 = axes[1]
    
    x = np.arange(len(case_study))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, case_study['Judge_Avg_Score'], width, 
                    label='Judge Score', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x + width/2, case_study['Estimated_Fan_Vote'] * 40, width, 
                    label='Fan Vote (×40)', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Contestant', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title('TOP 5 High Popularity, Low Score Cases', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:15] for name in case_study['Name']], 
                        rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q6_case_study.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: q6_case_study.png")
    plt.close()


create_case_study()


print("\n" + "="*80)
print("论文级图表生成完成！")
print("="*80)
print("\n生成的图表:")
print("  1. q3_sankey_chaos.png - 规则混乱桑基图（100%逆转率）")
print("  2. q5_tornado_plot.png - 特征重要性龙卷风图")
print("  3. q1_particle_cloud.png - 粒子云图（不确定性收敛）")
print("  4. q6_case_study.png - 案例研究对比图")
print("\n这些图表可以直接用于美赛论文！")
