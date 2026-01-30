"""
可视化Q5和Q6的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# Q5 可视化
# ============================================================================

print("Generating Q5 visualizations...")

df_q5 = pd.read_csv('Q5_Feature_Importance.csv')

# 图1：TOP 10特征对裁判分数的影响
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 裁判分数
df_q5['Abs_Coef_Judge'] = df_q5['Coef_Judge'].abs()
top_judge = df_q5.nlargest(10, 'Abs_Coef_Judge').sort_values('Coef_Judge')

colors_judge = ['red' if x < 0 else 'green' for x in top_judge['Coef_Judge']]
ax1.barh(range(len(top_judge)), top_judge['Coef_Judge'], color=colors_judge, alpha=0.7)
ax1.set_yticks(range(len(top_judge)))
ax1.set_yticklabels([f.replace('Partner_', 'P:').replace('Industry_', 'I:') 
                      for f in top_judge['Feature']], fontsize=9)
ax1.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
ax1.set_title('TOP 10 Features Influencing Judge Scores', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# 观众投票
df_q5['Abs_Coef_Fan'] = df_q5['Coef_Fan'].abs()
top_fan = df_q5.nlargest(10, 'Abs_Coef_Fan').sort_values('Coef_Fan')

colors_fan = ['red' if x < 0 else 'green' for x in top_fan['Coef_Fan']]
ax2.barh(range(len(top_fan)), top_fan['Coef_Fan'], color=colors_fan, alpha=0.7)
ax2.set_yticks(range(len(top_fan)))
ax2.set_yticklabels([f.replace('Partner_', 'P:').replace('Industry_', 'I:') 
                      for f in top_fan['Feature']], fontsize=9)
ax2.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
ax2.set_title('TOP 10 Features Influencing Fan Votes', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('q5_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q5_feature_importance.png")
plt.close()

# 图2：舞伴影响力对比
fig, ax = plt.subplots(figsize=(12, 8))

partner_features = df_q5[df_q5['Feature'].str.startswith('Partner_')].copy()
partner_features['Partner'] = partner_features['Feature'].str.replace('Partner_', '')
partner_features = partner_features.sort_values('Coef_Judge', ascending=False).head(15)

x = np.arange(len(partner_features))
width = 0.35

bars1 = ax.bar(x - width/2, partner_features['Coef_Judge'], width, 
               label='Judge Score Impact', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, partner_features['Coef_Fan'] * 100, width, 
               label='Fan Vote Impact (×100)', alpha=0.8, color='coral')

ax.set_xlabel('Partner', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient', fontsize=12, fontweight='bold')
ax.set_title('Partner Influence on Judge Scores and Fan Votes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(partner_features['Partner'], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('q5_partner_influence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q5_partner_influence.png")
plt.close()

# ============================================================================
# Q6 可视化
# ============================================================================

print("\nGenerating Q6 visualizations...")

df_q6 = pd.read_csv('Q6_New_System_Simulation.csv')

# 图3：被淘汰者排名分布对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 裁判排名分布
judge_rank_dist = df_q6['Judge_Rank'].value_counts().sort_index()
ax1.bar(judge_rank_dist.index, judge_rank_dist.values, alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('Judge Rank (1=Best)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Eliminated Contestants: Judge Rank Distribution', fontsize=14, fontweight='bold')
ax1.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Bottom 3 Threshold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 观众排名分布
fan_rank_dist = df_q6['Fan_Rank'].value_counts().sort_index()
ax2.bar(fan_rank_dist.index, fan_rank_dist.values, alpha=0.7, color='coral', edgecolor='black')
ax2.set_xlabel('Fan Rank (1=Most Popular)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Eliminated Contestants: Fan Rank Distribution', fontsize=14, fontweight='bold')
ax2.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Bottom 3 Threshold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('q6_rank_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q6_rank_distribution.png")
plt.close()

# 图4：冤案率对比
fig, ax = plt.subplots(figsize=(10, 6))

systems = ['Old System\n(Percentage)', 'New System\n(70/30 + Sigmoid)']
injustice_rates = [94.70, 93.43]
colors = ['#e74c3c', '#3498db']

bars = ax.bar(systems, injustice_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# 添加数值标签
for bar, rate in zip(bars, injustice_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Injustice Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Injustice Rate Comparison: Old vs New System', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# 添加改善标注
improvement = injustice_rates[0] - injustice_rates[1]
ax.annotate(f'Improvement: {improvement:.2f}%',
            xy=(0.5, (injustice_rates[0] + injustice_rates[1])/2),
            xytext=(0.5, 85),
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

plt.tight_layout()
plt.savefig('q6_injustice_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q6_injustice_comparison.png")
plt.close()

# 图5：新系统被淘汰者特征散点图
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(df_q6['Judge_Rank'], df_q6['Fan_Rank'], 
                     c=df_q6['Num_Contestants'], cmap='viridis', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Judge Rank (1=Best)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fan Rank (1=Most Popular)', fontsize=12, fontweight='bold')
ax.set_title('New System: Eliminated Contestants Profile', fontsize=14, fontweight='bold')

# 添加参考线
ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Judge Bottom 3')
ax.axhline(y=3, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Fan Bottom 3')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Contestants', fontsize=10)

ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('q6_eliminated_profile.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q6_eliminated_profile.png")
plt.close()

# 图6：综合对比仪表板
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 子图1：年龄影响
ax1 = fig.add_subplot(gs[0, 0])
age_impact = df_q5[df_q5['Feature'] == 'Age'].iloc[0]
ax1.bar(['Judge Score', 'Fan Vote'], 
        [age_impact['Coef_Judge'], age_impact['Coef_Fan'] * 100],
        color=['steelblue', 'coral'], alpha=0.8)
ax1.set_title('Age Impact', fontweight='bold')
ax1.set_ylabel('Coefficient')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3)

# 子图2：Derek Hough效应
ax2 = fig.add_subplot(gs[0, 1])
derek = df_q5[df_q5['Feature'] == 'Partner_Derek Hough'].iloc[0]
ax2.bar(['Judge Score', 'Fan Vote'], 
        [derek['Coef_Judge'], derek['Coef_Fan'] * 100],
        color=['green', 'lightgreen'], alpha=0.8)
ax2.set_title('Derek Hough Effect', fontweight='bold')
ax2.set_ylabel('Coefficient')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)

# 子图3：冤案率对比
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(['Old', 'New'], [94.70, 93.43], color=['#e74c3c', '#3498db'], alpha=0.8)
ax3.set_title('Injustice Rate (%)', fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(axis='y', alpha=0.3)

# 子图4：裁判排名分布
ax4 = fig.add_subplot(gs[1, :])
judge_rank_dist = df_q6['Judge_Rank'].value_counts().sort_index()
ax4.bar(judge_rank_dist.index, judge_rank_dist.values, alpha=0.7, color='steelblue')
ax4.set_xlabel('Judge Rank')
ax4.set_ylabel('Frequency')
ax4.set_title('New System: Eliminated Contestants Judge Rank Distribution', fontweight='bold')
ax4.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Bottom 3')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 子图5：观众排名分布
ax5 = fig.add_subplot(gs[2, :])
fan_rank_dist = df_q6['Fan_Rank'].value_counts().sort_index()
ax5.bar(fan_rank_dist.index, fan_rank_dist.values, alpha=0.7, color='coral')
ax5.set_xlabel('Fan Rank')
ax5.set_ylabel('Frequency')
ax5.set_title('New System: Eliminated Contestants Fan Rank Distribution', fontweight='bold')
ax5.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Bottom 3')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

plt.suptitle('Q5 & Q6 Comprehensive Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('q5_q6_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q5_q6_dashboard.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. q5_feature_importance.png - Feature coefficients")
print("  2. q5_partner_influence.png - Partner impact comparison")
print("  3. q6_rank_distribution.png - Eliminated contestants ranking")
print("  4. q6_injustice_comparison.png - Injustice rate comparison")
print("  5. q6_eliminated_profile.png - Scatter plot of eliminated contestants")
print("  6. q5_q6_dashboard.png - Comprehensive dashboard")
