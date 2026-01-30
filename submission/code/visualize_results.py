"""可视化SMC结果"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
df = pd.read_csv('Q1_Estimated_Fan_Votes.csv')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 不确定性分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 不确定性直方图
axes[0, 0].hist(df['Uncertainty_Std'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Uncertainty (Std)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Uncertainty')
axes[0, 0].grid(True, alpha=0.3)

# 1.2 按规则的不确定性对比
rule_data = df.groupby('Rule_Used')['Uncertainty_Std'].apply(list)
axes[0, 1].boxplot([rule_data['rank'], rule_data['percent']], labels=['Rank', 'Percent'])
axes[0, 1].set_ylabel('Uncertainty (Std)')
axes[0, 1].set_title('Uncertainty by Rule Type')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 不确定性随周次变化
week_uncertainty = df.groupby('Week')['Uncertainty_Std'].mean()
axes[1, 0].plot(week_uncertainty.index, week_uncertainty.values, marker='o', linewidth=2)
axes[1, 0].set_xlabel('Week')
axes[1, 0].set_ylabel('Average Uncertainty')
axes[1, 0].set_title('Uncertainty vs Week')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 投票分布示例（Season 1）
season1_week1 = df[(df['Season'] == 1) & (df['Week'] == 1)]
axes[1, 1].bar(range(len(season1_week1)), season1_week1['Estimated_Fan_Vote'].values)
axes[1, 1].errorbar(range(len(season1_week1)), 
                    season1_week1['Estimated_Fan_Vote'].values,
                    yerr=season1_week1['Uncertainty_Std'].values,
                    fmt='none', ecolor='red', capsize=5)
axes[1, 1].set_xlabel('Contestant Index')
axes[1, 1].set_ylabel('Estimated Fan Vote')
axes[1, 1].set_title('Season 1 Week 1: Fan Votes with Uncertainty')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smc_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: smc_uncertainty_analysis.png")

# 2. 选手人气轨迹（选择几个有代表性的选手）
fig, ax = plt.subplots(figsize=(14, 6))

# 选择Season 1的所有选手
season1_data = df[df['Season'] == 1]
for name in season1_data['Name'].unique():
    contestant_data = season1_data[season1_data['Name'] == name]
    ax.plot(contestant_data['Week'], contestant_data['Estimated_Fan_Vote'], 
            marker='o', label=name, linewidth=2)

ax.set_xlabel('Week')
ax.set_ylabel('Estimated Fan Vote')
ax.set_title('Season 1: Fan Vote Trajectories')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('season1_trajectories.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: season1_trajectories.png")

# 3. 赛季统计
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3.1 每个赛季的平均不确定性
season_uncertainty = df.groupby('Season')['Uncertainty_Std'].mean()
axes[0].bar(season_uncertainty.index, season_uncertainty.values, alpha=0.7)
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Average Uncertainty')
axes[0].set_title('Average Uncertainty by Season')
axes[0].grid(True, alpha=0.3)

# 3.2 每个赛季的数据点数量
season_counts = df.groupby('Season').size()
axes[1].bar(season_counts.index, season_counts.values, alpha=0.7, color='green')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('Number of Data Points')
axes[1].set_title('Data Points by Season')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('season_statistics.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: season_statistics.png")

print("\n✓ 所有可视化完成！")
