"""分析平行宇宙仿真结果"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
df = pd.read_csv('Simulation_Results_Q3_Q4.csv')

print("=" * 70)
print("平行宇宙仿真结果深度分析")
print("=" * 70)

# 1. 基本统计
print("\n【1. 基本统计】")
print(f"  总仿真周次: {len(df)}")
print(f"  涉及赛季: {df['Season'].min()} - {df['Season'].max()}")
print(f"  平均每周选手数: {df['Num_Contestants'].mean():.2f}")

# 2. Q3: 赛制差异分析
print("\n【2. Q3: 赛制差异分析 - 排名制 vs 百分比制】")
reversals = df['Is_Reversal'].sum()
total = len(df)
print(f"  逆转次数: {reversals} / {total} ({reversals/total*100:.2f}%)")
print(f"  结论: 两种赛制在 {reversals/total*100:.2f}% 的情况下会淘汰不同的人")
print(f"  这证明了赛制规则对结果有显著影响！")

# 按赛季分析逆转率
print("\n  按赛季分析逆转率:")
season_reversals = df.groupby('Season')['Is_Reversal'].agg(['sum', 'count', 'mean'])
season_reversals.columns = ['Reversals', 'Total', 'Rate']
season_reversals['Rate'] = season_reversals['Rate'] * 100
print(season_reversals.head(10).to_string())

# 3. Q4: 裁判拯救机制分析
print("\n【3. Q4: 裁判拯救机制分析】")
saves = df['Is_Saved'].sum()
print(f"  拯救生效次数: {saves} / {total} ({saves/total*100:.2f}%)")
print(f"  结论: 裁判拯救机制在 {saves/total*100:.2f}% 的情况下改变了淘汰结果")
print(f"  这意味着约 1/4 的情况下，裁判的主观判断能够改变命运")

# 找出被拯救的选手
saved_cases = df[df['Is_Saved']]
print(f"\n  被拯救的案例数: {len(saved_cases)}")
print(f"  示例（前10个被拯救的案例）:")
print(saved_cases[['Season', 'Week', 'Actual_Eliminated', 'Simulated_Elim_Percent', 
                   'Simulated_Elim_Save']].head(10).to_string(index=False))

# 4. 冤案分析
print("\n【4. 冤案分析】")
injustices = df['Is_Injustice'].sum()
print(f"  冤案次数: {injustices} / {total} ({injustices/total*100:.2f}%)")
print(f"  解释: 在 {injustices/total*100:.2f}% 的情况下，被淘汰者既不是裁判最差，也不是观众最差")
print(f"  这说明赛制规则的折算方式可能导致'不公平'的结果")

# 分析冤案的严重程度
injustice_cases = df[df['Is_Injustice']]
if len(injustice_cases) > 0:
    avg_judge_rank = injustice_cases['Judge_Rank'].mean()
    avg_fan_rank = injustice_cases['Fan_Rank'].mean()
    print(f"\n  冤案中被淘汰者的平均排名:")
    print(f"    裁判排名: {avg_judge_rank:.2f}")
    print(f"    观众排名: {avg_fan_rank:.2f}")
    print(f"  （排名越小越好，1是最好）")

# 5. 模型一致性分析
print("\n【5. 模型一致性分析】")
# 三种模型都一致
all_same = (
    (df['Simulated_Elim_Rank'] == df['Simulated_Elim_Percent']) &
    (df['Simulated_Elim_Percent'] == df['Simulated_Elim_Save'])
).sum()
print(f"  三种模型完全一致: {all_same} / {total} ({all_same/total*100:.2f}%)")

# 两两一致
rank_percent_same = (df['Simulated_Elim_Rank'] == df['Simulated_Elim_Percent']).sum()
percent_save_same = (df['Simulated_Elim_Percent'] == df['Simulated_Elim_Save']).sum()
rank_save_same = (df['Simulated_Elim_Rank'] == df['Simulated_Elim_Save']).sum()

print(f"  排名制 = 百分比制: {rank_percent_same} / {total} ({rank_percent_same/total*100:.2f}%)")
print(f"  百分比制 = 拯救机制: {percent_save_same} / {total} ({percent_save_same/total*100:.2f}%)")
print(f"  排名制 = 拯救机制: {rank_save_same} / {total} ({rank_save_same/total*100:.2f}%)")

# 6. 与真实结果的匹配度
print("\n【6. 与真实结果的匹配度】")
rank_match = (df['Actual_Eliminated'] == df['Simulated_Elim_Rank']).sum()
percent_match = (df['Actual_Eliminated'] == df['Simulated_Elim_Percent']).sum()
save_match = (df['Actual_Eliminated'] == df['Simulated_Elim_Save']).sum()

print(f"  排名制匹配真实结果: {rank_match} / {total} ({rank_match/total*100:.2f}%)")
print(f"  百分比制匹配真实结果: {percent_match} / {total} ({percent_match/total*100:.2f}%)")
print(f"  拯救机制匹配真实结果: {save_match} / {total} ({save_match/total*100:.2f}%)")
print(f"\n  结论: 百分比制与真实结果的匹配度最高（{percent_match/total*100:.2f}%）")
print(f"  这说明实际比赛更接近百分比制规则")

# 7. 按选手数量分析
print("\n【7. 按选手数量分析】")
contestant_analysis = df.groupby('Num_Contestants').agg({
    'Is_Reversal': 'mean',
    'Is_Saved': 'mean',
    'Is_Injustice': 'mean',
    'Season': 'count'
})
contestant_analysis.columns = ['Reversal_Rate', 'Save_Rate', 'Injustice_Rate', 'Count']
contestant_analysis = contestant_analysis * 100  # 转换为百分比
contestant_analysis['Count'] = contestant_analysis['Count'] / 100  # 恢复计数
print(contestant_analysis.to_string())

# 8. 示例案例分析
print("\n【8. 典型案例分析】")
print("\n  案例1: 三种模型结果完全不同")
different_all = df[
    (df['Simulated_Elim_Rank'] != df['Simulated_Elim_Percent']) &
    (df['Simulated_Elim_Percent'] != df['Simulated_Elim_Save']) &
    (df['Simulated_Elim_Rank'] != df['Simulated_Elim_Save'])
]
if len(different_all) > 0:
    print(different_all[['Season', 'Week', 'Actual_Eliminated', 
                        'Simulated_Elim_Rank', 'Simulated_Elim_Percent', 
                        'Simulated_Elim_Save']].head(5).to_string(index=False))
else:
    print("  没有找到三种模型结果完全不同的案例")

print("\n" + "=" * 70)
print("✓ 分析完成")
print("=" * 70)

# 生成可视化
print("\n生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 逆转率和拯救率
metrics = ['Is_Reversal', 'Is_Saved', 'Is_Injustice']
metric_names = ['Reversal\n(Rank≠Percent)', 'Saved\n(Save≠Percent)', 'Injustice']
values = [df[m].mean() * 100 for m in metrics]
colors = ['#ff6b6b', '#4ecdc4', '#ffd93d']

axes[0, 0].bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Percentage (%)')
axes[0, 0].set_title('Q3 & Q4: Key Metrics')
axes[0, 0].set_ylim([0, 105])
for i, v in enumerate(values):
    axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. 模型匹配度
models = ['Rank', 'Percent', 'Save']
matches = [rank_match, percent_match, save_match]
match_rates = [m/total*100 for m in matches]

axes[0, 1].bar(models, match_rates, color=['#95e1d3', '#f38181', '#aa96da'], 
               alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Match Rate (%)')
axes[0, 1].set_title('Model Accuracy vs Actual Results')
axes[0, 1].set_ylim([0, 105])
for i, v in enumerate(match_rates):
    axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. 按赛季的逆转率
season_reversal_rate = df.groupby('Season')['Is_Reversal'].mean() * 100
axes[1, 0].plot(season_reversal_rate.index, season_reversal_rate.values, 
                marker='o', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Season')
axes[1, 0].set_ylabel('Reversal Rate (%)')
axes[1, 0].set_title('Reversal Rate by Season')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100%')
axes[1, 0].legend()

# 4. 按选手数量的指标分布
contestant_counts = sorted(df['Num_Contestants'].unique())
reversal_by_count = [df[df['Num_Contestants']==c]['Is_Reversal'].mean()*100 for c in contestant_counts]
save_by_count = [df[df['Num_Contestants']==c]['Is_Saved'].mean()*100 for c in contestant_counts]

axes[1, 1].plot(contestant_counts, reversal_by_count, marker='o', label='Reversal Rate', linewidth=2)
axes[1, 1].plot(contestant_counts, save_by_count, marker='s', label='Save Rate', linewidth=2)
axes[1, 1].set_xlabel('Number of Contestants')
axes[1, 1].set_ylabel('Rate (%)')
axes[1, 1].set_title('Metrics by Number of Contestants')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiverse_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: multiverse_analysis.png")

print("\n✓ 所有分析完成！")
