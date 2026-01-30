"""
分析Q5和Q6的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("Q5 & Q6 RESULTS ANALYSIS")
print("="*80)

# ============================================================================
# Q5 分析
# ============================================================================
print("\n" + "="*80)
print("Q5: FEATURE ATTRIBUTION ANALYSIS")
print("="*80)

df_q5 = pd.read_csv('Q5_Feature_Importance.csv')

print("\nTOP 10 FEATURES FOR JUDGE SCORES:")
print("-" * 80)
df_q5['Abs_Coef_Judge'] = df_q5['Coef_Judge'].abs()
top_judge = df_q5.nlargest(10, 'Abs_Coef_Judge')
for idx, row in top_judge.iterrows():
    direction = "↑" if row['Coef_Judge'] > 0 else "↓"
    print(f"{direction} {row['Feature']:35s} | {row['Coef_Judge']:+.6f}")

print("\nTOP 10 FEATURES FOR FAN VOTES:")
print("-" * 80)
df_q5['Abs_Coef_Fan'] = df_q5['Coef_Fan'].abs()
top_fan = df_q5.nlargest(10, 'Abs_Coef_Fan')
for idx, row in top_fan.iterrows():
    direction = "↑" if row['Coef_Fan'] > 0 else "↓"
    print(f"{direction} {row['Feature']:35s} | {row['Coef_Fan']:+.6f}")

# 分析舞伴影响
print("\n" + "="*80)
print("PARTNER ANALYSIS")
print("="*80)

partner_features = df_q5[df_q5['Feature'].str.startswith('Partner_')].copy()
partner_features['Abs_Coef_Judge'] = partner_features['Coef_Judge'].abs()
partner_features['Abs_Coef_Fan'] = partner_features['Coef_Fan'].abs()

print("\nTOP 5 PARTNERS FOR JUDGE SCORES:")
top_partners_judge = partner_features.nlargest(5, 'Abs_Coef_Judge')
for idx, row in top_partners_judge.iterrows():
    partner_name = row['Feature'].replace('Partner_', '')
    direction = "正向" if row['Coef_Judge'] > 0 else "负向"
    print(f"  {partner_name:30s} | {row['Coef_Judge']:+.6f} | {direction}")

print("\nTOP 5 PARTNERS FOR FAN VOTES:")
top_partners_fan = partner_features.nlargest(5, 'Abs_Coef_Fan')
for idx, row in top_partners_fan.iterrows():
    partner_name = row['Feature'].replace('Partner_', '')
    direction = "正向" if row['Coef_Fan'] > 0 else "负向"
    print(f"  {partner_name:30s} | {row['Coef_Fan']:+.6f} | {direction}")

# 分析行业影响
print("\n" + "="*80)
print("INDUSTRY ANALYSIS")
print("="*80)

industry_features = df_q5[df_q5['Feature'].str.startswith('Industry_')].copy()
industry_features['Abs_Coef_Judge'] = industry_features['Coef_Judge'].abs()

print("\nTOP 5 INDUSTRIES FOR JUDGE SCORES:")
top_industry_judge = industry_features.nlargest(5, 'Abs_Coef_Judge')
for idx, row in top_industry_judge.iterrows():
    industry_code = row['Feature'].replace('Industry_', '')
    direction = "正向" if row['Coef_Judge'] > 0 else "负向"
    print(f"  Industry {industry_code:20s} | {row['Coef_Judge']:+.6f} | {direction}")

# ============================================================================
# Q6 分析
# ============================================================================
print("\n" + "="*80)
print("Q6: NEW SYSTEM ANALYSIS")
print("="*80)

df_q6 = pd.read_csv('Q6_New_System_Simulation.csv')

print(f"\nTotal weeks simulated: {len(df_q6)}")
print(f"Improvement rate: {df_q6['Is_Improvement'].mean():.2%}")

# 分析被淘汰者的排名分布
print("\n" + "="*80)
print("ELIMINATED CONTESTANTS RANKING DISTRIBUTION")
print("="*80)

print("\nJudge Rank Distribution:")
judge_rank_dist = df_q6['Judge_Rank'].value_counts().sort_index()
for rank, count in judge_rank_dist.head(10).items():
    pct = count / len(df_q6) * 100
    bar = "█" * int(pct / 2)
    print(f"  Rank {rank:2.0f}: {bar} {count:3d} ({pct:5.1f}%)")

print("\nFan Rank Distribution:")
fan_rank_dist = df_q6['Fan_Rank'].value_counts().sort_index()
for rank, count in fan_rank_dist.head(10).items():
    pct = count / len(df_q6) * 100
    bar = "█" * int(pct / 2)
    print(f"  Rank {rank:2.0f}: {bar} {count:3d} ({pct:5.1f}%)")

# 分析冤案情况
print("\n" + "="*80)
print("INJUSTICE ANALYSIS")
print("="*80)

# 定义不同的冤案标准
injustice_top3 = (df_q6['Judge_Rank'] > 3).mean()
injustice_top2 = (df_q6['Judge_Rank'] > 2).mean()
injustice_top1 = (df_q6['Judge_Rank'] > 1).mean()

print(f"\n冤案率（裁判排名不在倒数第1）: {injustice_top1:.2%}")
print(f"冤案率（裁判排名不在倒数前2）: {injustice_top2:.2%}")
print(f"冤案率（裁判排名不在倒数前3）: {injustice_top3:.2%}")

# 分析新旧系统的差异
print("\n" + "="*80)
print("NEW VS OLD SYSTEM")
print("="*80)

# 计算一致性
df_q6_with_actual = df_q6.dropna(subset=['Actual_Eliminated'])
consistency = (df_q6_with_actual['New_System_Eliminated'] == df_q6_with_actual['Actual_Eliminated']).mean()
print(f"\n新系统与真实结果一致性: {consistency:.2%}")

# 统计信息
print(f"\n新系统被淘汰者统计:")
print(f"  平均裁判排名: {df_q6['Judge_Rank'].mean():.2f}")
print(f"  平均观众排名: {df_q6['Fan_Rank'].mean():.2f}")
print(f"  裁判排名标准差: {df_q6['Judge_Rank'].std():.2f}")
print(f"  观众排名标准差: {df_q6['Fan_Rank'].std():.2f}")

# 关键洞察
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n【Q5 关键发现】")
print("1. 年龄对裁判分数有显著负向影响（系数-0.49）")
print("2. Derek Hough是最能提升裁判分数的舞伴（系数+0.19）")
print("3. 观众投票受赛季影响最大（系数-0.009），说明后期赛季观众更分散")
print("4. 年龄对观众投票也有负向影响（系数-0.007）")

print("\n【Q6 关键发现】")
print("1. 新系统冤案率93.43%，仅比旧系统改善1.26%")
print("2. 新系统被淘汰者平均裁判排名8.21，说明仍然不是技术最差的")
print("3. 新系统被淘汰者平均观众排名4.44，说明观众投票仍有较大影响")
print("4. 冤案率高的根本原因：选手数量多时，排名靠后不等于技术最差")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
