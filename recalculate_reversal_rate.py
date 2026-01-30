"""
重新计算合理的逆转率
真正的逆转：Judge/Fan排名与最终结果严重不符
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("重新计算逆转率 - 更合理的定义")
print("=" * 80)

# Load data
df = pd.read_csv('submission/results/Problem_Driven_Dataset.csv')

print(f"\nLoaded {len(df)} records")

# Calculate Combined_Score if not exists
if 'Combined_Score' not in df.columns:
    df['Combined_Score'] = df['Judge_Avg_Score'] + df['Estimated_Fan_Vote']

# 计算每周的Judge和Fan排名
df['Judge_Rank_Week'] = df.groupby(['Season', 'Week'])['Judge_Avg_Score'].rank(ascending=False, method='min')
df['Fan_Rank_Week'] = df.groupby(['Season', 'Week'])['Estimated_Fan_Vote'].rank(ascending=False, method='min')
df['Combined_Rank_Week'] = df.groupby(['Season', 'Week'])['Combined_Score'].rank(ascending=False, method='min')

# 定义逆转的几种情况
reversals = []

for (season, week), group in df.groupby(['Season', 'Week']):
    if len(group) < 2:
        continue
    
    # 找出被淘汰的选手（Combined_Score最低）
    eliminated_idx = group['Combined_Score'].idxmin()
    eliminated = group.loc[eliminated_idx]
    
    # 找出Judge和Fan的最爱（排名第1）
    judge_favorite_idx = group['Judge_Rank_Week'].idxmin()
    fan_favorite_idx = group['Fan_Rank_Week'].idxmin()
    
    # 找出Judge和Fan的最不喜欢（排名最后）
    judge_worst_idx = group['Judge_Rank_Week'].idxmax()
    fan_worst_idx = group['Fan_Rank_Week'].idxmax()
    
    # 类型1: Judge最爱被淘汰
    judge_favorite_eliminated = (eliminated_idx == judge_favorite_idx)
    
    # 类型2: Fan最爱被淘汰
    fan_favorite_eliminated = (eliminated_idx == fan_favorite_idx)
    
    # 类型3: 被淘汰的人在Judge排名前50%
    eliminated_judge_rank = eliminated['Judge_Rank_Week']
    judge_top_half_eliminated = (eliminated_judge_rank <= len(group) / 2)
    
    # 类型4: 被淘汰的人在Fan排名前50%
    eliminated_fan_rank = eliminated['Fan_Rank_Week']
    fan_top_half_eliminated = (eliminated_fan_rank <= len(group) / 2)
    
    # 类型5: 被淘汰的人既不是Judge最差也不是Fan最差
    neither_worst = (eliminated_idx != judge_worst_idx) and (eliminated_idx != fan_worst_idx)
    
    reversals.append({
        'Season': season,
        'Week': week,
        'Num_Contestants': len(group),
        'Eliminated': eliminated['Name'],
        'Judge_Rank': eliminated_judge_rank,
        'Fan_Rank': eliminated_fan_rank,
        'Combined_Rank': eliminated['Combined_Rank_Week'],
        'Judge_Favorite_Eliminated': judge_favorite_eliminated,
        'Fan_Favorite_Eliminated': fan_favorite_eliminated,
        'Judge_Top_Half_Eliminated': judge_top_half_eliminated,
        'Fan_Top_Half_Eliminated': fan_top_half_eliminated,
        'Neither_Worst': neither_worst
    })

df_reversals = pd.DataFrame(reversals)

# 计算各种逆转率
print("\n" + "=" * 80)
print("逆转率分析（更合理的定义）")
print("=" * 80)

total_weeks = len(df_reversals)

print(f"\n总周数: {total_weeks}")

print("\n【类型1】Judge最爱被淘汰:")
judge_fav_elim = df_reversals['Judge_Favorite_Eliminated'].sum()
print(f"  发生次数: {judge_fav_elim}")
print(f"  逆转率: {judge_fav_elim/total_weeks*100:.2f}%")

print("\n【类型2】Fan最爱被淘汰:")
fan_fav_elim = df_reversals['Fan_Favorite_Eliminated'].sum()
print(f"  发生次数: {fan_fav_elim}")
print(f"  逆转率: {fan_fav_elim/total_weeks*100:.2f}%")

print("\n【类型3】Judge排名前50%的选手被淘汰:")
judge_top_half = df_reversals['Judge_Top_Half_Eliminated'].sum()
print(f"  发生次数: {judge_top_half}")
print(f"  逆转率: {judge_top_half/total_weeks*100:.2f}%")

print("\n【类型4】Fan排名前50%的选手被淘汰:")
fan_top_half = df_reversals['Fan_Top_Half_Eliminated'].sum()
print(f"  发生次数: {fan_top_half}")
print(f"  逆转率: {fan_top_half/total_weeks*100:.2f}%")

print("\n【类型5】被淘汰的人既不是Judge最差也不是Fan最差（冤案）:")
neither_worst = df_reversals['Neither_Worst'].sum()
print(f"  发生次数: {neither_worst}")
print(f"  冤案率: {neither_worst/total_weeks*100:.2f}%")

# 分析被淘汰选手的排名分布
print("\n" + "=" * 80)
print("被淘汰选手的排名分布")
print("=" * 80)

print("\nJudge排名分布:")
print(df_reversals['Judge_Rank'].value_counts().sort_index())

print("\nFan排名分布:")
print(df_reversals['Fan_Rank'].value_counts().sort_index())

# 计算平均排名
print(f"\n被淘汰选手的平均Judge排名: {df_reversals['Judge_Rank'].mean():.2f}")
print(f"被淘汰选手的平均Fan排名: {df_reversals['Fan_Rank'].mean():.2f}")

# 保存结果
df_reversals.to_csv('Reversal_Analysis_Corrected.csv', index=False)
print(f"\n✓ 结果已保存到 Reversal_Analysis_Corrected.csv")

# 总结
print("\n" + "=" * 80)
print("结论")
print("=" * 80)

print("\n之前的100%逆转率定义有误，它只是说明Rank制和Percent制")
print("会淘汰不同的人，但这不代表'排名被完全逆转'。")

print(f"\n更合理的逆转率应该是:")
print(f"  • Judge最爱被淘汰: {judge_fav_elim/total_weeks*100:.1f}%")
print(f"  • Fan最爱被淘汰: {fan_fav_elim/total_weeks*100:.1f}%")
print(f"  • Judge前50%被淘汰: {judge_top_half/total_weeks*100:.1f}%")
print(f"  • Fan前50%被淘汰: {fan_top_half/total_weeks*100:.1f}%")
print(f"  • 冤案率（既不是Judge最差也不是Fan最差）: {neither_worst/total_weeks*100:.1f}%")

print("\n这些数字更能反映系统的真实'不公平'程度。")
