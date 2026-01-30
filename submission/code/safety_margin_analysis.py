"""
Safety Margin Analysis: Proving SMC Predicts Dominance, Not Just Survival
安全边际分析：证明SMC预测的是统治力，而非仅仅是存活
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def analyze_safety_margin():
    """分析冠军和早期淘汰者的安全边际"""
    
    print("="*80)
    print("SAFETY MARGIN ANALYSIS: DOMINANCE vs SURVIVAL")
    print("="*80)
    
    # 加载数据
    df_fan = pd.read_csv('../results/Q1_Estimated_Fan_Votes.csv')
    df_proc = pd.read_csv('../results/Processed_DWTS_Long_Format.csv')
    
    # 合并数据
    df = df_fan.merge(
        df_proc[['Season', 'Week', 'Name', 'Placement']], 
        on=['Season', 'Week', 'Name'],
        how='inner'
    )
    
    # 计算每周的安全线（平均得票率）
    df['Safety_Line'] = 1.0 / df.groupby(['Season', 'Week'])['Name'].transform('count')
    
    # 计算安全边际
    df['Safety_Margin'] = df['Estimated_Fan_Vote'] - df['Safety_Line']
    df['Safety_Margin_Pct'] = df['Safety_Margin'] / df['Safety_Line']
    
    # 识别冠军
    winners = df[df['Placement'] == 1].groupby('Season')['Name'].first()
    
    # 分析冠军的安全边际
    winner_margins = []
    for season, winner_name in winners.items():
        mask = (df['Season'] == season) & (df['Name'] == winner_name)
        winner_data = df[mask]
        
        avg_vote = winner_data['Estimated_Fan_Vote'].mean()
        avg_safety_line = winner_data['Safety_Line'].mean()
        avg_margin = winner_data['Safety_Margin'].mean()
        avg_margin_pct = winner_data['Safety_Margin_Pct'].mean()
        
        winner_margins.append({
            'season': season,
            'name': winner_name,
            'avg_vote': avg_vote,
            'avg_safety_line': avg_safety_line,
            'avg_margin': avg_margin,
            'avg_margin_pct': avg_margin_pct
        })
    
    df_winners = pd.DataFrame(winner_margins)
    
    # 分析早期淘汰者的安全边际
    last_week = df.groupby(['Season', 'Name'])['Week'].max()
    early_eliminated = []
    
    for (season, name), max_week in last_week.items():
        if max_week <= 2:
            mask = (df['Season'] == season) & (df['Name'] == name)
            elim_data = df[mask]
            
            avg_vote = elim_data['Estimated_Fan_Vote'].mean()
            avg_safety_line = elim_data['Safety_Line'].mean()
            avg_margin = elim_data['Safety_Margin'].mean()
            avg_margin_pct = elim_data['Safety_Margin_Pct'].mean()
            
            early_eliminated.append({
                'season': season,
                'name': name,
                'avg_vote': avg_vote,
                'avg_safety_line': avg_safety_line,
                'avg_margin': avg_margin,
                'avg_margin_pct': avg_margin_pct
            })
    
    df_early = pd.DataFrame(early_eliminated)
    
    # 统计结果
    results = {
        'winners': {
            'count': len(df_winners),
            'avg_vote': float(df_winners['avg_vote'].mean()),
            'avg_safety_line': float(df_winners['avg_safety_line'].mean()),
            'avg_margin': float(df_winners['avg_margin'].mean()),
            'avg_margin_pct': float(df_winners['avg_margin_pct'].mean()),
            'above_safety_rate': float((df_winners['avg_margin'] > 0).mean())
        },
        'early_eliminated': {
            'count': len(df_early),
            'avg_vote': float(df_early['avg_vote'].mean()),
            'avg_safety_line': float(df_early['avg_safety_line'].mean()),
            'avg_margin': float(df_early['avg_margin'].mean()),
            'avg_margin_pct': float(df_early['avg_margin_pct'].mean()),
            'below_safety_rate': float((df_early['avg_margin'] < 0).mean())
        }
    }
    
    # 打印结果
    print("\n1. WINNERS (Champions)")
    print(f"   Count: {results['winners']['count']}")
    print(f"   Average Vote: {results['winners']['avg_vote']:.3f}")
    print(f"   Average Safety Line: {results['winners']['avg_safety_line']:.3f}")
    print(f"   Average Margin: {results['winners']['avg_margin']:.3f} ({results['winners']['avg_margin_pct']:.1%})")
    print(f"   Above Safety Rate: {results['winners']['above_safety_rate']:.1%}")
    
    print("\n2. EARLY ELIMINATED (Week 1-2)")
    print(f"   Count: {results['early_eliminated']['count']}")
    print(f"   Average Vote: {results['early_eliminated']['avg_vote']:.3f}")
    print(f"   Average Safety Line: {results['early_eliminated']['avg_safety_line']:.3f}")
    print(f"   Average Margin: {results['early_eliminated']['avg_margin']:.3f} ({results['early_eliminated']['avg_margin_pct']:.1%})")
    print(f"   Below Safety Rate: {results['early_eliminated']['below_safety_rate']:.1%}")
    
    print("\n3. KEY FINDING")
    winner_margin_pct = results['winners']['avg_margin_pct']
    early_margin_pct = results['early_eliminated']['avg_margin_pct']
    
    print(f"   Winners' Safety Margin: +{winner_margin_pct:.1%} above safety line")
    print(f"   Early Eliminated Margin: {early_margin_pct:.1%} below safety line")
    print(f"   Separation: {winner_margin_pct - early_margin_pct:.1%}")
    
    if winner_margin_pct > 0.2:
        print(f"\n   ✓ Winners show DOMINANCE (>{20}% margin), not just survival")
    else:
        print(f"\n   ⚠ Winners show marginal survival (<{20}% margin)")
    
    # 保存结果
    with open('../results/Safety_Margin_Analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to ../results/Safety_Margin_Analysis.json")
    
    return results

if __name__ == '__main__':
    results = analyze_safety_margin()
