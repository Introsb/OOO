"""
Relative Strength Analysis: Testing Week's Explanatory Power Beyond Denominator Effect
相对强度分析：测试Week特征在剔除分母效应后的解释力
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import json

def analyze_relative_strength():
    """分析Week对相对强度的解释力"""
    
    print("="*80)
    print("RELATIVE STRENGTH ANALYSIS: BEYOND DENOMINATOR EFFECT")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('../results/Processed_DWTS_Long_Format.csv')
    
    # 计算相对强度：实际得票率 / 平均得票率
    df['Avg_Vote_Share'] = 1.0 / df.groupby(['Season', 'Week'])['Name'].transform('count')
    df['Relative_Strength'] = df['Fan_Vote_Pct'] / df['Avg_Vote_Share']
    
    # 准备特征
    # 基础特征（不含Week）
    basic_features = ['Age', 'Industry_Code', 'Season']
    
    # Week特征
    week_features = ['Week', 'Week_Squared', 'Week_Cubed']
    df['Week_Squared'] = df['Week'] ** 2
    df['Week_Cubed'] = df['Week'] ** 3
    
    # 移除缺失值
    df_clean = df.dropna(subset=basic_features + week_features + ['Relative_Strength'])
    
    # 实验1：预测绝对得票率（Fan_Vote_Pct）
    print("\n1. PREDICTING ABSOLUTE VOTE SHARE (Fan_Vote_Pct)")
    
    X_basic = df_clean[basic_features]
    X_with_week = df_clean[basic_features + week_features]
    y_absolute = df_clean['Fan_Vote_Pct']
    
    # 基础模型
    model_basic_abs = Lasso(alpha=0.01, random_state=42)
    scores_basic_abs = cross_val_score(model_basic_abs, X_basic, y_absolute, cv=5, scoring='r2')
    r2_basic_abs = scores_basic_abs.mean()
    
    # Week模型
    model_week_abs = Lasso(alpha=0.01, random_state=42)
    scores_week_abs = cross_val_score(model_week_abs, X_with_week, y_absolute, cv=5, scoring='r2')
    r2_week_abs = scores_week_abs.mean()
    
    print(f"   Basic Features: R² = {r2_basic_abs:.4f}")
    print(f"   Basic + Week: R² = {r2_week_abs:.4f}")
    print(f"   Improvement: ΔR² = {r2_week_abs - r2_basic_abs:.4f} ({(r2_week_abs - r2_basic_abs)/r2_basic_abs*100:.1f}%)")
    
    # 实验2：预测相对强度（Relative_Strength）
    print("\n2. PREDICTING RELATIVE STRENGTH (Vote Share / Average)")
    
    y_relative = df_clean['Relative_Strength']
    
    # 基础模型
    model_basic_rel = Lasso(alpha=0.01, random_state=42)
    scores_basic_rel = cross_val_score(model_basic_rel, X_basic, y_relative, cv=5, scoring='r2')
    r2_basic_rel = scores_basic_rel.mean()
    
    # Week模型
    model_week_rel = Lasso(alpha=0.01, random_state=42)
    scores_week_rel = cross_val_score(model_week_rel, X_with_week, y_relative, cv=5, scoring='r2')
    r2_week_rel = scores_week_rel.mean()
    
    print(f"   Basic Features: R² = {r2_basic_rel:.4f}")
    print(f"   Basic + Week: R² = {r2_week_rel:.4f}")
    print(f"   Improvement: ΔR² = {r2_week_rel - r2_basic_rel:.4f} ({(r2_week_rel - r2_basic_rel)/abs(r2_basic_rel)*100:.1f}%)")
    
    # 实验3：预测排名（Rank）
    print("\n3. PREDICTING RANK (Ordinal Position)")
    
    # 计算每周的排名
    df_clean['Fan_Rank'] = df_clean.groupby(['Season', 'Week'])['Fan_Vote_Pct'].rank(ascending=False)
    y_rank = df_clean['Fan_Rank']
    
    # 基础模型
    model_basic_rank = Lasso(alpha=0.01, random_state=42)
    scores_basic_rank = cross_val_score(model_basic_rank, X_basic, y_rank, cv=5, scoring='r2')
    r2_basic_rank = scores_basic_rank.mean()
    
    # Week模型
    model_week_rank = Lasso(alpha=0.01, random_state=42)
    scores_week_rank = cross_val_score(model_week_rank, X_with_week, y_rank, cv=5, scoring='r2')
    r2_week_rank = scores_week_rank.mean()
    
    print(f"   Basic Features: R² = {r2_basic_rank:.4f}")
    print(f"   Basic + Week: R² = {r2_week_rank:.4f}")
    print(f"   Improvement: ΔR² = {r2_week_rank - r2_basic_rank:.4f}")
    
    # 结果汇总
    results = {
        'absolute_vote_share': {
            'basic_r2': float(r2_basic_abs),
            'week_r2': float(r2_week_abs),
            'improvement': float(r2_week_abs - r2_basic_abs),
            'improvement_pct': float((r2_week_abs - r2_basic_abs) / r2_basic_abs * 100)
        },
        'relative_strength': {
            'basic_r2': float(r2_basic_rel),
            'week_r2': float(r2_week_rel),
            'improvement': float(r2_week_rel - r2_basic_rel),
            'improvement_pct': float((r2_week_rel - r2_basic_rel) / abs(r2_basic_rel) * 100) if r2_basic_rel != 0 else 0
        },
        'rank': {
            'basic_r2': float(r2_basic_rank),
            'week_r2': float(r2_week_rank),
            'improvement': float(r2_week_rank - r2_basic_rank)
        }
    }
    
    # 判断
    print("\n4. CONCLUSION")
    
    if r2_week_rel > 0.1:
        print(f"   ✓ Week retains explanatory power ({r2_week_rel:.1%}) for RELATIVE strength")
        print(f"   → Week captures structural evolution, not just denominator effect")
    else:
        print(f"   ⚠ Week's explanatory power drops to {r2_week_rel:.1%} for relative strength")
        print(f"   → Week primarily captures denominator effect (contestant reduction)")
        print(f"   → This still supports Q6: structural evolution requires dynamic weighting")
    
    # 保存结果
    with open('../results/Relative_Strength_Analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to ../results/Relative_Strength_Analysis.json")
    
    return results

if __name__ == '__main__':
    results = analyze_relative_strength()
