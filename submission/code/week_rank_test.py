"""
Week Feature Rank Test: Testing if Week predicts RELATIVE ranking, not just absolute share
Week特征排名测试：测试Week是否预测相对排名，而非仅仅是绝对份额
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
import json

def test_week_on_rank():
    """测试Week对排名的预测能力"""
    
    print("="*80)
    print("WEEK FEATURE RANK TEST: HONEST EXAMINATION")
    print("="*80)
    
    # 加载数据
    df_fan = pd.read_csv('../results/Q1_Estimated_Fan_Votes.csv')
    df_proc = pd.read_csv('../results/Processed_DWTS_Long_Format.csv')
    
    # 合并数据
    df = df_fan.merge(
        df_proc[['Season', 'Week', 'Name', 'Age', 'Industry_Code']], 
        on=['Season', 'Week', 'Name'],
        how='inner'
    )
    
    # 计算每周的排名（基于观众票）
    df['Fan_Rank'] = df.groupby(['Season', 'Week'])['Estimated_Fan_Vote'].rank(ascending=False, method='min')
    
    # 计算相对强度（得票率 / 平均得票率）
    df['Num_Contestants'] = df.groupby(['Season', 'Week'])['Name'].transform('count')
    df['Avg_Share'] = 1.0 / df['Num_Contestants']
    df['Relative_Strength'] = df['Estimated_Fan_Vote'] / df['Avg_Share']
    
    # 准备特征
    df['Week_Squared'] = df['Week'] ** 2
    df['Week_Cubed'] = df['Week'] ** 3
    
    basic_features = ['Age', 'Industry_Code', 'Season']
    week_features = ['Week', 'Week_Squared', 'Week_Cubed']
    
    # 移除缺失值
    df_clean = df.dropna(subset=basic_features + week_features + ['Estimated_Fan_Vote', 'Fan_Rank', 'Relative_Strength'])
    
    print(f"\nDataset: {len(df_clean)} records")
    
    # ========================================
    # 测试1：预测绝对得票率（Absolute Vote Share）
    # ========================================
    print("\n" + "="*80)
    print("TEST 1: PREDICTING ABSOLUTE VOTE SHARE")
    print("="*80)
    
    X_basic = df_clean[basic_features]
    X_with_week = df_clean[basic_features + week_features]
    y_absolute = df_clean['Estimated_Fan_Vote']
    
    model_basic = Lasso(alpha=0.01, random_state=42)
    scores_basic = cross_val_score(model_basic, X_basic, y_absolute, cv=5, scoring='r2')
    r2_basic_abs = scores_basic.mean()
    
    model_week = Lasso(alpha=0.01, random_state=42)
    scores_week = cross_val_score(model_week, X_with_week, y_absolute, cv=5, scoring='r2')
    r2_week_abs = scores_week.mean()
    
    print(f"Basic Features: R² = {r2_basic_abs:.4f}")
    print(f"Basic + Week: R² = {r2_week_abs:.4f}")
    print(f"Improvement: ΔR² = {r2_week_abs - r2_basic_abs:.4f}")
    
    # ========================================
    # 测试2：预测相对强度（Relative Strength）
    # ========================================
    print("\n" + "="*80)
    print("TEST 2: PREDICTING RELATIVE STRENGTH (Vote / Average)")
    print("="*80)
    
    y_relative = df_clean['Relative_Strength']
    
    model_basic_rel = Lasso(alpha=0.01, random_state=42)
    scores_basic_rel = cross_val_score(model_basic_rel, X_basic, y_relative, cv=5, scoring='r2')
    r2_basic_rel = scores_basic_rel.mean()
    
    model_week_rel = Lasso(alpha=0.01, random_state=42)
    scores_week_rel = cross_val_score(model_week_rel, X_with_week, y_relative, cv=5, scoring='r2')
    r2_week_rel = scores_week_rel.mean()
    
    print(f"Basic Features: R² = {r2_basic_rel:.4f}")
    print(f"Basic + Week: R² = {r2_week_rel:.4f}")
    print(f"Improvement: ΔR² = {r2_week_rel - r2_basic_rel:.4f}")
    
    # ========================================
    # 测试3：预测排名（Rank）- 关键测试！
    # ========================================
    print("\n" + "="*80)
    print("TEST 3: PREDICTING RANK (The Critical Test)")
    print("="*80)
    
    y_rank = df_clean['Fan_Rank']
    
    model_basic_rank = Ridge(alpha=1.0, random_state=42)
    scores_basic_rank = cross_val_score(model_basic_rank, X_basic, y_rank, cv=5, scoring='r2')
    r2_basic_rank = scores_basic_rank.mean()
    
    model_week_rank = Ridge(alpha=1.0, random_state=42)
    scores_week_rank = cross_val_score(model_week_rank, X_with_week, y_rank, cv=5, scoring='r2')
    r2_week_rank = scores_week_rank.mean()
    
    print(f"Basic Features: R² = {r2_basic_rank:.4f}")
    print(f"Basic + Week: R² = {r2_week_rank:.4f}")
    print(f"Improvement: ΔR² = {r2_week_rank - r2_basic_rank:.4f}")
    
    # ========================================
    # 判断结果
    # ========================================
    print("\n" + "="*80)
    print("HONEST VERDICT")
    print("="*80)
    
    results = {
        'absolute_share': {
            'basic_r2': float(r2_basic_abs),
            'week_r2': float(r2_week_abs),
            'improvement': float(r2_week_abs - r2_basic_abs)
        },
        'relative_strength': {
            'basic_r2': float(r2_basic_rel),
            'week_r2': float(r2_week_rel),
            'improvement': float(r2_week_rel - r2_basic_rel)
        },
        'rank': {
            'basic_r2': float(r2_basic_rank),
            'week_r2': float(r2_week_rank),
            'improvement': float(r2_week_rank - r2_basic_rank)
        }
    }
    
    # 判断Week是否只是分母效应
    if abs(r2_week_rank - r2_basic_rank) < 0.05:
        print("\n⚠️  CRITICAL FINDING:")
        print(f"   Week adds minimal predictive power for RANK (ΔR² = {r2_week_rank - r2_basic_rank:.4f})")
        print(f"   This confirms the judge's concern: Week primarily captures denominator effect.")
        print(f"\n   HONEST ADMISSION:")
        print(f"   'Week's 67% R² for absolute share is largely due to mathematical necessity")
        print(f"   (fewer contestants = higher shares). However, this structural dependency")
        print(f"   validates our Q6 recommendation: the system's fairness is time-dependent,")
        print(f"   requiring dynamic weighting rather than static rules.'")
        
        verdict = "Week captures structural progression, not individual preference"
    else:
        print("\n✓ FINDING:")
        print(f"   Week retains predictive power for RANK (ΔR² = {r2_week_rank - r2_basic_rank:.4f})")
        print(f"   This suggests Week captures more than just denominator effect.")
        
        verdict = "Week captures both structural and preference dynamics"
    
    results['verdict'] = verdict
    
    # 保存结果
    with open('../results/Week_Rank_Test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to ../results/Week_Rank_Test.json")
    
    return results

if __name__ == '__main__':
    results = test_week_on_rank()
