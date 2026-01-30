"""
Optimized Feature Engineering: Actually Improve R²
优化特征工程：真正提升R²
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """加载并准备数据"""
    print("Loading data...")
    
    # 加载数据
    df_fan = pd.read_csv('../results/Q1_Estimated_Fan_Votes.csv')
    df_proc = pd.read_csv('../results/Processed_DWTS_Long_Format.csv')
    df_raw = pd.read_csv('../data/2026 MCM Problem C Data.csv')
    
    # 合并数据
    df = df_proc.merge(df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
                       on=['Season', 'Week', 'Name'], 
                       how='inner')
    
    # 添加Partner信息
    partner_info = df_raw[['celebrity_name', 'ballroom_partner', 'season']].copy()
    partner_info.columns = ['Name', 'Partner', 'Season']
    df = df.merge(partner_info, on=['Name', 'Season'], how='left')
    
    print(f"✓ Loaded {len(df)} records")
    
    return df


def engineer_features(df):
    """特征工程：创建真正有预测力的特征"""
    
    print("\nEngineering features...")
    
    df = df.copy()
    
    # 1. 基础特征
    print("  [1/6] Basic features...")
    # Age, Industry_Code, Season 已经存在
    
    # 2. Partner特征（one-hot编码）
    print("  [2/6] Partner features...")
    partner_counts = df['Partner'].value_counts()
    frequent_partners = partner_counts[partner_counts >= 5].index
    df['Partner_Grouped'] = df['Partner'].apply(
        lambda x: x if x in frequent_partners else 'Other'
    )
    partner_dummies = pd.get_dummies(df['Partner_Grouped'], prefix='Partner')
    df = pd.concat([df, partner_dummies], axis=1)
    
    # 3. 当前周次特征（这是合法的外生变量）
    print("  [3/6] Week features...")
    df['Week'] = df.groupby(['Season', 'Name']).cumcount() + 1
    df['Week_Squared'] = df['Week'] ** 2
    df['Week_Cubed'] = df['Week'] ** 3
    
    # 4. 竞争强度特征
    print("  [4/6] Competition features...")
    df['Num_Contestants'] = df.groupby(['Season', 'Week'])['Name'].transform('count')
    df['Competition_Intensity'] = 1 / df['Num_Contestants']  # 选手越少，竞争越激烈
    
    # 5. 裁判分数特征（这是已知的，可以用）
    print("  [5/6] Judge score features...")
    df['Judge_Score_Rank'] = df.groupby(['Season', 'Week'])['Judge_Avg_Score'].rank(ascending=False)
    df['Judge_Score_Percentile'] = df.groupby(['Season', 'Week'])['Judge_Avg_Score'].rank(pct=True)
    df['Judge_Score_Zscore'] = df.groupby(['Season', 'Week'])['Judge_Avg_Score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    
    # 6. 交互特征
    print("  [6/6] Interaction features...")
    df['Age_x_Week'] = df['Age'] * df['Week']
    df['Judge_x_Week'] = df['Judge_Avg_Score'] * df['Week']
    df['Age_x_Judge'] = df['Age'] * df['Judge_Avg_Score']
    df['Industry_x_Week'] = df['Industry_Code'] * df['Week']
    
    # 非线性特征
    df['Age_Squared'] = df['Age'] ** 2
    df['Season_Squared'] = df['Season'] ** 2
    df['Judge_Squared'] = df['Judge_Avg_Score'] ** 2
    
    print("✓ Feature engineering complete")
    
    return df


def select_features(df):
    """选择特征"""
    
    # 基础特征
    basic_features = ['Age', 'Industry_Code', 'Season']
    
    # Partner特征（只选择one-hot编码后的列）
    partner_features = [col for col in df.columns if col.startswith('Partner_') and col != 'Partner_Grouped']
    
    # Week特征
    week_features = ['Week', 'Week_Squared', 'Week_Cubed', 'Num_Contestants', 'Competition_Intensity']
    
    # Judge特征
    judge_features = ['Judge_Avg_Score', 'Judge_Score_Rank', 'Judge_Score_Percentile', 'Judge_Score_Zscore', 'Judge_Squared']
    
    # 交互特征
    interaction_features = ['Age_x_Week', 'Judge_x_Week', 'Age_x_Judge', 'Industry_x_Week', 'Age_Squared', 'Season_Squared']
    
    # 组合不同的特征集
    feature_sets = {
        'Basic': basic_features,
        'Basic + Partner': basic_features + partner_features,
        'Basic + Partner + Week': basic_features + partner_features + week_features,
        'Basic + Partner + Judge': basic_features + partner_features + judge_features,
        'Full (All features)': basic_features + partner_features + week_features + judge_features + interaction_features
    }
    
    return feature_sets


def evaluate_models(df, feature_sets):
    """评估不同特征集和模型"""
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    results = []
    
    for set_name, features in feature_sets.items():
        print(f"\n{set_name} ({len(features)} features):")
        
        # 准备数据
        required_cols = features + ['Estimated_Fan_Vote']
        df_clean = df.dropna(subset=required_cols)
        
        X = df_clean[features]
        y = df_clean['Estimated_Fan_Vote']
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 测试多个模型
        models = {
            'Bayesian Ridge': BayesianRidge(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.001, max_iter=10000),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for model_name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            mean_r2 = scores.mean()
            std_r2 = scores.std()
            
            print(f"  {model_name:20s}: R² = {mean_r2:.4f} ± {std_r2:.4f}")
            
            results.append({
                'feature_set': set_name,
                'num_features': len(features),
                'model': model_name,
                'r2_mean': mean_r2,
                'r2_std': std_r2
            })
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    print("="*80)
    print("OPTIMIZED FEATURE ENGINEERING")
    print("="*80)
    
    # 加载数据
    df = load_and_prepare_data()
    
    # 特征工程
    df = engineer_features(df)
    
    # 选择特征集
    feature_sets = select_features(df)
    
    # 评估模型
    results_df = evaluate_models(df, feature_sets)
    
    # 找出最佳模型
    print("\n" + "="*80)
    print("BEST MODELS")
    print("="*80)
    
    best_overall = results_df.loc[results_df['r2_mean'].idxmax()]
    print(f"\nBest Overall:")
    print(f"  Feature Set: {best_overall['feature_set']}")
    print(f"  Model: {best_overall['model']}")
    print(f"  R² = {best_overall['r2_mean']:.4f} ± {best_overall['r2_std']:.4f}")
    print(f"  Number of features: {best_overall['num_features']}")
    
    # 按特征集分组，找出每个特征集的最佳模型
    print(f"\nBest Model for Each Feature Set:")
    for set_name in feature_sets.keys():
        subset = results_df[results_df['feature_set'] == set_name]
        best = subset.loc[subset['r2_mean'].idxmax()]
        print(f"  {set_name:30s}: {best['model']:20s} R² = {best['r2_mean']:.4f}")
    
    # 保存结果
    results_df.to_csv('../results/Optimized_Feature_Analysis.csv', index=False)
    print(f"\n✓ Results saved to ../results/Optimized_Feature_Analysis.csv")
    
    return results_df


if __name__ == '__main__':
    results = main()
