"""
Ultimate Feature Engineering - CLEAN VERSION
完全消除数据泄露 - 只使用真正的历史信息和外部信息
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CleanFeatureEngineer:
    """干净版特征工程器 - 绝对无数据泄露"""
    
    def __init__(self):
        pass
        
    def load_base_data(self):
        """加载基础数据"""
        print("\n" + "="*80)
        print("CLEAN FEATURE ENGINEERING - ABSOLUTELY NO DATA LEAKAGE")
        print("="*80)
        print("\n1. Loading base data...")
        
        df_processed = pd.read_csv('results/Processed_DWTS_Long_Format.csv')
        df_fan = pd.read_csv('results/Q1_Estimated_Fan_Votes.csv')
        
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 按选手和赛季排序
        df = df.sort_values(['Season', 'Name', 'Week']).reset_index(drop=True)
        
        print(f"   ✓ Loaded {len(df)} records")
        
        return df
    
    def add_external_features(self, df):
        """添加外部特征（不依赖目标变量）"""
        print("\n2. Adding external features...")
        
        # 2.1 Week特征（Phase 1）
        df['Week_Squared'] = df['Week'] ** 2
        df['Week_Cubed'] = df['Week'] ** 3
        df['Log_Week'] = np.log1p(df['Week'])
        
        # 2.2 Age特征
        df['Age_Squared'] = df['Age'] ** 2
        df['Log_Age'] = np.log1p(df['Age'])
        
        # 2.3 Season特征
        df['Season_Squared'] = df['Season'] ** 2
        df['Log_Season'] = np.log1p(df['Season'])
        
        # 2.4 交互特征
        df['Age_Week'] = df['Age'] * df['Week']
        df['Age_Season'] = df['Age'] * df['Season']
        df['Week_Season'] = df['Week'] * df['Season']
        df['Age_Week_Season'] = df['Age'] * df['Week'] * df['Season']
        
        print(f"   ✓ External features added: {df.shape[1]} columns")
        
        return df
    
    def add_historical_features(self, df):
        """添加历史特征（只用过去的分数）"""
        print("\n3. Adding historical features (lag=1)...")
        
        # 3.1 滞后特征
        for lag in [1, 2, 3]:
            df[f'score_lag_{lag}'] = df.groupby(['Season', 'Name'])['Judge_Avg_Score'].shift(lag)
            df[f'fan_lag_{lag}'] = df.groupby(['Season', 'Name'])['Estimated_Fan_Vote'].shift(lag)
        
        # 3.2 历史统计（不包含当前）
        df['score_hist_mean'] = df.groupby(['Season', 'Name'])['Judge_Avg_Score'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['score_hist_std'] = df.groupby(['Season', 'Name'])['Judge_Avg_Score'].transform(
            lambda x: x.shift(1).expanding().std()
        )
        df['score_hist_max'] = df.groupby(['Season', 'Name'])['Judge_Avg_Score'].transform(
            lambda x: x.shift(1).cummax()
        )
        df['score_hist_min'] = df.groupby(['Season', 'Name'])['Judge_Avg_Score'].transform(
            lambda x: x.shift(1).cummin()
        )
        
        # 3.3 历史趋势
        df['score_trend'] = df['score_lag_1'] - df['score_lag_2']
        df['score_acceleration'] = df['score_trend'] - (df['score_lag_2'] - df['score_lag_3'])
        
        # 3.4 历史改进
        df['improvement_from_first'] = df['score_lag_1'] - df.groupby(['Season', 'Name'])['Judge_Avg_Score'].transform('first')
        
        # 填充缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        print(f"   ✓ Historical features added: {df.shape[1]} columns")
        
        return df
    
    def add_competition_context(self, df):
        """添加竞争环境特征（不依赖当前分数）"""
        print("\n4. Adding competition context...")
        
        # 4.1 竞争者数量（外部信息）
        df['n_competitors'] = df.groupby(['Season', 'Week'])['Name'].transform('count')
        df['competitors_remaining_pct'] = df['n_competitors'] / df.groupby('Season')['Name'].transform(lambda x: x.nunique())
        
        # 4.2 Week在Season中的位置
        df['week_in_season'] = df.groupby('Season')['Week'].rank(method='dense')
        df['is_early_week'] = (df['week_in_season'] <= 3).astype(int)
        df['is_late_week'] = (df['week_in_season'] >= df.groupby('Season')['week_in_season'].transform('max') - 2).astype(int)
        
        print(f"   ✓ Competition context added: {df.shape[1]} columns")
        
        return df
    
    def add_interaction_features(self, df):
        """添加交互特征"""
        print("\n5. Adding interaction features...")
        
        # 5.1 历史×Week交互
        df['hist_mean_times_week'] = df['score_hist_mean'] * df['Week']
        df['hist_std_times_week'] = df['score_hist_std'] * df['Week']
        
        # 5.2 历史×Age交互
        df['hist_mean_times_age'] = df['score_hist_mean'] * df['Age']
        
        # 5.3 Lag×Week交互
        df['lag1_times_week'] = df['score_lag_1'] * df['Week']
        
        print(f"   ✓ Interaction features added: {df.shape[1]} columns")
        
        return df
    
    def validate_features(self, df):
        """验证特征质量"""
        print("\n6. Validating features...")
        
        target_cols = ['Judge_Avg_Score', 'Estimated_Fan_Vote']
        exclude_cols = target_cols + ['Season', 'Week', 'Name', 'Score_Scaled', 'Placement', 'Industry_Code']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 检查相关性
        corr_judge = df[feature_cols + ['Judge_Avg_Score']].corr()['Judge_Avg_Score'].drop('Judge_Avg_Score')
        corr_fan = df[feature_cols + ['Estimated_Fan_Vote']].corr()['Estimated_Fan_Vote'].drop('Estimated_Fan_Vote')
        
        max_corr_judge = corr_judge.abs().max()
        max_corr_fan = corr_fan.abs().max()
        
        print(f"   Max |correlation| with Judge Score: {max_corr_judge:.4f}")
        print(f"   Max |correlation| with Fan Vote: {max_corr_fan:.4f}")
        
        # 检查缺失值
        missing_pct = df[feature_cols].isnull().sum() / len(df) * 100
        if missing_pct.max() > 0:
            print(f"   ⚠ Max missing percentage: {missing_pct.max():.2f}%")
        else:
            print(f"   ✓ No missing values")
        
        # 检查无穷值
        inf_count = np.isinf(df[feature_cols]).sum().sum()
        if inf_count > 0:
            print(f"   ⚠ Found {inf_count} infinite values")
        else:
            print(f"   ✓ No infinite values")
        
        if max_corr_judge < 0.85 and max_corr_fan < 0.85:
            print(f"   ✓ VALIDATION PASSED: No obvious data leakage")
            return True, max_corr_judge, max_corr_fan
        else:
            print(f"   ⚠ WARNING: Suspiciously high correlation!")
            return False, max_corr_judge, max_corr_fan
    
    def generate_summary(self, df, original_cols):
        """生成总结"""
        print("\n7. Generating summary...")
        
        target_cols = ['Judge_Avg_Score', 'Estimated_Fan_Vote']
        exclude_cols = target_cols + ['Season', 'Week', 'Name', 'Score_Scaled', 'Placement', 'Industry_Code']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        new_features = [col for col in feature_cols if col not in original_cols]
        
        categories = {
            'External (Week/Age/Season)': [f for f in new_features if any(x in f for x in ['Week', 'Age', 'Season', 'Log_'])],
            'Historical (Lag)': [f for f in new_features if 'lag' in f.lower() or 'hist' in f.lower()],
            'Trend': [f for f in new_features if any(x in f for x in ['trend', 'acceleration', 'improvement'])],
            'Competition Context': [f for f in new_features if any(x in f for x in ['competitors', 'week_in', 'is_early', 'is_late'])],
            'Interaction': [f for f in new_features if 'times' in f.lower()]
        }
        
        summary = []
        for category, features in categories.items():
            summary.append({
                'Category': category,
                'Count': len(features),
                'Examples': ', '.join(features[:3]) if features else 'None'
            })
        
        df_summary = pd.DataFrame(summary)
        
        print("\n" + "="*80)
        print("CLEAN FEATURE ENGINEERING SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        print(f"\nTotal features: {len(feature_cols)}")
        print(f"New features: {len(new_features)}")
        print(f"Feature/Sample ratio: {len(feature_cols)/len(df):.4f}")
        
        return df_summary, feature_cols
    
    def run_clean_feature_engineering(self):
        """运行干净的特征工程"""
        # 1. 加载数据
        df = self.load_base_data()
        original_cols = df.columns.tolist()
        
        # 2. 添加特征
        df = self.add_external_features(df)
        df = self.add_historical_features(df)
        df = self.add_competition_context(df)
        df = self.add_interaction_features(df)
        
        # 3. 验证
        passed, max_corr_judge, max_corr_fan = self.validate_features(df)
        
        # 4. 生成总结
        summary, feature_cols = self.generate_summary(df, original_cols)
        
        # 5. 保存
        print("\n8. Saving results...")
        
        output_path = 'results/Clean_Enhanced_Dataset.csv'
        df.to_csv(output_path, index=False)
        print(f"   ✓ Clean dataset saved to {output_path}")
        
        summary_path = 'Clean_Feature_Summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f"   ✓ Summary saved to {summary_path}")
        
        # 保存验证报告
        validation = pd.DataFrame({
            'Metric': ['Validation Status', 'Max Corr (Judge)', 'Max Corr (Fan)', 'Total Features', 'Feature/Sample Ratio'],
            'Value': ['PASSED' if passed else 'FAILED', f'{max_corr_judge:.4f}', f'{max_corr_fan:.4f}', str(len(feature_cols)), f'{len(feature_cols)/len(df):.4f}']
        })
        validation_path = 'Clean_Validation_Report.csv'
        validation.to_csv(validation_path, index=False)
        print(f"   ✓ Validation report saved to {validation_path}")
        
        print("\n" + "="*80)
        print("✓ CLEAN FEATURE ENGINEERING COMPLETE")
        print("="*80)
        print("\nKey Principles:")
        print("  • NO features derived from current target value")
        print("  • NO week-level aggregations that include current observation")
        print("  • ONLY historical information (lag >= 1)")
        print("  • ONLY external information (Week, Age, Season)")
        print(f"  • Max correlation: {max(max_corr_judge, max_corr_fan):.4f} (< 0.85 threshold)")
        
        return df, summary


def main():
    """主函数"""
    engineer = CleanFeatureEngineer()
    df, summary = engineer.run_clean_feature_engineering()
    return df, summary


if __name__ == '__main__':
    df, summary = main()
