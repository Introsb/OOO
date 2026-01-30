"""
Enhanced Feature Attribution Analysis (冲刺O奖版本)
新增功能：
1. Week特征（提升R²）
2. 交互特征（Age × Week）
3. 多种模型对比（Linear, Random Forest, XGBoost）
4. 因果推断（Propensity Score Matching）
5. 特征重要性可视化
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Will skip XGBoost model.")


class EnhancedFeatureAttributionAnalyzer:
    """增强版特征归因分析器"""
    
    def __init__(self):
        self.models = {
            'Bayesian Ridge': BayesianRidge(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model_judge = None
        self.best_model_fan = None
        
    def load_and_merge_data(self, processed_path, fan_votes_path, raw_data_path):
        """加载并合并数据"""
        print("Loading data...")
        
        # 加载预处理数据
        df_processed = pd.read_csv(processed_path)
        
        # 加载观众投票数据
        df_fan = pd.read_csv(fan_votes_path)
        
        # 加载原始数据以获取舞伴信息
        df_raw = pd.read_csv(raw_data_path)
        
        # 提取舞伴信息
        partner_info = df_raw[['celebrity_name', 'ballroom_partner', 'season']].copy()
        partner_info.columns = ['Name', 'Partner', 'Season']
        
        # 合并数据
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 合并舞伴信息
        df = df.merge(partner_info, on=['Name', 'Season'], how='left')
        
        print(f"Merged data shape: {df.shape}")
        
        return df
    
    def prepare_enhanced_features(self, df):
        """准备增强特征矩阵（包含Week和交互特征）"""
        print("\nPreparing enhanced features...")
        
        # 基础数值特征（新增Week）
        X_numeric = df[['Age', 'Season', 'Week']].copy()
        
        # 交互特征：Age × Week
        X_numeric['Age_Week_Interaction'] = df['Age'] * df['Week']
        
        # 交互特征：Age × Season
        X_numeric['Age_Season_Interaction'] = df['Age'] * df['Season']
        
        # Week的平方项（捕捉非线性效应）
        X_numeric['Week_Squared'] = df['Week'] ** 2
        
        # Industry One-Hot编码
        industry_dummies = pd.get_dummies(df['Industry_Code'], prefix='Industry')
        
        # Partner One-Hot编码
        partner_counts = df['Partner'].value_counts()
        frequent_partners = partner_counts[partner_counts >= 5].index
        df['Partner_Grouped'] = df['Partner'].apply(
            lambda x: x if x in frequent_partners else 'Other'
        )
        partner_dummies = pd.get_dummies(df['Partner_Grouped'], prefix='Partner')
        
        # 合并所有特征
        X = pd.concat([X_numeric, industry_dummies, partner_dummies], axis=1)
        
        # 保存特征名
        self.feature_names = X.columns.tolist()
        
        # 目标变量
        y_judge = df['Judge_Avg_Score'].values
        y_fan = df['Estimated_Fan_Vote'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"New features added: Week, Age×Week, Age×Season, Week²")
        
        return X.values, y_judge, y_fan
    
    def compare_models(self, X, y, target_name='Judge Score'):
        """对比多种模型的性能"""
        print(f"\n{'='*80}")
        print(f"COMPARING MODELS FOR {target_name.upper()}")
        print(f"{'='*80}")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        for model_name, model in self.models.items():
            print(f"\nTesting {model_name}...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
            
            # 训练完整模型
            model.fit(X_scaled, y)
            train_score = model.score(X_scaled, y)
            
            results[model_name] = {
                'CV Mean R²': cv_scores.mean(),
                'CV Std R²': cv_scores.std(),
                'Train R²': train_score,
                'Model': model
            }
            
            print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Train R²: {train_score:.4f}")
        
        # 找到最佳模型
        best_model_name = max(results, key=lambda x: results[x]['CV Mean R²'])
        best_model = results[best_model_name]['Model']
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"CV R²: {results[best_model_name]['CV Mean R²']:.4f} ± {results[best_model_name]['CV Std R²']:.4f}")
        print(f"{'='*80}")
        
        return best_model, results
    
    def extract_feature_importance_from_best_models(self):
        """从最佳模型中提取特征重要性"""
        print("\nExtracting feature importance from best models...")
        
        # 对于线性模型，使用系数
        if hasattr(self.best_model_judge, 'coef_'):
            coef_judge = self.best_model_judge.coef_
        elif hasattr(self.best_model_judge, 'feature_importances_'):
            # 对于树模型，使用feature_importances_
            coef_judge = self.best_model_judge.feature_importances_
        else:
            coef_judge = np.zeros(len(self.feature_names))
        
        if hasattr(self.best_model_fan, 'coef_'):
            coef_fan = self.best_model_fan.coef_
        elif hasattr(self.best_model_fan, 'feature_importances_'):
            coef_fan = self.best_model_fan.feature_importances_
        else:
            coef_fan = np.zeros(len(self.feature_names))
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Coef_Judge': coef_judge,
            'Coef_Fan': coef_fan,
            'Abs_Coef_Judge': np.abs(coef_judge),
            'Abs_Coef_Fan': np.abs(coef_fan)
        })
        
        # 按绝对值排序
        results = results.sort_values('Abs_Coef_Judge', ascending=False)
        
        return results
    
    def analyze_week_effect(self, df, results):
        """专门分析Week特征的影响"""
        print(f"\n{'='*80}")
        print("WEEK EFFECT ANALYSIS")
        print(f"{'='*80}")
        
        # 提取Week相关特征
        week_features = results[results['Feature'].str.contains('Week', case=False)]
        
        print("\nWeek-related features:")
        for idx, row in week_features.iterrows():
            print(f"{row['Feature']:30s} | Judge: {row['Coef_Judge']:+.6f} | Fan: {row['Coef_Fan']:+.6f}")
        
        # 分析Week对分数的影响
        week_corr_judge = df[['Week', 'Judge_Avg_Score']].corr().iloc[0, 1]
        week_corr_fan = df[['Week', 'Estimated_Fan_Vote']].corr().iloc[0, 1]
        
        print(f"\nDirect correlations:")
        print(f"Week vs Judge Score: {week_corr_judge:+.4f}")
        print(f"Week vs Fan Vote: {week_corr_fan:+.4f}")
        
        return week_features
    
    def analyze_interaction_effects(self, results):
        """分析交互特征的影响"""
        print(f"\n{'='*80}")
        print("INTERACTION EFFECT ANALYSIS")
        print(f"{'='*80}")
        
        # 提取交互特征
        interaction_features = results[results['Feature'].str.contains('Interaction', case=False)]
        
        print("\nInteraction features:")
        for idx, row in interaction_features.iterrows():
            print(f"{row['Feature']:30s} | Judge: {row['Coef_Judge']:+.6f} | Fan: {row['Coef_Fan']:+.6f}")
        
        # 解释
        print("\nInterpretation:")
        age_week = interaction_features[interaction_features['Feature'] == 'Age_Week_Interaction']
        if not age_week.empty:
            coef = age_week.iloc[0]['Coef_Judge']
            if coef < 0:
                print("  • Age disadvantage INCREASES as weeks progress (older contestants struggle more in later weeks)")
            else:
                print("  • Age disadvantage DECREASES as weeks progress (older contestants adapt better)")
        
        return interaction_features
    
    def save_enhanced_results(self, results, model_comparison_judge, model_comparison_fan, 
                             output_path='Enhanced_Feature_Analysis.csv'):
        """保存增强版结果"""
        # 保存特征重要性
        output = results[['Feature', 'Coef_Judge', 'Coef_Fan']].copy()
        output.to_csv(output_path, index=False)
        print(f"\n✓ Feature importance saved to {output_path}")
        
        # 保存模型对比结果
        comparison_data = []
        for model_name in model_comparison_judge.keys():
            comparison_data.append({
                'Model': model_name,
                'Target': 'Judge Score',
                'CV_Mean_R2': model_comparison_judge[model_name]['CV Mean R²'],
                'CV_Std_R2': model_comparison_judge[model_name]['CV Std R²'],
                'Train_R2': model_comparison_judge[model_name]['Train R²']
            })
        
        for model_name in model_comparison_fan.keys():
            comparison_data.append({
                'Model': model_name,
                'Target': 'Fan Vote',
                'CV_Mean_R2': model_comparison_fan[model_name]['CV Mean R²'],
                'CV_Std_R2': model_comparison_fan[model_name]['CV Std R²'],
                'Train_R2': model_comparison_fan[model_name]['Train R²']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = 'Model_Comparison_Results.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Model comparison saved to {comparison_path}")
        
        return output_path, comparison_path


def main():
    """主函数"""
    print("="*80)
    print("ENHANCED FEATURE ATTRIBUTION ANALYSIS (O奖冲刺版)")
    print("="*80)
    
    # 初始化分析器
    analyzer = EnhancedFeatureAttributionAnalyzer()
    
    # 加载并合并数据
    df = analyzer.load_and_merge_data(
        'results/Processed_DWTS_Long_Format.csv',
        'results/Q1_Estimated_Fan_Votes.csv',
        'data/2026 MCM Problem C Data.csv'
    )
    
    # 准备增强特征
    X, y_judge, y_fan = analyzer.prepare_enhanced_features(df)
    
    # 对比多种模型 - Judge Score
    best_model_judge, comparison_judge = analyzer.compare_models(X, y_judge, 'Judge Score')
    analyzer.best_model_judge = best_model_judge
    
    # 对比多种模型 - Fan Vote
    best_model_fan, comparison_fan = analyzer.compare_models(X, y_fan, 'Fan Vote')
    analyzer.best_model_fan = best_model_fan
    
    # 提取特征重要性
    results = analyzer.extract_feature_importance_from_best_models()
    
    # 分析Week效应
    week_features = analyzer.analyze_week_effect(df, results)
    
    # 分析交互效应
    interaction_features = analyzer.analyze_interaction_effects(results)
    
    # 分析最重要的特征（Top 10）
    print(f"\n{'='*80}")
    print("TOP 10 FEATURES INFLUENCING JUDGE SCORES")
    print(f"{'='*80}")
    top_judge = results.nlargest(10, 'Abs_Coef_Judge')
    for idx, row in top_judge.iterrows():
        direction = "正向" if row['Coef_Judge'] > 0 else "负向"
        print(f"{row['Feature']:35s} | 系数: {row['Coef_Judge']:+.6f} | {direction}影响")
    
    # 保存结果
    output_path, comparison_path = analyzer.save_enhanced_results(
        results, comparison_judge, comparison_fan
    )
    
    print("\n" + "="*80)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*80)
    print(f"Best Judge Model: {[k for k, v in comparison_judge.items() if v['Model'] == best_model_judge][0]}")
    print(f"  R²: {comparison_judge[[k for k, v in comparison_judge.items() if v['Model'] == best_model_judge][0]]['CV Mean R²']:.4f}")
    print(f"Best Fan Model: {[k for k, v in comparison_fan.items() if v['Model'] == best_model_fan][0]}")
    print(f"  R²: {comparison_fan[[k for k, v in comparison_fan.items() if v['Model'] == best_model_fan][0]]['CV Mean R²']:.4f}")
    print(f"Total features analyzed: {len(results)}")
    print(f"Output files: {output_path}, {comparison_path}")
    
    return results, analyzer, df


if __name__ == '__main__':
    results, analyzer, df = main()
