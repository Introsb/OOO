"""
Temporal Dynamics Analysis for O Award
时间动态分析 - 探索比赛公平性的时间演化
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TemporalDynamicsAnalyzer:
    """时间动态分析器"""
    
    def __init__(self):
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        print("\n1. Loading data...")
        
        df_processed = pd.read_csv('results/Processed_DWTS_Long_Format.csv')
        df_fan = pd.read_csv('results/Q1_Estimated_Fan_Votes.csv')
        
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        print(f"   ✓ Loaded {len(df)} records across {df['Season'].nunique()} seasons")
        return df
    
    def analyze_score_inflation(self, df):
        """分析分数通胀 - 评委是否随时间变得更慷慨"""
        print("\n2. Score Inflation Analysis...")
        print("   Testing: Do judges become more generous over weeks?")
        
        # 按周次分组
        week_stats = df.groupby('Week').agg({
            'Judge_Avg_Score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        week_stats.columns = ['Week', 'Mean', 'Std', 'Min', 'Max', 'Count']
        
        # 线性回归：Mean Score ~ Week
        X = week_stats['Week'].values.reshape(-1, 1)
        y = week_stats['Mean'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        # 统计检验
        _, p_value = stats.pearsonr(week_stats['Week'], week_stats['Mean'])
        
        print(f"   ✓ Inflation Rate: {slope:.4f} points per week")
        print(f"   ✓ R²: {r2:.4f}")
        print(f"   ✓ P-value: {p_value:.4e}")
        print(f"   ✓ Interpretation: Judges give {slope:.4f} more points each week")
        
        # 计算总通胀
        first_week_score = week_stats.iloc[0]['Mean']
        last_week_score = week_stats.iloc[-1]['Mean']
        total_inflation = last_week_score - first_week_score
        
        print(f"\n   Total Inflation:")
        print(f"   Week 1 average: {first_week_score:.2f}")
        print(f"   Week {week_stats['Week'].max()} average: {last_week_score:.2f}")
        print(f"   Total increase: {total_inflation:.2f} points ({total_inflation/first_week_score*100:.1f}%)")
        
        return {
            'analysis': 'Score Inflation',
            'slope': slope,
            'r2': r2,
            'p_value': p_value,
            'first_week_mean': first_week_score,
            'last_week_mean': last_week_score,
            'total_inflation': total_inflation,
            'inflation_pct': total_inflation/first_week_score*100
        }
    
    def analyze_variance_convergence(self, df):
        """分析方差收敛 - 分数是否随时间变得更集中"""
        print("\n3. Variance Convergence Analysis...")
        print("   Testing: Do scores become more homogeneous over weeks?")
        
        week_stats = df.groupby('Week').agg({
            'Judge_Avg_Score': ['std', 'count']
        }).reset_index()
        
        week_stats.columns = ['Week', 'Std', 'Count']
        
        # 线性回归：Std ~ Week
        X = week_stats['Week'].values.reshape(-1, 1)
        y = week_stats['Std'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        _, p_value = stats.pearsonr(week_stats['Week'], week_stats['Std'])
        
        print(f"   ✓ Convergence Rate: {slope:.4f} std per week")
        print(f"   ✓ R²: {r2:.4f}")
        print(f"   ✓ P-value: {p_value:.4e}")
        
        if slope < 0:
            print(f"   ✓ Interpretation: Scores converge by {abs(slope):.4f} std each week")
            print(f"      (contestants become more similar)")
        else:
            print(f"   ✓ Interpretation: Scores diverge by {slope:.4f} std each week")
            print(f"      (contestants become more different)")
        
        return {
            'analysis': 'Variance Convergence',
            'slope': slope,
            'r2': r2,
            'p_value': p_value,
            'direction': 'convergence' if slope < 0 else 'divergence'
        }
    
    def analyze_survival_bias(self, df):
        """分析生存偏差 - 留下的选手是否更强"""
        print("\n4. Survival Bias Analysis...")
        print("   Testing: Do surviving contestants have higher baseline quality?")
        
        # 计算每个选手的首周表现
        first_week = df.groupby(['Season', 'Name']).first().reset_index()
        first_week['First_Week_Score'] = first_week['Judge_Avg_Score']
        
        # 计算每个选手的最后一周
        last_week = df.groupby(['Season', 'Name']).last().reset_index()
        last_week['Last_Week'] = last_week['Week']
        
        # 合并
        survival_df = first_week[['Season', 'Name', 'First_Week_Score']].merge(
            last_week[['Season', 'Name', 'Last_Week']],
            on=['Season', 'Name']
        )
        
        # 按最后一周分组
        survival_stats = survival_df.groupby('Last_Week').agg({
            'First_Week_Score': ['mean', 'std', 'count']
        }).reset_index()
        
        survival_stats.columns = ['Last_Week', 'Mean_First_Score', 'Std', 'Count']
        
        # 线性回归
        X = survival_stats['Last_Week'].values.reshape(-1, 1)
        y = survival_stats['Mean_First_Score'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        _, p_value = stats.pearsonr(survival_stats['Last_Week'], survival_stats['Mean_First_Score'])
        
        print(f"   ✓ Survival Bias: {slope:.4f} points per week survived")
        print(f"   ✓ R²: {r2:.4f}")
        print(f"   ✓ P-value: {p_value:.4e}")
        print(f"   ✓ Interpretation: Each additional week survived requires")
        print(f"      {slope:.4f} higher first-week score")
        
        return {
            'analysis': 'Survival Bias',
            'slope': slope,
            'r2': r2,
            'p_value': p_value
        }
    
    def analyze_momentum_effect(self, df):
        """分析动量效应 - 上周表现是否影响本周"""
        print("\n5. Momentum Effect Analysis...")
        print("   Testing: Does last week's score predict this week's score?")
        
        # 按选手和赛季排序
        df_sorted = df.sort_values(['Season', 'Name', 'Week'])
        
        # 创建滞后变量
        df_sorted['Score_Lag1'] = df_sorted.groupby(['Season', 'Name'])['Judge_Avg_Score'].shift(1)
        df_sorted['Score_Change'] = df_sorted['Judge_Avg_Score'] - df_sorted['Score_Lag1']
        
        # 删除缺失值
        df_clean = df_sorted.dropna(subset=['Score_Lag1', 'Score_Change'])
        
        # 分析：本周变化 ~ 上周分数
        X = df_clean['Score_Lag1'].values.reshape(-1, 1)
        y = df_clean['Score_Change'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        _, p_value = stats.pearsonr(df_clean['Score_Lag1'], df_clean['Score_Change'])
        
        print(f"   ✓ Momentum Coefficient: {slope:.4f}")
        print(f"   ✓ R²: {r2:.4f}")
        print(f"   ✓ P-value: {p_value:.4e}")
        
        if slope < 0:
            print(f"   ✓ Interpretation: Mean reversion - high scorers tend to drop")
        else:
            print(f"   ✓ Interpretation: Momentum - high scorers tend to rise further")
        
        # 计算正负动量比例
        positive_momentum = (df_clean['Score_Change'] > 0).sum()
        negative_momentum = (df_clean['Score_Change'] < 0).sum()
        
        print(f"\n   Momentum Distribution:")
        print(f"   Positive (improving): {positive_momentum} ({positive_momentum/len(df_clean)*100:.1f}%)")
        print(f"   Negative (declining):  {negative_momentum} ({negative_momentum/len(df_clean)*100:.1f}%)")
        
        return {
            'analysis': 'Momentum Effect',
            'slope': slope,
            'r2': r2,
            'p_value': p_value,
            'positive_pct': positive_momentum/len(df_clean)*100,
            'negative_pct': negative_momentum/len(df_clean)*100
        }
    
    def analyze_elimination_threshold(self, df):
        """分析淘汰阈值 - 每周的安全分数线"""
        print("\n6. Elimination Threshold Analysis...")
        print("   Testing: What score is needed to survive each week?")
        
        # 找出每周被淘汰的选手（下周没有出现）
        df_sorted = df.sort_values(['Season', 'Name', 'Week'])
        
        # 标记是否有下一周
        df_sorted['Has_Next_Week'] = df_sorted.groupby(['Season', 'Name'])['Week'].shift(-1).notna()
        
        # 按周次分组，计算淘汰阈值
        threshold_stats = []
        
        for week in sorted(df_sorted['Week'].unique()):
            week_data = df_sorted[df_sorted['Week'] == week]
            
            eliminated = week_data[~week_data['Has_Next_Week']]
            survived = week_data[week_data['Has_Next_Week']]
            
            if len(eliminated) > 0 and len(survived) > 0:
                threshold = eliminated['Judge_Avg_Score'].max()
                safe_min = survived['Judge_Avg_Score'].min()
                
                threshold_stats.append({
                    'Week': week,
                    'Elimination_Threshold': threshold,
                    'Safe_Minimum': safe_min,
                    'Gap': safe_min - threshold,
                    'N_Eliminated': len(eliminated),
                    'N_Survived': len(survived)
                })
        
        threshold_df = pd.DataFrame(threshold_stats)
        
        # 线性回归：Threshold ~ Week
        X = threshold_df['Week'].values.reshape(-1, 1)
        y = threshold_df['Elimination_Threshold'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        print(f"   ✓ Threshold Increase: {slope:.4f} points per week")
        print(f"   ✓ R²: {r2:.4f}")
        print(f"   ✓ Interpretation: Survival threshold rises {slope:.4f} points each week")
        
        print(f"\n   Threshold Evolution:")
        print(f"   Week 1 threshold: {threshold_df.iloc[0]['Elimination_Threshold']:.2f}")
        print(f"   Week {threshold_df['Week'].max()} threshold: {threshold_df.iloc[-1]['Elimination_Threshold']:.2f}")
        print(f"   Average gap (safe - eliminated): {threshold_df['Gap'].mean():.2f}")
        
        return {
            'analysis': 'Elimination Threshold',
            'slope': slope,
            'r2': r2,
            'first_week_threshold': threshold_df.iloc[0]['Elimination_Threshold'],
            'last_week_threshold': threshold_df.iloc[-1]['Elimination_Threshold'],
            'avg_gap': threshold_df['Gap'].mean()
        }
    
    def run_all_analyses(self):
        """运行所有时间动态分析"""
        print("="*80)
        print("TEMPORAL DYNAMICS ANALYSIS")
        print("="*80)
        
        df = self.load_data()
        
        results = []
        
        # 1. Score Inflation
        try:
            result = self.analyze_score_inflation(df.copy())
            results.append(result)
        except Exception as e:
            print(f"   ✗ Score Inflation failed: {e}")
        
        # 2. Variance Convergence
        try:
            result = self.analyze_variance_convergence(df.copy())
            results.append(result)
        except Exception as e:
            print(f"   ✗ Variance Convergence failed: {e}")
        
        # 3. Survival Bias
        try:
            result = self.analyze_survival_bias(df.copy())
            results.append(result)
        except Exception as e:
            print(f"   ✗ Survival Bias failed: {e}")
        
        # 4. Momentum Effect
        try:
            result = self.analyze_momentum_effect(df.copy())
            results.append(result)
        except Exception as e:
            print(f"   ✗ Momentum Effect failed: {e}")
        
        # 5. Elimination Threshold
        try:
            result = self.analyze_elimination_threshold(df.copy())
            results.append(result)
        except Exception as e:
            print(f"   ✗ Elimination Threshold failed: {e}")
        
        # 保存结果
        if results:
            df_results = pd.DataFrame(results)
            output_path = 'Temporal_Dynamics_Results.csv'
            df_results.to_csv(output_path, index=False)
            
            print("\n" + "="*80)
            print("TEMPORAL DYNAMICS SUMMARY")
            print("="*80)
            print(f"\n✓ All {len(results)} temporal analyses completed")
            print(f"✓ Results saved to {output_path}")
            
            print("\n📊 Key Findings:")
            for result in results:
                print(f"\n{result['analysis']}:")
                for key, value in result.items():
                    if key != 'analysis' and isinstance(value, (int, float)):
                        print(f"  • {key}: {value:.4f}")
            
            return df_results
        else:
            print("\n⚠ No analyses completed successfully")
            return None


def main():
    """主函数"""
    analyzer = TemporalDynamicsAnalyzer()
    results = analyzer.run_all_analyses()
    return results


if __name__ == '__main__':
    results = main()
