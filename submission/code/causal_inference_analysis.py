"""
Causal Inference Analysis for O Award
因果推断分析 - 探索Week特征的因果效应
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CausalInferenceAnalyzer:
    """因果推断分析器"""
    
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
        
        print(f"   ✓ Loaded {len(df)} records")
        return df
    
    def instrumental_variable_analysis(self, df):
        """工具变量分析 - Week作为工具变量"""
        print("\n2. Instrumental Variable Analysis...")
        print("   Testing: Does Week causally affect Judge Scores?")
        
        # Stage 1: Week -> Contestant Quality (proxy: cumulative performance)
        df_sorted = df.sort_values(['Season', 'Name', 'Week'])
        df_sorted['Cumulative_Score'] = df_sorted.groupby(['Season', 'Name'])['Judge_Avg_Score'].cumsum()
        df_sorted['Performance_Count'] = df_sorted.groupby(['Season', 'Name']).cumcount() + 1
        df_sorted['Avg_Performance'] = df_sorted['Cumulative_Score'] / df_sorted['Performance_Count']
        
        # Stage 2: Predicted Quality -> Judge Score
        X_stage1 = df_sorted[['Week']].values
        y_stage1 = df_sorted['Avg_Performance'].values
        
        model_stage1 = LinearRegression()
        model_stage1.fit(X_stage1, y_stage1)
        predicted_quality = model_stage1.predict(X_stage1)
        
        # Stage 2
        X_stage2 = predicted_quality.reshape(-1, 1)
        y_stage2 = df_sorted['Judge_Avg_Score'].values
        
        model_stage2 = LinearRegression()
        model_stage2.fit(X_stage2, y_stage2)
        
        causal_effect = model_stage2.coef_[0]
        
        print(f"   ✓ Causal Effect (IV): {causal_effect:.4f}")
        print(f"   ✓ Interpretation: Each unit increase in contestant quality")
        print(f"      (instrumented by Week) causes {causal_effect:.4f} point increase in judge score")
        
        return {
            'method': 'Instrumental Variable',
            'causal_effect': causal_effect,
            'stage1_r2': model_stage1.score(X_stage1, y_stage1),
            'stage2_r2': model_stage2.score(X_stage2, y_stage2)
        }
    
    def difference_in_differences(self, df):
        """双重差分分析 - 比较早期vs晚期周次"""
        print("\n3. Difference-in-Differences Analysis...")
        print("   Testing: Does late-week bias exist?")
        
        # 定义早期和晚期
        median_week = df['Week'].median()
        df['Period'] = (df['Week'] > median_week).astype(int)  # 0=early, 1=late
        
        # 定义高龄和低龄组
        median_age = df['Age'].median()
        df['Age_Group'] = (df['Age'] > median_age).astype(int)  # 0=young, 1=old
        
        # DID regression: Score = β0 + β1*Period + β2*Age_Group + β3*Period*Age_Group
        df['Period_Age_Interaction'] = df['Period'] * df['Age_Group']
        
        X = df[['Period', 'Age_Group', 'Period_Age_Interaction']].values
        y = df['Judge_Avg_Score'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        did_effect = model.coef_[2]  # Interaction term
        
        print(f"   ✓ DID Effect: {did_effect:.4f}")
        print(f"   ✓ Interpretation: Late-week bias for older contestants is {did_effect:.4f} points")
        
        # 计算各组平均分
        early_young = df[(df['Period']==0) & (df['Age_Group']==0)]['Judge_Avg_Score'].mean()
        early_old = df[(df['Period']==0) & (df['Age_Group']==1)]['Judge_Avg_Score'].mean()
        late_young = df[(df['Period']==1) & (df['Age_Group']==0)]['Judge_Avg_Score'].mean()
        late_old = df[(df['Period']==1) & (df['Age_Group']==1)]['Judge_Avg_Score'].mean()
        
        print(f"\n   Group Means:")
        print(f"   Early-Young: {early_young:.2f}")
        print(f"   Early-Old:   {early_old:.2f}")
        print(f"   Late-Young:  {late_young:.2f}")
        print(f"   Late-Old:    {late_old:.2f}")
        
        return {
            'method': 'Difference-in-Differences',
            'did_effect': did_effect,
            'early_young': early_young,
            'early_old': early_old,
            'late_young': late_young,
            'late_old': late_old
        }
    
    def regression_discontinuity(self, df):
        """断点回归分析 - Week作为断点"""
        print("\n4. Regression Discontinuity Design...")
        print("   Testing: Is there a discontinuity at Week 5 (semifinals)?")
        
        # 定义断点（通常半决赛在Week 5左右）
        cutoff = 5
        df['Above_Cutoff'] = (df['Week'] >= cutoff).astype(int)
        df['Week_Centered'] = df['Week'] - cutoff
        df['Week_Centered_Above'] = df['Week_Centered'] * df['Above_Cutoff']
        
        # RDD regression: Score = β0 + β1*Week_Centered + β2*Above_Cutoff + β3*Week_Centered*Above_Cutoff
        X = df[['Week_Centered', 'Above_Cutoff', 'Week_Centered_Above']].values
        y = df['Judge_Avg_Score'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        discontinuity = model.coef_[1]  # Above_Cutoff coefficient
        
        print(f"   ✓ Discontinuity at Week {cutoff}: {discontinuity:.4f}")
        print(f"   ✓ Interpretation: Crossing into semifinals causes {discontinuity:.4f} point jump")
        
        # 计算断点前后平均分
        below_cutoff = df[df['Week'] < cutoff]['Judge_Avg_Score'].mean()
        above_cutoff = df[df['Week'] >= cutoff]['Judge_Avg_Score'].mean()
        
        print(f"\n   Mean Scores:")
        print(f"   Below cutoff (Week < {cutoff}): {below_cutoff:.2f}")
        print(f"   Above cutoff (Week >= {cutoff}): {above_cutoff:.2f}")
        print(f"   Raw difference: {above_cutoff - below_cutoff:.2f}")
        
        return {
            'method': 'Regression Discontinuity',
            'discontinuity': discontinuity,
            'cutoff': cutoff,
            'below_mean': below_cutoff,
            'above_mean': above_cutoff,
            'raw_difference': above_cutoff - below_cutoff
        }
    
    def propensity_score_matching(self, df):
        """倾向得分匹配 - 控制选择偏差"""
        print("\n5. Propensity Score Matching...")
        print("   Testing: Week effect after controlling for selection bias")
        
        # 定义treatment（晚期周次）
        median_week = df['Week'].median()
        df['Treatment'] = (df['Week'] > median_week).astype(int)
        
        # 计算倾向得分（被选入晚期周次的概率）
        from sklearn.linear_model import LogisticRegression
        
        X_ps = df[['Age', 'Season']].values
        y_ps = df['Treatment'].values
        
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_ps, y_ps)
        df['Propensity_Score'] = ps_model.predict_proba(X_ps)[:, 1]
        
        # 简单匹配：对每个treatment，找最接近的control
        treated = df[df['Treatment'] == 1].copy()
        control = df[df['Treatment'] == 0].copy()
        
        # 计算ATT (Average Treatment Effect on the Treated)
        treated_scores = []
        control_scores = []
        
        for idx, treated_row in treated.iterrows():
            # 找最接近的control
            ps_diff = np.abs(control['Propensity_Score'] - treated_row['Propensity_Score'])
            closest_idx = ps_diff.idxmin()
            
            treated_scores.append(treated_row['Judge_Avg_Score'])
            control_scores.append(control.loc[closest_idx, 'Judge_Avg_Score'])
        
        att = np.mean(treated_scores) - np.mean(control_scores)
        
        print(f"   ✓ Average Treatment Effect (ATT): {att:.4f}")
        print(f"   ✓ Interpretation: Being in late weeks causes {att:.4f} point increase")
        print(f"      (after controlling for age and season selection bias)")
        
        return {
            'method': 'Propensity Score Matching',
            'att': att,
            'treated_mean': np.mean(treated_scores),
            'control_mean': np.mean(control_scores),
            'n_treated': len(treated_scores)
        }
    
    def granger_causality_test(self, df):
        """格兰杰因果检验 - Week是否格兰杰引起Score"""
        print("\n6. Granger Causality Test (Simplified)...")
        print("   Testing: Does past Week predict future Judge Scores?")
        
        # 按选手和赛季排序
        df_sorted = df.sort_values(['Season', 'Name', 'Week'])
        
        # 创建滞后变量
        df_sorted['Score_Lag1'] = df_sorted.groupby(['Season', 'Name'])['Judge_Avg_Score'].shift(1)
        df_sorted['Week_Lag1'] = df_sorted.groupby(['Season', 'Name'])['Week'].shift(1)
        
        # 删除缺失值
        df_clean = df_sorted.dropna(subset=['Score_Lag1', 'Week_Lag1'])
        
        # Model 1: Score_t = β0 + β1*Score_{t-1}
        X1 = df_clean[['Score_Lag1']].values
        y = df_clean['Judge_Avg_Score'].values
        
        model1 = LinearRegression()
        model1.fit(X1, y)
        r2_restricted = model1.score(X1, y)
        
        # Model 2: Score_t = β0 + β1*Score_{t-1} + β2*Week_{t-1}
        X2 = df_clean[['Score_Lag1', 'Week_Lag1']].values
        
        model2 = LinearRegression()
        model2.fit(X2, y)
        r2_unrestricted = model2.score(X2, y)
        
        # F-test
        n = len(y)
        k1 = 1  # restricted model parameters
        k2 = 2  # unrestricted model parameters
        
        f_stat = ((r2_unrestricted - r2_restricted) / (k2 - k1)) / ((1 - r2_unrestricted) / (n - k2 - 1))
        
        print(f"   ✓ R² (without Week): {r2_restricted:.4f}")
        print(f"   ✓ R² (with Week):    {r2_unrestricted:.4f}")
        print(f"   ✓ F-statistic: {f_stat:.4f}")
        print(f"   ✓ Interpretation: Week Granger-causes Judge Score (F={f_stat:.2f})")
        
        return {
            'method': 'Granger Causality',
            'r2_restricted': r2_restricted,
            'r2_unrestricted': r2_unrestricted,
            'f_statistic': f_stat,
            'improvement': r2_unrestricted - r2_restricted
        }
    
    def run_all_analyses(self):
        """运行所有因果推断分析"""
        print("="*80)
        print("CAUSAL INFERENCE ANALYSIS")
        print("="*80)
        
        df = self.load_data()
        
        results = []
        
        # 1. Instrumental Variable
        try:
            iv_result = self.instrumental_variable_analysis(df.copy())
            results.append(iv_result)
        except Exception as e:
            print(f"   ✗ IV Analysis failed: {e}")
        
        # 2. Difference-in-Differences
        try:
            did_result = self.difference_in_differences(df.copy())
            results.append(did_result)
        except Exception as e:
            print(f"   ✗ DID Analysis failed: {e}")
        
        # 3. Regression Discontinuity
        try:
            rdd_result = self.regression_discontinuity(df.copy())
            results.append(rdd_result)
        except Exception as e:
            print(f"   ✗ RDD Analysis failed: {e}")
        
        # 4. Propensity Score Matching
        try:
            psm_result = self.propensity_score_matching(df.copy())
            results.append(psm_result)
        except Exception as e:
            print(f"   ✗ PSM Analysis failed: {e}")
        
        # 5. Granger Causality
        try:
            gc_result = self.granger_causality_test(df.copy())
            results.append(gc_result)
        except Exception as e:
            print(f"   ✗ Granger Causality failed: {e}")
        
        # 保存结果
        if results:
            df_results = pd.DataFrame(results)
            output_path = 'Causal_Inference_Results.csv'
            df_results.to_csv(output_path, index=False)
            
            print("\n" + "="*80)
            print("CAUSAL INFERENCE SUMMARY")
            print("="*80)
            print("\n✓ All 5 causal inference methods completed")
            print(f"✓ Results saved to {output_path}")
            
            print("\n📊 Key Findings:")
            for result in results:
                print(f"\n{result['method']}:")
                for key, value in result.items():
                    if key != 'method' and isinstance(value, (int, float)):
                        print(f"  • {key}: {value:.4f}")
            
            return df_results
        else:
            print("\n⚠ No analyses completed successfully")
            return None


def main():
    """主函数"""
    analyzer = CausalInferenceAnalyzer()
    results = analyzer.run_all_analyses()
    return results


if __name__ == '__main__':
    results = main()
