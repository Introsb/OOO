"""
Q5: Feature Attribution Analysis
使用贝叶斯岭回归分析影响裁判分数和观众投票的关键特征
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureAttributionAnalyzer:
    """特征归因分析器"""
    
    def __init__(self):
        self.model_judge = BayesianRidge()
        self.model_fan = BayesianRidge()
        self.scaler = StandardScaler()
        self.feature_names = []
        
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
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def prepare_features(self, df):
        """准备特征矩阵"""
        print("\nPreparing features...")
        
        # 数值特征
        X_numeric = df[['Age', 'Season']].copy()
        
        # Industry One-Hot编码
        industry_dummies = pd.get_dummies(df['Industry_Code'], prefix='Industry')
        
        # Partner One-Hot编码
        # 只保留出现次数>=5的舞伴，避免过拟合
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
        
        return X.values, y_judge, y_fan
    
    def train_models(self, X, y_judge, y_fan):
        """训练贝叶斯岭回归模型"""
        print("\nTraining Bayesian Ridge models...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练裁判分数模型
        print("Training Judge Score model...")
        self.model_judge.fit(X_scaled, y_judge)
        judge_score = self.model_judge.score(X_scaled, y_judge)
        print(f"Judge model R² score: {judge_score:.4f}")
        
        # 训练观众投票模型
        print("Training Fan Vote model...")
        self.model_fan.fit(X_scaled, y_fan)
        fan_score = self.model_fan.score(X_scaled, y_fan)
        print(f"Fan model R² score: {fan_score:.4f}")
        
        return judge_score, fan_score
    
    def extract_feature_importance(self):
        """提取特征重要性"""
        print("\nExtracting feature importance...")
        
        # 获取系数
        coef_judge = self.model_judge.coef_
        coef_fan = self.model_fan.coef_
        
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
    
    def analyze_top_features(self, results, top_n=5):
        """分析最重要的特征"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} FEATURES INFLUENCING JUDGE SCORES")
        print(f"{'='*80}")
        
        top_judge = results.nlargest(top_n, 'Abs_Coef_Judge')
        for idx, row in top_judge.iterrows():
            direction = "正向" if row['Coef_Judge'] > 0 else "负向"
            print(f"{row['Feature']:30s} | 系数: {row['Coef_Judge']:+.6f} | {direction}影响")
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} FEATURES INFLUENCING FAN VOTES")
        print(f"{'='*80}")
        
        top_fan = results.nlargest(top_n, 'Abs_Coef_Fan')
        for idx, row in top_fan.iterrows():
            direction = "正向" if row['Coef_Fan'] > 0 else "负向"
            print(f"{row['Feature']:30s} | 系数: {row['Coef_Fan']:+.6f} | {direction}影响")
        
        return top_judge, top_fan
    
    def save_results(self, results, output_path='Q5_Feature_Importance.csv'):
        """保存结果"""
        # 只保存需要的列
        output = results[['Feature', 'Coef_Judge', 'Coef_Fan']].copy()
        output.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        
        return output_path


def main():
    """主函数"""
    print("="*80)
    print("Q5: FEATURE ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # 初始化分析器
    analyzer = FeatureAttributionAnalyzer()
    
    # 加载并合并数据
    df = analyzer.load_and_merge_data(
        'Processed_DWTS_Long_Format.csv',
        'Q1_Estimated_Fan_Votes.csv',
        '2026 MCM Problem C Data.csv'
    )
    
    # 准备特征
    X, y_judge, y_fan = analyzer.prepare_features(df)
    
    # 训练模型
    judge_r2, fan_r2 = analyzer.train_models(X, y_judge, y_fan)
    
    # 提取特征重要性
    results = analyzer.extract_feature_importance()
    
    # 分析最重要的特征
    top_judge, top_fan = analyzer.analyze_top_features(results, top_n=5)
    
    # 保存结果
    output_path = analyzer.save_results(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Judge Model R²: {judge_r2:.4f}")
    print(f"Fan Model R²: {fan_r2:.4f}")
    print(f"Total features analyzed: {len(results)}")
    print(f"Output file: {output_path}")
    
    return results, analyzer


if __name__ == '__main__':
    results, analyzer = main()
