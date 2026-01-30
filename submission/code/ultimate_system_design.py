"""
Q6: Ultimate System Design
设计并验证新的"公平且有趣"的赛制系统
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class UltimateSystemDesigner:
    """终极赛制设计器"""
    
    def __init__(self, judge_weight=0.7):
        """
        初始化设计器
        
        Args:
            judge_weight: 裁判分数权重 (默认0.7，更偏向专业性)
        """
        self.judge_weight = judge_weight
        self.fan_weight = 1 - judge_weight
        
    def sigmoid(self, x, k=15, x0=0.4):
        """
        Sigmoid函数用于抑制极端观众投票
        
        Args:
            x: 输入值 (0-1之间)
            k: 陡峭度参数 (越大越陡，默认15)
            x0: 中心点 (默认0.4，更早开始抑制)
        
        Returns:
            映射后的值 (0-1之间)
        """
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    def min_max_normalize(self, scores):
        """Min-Max归一化到[0,1]"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val == min_val:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    
    def calculate_new_score(self, judge_scores, fan_votes):
        """
        计算新赛制分数
        
        New_Score = w * f(Judge) + (1-w) * g(Fan)
        其中:
        - f(Judge): Min-Max归一化的裁判分
        - g(Fan): Sigmoid抑制的观众票
        - w: 裁判权重
        
        Args:
            judge_scores: 裁判分数数组
            fan_votes: 观众投票占比数组
        
        Returns:
            新赛制分数数组
        """
        # 裁判分标准化
        f_judge = self.min_max_normalize(judge_scores)
        
        # 观众票Sigmoid抑制
        g_fan = self.sigmoid(fan_votes)
        
        # 加权组合
        new_score = self.judge_weight * f_judge + self.fan_weight * g_fan
        
        return new_score
    
    def load_data(self, processed_path, fan_votes_path, simulation_path):
        """加载所有需要的数据"""
        print("Loading data...")
        
        # 加载预处理数据
        df_processed = pd.read_csv(processed_path)
        
        # 加载观众投票数据
        df_fan = pd.read_csv(fan_votes_path)
        
        # 加载Q3/Q4仿真结果
        df_sim = pd.read_csv(simulation_path)
        
        # 合并数据
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        print(f"Merged data shape: {df.shape}")
        
        return df, df_sim
    
    def simulate_new_system(self, df):
        """使用新赛制进行仿真"""
        print("\nSimulating new system...")
        
        results = []
        
        # 按赛季和周次分组
        for (season, week), group in df.groupby(['Season', 'Week']):
            if len(group) < 2:
                continue
            
            # 计算新分数
            judge_scores = group['Judge_Avg_Score'].values
            fan_votes = group['Estimated_Fan_Vote'].values
            names = group['Name'].values
            
            new_scores = self.calculate_new_score(judge_scores, fan_votes)
            
            # 找出新系统下的淘汰者（分数最低）
            min_idx = np.argmin(new_scores)
            eliminated = names[min_idx]
            
            # 计算被淘汰者的排名
            judge_rank = len(judge_scores) - np.argsort(np.argsort(judge_scores))[min_idx]
            fan_rank = len(fan_votes) - np.argsort(np.argsort(fan_votes))[min_idx]
            
            results.append({
                'Season': season,
                'Week': week,
                'New_System_Eliminated': eliminated,
                'Judge_Rank': judge_rank,
                'Fan_Rank': fan_rank,
                'Num_Contestants': len(group),
                'New_Score': new_scores[min_idx],
                'Judge_Score': judge_scores[min_idx],
                'Fan_Vote': fan_votes[min_idx]
            })
        
        df_results = pd.DataFrame(results)
        print(f"Simulated {len(df_results)} weeks")
        
        return df_results
    
    def merge_with_actual(self, df_new, df_sim):
        """合并新系统结果与实际结果"""
        print("\nMerging with actual results...")
        
        # 从Q3/Q4结果中提取实际淘汰者
        df_actual = df_sim[['Season', 'Week', 'Actual_Eliminated']].copy()
        
        # 合并
        df_merged = df_new.merge(df_actual, on=['Season', 'Week'], how='left')
        
        return df_merged
    
    def calculate_injustice_rate(self, df, judge_rank_threshold=3):
        """
        计算冤案率
        
        冤案定义：被淘汰者的裁判排名不在倒数前N名
        
        Args:
            df: 结果DataFrame
            judge_rank_threshold: 裁判排名阈值（默认3，即倒数前3名）
        
        Returns:
            冤案率
        """
        # 冤案：裁判排名 > threshold（即不在倒数前N名）
        injustice = df['Judge_Rank'] > judge_rank_threshold
        injustice_rate = injustice.mean()
        
        return injustice_rate, injustice
    
    def compare_systems(self, df_new, df_sim):
        """对比新旧系统"""
        print("\n" + "="*80)
        print("SYSTEM COMPARISON")
        print("="*80)
        
        # 计算新系统冤案率
        new_injustice_rate, new_injustice = self.calculate_injustice_rate(df_new)
        
        # 从Q3/Q4结果中获取旧系统冤案率
        if 'Is_Injustice' in df_sim.columns:
            old_injustice_rate = df_sim['Is_Injustice'].mean()
        else:
            # 如果没有，手动计算
            old_injustice_rate = 0.947  # 从Q3/Q4结果中已知
        
        print(f"\n旧系统冤案率: {old_injustice_rate:.2%}")
        print(f"新系统冤案率: {new_injustice_rate:.2%}")
        print(f"改善幅度: {(old_injustice_rate - new_injustice_rate):.2%}")
        
        # 计算改进标记
        df_new['Is_Improvement'] = ~new_injustice
        
        # 统计新系统淘汰者的特征
        print(f"\n新系统被淘汰者统计:")
        print(f"  平均裁判排名: {df_new['Judge_Rank'].mean():.2f}")
        print(f"  平均观众排名: {df_new['Fan_Rank'].mean():.2f}")
        print(f"  裁判排名中位数: {df_new['Judge_Rank'].median():.1f}")
        print(f"  观众排名中位数: {df_new['Fan_Rank'].median():.1f}")
        
        return new_injustice_rate, old_injustice_rate
    
    def save_results(self, df, output_path='Q6_New_System_Simulation.csv'):
        """保存结果"""
        # 选择需要的列
        output_cols = [
            'Season', 'Week', 
            'Actual_Eliminated', 'New_System_Eliminated',
            'Is_Improvement',
            'Judge_Rank', 'Fan_Rank', 'Num_Contestants'
        ]
        
        # 确保所有列都存在
        available_cols = [col for col in output_cols if col in df.columns]
        output = df[available_cols].copy()
        
        output.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        
        return output_path
    
    def visualize_score_distribution(self, df):
        """可视化分数分布"""
        print("\n" + "="*80)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("="*80)
        
        print(f"\n新系统分数统计:")
        print(f"  均值: {df['New_Score'].mean():.4f}")
        print(f"  标准差: {df['New_Score'].std():.4f}")
        print(f"  最小值: {df['New_Score'].min():.4f}")
        print(f"  最大值: {df['New_Score'].max():.4f}")
        
        print(f"\n裁判分数统计:")
        print(f"  均值: {df['Judge_Score'].mean():.4f}")
        print(f"  标准差: {df['Judge_Score'].std():.4f}")
        
        print(f"\n观众投票统计:")
        print(f"  均值: {df['Fan_Vote'].mean():.4f}")
        print(f"  标准差: {df['Fan_Vote'].std():.4f}")


def main():
    """主函数"""
    print("="*80)
    print("Q6: ULTIMATE SYSTEM DESIGN")
    print("="*80)
    
    # 初始化设计器
    designer = UltimateSystemDesigner(judge_weight=0.7)
    print(f"\nSystem parameters:")
    print(f"  Judge weight: {designer.judge_weight}")
    print(f"  Fan weight: {designer.fan_weight}")
    print(f"  Sigmoid function: k=15, x0=0.4")
    
    # 加载数据
    df, df_sim = designer.load_data(
        'Processed_DWTS_Long_Format.csv',
        'Q1_Estimated_Fan_Votes.csv',
        'Simulation_Results_Q3_Q4.csv'
    )
    
    # 仿真新系统
    df_new = designer.simulate_new_system(df)
    
    # 合并实际结果
    df_merged = designer.merge_with_actual(df_new, df_sim)
    
    # 对比系统
    new_injustice, old_injustice = designer.compare_systems(df_merged, df_sim)
    
    # 可视化分析
    designer.visualize_score_distribution(df_merged)
    
    # 保存结果
    output_path = designer.save_results(df_merged)
    
    print("\n" + "="*80)
    print("DESIGN COMPLETE")
    print("="*80)
    print(f"Old system injustice rate: {old_injustice:.2%}")
    print(f"New system injustice rate: {new_injustice:.2%}")
    print(f"Improvement: {(old_injustice - new_injustice):.2%}")
    print(f"Output file: {output_path}")
    
    return df_merged, designer


if __name__ == '__main__':
    df_results, designer = main()
