"""
Optimized System Design (参数优化版本)
使用网格搜索找到最优的权重和Sigmoid参数
目标：最大化技术公平性，同时保持娱乐性
"""

import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class OptimizedSystemDesigner:
    """优化系统设计器"""
    
    def __init__(self):
        self.best_params = None
        self.best_score = -np.inf
        self.all_results = []
        
    def load_data(self, processed_path, fan_votes_path):
        """加载数据"""
        print("Loading data...")
        df_processed = pd.read_csv(processed_path)
        df_fan = pd.read_csv(fan_votes_path)
        
        # 合并数据
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        print(f"Data shape: {df.shape}")
        return df
    
    def sigmoid(self, x, k=15, x0=0.4):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    def min_max_normalize(self, scores):
        """Min-Max归一化"""
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)
    
    def calculate_new_scores(self, df, judge_weight, fan_weight, sigmoid_k, sigmoid_x0):
        """计算新系统的综合分数"""
        # 按周次分组计算
        results = []
        
        for (season, week), group in df.groupby(['Season', 'Week']):
            if len(group) < 2:
                continue
            
            # 归一化裁判分数
            judge_normalized = self.min_max_normalize(group['Judge_Avg_Score'].values)
            
            # Sigmoid变换观众投票
            fan_sigmoid = self.sigmoid(group['Estimated_Fan_Vote'].values, k=sigmoid_k, x0=sigmoid_x0)
            
            # 计算综合分数
            new_scores = judge_weight * judge_normalized + fan_weight * fan_sigmoid
            
            # 找到最低分（被淘汰者）
            eliminated_idx = np.argmin(new_scores)
            
            results.append({
                'Season': season,
                'Week': week,
                'Eliminated_Name': group.iloc[eliminated_idx]['Name'],
                'Eliminated_Judge_Rank': group['Judge_Avg_Score'].rank(ascending=True).iloc[eliminated_idx],
                'Eliminated_Fan_Rank': group['Estimated_Fan_Vote'].rank(ascending=True).iloc[eliminated_idx],
                'Num_Contestants': len(group)
            })
        
        return pd.DataFrame(results)
    
    def evaluate_system(self, results_df):
        """评估系统性能"""
        # 指标1：被淘汰者的平均裁判排名（越高越好）
        avg_judge_rank = results_df['Eliminated_Judge_Rank'].mean()
        
        # 指标2：冤案率（被淘汰者既不是裁判最低也不是观众最低）
        is_judge_lowest = results_df['Eliminated_Judge_Rank'] == 1
        is_fan_lowest = results_df['Eliminated_Fan_Rank'] == 1
        injustice_rate = (~(is_judge_lowest | is_fan_lowest)).mean()
        
        # 指标3：技术公平性（被淘汰者在裁判排名后50%的比例）
        total_contestants = results_df['Num_Contestants']
        is_bottom_half = results_df['Eliminated_Judge_Rank'] <= (total_contestants / 2)
        technical_fairness = is_bottom_half.mean()
        
        return {
            'avg_judge_rank': avg_judge_rank,
            'injustice_rate': injustice_rate,
            'technical_fairness': technical_fairness
        }
    
    def calculate_综合得分(self, metrics):
        """计算综合得分（多目标优化）"""
        # 归一化各指标到[0, 1]
        # avg_judge_rank: 越高越好，假设范围[1, 10]
        norm_judge_rank = metrics['avg_judge_rank'] / 10.0
        
        # injustice_rate: 越低越好，范围[0, 1]
        norm_injustice = 1 - metrics['injustice_rate']
        
        # technical_fairness: 越高越好，范围[0, 1]
        norm_technical = metrics['technical_fairness']
        
        # 加权综合得分
        # 权重：裁判排名40%，冤案率30%，技术公平性30%
        score = 0.4 * norm_judge_rank + 0.3 * norm_injustice + 0.3 * norm_technical
        
        return score
    
    def grid_search(self, df, judge_weights, sigmoid_ks, sigmoid_x0s):
        """网格搜索最优参数"""
        print(f"\n{'='*80}")
        print("GRID SEARCH FOR OPTIMAL PARAMETERS")
        print(f"{'='*80}")
        print(f"Judge weights: {judge_weights}")
        print(f"Sigmoid k values: {sigmoid_ks}")
        print(f"Sigmoid x0 values: {sigmoid_x0s}")
        print(f"Total combinations: {len(judge_weights) * len(sigmoid_ks) * len(sigmoid_x0s)}")
        
        # 遍历所有参数组合
        for judge_w in judge_weights:
            fan_w = 1 - judge_w
            
            for sigmoid_k in sigmoid_ks:
                for sigmoid_x0 in sigmoid_x0s:
                    # 计算新系统分数
                    results_df = self.calculate_new_scores(df, judge_w, fan_w, sigmoid_k, sigmoid_x0)
                    
                    # 评估系统
                    metrics = self.evaluate_system(results_df)
                    
                    # 计算综合得分
                    score = self.calculate_综合得分(metrics)
                    
                    # 保存结果
                    self.all_results.append({
                        'judge_weight': judge_w,
                        'fan_weight': fan_w,
                        'sigmoid_k': sigmoid_k,
                        'sigmoid_x0': sigmoid_x0,
                        'avg_judge_rank': metrics['avg_judge_rank'],
                        'injustice_rate': metrics['injustice_rate'],
                        'technical_fairness': metrics['technical_fairness'],
                        'composite_score': score
                    })
                    
                    # 更新最佳参数
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = {
                            'judge_weight': judge_w,
                            'fan_weight': fan_w,
                            'sigmoid_k': sigmoid_k,
                            'sigmoid_x0': sigmoid_x0,
                            'metrics': metrics,
                            'score': score
                        }
        
        print(f"\n✓ Grid search complete. Tested {len(self.all_results)} combinations.")
        
        return self.best_params
    
    def display_best_params(self):
        """显示最佳参数"""
        print(f"\n{'='*80}")
        print("OPTIMAL PARAMETERS FOUND")
        print(f"{'='*80}")
        print(f"Judge Weight: {self.best_params['judge_weight']:.2f}")
        print(f"Fan Weight: {self.best_params['fan_weight']:.2f}")
        print(f"Sigmoid k: {self.best_params['sigmoid_k']}")
        print(f"Sigmoid x0: {self.best_params['sigmoid_x0']}")
        print(f"\nPerformance Metrics:")
        print(f"  Average Judge Rank: {self.best_params['metrics']['avg_judge_rank']:.2f}")
        print(f"  Injustice Rate: {self.best_params['metrics']['injustice_rate']:.2%}")
        print(f"  Technical Fairness: {self.best_params['metrics']['technical_fairness']:.2%}")
        print(f"  Composite Score: {self.best_params['score']:.4f}")
        print(f"{'='*80}")
    
    def compare_with_baseline(self, df):
        """与基线系统（70/30, k=15, x0=0.4）对比"""
        print(f"\n{'='*80}")
        print("COMPARISON WITH BASELINE SYSTEM")
        print(f"{'='*80}")
        
        # 基线系统
        baseline_results = self.calculate_new_scores(df, 0.7, 0.3, 15, 0.4)
        baseline_metrics = self.evaluate_system(baseline_results)
        baseline_score = self.calculate_综合得分(baseline_metrics)
        
        print("\nBaseline System (70/30, k=15, x0=0.4):")
        print(f"  Average Judge Rank: {baseline_metrics['avg_judge_rank']:.2f}")
        print(f"  Injustice Rate: {baseline_metrics['injustice_rate']:.2%}")
        print(f"  Technical Fairness: {baseline_metrics['technical_fairness']:.2%}")
        print(f"  Composite Score: {baseline_score:.4f}")
        
        print("\nOptimized System:")
        print(f"  Average Judge Rank: {self.best_params['metrics']['avg_judge_rank']:.2f}")
        print(f"  Injustice Rate: {self.best_params['metrics']['injustice_rate']:.2%}")
        print(f"  Technical Fairness: {self.best_params['metrics']['technical_fairness']:.2%}")
        print(f"  Composite Score: {self.best_params['score']:.4f}")
        
        print("\nImprovement:")
        print(f"  Judge Rank: {self.best_params['metrics']['avg_judge_rank'] - baseline_metrics['avg_judge_rank']:+.2f}")
        print(f"  Injustice Rate: {(self.best_params['metrics']['injustice_rate'] - baseline_metrics['injustice_rate'])*100:+.2f}%")
        print(f"  Technical Fairness: {(self.best_params['metrics']['technical_fairness'] - baseline_metrics['technical_fairness'])*100:+.2f}%")
        print(f"  Composite Score: {self.best_params['score'] - baseline_score:+.4f}")
        
        return baseline_metrics, baseline_score
    
    def save_results(self, output_path='Optimized_System_Parameters.csv'):
        """保存所有结果"""
        results_df = pd.DataFrame(self.all_results)
        
        # 按综合得分排序
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ All results saved to {output_path}")
        
        # 保存最佳参数
        best_params_df = pd.DataFrame([{
            'Parameter': k,
            'Value': v
        } for k, v in self.best_params.items() if k != 'metrics'])
        
        best_params_path = 'Best_System_Parameters.csv'
        best_params_df.to_csv(best_params_path, index=False)
        print(f"✓ Best parameters saved to {best_params_path}")
        
        return output_path, best_params_path
    
    def analyze_parameter_sensitivity(self):
        """分析参数灵敏度"""
        print(f"\n{'='*80}")
        print("PARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(self.all_results)
        
        # 分析judge_weight的影响
        print("\nJudge Weight Sensitivity:")
        for jw in sorted(results_df['judge_weight'].unique()):
            subset = results_df[results_df['judge_weight'] == jw]
            avg_score = subset['composite_score'].mean()
            std_score = subset['composite_score'].std()
            print(f"  {jw:.2f}: Score = {avg_score:.4f} ± {std_score:.4f}")
        
        # 分析sigmoid_k的影响
        print("\nSigmoid k Sensitivity:")
        for k in sorted(results_df['sigmoid_k'].unique()):
            subset = results_df[results_df['sigmoid_k'] == k]
            avg_score = subset['composite_score'].mean()
            std_score = subset['composite_score'].std()
            print(f"  {k}: Score = {avg_score:.4f} ± {std_score:.4f}")
        
        # 分析sigmoid_x0的影响
        print("\nSigmoid x0 Sensitivity:")
        for x0 in sorted(results_df['sigmoid_x0'].unique()):
            subset = results_df[results_df['sigmoid_x0'] == x0]
            avg_score = subset['composite_score'].mean()
            std_score = subset['composite_score'].std()
            print(f"  {x0:.2f}: Score = {avg_score:.4f} ± {std_score:.4f}")


def main():
    """主函数"""
    print("="*80)
    print("OPTIMIZED SYSTEM DESIGN (参数优化版本)")
    print("="*80)
    
    # 初始化设计器
    designer = OptimizedSystemDesigner()
    
    # 加载数据
    df = designer.load_data(
        'results/Processed_DWTS_Long_Format.csv',
        'results/Q1_Estimated_Fan_Votes.csv'
    )
    
    # 定义参数搜索空间
    judge_weights = np.arange(0.5, 0.91, 0.05)  # 50%-90%，步长5%
    sigmoid_ks = [5, 10, 15, 20, 25, 30]
    sigmoid_x0s = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    # 网格搜索
    best_params = designer.grid_search(df, judge_weights, sigmoid_ks, sigmoid_x0s)
    
    # 显示最佳参数
    designer.display_best_params()
    
    # 与基线对比
    baseline_metrics, baseline_score = designer.compare_with_baseline(df)
    
    # 参数灵敏度分析
    designer.analyze_parameter_sensitivity()
    
    # 保存结果
    output_path, best_params_path = designer.save_results()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best composite score: {designer.best_score:.4f}")
    print(f"Total combinations tested: {len(designer.all_results)}")
    print(f"Output files: {output_path}, {best_params_path}")
    
    return designer


if __name__ == '__main__':
    designer = main()
