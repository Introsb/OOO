"""
参数空间灵敏度分析 (Sensitivity Analysis)
验证新赛制在不同参数组合下的鲁棒性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SensitivityAnalyzer:
    """参数灵敏度分析器"""
    
    def __init__(self):
        self.results = []
        
    def sigmoid(self, x, k=10, x0=0.5):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    def min_max_normalize(self, scores):
        """Min-Max归一化"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val == min_val:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    
    def calculate_new_score(self, judge_scores, fan_votes, w, k):
        """
        计算新赛制分数
        
        Args:
            judge_scores: 裁判分数数组
            fan_votes: 观众投票占比数组
            w: 裁判权重
            k: Sigmoid斜率
        """
        # 裁判分标准化
        f_judge = self.min_max_normalize(judge_scores)
        
        # 观众票Sigmoid抑制
        g_fan = self.sigmoid(fan_votes, k=k, x0=0.5)
        
        # 加权组合
        new_score = w * f_judge + (1 - w) * g_fan
        
        return new_score
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 加载预处理数据
        df_processed = pd.read_csv('Processed_DWTS_Long_Format.csv')
        
        # 加载观众投票数据
        df_fan = pd.read_csv('Q1_Estimated_Fan_Votes.csv')
        
        # 加载Q3/Q4仿真结果（用于对比）
        df_sim = pd.read_csv('Simulation_Results_Q3_Q4.csv')
        
        # 合并数据
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        print(f"✓ 数据加载完成: {df.shape[0]} 行")
        
        return df, df_sim
    
    def simulate_with_params(self, df, w, k):
        """使用指定参数进行仿真"""
        results = []
        
        # 按赛季和周次分组
        for (season, week), group in df.groupby(['Season', 'Week']):
            if len(group) < 2:
                continue
            
            # 计算新分数
            judge_scores = group['Judge_Avg_Score'].values
            fan_votes = group['Estimated_Fan_Vote'].values
            names = group['Name'].values
            
            new_scores = self.calculate_new_score(judge_scores, fan_votes, w, k)
            
            # 找出新系统下的淘汰者（分数最低）
            min_idx = np.argmin(new_scores)
            eliminated = names[min_idx]
            
            # 计算被淘汰者的排名
            judge_rank = len(judge_scores) - np.argsort(np.argsort(judge_scores))[min_idx]
            
            results.append({
                'Season': season,
                'Week': week,
                'Eliminated': eliminated,
                'Judge_Rank': judge_rank,
                'Judge_Score': judge_scores[min_idx]
            })
        
        return pd.DataFrame(results)
    
    def calculate_fairness_score(self, df_new, df_baseline):
        """
        计算公平性得分
        
        定义：Score = (改善案例数 - 恶化案例数) / 总争议周次
        - 改善：新赛制淘汰了裁判分更低的人
        - 恶化：新赛制淘汰了裁判分更高的人
        - 争议周次：新旧赛制结果不一样的周次
        """
        # 合并新旧结果
        merged = df_new.merge(
            df_baseline[['Season', 'Week', 'Simulated_Elim_Percent', 'Judge_Rank']], 
            on=['Season', 'Week'],
            how='inner',
            suffixes=('_new', '_baseline')
        )
        
        # 只统计新旧赛制结果不同的周次
        disputed = merged[merged['Eliminated'] != merged['Simulated_Elim_Percent']]
        
        if len(disputed) == 0:
            return 0.0, 0, 0, 0
        
        # 改善：新赛制的裁判排名更靠后（技术更差）
        improved = (disputed['Judge_Rank_new'] > disputed['Judge_Rank_baseline']).sum()
        
        # 恶化：新赛制的裁判排名更靠前（技术更好）
        regressed = (disputed['Judge_Rank_new'] < disputed['Judge_Rank_baseline']).sum()
        
        # 公平性得分
        fairness_score = (improved - regressed) / len(disputed)
        
        return fairness_score, improved, regressed, len(disputed)
    
    def grid_search(self, df, df_baseline, weight_range, slope_range):
        """网格搜索"""
        print(f"\n{'='*80}")
        print("参数空间网格搜索")
        print(f"{'='*80}")
        print(f"权重范围: {weight_range[0]:.2f} - {weight_range[-1]:.2f} (步长 {weight_range[1]-weight_range[0]:.2f})")
        print(f"斜率范围: {slope_range[0]} - {slope_range[-1]}")
        print(f"总组合数: {len(weight_range) * len(slope_range)}")
        
        results = []
        
        # 使用tqdm显示进度
        total = len(weight_range) * len(slope_range)
        with tqdm(total=total, desc="网格搜索进度") as pbar:
            for w in weight_range:
                for k in slope_range:
                    # 使用当前参数进行仿真
                    df_new = self.simulate_with_params(df, w, k)
                    
                    # 计算公平性得分
                    fairness_score, improved, regressed, disputed = \
                        self.calculate_fairness_score(df_new, df_baseline)
                    
                    results.append({
                        'Weight_W': w,
                        'Slope_K': k,
                        'Fairness_Score': fairness_score,
                        'Improved_Cases': improved,
                        'Regressed_Cases': regressed,
                        'Disputed_Weeks': disputed
                    })
                    
                    pbar.update(1)
        
        df_results = pd.DataFrame(results)
        
        print(f"\n✓ 网格搜索完成")
        print(f"  最高公平性得分: {df_results['Fairness_Score'].max():.4f}")
        print(f"  最低公平性得分: {df_results['Fairness_Score'].min():.4f}")
        print(f"  平均公平性得分: {df_results['Fairness_Score'].mean():.4f}")
        
        # 找出最优参数
        best_idx = df_results['Fairness_Score'].idxmax()
        best_params = df_results.loc[best_idx]
        print(f"\n最优参数组合:")
        print(f"  权重 w: {best_params['Weight_W']:.2f}")
        print(f"  斜率 k: {best_params['Slope_K']:.0f}")
        print(f"  公平性得分: {best_params['Fairness_Score']:.4f}")
        
        return df_results
    
    def visualize_heatmap(self, df_results, output_path='sensitivity_heatmap.png'):
        """生成参数热力图"""
        print(f"\n生成参数热力图...")
        
        # 创建透视表
        pivot_table = df_results.pivot(
            index='Slope_K', 
            columns='Weight_W', 
            values='Fairness_Score'
        )
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制热力图
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Fairness Score'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_xlabel('Judge Weight (w)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sigmoid Slope (k)', fontsize=14, fontweight='bold')
        ax.set_title('Parameter Sensitivity Analysis: Fairness Score Heatmap', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # 标注最优参数
        best_idx = df_results['Fairness_Score'].idxmax()
        best_w = df_results.loc[best_idx, 'Weight_W']
        best_k = df_results.loc[best_idx, 'Slope_K']
        best_score = df_results.loc[best_idx, 'Fairness_Score']
        
        # 在图上标注
        ax.text(0.02, 0.98, 
                f'Optimal: w={best_w:.2f}, k={best_k:.0f}\nScore={best_score:.4f}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存: {output_path}")
        plt.close()
    
    def visualize_3d_surface(self, df_results, output_path='sensitivity_3d.png'):
        """生成3D曲面图"""
        print(f"\n生成3D曲面图...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # 准备数据
        pivot_table = df_results.pivot(
            index='Slope_K', 
            columns='Weight_W', 
            values='Fairness_Score'
        )
        
        X = pivot_table.columns.values
        Y = pivot_table.index.values
        X, Y = np.meshgrid(X, Y)
        Z = pivot_table.values
        
        # 创建3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, 
                               edgecolor='none', antialiased=True)
        
        # 添加等高线
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='RdYlGn', alpha=0.5)
        
        ax.set_xlabel('Judge Weight (w)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sigmoid Slope (k)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Fairness Score', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Sensitivity: 3D Surface', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存: {output_path}")
        plt.close()
    
    def analyze_sweet_spot(self, df_results, threshold=0.0):
        """分析"甜点区"（Sweet Spot）"""
        print(f"\n{'='*80}")
        print("甜点区分析 (Sweet Spot Analysis)")
        print(f"{'='*80}")
        
        # 找出公平性得分 > threshold 的参数组合
        sweet_spot = df_results[df_results['Fairness_Score'] > threshold]
        
        print(f"\n公平性得分 > {threshold} 的参数组合:")
        print(f"  数量: {len(sweet_spot)} / {len(df_results)} ({len(sweet_spot)/len(df_results)*100:.1f}%)")
        
        if len(sweet_spot) > 0:
            print(f"\n权重 (w) 范围:")
            print(f"  最小值: {sweet_spot['Weight_W'].min():.2f}")
            print(f"  最大值: {sweet_spot['Weight_W'].max():.2f}")
            print(f"  跨度: {sweet_spot['Weight_W'].max() - sweet_spot['Weight_W'].min():.2f}")
            
            print(f"\n斜率 (k) 范围:")
            print(f"  最小值: {sweet_spot['Slope_K'].min():.0f}")
            print(f"  最大值: {sweet_spot['Slope_K'].max():.0f}")
            print(f"  跨度: {sweet_spot['Slope_K'].max() - sweet_spot['Slope_K'].min():.0f}")
            
            print(f"\n结论:")
            if len(sweet_spot) / len(df_results) > 0.3:
                print("  ✓ 模型非常稳健！存在广阔的参数甜点区")
                print("  ✓ 参数选择具有较大的灵活性")
            elif len(sweet_spot) / len(df_results) > 0.1:
                print("  ✓ 模型较为稳健，存在明确的参数甜点区")
            else:
                print("  ⚠ 模型对参数较为敏感，需要精确调参")
        
        return sweet_spot


def main():
    """主函数"""
    print("="*80)
    print("参数空间灵敏度分析 (Sensitivity Analysis)")
    print("="*80)
    
    # 初始化分析器
    analyzer = SensitivityAnalyzer()
    
    # 加载数据
    df, df_baseline = analyzer.load_data()
    
    # 定义参数范围
    weight_range = np.arange(0.4, 0.91, 0.05)  # 0.4 到 0.9，步长0.05
    slope_range = [5, 10, 15, 20, 25]  # Sigmoid斜率
    
    print(f"\n参数空间:")
    print(f"  权重 w: {len(weight_range)} 个值 ({weight_range[0]:.2f} - {weight_range[-1]:.2f})")
    print(f"  斜率 k: {len(slope_range)} 个值 ({slope_range})")
    print(f"  总组合: {len(weight_range) * len(slope_range)} 个")
    
    # 网格搜索
    df_results = analyzer.grid_search(df, df_baseline, weight_range, slope_range)
    
    # 保存结果
    output_csv = 'Sensitivity_Grid_Search.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ 结果保存到: {output_csv}")
    
    # 生成热力图
    analyzer.visualize_heatmap(df_results, 'sensitivity_heatmap.png')
    
    # 生成3D曲面图
    analyzer.visualize_3d_surface(df_results, 'sensitivity_3d.png')
    
    # 分析甜点区
    sweet_spot = analyzer.analyze_sweet_spot(df_results, threshold=0.0)
    
    print(f"\n{'='*80}")
    print("灵敏度分析完成")
    print(f"{'='*80}")
    
    return analyzer, df_results


if __name__ == '__main__':
    analyzer, df_results = main()
