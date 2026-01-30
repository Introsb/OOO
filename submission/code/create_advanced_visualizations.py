"""
Create Advanced Visualizations for Phase 2
创建高级可视化图表 - 因果推断和时间动态
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class AdvancedVisualizer:
    """高级可视化器"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 300
        
    def plot_causal_inference_comparison(self):
        """绘制因果推断方法对比图"""
        print("\n1. Creating Causal Inference Comparison Chart...")
        
        # 手动输入结果
        methods = ['IV', 'DID', 'RDD', 'PSM', 'Granger']
        estimates = [1.4605, 0.0965, 0.0820, 1.9236, 0.0299]  # Granger用R²改进
        r2_scores = [0.4384, np.nan, np.nan, np.nan, 0.5703]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：因果效应估计
        colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'plum']
        bars = ax1.barh(methods, estimates, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Causal Effect Estimate', fontsize=12, fontweight='bold')
        ax1.set_title('Causal Inference Methods Comparison\n(Week Effect on Judge Scores)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (method, est) in enumerate(zip(methods, estimates)):
            ax1.text(est + 0.05, i, f'{est:.4f}', va='center', fontsize=10)
        
        # 右图：方法说明
        ax2.axis('off')
        
        method_descriptions = """
        CAUSAL INFERENCE METHODS
        
        IV (Instrumental Variable)
        • Estimate: 1.4605 points
        • Uses Week as instrument for quality
        • R² = 0.44
        
        DID (Difference-in-Differences)
        • Estimate: 0.0965 points
        • Compares early vs late weeks
        • Controls for time-invariant factors
        
        RDD (Regression Discontinuity)
        • Estimate: 0.0820 points
        • Discontinuity at Week 5 (semifinals)
        • Identifies local treatment effect
        
        PSM (Propensity Score Matching)
        • Estimate: 1.9236 points (ATT)
        • Controls for selection bias
        • Matches treated and control units
        
        Granger Causality
        • R² improvement: 0.0299
        • F-statistic: 163.60 (p < 0.001)
        • Tests temporal precedence
        """
        
        ax2.text(0.05, 0.5, method_descriptions, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        output_path = 'figures/causal_inference_comparison.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_temporal_dynamics_dashboard(self):
        """绘制时间动态4面板仪表板"""
        print("\n2. Creating Temporal Dynamics Dashboard...")
        
        # 加载数据
        df_processed = pd.read_csv('results/Processed_DWTS_Long_Format.csv')
        df_fan = pd.read_csv('results/Q1_Estimated_Fan_Votes.csv')
        
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel A: Score Inflation
        week_stats = df.groupby('Week').agg({
            'Judge_Avg_Score': ['mean', 'std']
        }).reset_index()
        week_stats.columns = ['Week', 'Mean', 'Std']
        
        ax1.plot(week_stats['Week'], week_stats['Mean'], 'o-', 
                linewidth=2, markersize=8, color='steelblue')
        ax1.fill_between(week_stats['Week'], 
                        week_stats['Mean'] - week_stats['Std'],
                        week_stats['Mean'] + week_stats['Std'],
                        alpha=0.3, color='steelblue')
        
        # 添加趋势线
        z = np.polyfit(week_stats['Week'], week_stats['Mean'], 1)
        p = np.poly1d(z)
        ax1.plot(week_stats['Week'], p(week_stats['Week']), "--", 
                color='red', linewidth=2, label=f'Trend: +{z[0]:.3f} pts/week')
        
        ax1.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Judge Average Score', fontsize=11, fontweight='bold')
        ax1.set_title('A. Score Inflation (49.8% increase)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Variance Convergence
        ax2.plot(week_stats['Week'], week_stats['Std'], 'o-', 
                linewidth=2, markersize=8, color='coral')
        
        # 添加趋势线
        z = np.polyfit(week_stats['Week'], week_stats['Std'], 1)
        p = np.poly1d(z)
        ax2.plot(week_stats['Week'], p(week_stats['Week']), "--", 
                color='red', linewidth=2, label=f'Trend: {z[0]:.3f} std/week')
        
        ax2.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        ax2.set_title('B. Variance Convergence (R²=0.74)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Survival Bias
        first_week = df.groupby(['Season', 'Name']).first().reset_index()
        first_week['First_Week_Score'] = first_week['Judge_Avg_Score']
        
        last_week = df.groupby(['Season', 'Name']).last().reset_index()
        last_week['Last_Week'] = last_week['Week']
        
        survival_df = first_week[['Season', 'Name', 'First_Week_Score']].merge(
            last_week[['Season', 'Name', 'Last_Week']],
            on=['Season', 'Name']
        )
        
        survival_stats = survival_df.groupby('Last_Week').agg({
            'First_Week_Score': ['mean', 'std', 'count']
        }).reset_index()
        survival_stats.columns = ['Last_Week', 'Mean', 'Std', 'Count']
        
        ax3.scatter(survival_stats['Last_Week'], survival_stats['Mean'], 
                   s=survival_stats['Count']*5, alpha=0.6, color='lightgreen')
        
        # 添加趋势线
        z = np.polyfit(survival_stats['Last_Week'], survival_stats['Mean'], 1)
        p = np.poly1d(z)
        ax3.plot(survival_stats['Last_Week'], p(survival_stats['Last_Week']), "--", 
                color='red', linewidth=2, label=f'Trend: +{z[0]:.3f} pts/week')
        
        ax3.set_xlabel('Weeks Survived', fontsize=11, fontweight='bold')
        ax3.set_ylabel('First Week Score', fontsize=11, fontweight='bold')
        ax3.set_title('C. Survival Bias (R²=0.90)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Elimination Threshold
        df_sorted = df.sort_values(['Season', 'Name', 'Week'])
        df_sorted['Has_Next_Week'] = df_sorted.groupby(['Season', 'Name'])['Week'].shift(-1).notna()
        
        threshold_stats = []
        for week in sorted(df_sorted['Week'].unique()):
            week_data = df_sorted[df_sorted['Week'] == week]
            eliminated = week_data[~week_data['Has_Next_Week']]
            survived = week_data[week_data['Has_Next_Week']]
            
            if len(eliminated) > 0 and len(survived) > 0:
                threshold_stats.append({
                    'Week': week,
                    'Elimination_Threshold': eliminated['Judge_Avg_Score'].max(),
                    'Safe_Minimum': survived['Judge_Avg_Score'].min()
                })
        
        threshold_df = pd.DataFrame(threshold_stats)
        
        ax4.plot(threshold_df['Week'], threshold_df['Elimination_Threshold'], 
                'o-', linewidth=2, markersize=8, color='red', label='Elimination Threshold')
        ax4.plot(threshold_df['Week'], threshold_df['Safe_Minimum'], 
                'o-', linewidth=2, markersize=8, color='green', label='Safe Minimum')
        ax4.fill_between(threshold_df['Week'], 
                        threshold_df['Elimination_Threshold'],
                        threshold_df['Safe_Minimum'],
                        alpha=0.3, color='yellow', label='Danger Zone')
        
        ax4.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Judge Score', fontsize=11, fontweight='bold')
        ax4.set_title('D. Elimination Threshold Evolution (R²=0.54)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Temporal Dynamics Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        output_path = 'figures/temporal_dynamics_dashboard.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_causal_dag(self):
        """绘制因果关系图（DAG）"""
        print("\n3. Creating Causal DAG...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 定义节点位置
        nodes = {
            'Week': (0.2, 0.7),
            'Contestant\nQuality': (0.5, 0.7),
            'Judge\nScore': (0.8, 0.7),
            'Age': (0.2, 0.3),
            'Season': (0.35, 0.3),
            'Industry': (0.5, 0.3),
            'Partner': (0.65, 0.3),
            'Fan Vote': (0.8, 0.3)
        }
        
        # 绘制节点
        for node, (x, y) in nodes.items():
            if node in ['Week', 'Contestant\nQuality', 'Judge\nScore']:
                color = 'lightblue'
                size = 0.08
            else:
                color = 'lightgray'
                size = 0.06
            
            circle = plt.Circle((x, y), size, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制箭头（因果关系）
        arrows = [
            ('Week', 'Contestant\nQuality', 'red', 3),
            ('Contestant\nQuality', 'Judge\nScore', 'red', 3),
            ('Age', 'Contestant\nQuality', 'blue', 1.5),
            ('Season', 'Contestant\nQuality', 'blue', 1.5),
            ('Industry', 'Contestant\nQuality', 'blue', 1.5),
            ('Partner', 'Contestant\nQuality', 'blue', 1.5),
            ('Contestant\nQuality', 'Fan Vote', 'green', 1.5),
            ('Week', 'Judge\nScore', 'orange', 1.5)  # Direct effect
        ]
        
        for start, end, color, width in arrows:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            
            # 计算箭头方向
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # 调整起点和终点（避免覆盖节点）
            offset = 0.09
            x1_adj = x1 + (dx / length) * offset
            y1_adj = y1 + (dy / length) * offset
            x2_adj = x2 - (dx / length) * offset
            y2_adj = y2 - (dy / length) * offset
            
            ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                       arrowprops=dict(arrowstyle='->', lw=width, color=color))
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=3, label='Causal Path (IV)'),
            plt.Line2D([0], [0], color='blue', lw=1.5, label='Confounders'),
            plt.Line2D([0], [0], color='orange', lw=1.5, label='Direct Effect'),
            plt.Line2D([0], [0], color='green', lw=1.5, label='Outcome')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        # 添加标题和说明
        ax.text(0.5, 0.95, 'Causal Directed Acyclic Graph (DAG)', 
               ha='center', fontsize=16, fontweight='bold')
        
        explanation = """
        Causal Identification Strategy:
        • Week → Quality → Score (Instrumental Variable path)
        • Confounders (Age, Season, Industry, Partner) controlled
        • Direct effect (Week → Score) estimated separately
        • Fan Vote as alternative outcome measure
        """
        
        ax.text(0.5, 0.05, explanation, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = 'figures/causal_dag.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def create_all_visualizations(self):
        """创建所有高级可视化"""
        print("="*80)
        print("CREATING ADVANCED VISUALIZATIONS (PHASE 2)")
        print("="*80)
        
        created_files = []
        
        # 1. Causal Inference Comparison
        try:
            path = self.plot_causal_inference_comparison()
            created_files.append(path)
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # 2. Temporal Dynamics Dashboard
        try:
            path = self.plot_temporal_dynamics_dashboard()
            created_files.append(path)
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # 3. Causal DAG
        try:
            path = self.plot_causal_dag()
            created_files.append(path)
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print("\n" + "="*80)
        print("ADVANCED VISUALIZATION COMPLETE")
        print("="*80)
        print(f"Created {len(created_files)} visualizations:")
        for f in created_files:
            print(f"  • {f}")
        
        return created_files


def main():
    """主函数"""
    visualizer = AdvancedVisualizer()
    files = visualizer.create_all_visualizations()
    return files


if __name__ == '__main__':
    files = main()
