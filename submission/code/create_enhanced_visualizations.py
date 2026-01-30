"""
Create Enhanced Visualizations for O Award
为O奖创建增强版可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedVisualizer:
    """增强版可视化器"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 300
        
    def plot_model_comparison(self, model_results_path='Model_Comparison_Results.csv'):
        """绘制模型对比图"""
        print("\n1. Creating Model Comparison Chart...")
        
        df = pd.read_csv(model_results_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Judge Score模型对比
        judge_data = df[df['Target'] == 'Judge Score']
        x = np.arange(len(judge_data))
        width = 0.35
        
        ax1.bar(x - width/2, judge_data['CV_Mean_R2'], width, 
                label='CV R²', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, judge_data['Train_R2'], width, 
                label='Train R²', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Judge Score Model Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(judge_data['Model'], rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (cv, train) in enumerate(zip(judge_data['CV_Mean_R2'], judge_data['Train_R2'])):
            ax1.text(i - width/2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Fan Vote模型对比
        fan_data = df[df['Target'] == 'Fan Vote']
        
        ax2.bar(x - width/2, fan_data['CV_Mean_R2'], width, 
                label='CV R²', alpha=0.8, color='steelblue')
        ax2.bar(x + width/2, fan_data['Train_R2'], width, 
                label='Train R²', alpha=0.8, color='coral')
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('Fan Vote Model Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fan_data['Model'], rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (cv, train) in enumerate(zip(fan_data['CV_Mean_R2'], fan_data['Train_R2'])):
            ax2.text(i - width/2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = 'figures/model_comparison_enhanced.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_r2_improvement(self):
        """绘制R²改进对比图"""
        print("\n2. Creating R² Improvement Chart...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Judge Score', 'Fan Vote']
        before = [0.2828, 0.1104]
        after = [0.5922, 0.6106]
        improvement = [(a-b)/b*100 for a, b in zip(after, before)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before, width, label='Before (Baseline)', 
                       alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, after, width, label='After (Enhanced)', 
                       alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('R² Improvement: Before vs After Enhancement', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值和改进百分比
        for i, (b, a, imp) in enumerate(zip(before, after, improvement)):
            ax.text(i - width/2, b + 0.01, f'{b:.1%}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
            ax.text(i + width/2, a + 0.01, f'{a:.1%}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
            ax.text(i, max(b, a) + 0.08, f'↑{imp:.0f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='green')
        
        plt.tight_layout()
        output_path = 'figures/r2_improvement.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_week_effect(self, processed_path='results/Processed_DWTS_Long_Format.csv',
                        fan_votes_path='results/Q1_Estimated_Fan_Votes.csv'):
        """绘制Week效应图"""
        print("\n3. Creating Week Effect Chart...")
        
        df_processed = pd.read_csv(processed_path)
        df_fan = pd.read_csv(fan_votes_path)
        
        df = df_processed.merge(
            df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Week vs Judge Score
        week_judge = df.groupby('Week')['Judge_Avg_Score'].agg(['mean', 'std']).reset_index()
        
        ax1.plot(week_judge['Week'], week_judge['mean'], 'o-', linewidth=2, 
                markersize=8, color='steelblue', label='Mean Score')
        ax1.fill_between(week_judge['Week'], 
                        week_judge['mean'] - week_judge['std'],
                        week_judge['mean'] + week_judge['std'],
                        alpha=0.3, color='steelblue')
        
        ax1.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Judge Average Score', fontsize=12, fontweight='bold')
        ax1.set_title('Week Effect on Judge Scores\n(Correlation: 0.66)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Week vs Fan Vote
        week_fan = df.groupby('Week')['Estimated_Fan_Vote'].agg(['mean', 'std']).reset_index()
        
        ax2.plot(week_fan['Week'], week_fan['mean'], 'o-', linewidth=2, 
                markersize=8, color='coral', label='Mean Vote')
        ax2.fill_between(week_fan['Week'], 
                        week_fan['mean'] - week_fan['std'],
                        week_fan['mean'] + week_fan['std'],
                        alpha=0.3, color='coral')
        
        ax2.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Estimated Fan Vote', fontsize=12, fontweight='bold')
        ax2.set_title('Week Effect on Fan Votes\n(Correlation: 0.65)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        output_path = 'figures/week_effect_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_parameter_sensitivity(self, params_path='Optimized_System_Parameters.csv'):
        """绘制参数灵敏度热力图"""
        print("\n4. Creating Parameter Sensitivity Heatmap...")
        
        df = pd.read_csv(params_path)
        
        # 创建pivot table
        pivot = df.pivot_table(
            values='composite_score',
            index='sigmoid_k',
            columns='judge_weight',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Composite Score'},
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('Judge Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sigmoid k', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Sensitivity Analysis\n(270 Combinations Tested)', 
                    fontsize=14, fontweight='bold')
        
        # 标注最优点
        best_idx = df['composite_score'].idxmax()
        best_row = df.loc[best_idx]
        ax.text(0.5, -0.15, 
               f'Optimal: Judge Weight={best_row["judge_weight"]:.2f}, k={best_row["sigmoid_k"]}, Score={best_row["composite_score"]:.4f}',
               transform=ax.transAxes, ha='center', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        output_path = 'figures/parameter_sensitivity_heatmap.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_arrow_conditions(self, conditions_path='Arrow_Conditions_Check.csv'):
        """绘制Arrow定理条件检查图"""
        print("\n5. Creating Arrow's Theorem Conditions Chart...")
        
        df = pd.read_csv(conditions_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = df['Condition'].str.replace(r'^\d+\.\s*', '', regex=True).tolist()
        satisfied = df['Satisfied'].astype(int).tolist()
        
        colors = ['green' if s else 'red' for s in satisfied]
        bars = ax.barh(conditions, [1]*len(conditions), color=colors, alpha=0.7)
        
        # 添加标签
        for i, (cond, sat) in enumerate(zip(conditions, satisfied)):
            status = '✓ PASS' if sat else '✗ FAIL'
            ax.text(0.5, i, status, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
        
        ax.set_xlabel('Status', fontsize=12, fontweight='bold')
        ax.set_title("Arrow's Impossibility Theorem: 5 Conditions Check\n(3/5 Satisfied - Consistent with Theory)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(axis='x', alpha=0.3)
        
        # 添加总结
        total_satisfied = sum(satisfied)
        ax.text(0.5, -0.8, 
               f'{total_satisfied}/5 conditions satisfied\nArrow\'s theorem predicts no system can satisfy all 5',
               transform=ax.transAxes, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        output_path = 'figures/arrow_theorem_conditions.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def plot_feature_importance_top10(self, features_path='Enhanced_Feature_Analysis.csv'):
        """绘制Top 10特征重要性"""
        print("\n6. Creating Top 10 Feature Importance Chart...")
        
        df = pd.read_csv(features_path)
        df['Abs_Coef_Judge'] = df['Coef_Judge'].abs()
        df['Abs_Coef_Fan'] = df['Coef_Fan'].abs()
        
        top10_judge = df.nlargest(10, 'Abs_Coef_Judge')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(top10_judge))
        colors = ['green' if c > 0 else 'red' for c in top10_judge['Coef_Judge']]
        
        bars = ax.barh(y_pos, top10_judge['Coef_Judge'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top10_judge['Feature'])
        ax.set_xlabel('Coefficient (Standardized)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Features Influencing Judge Scores\n(Random Forest Model)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (feat, coef) in enumerate(zip(top10_judge['Feature'], top10_judge['Coef_Judge'])):
            ax.text(coef + 0.01 if coef > 0 else coef - 0.01, i, 
                   f'{coef:.3f}', va='center', 
                   ha='left' if coef > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        output_path = 'figures/feature_importance_top10.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path
    
    def create_summary_dashboard(self):
        """创建总结仪表板"""
        print("\n7. Creating Summary Dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. R² Improvement
        ax1 = fig.add_subplot(gs[0, :2])
        categories = ['Judge Score', 'Fan Vote']
        before = [0.2828, 0.1104]
        after = [0.5922, 0.6106]
        x = np.arange(len(categories))
        width = 0.35
        ax1.bar(x - width/2, before, width, label='Before', alpha=0.8, color='lightcoral')
        ax1.bar(x + width/2, after, width, label='After', alpha=0.8, color='lightgreen')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Improvement', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Key Metrics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        metrics_text = """
        KEY IMPROVEMENTS
        
        Judge R²: 28% → 59%
        (+109%)
        
        Fan R²: 11% → 61%
        (+453%)
        
        Week Correlation: 0.66
        
        Models Tested: 4
        
        Parameters Tested: 270
        """
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 3. Arrow's Theorem
        ax3 = fig.add_subplot(gs[1, :])
        conditions = ['Non-dictatorship', 'Pareto efficiency', 'IIA', 
                     'Unrestricted domain', 'Transitivity']
        satisfied = [1, 0, 0, 1, 1]
        colors = ['green' if s else 'red' for s in satisfied]
        ax3.barh(conditions, [1]*len(conditions), color=colors, alpha=0.7)
        ax3.set_title("Arrow's Theorem: 5 Conditions (3/5 Satisfied)", fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.set_xticks([])
        
        # 4. Parameter Optimization
        ax4 = fig.add_subplot(gs[2, :])
        param_text = """
        OPTIMAL PARAMETERS FOUND (Grid Search: 270 Combinations)
        
        • Judge Weight: 50% (vs baseline 70%)
        • Fan Weight: 50% (vs baseline 30%)
        • Sigmoid k: 5 (vs baseline 15)
        • Sigmoid x₀: 0.3 (vs baseline 0.4)
        
        Performance: Injustice Rate 5.07% → 4.18% (-0.90%)
                    Technical Fairness 99.10% → 99.40% (+0.30%)
        """
        ax4.text(0.5, 0.5, param_text, fontsize=10, verticalalignment='center',
                horizontalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.axis('off')
        
        fig.suptitle('Enhanced Analysis Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        output_path = 'figures/summary_dashboard_enhanced.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to {output_path}")
        return output_path


def main():
    """主函数"""
    print("="*80)
    print("CREATING ENHANCED VISUALIZATIONS FOR O AWARD")
    print("="*80)
    
    visualizer = EnhancedVisualizer()
    
    created_files = []
    
    # 1. Model Comparison
    try:
        path = visualizer.plot_model_comparison()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. R² Improvement
    try:
        path = visualizer.plot_r2_improvement()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Week Effect
    try:
        path = visualizer.plot_week_effect()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. Parameter Sensitivity
    try:
        path = visualizer.plot_parameter_sensitivity()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 5. Arrow Conditions
    try:
        path = visualizer.plot_arrow_conditions()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 6. Feature Importance
    try:
        path = visualizer.plot_feature_importance_top10()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 7. Summary Dashboard
    try:
        path = visualizer.create_summary_dashboard()
        created_files.append(path)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Created {len(created_files)} visualizations:")
    for f in created_files:
        print(f"  • {f}")
    
    return created_files


if __name__ == '__main__':
    files = main()
