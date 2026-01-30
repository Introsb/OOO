"""
Enhanced SMC Validation: 8 Independent Consistency Checks + Visualizations
增强版SMC验证：8个独立检验 + 可视化
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, ks_2samp, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class EnhancedSMCValidator:
    """增强版SMC算法验证器"""
    
    def __init__(self, df_fan_votes, df_processed):
        self.df_fan = df_fan_votes
        self.df_proc = df_processed
        
    def validate_consistency(self):
        """一致性验证：8个独立证据"""
        
        results = {}
        
        print("\n" + "="*80)
        print("ENHANCED SMC VALIDATION: 8 INDEPENDENT CONSISTENCY CHECKS")
        print("="*80)
        
        # 原有的5个验证
        print("\n[1/8] Checking normalization...")
        results['normalization'] = self._check_normalization()
        
        print("[2/8] Checking elimination consistency...")
        results['elimination_consistency'] = self._check_elimination_consistency()
        
        print("[3/8] Checking temporal smoothness...")
        results['temporal_smoothness'] = self._check_temporal_smoothness()
        
        print("[4/8] Checking cross-season stability...")
        results['cross_season_stability'] = self._check_cross_season_stability()
        
        print("[5/8] Checking judge correlation...")
        results['judge_correlation'] = self._check_judge_correlation()
        
        # 新增的3个验证
        print("[6/8] Checking winner consistency...")
        results['winner_consistency'] = self._check_winner_consistency()
        
        print("[7/8] Checking early elimination consistency...")
        results['early_elimination'] = self._check_early_elimination()
        
        print("[8/8] Checking vote distribution reasonableness...")
        results['vote_distribution'] = self._check_vote_distribution()
        
        return results
    
    def _check_normalization(self):
        """验证投票总和=1.0"""
        grouped = self.df_fan.groupby(['Season', 'Week'])
        sums = grouped['Estimated_Fan_Vote'].sum()
        
        deviations = np.abs(sums - 1.0)
        max_deviation = deviations.max()
        pass_rate = (deviations < 1e-6).mean()
        
        status = 'PASS' if pass_rate > 0.99 else 'FAIL'
        
        return {
            'max_deviation': float(max_deviation),
            'pass_rate': float(pass_rate),
            'total_weeks': len(sums),
            'passed_weeks': int((deviations < 1e-6).sum()),
            'status': status
        }
    
    def _check_elimination_consistency(self):
        """验证与真实淘汰结果的一致性"""
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Placement']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 相关性（负相关：票低→排名低）
        corr = df['Estimated_Fan_Vote'].corr(df['Placement'])
        
        # 底部四分位一致性
        df['Fan_Vote_Quartile'] = pd.qcut(df['Estimated_Fan_Vote'], q=4, labels=[1,2,3,4])
        df['Placement_Quartile'] = pd.qcut(df['Placement'], q=4, labels=[4,3,2,1])
        
        bottom_fan = df['Fan_Vote_Quartile'] == 1
        bottom_placement = df['Placement_Quartile'] == 1
        consistency_rate = (bottom_fan & bottom_placement).sum() / bottom_fan.sum()
        
        status = 'PASS' if consistency_rate > 0.25 else 'FAIL'
        
        return {
            'consistency_rate': float(consistency_rate),
            'correlation': float(corr),
            'bottom_quartile_overlap': int((bottom_fan & bottom_placement).sum()),
            'total_bottom_quartile': int(bottom_fan.sum()),
            'status': status
        }
    
    def _check_temporal_smoothness(self):
        """验证时间序列平滑性"""
        contestant_weeks = self.df_fan.groupby(['Season', 'Name']).size()
        long_contestants = contestant_weeks[contestant_weeks >= 5].index
        
        smoothness_scores = []
        
        for (season, name) in long_contestants:
            mask = (self.df_fan['Season'] == season) & (self.df_fan['Name'] == name)
            votes = self.df_fan.loc[mask].sort_values('Week')['Estimated_Fan_Vote'].values
            
            if len(votes) > 1:
                diffs = np.diff(votes)
                smoothness = np.mean(np.abs(diffs))
                smoothness_scores.append(smoothness)
        
        avg_smoothness = np.mean(smoothness_scores)
        status = 'PASS' if avg_smoothness < 0.1 else 'FAIL'
        
        return {
            'avg_smoothness': float(avg_smoothness),
            'num_contestants': len(smoothness_scores),
            'status': status
        }
    
    def _check_cross_season_stability(self):
        """验证跨赛季稳定性"""
        avg_votes = self.df_fan.groupby(['Season', 'Name'])['Estimated_Fan_Vote'].mean()
        seasons = sorted(self.df_fan['Season'].unique())
        
        similarities = []
        
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i+1]
            
            votes1 = avg_votes[s1].values if s1 in avg_votes.index else []
            votes2 = avg_votes[s2].values if s2 in avg_votes.index else []
            
            if len(votes1) > 5 and len(votes2) > 5:
                stat, pval = ks_2samp(votes1, votes2)
                similarity = 1 - stat
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        status = 'PASS' if avg_similarity > 0.5 else 'FAIL'
        
        return {
            'avg_similarity': float(avg_similarity),
            'num_comparisons': len(similarities),
            'status': status
        }
    
    def _check_judge_correlation(self):
        """验证与裁判分数的相关性"""
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Judge_Avg_Score']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        corr_pearson = df['Estimated_Fan_Vote'].corr(df['Judge_Avg_Score'])
        corr_spearman, _ = spearmanr(df['Estimated_Fan_Vote'], df['Judge_Avg_Score'])
        
        # 理想范围：0.1 < |r| < 0.7
        status = 'PASS' if 0.1 < abs(corr_spearman) < 0.7 else 'FAIL'
        
        return {
            'pearson_correlation': float(corr_pearson),
            'spearman_correlation': float(corr_spearman),
            'sample_size': len(df),
            'status': status
        }
    
    def _check_winner_consistency(self):
        """新增验证6：冠军一致性"""
        # 检查每个赛季的冠军（Placement=1）是否有较高的观众票
        
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Placement']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 找出每个赛季的冠军
        winners = df[df['Placement'] == 1].groupby('Season')['Name'].first()
        
        winner_stats = []
        for season, winner_name in winners.items():
            # 获取冠军在整个赛季的平均观众票
            mask = (df['Season'] == season) & (df['Name'] == winner_name)
            avg_vote = df.loc[mask, 'Estimated_Fan_Vote'].mean()
            
            # 获取该赛季所有选手的平均观众票
            season_avg = df[df['Season'] == season].groupby('Name')['Estimated_Fan_Vote'].mean()
            
            # 冠军的排名（1=最高票）
            winner_rank = (season_avg > avg_vote).sum() + 1
            total_contestants = len(season_avg)
            
            winner_stats.append({
                'season': season,
                'winner': winner_name,
                'avg_vote': avg_vote,
                'rank': winner_rank,
                'total': total_contestants,
                'percentile': 1 - (winner_rank / total_contestants)
            })
        
        # 计算冠军在前50%的比例
        top_half_rate = sum(1 for s in winner_stats if s['percentile'] >= 0.5) / len(winner_stats)
        
        # 计算冠军的平均百分位
        avg_percentile = np.mean([s['percentile'] for s in winner_stats])
        
        status = 'PASS' if avg_percentile > 0.5 else 'FAIL'
        
        return {
            'top_half_rate': float(top_half_rate),
            'avg_percentile': float(avg_percentile),
            'num_winners': len(winner_stats),
            'status': status,
            'interpretation': f'{top_half_rate:.1%} of winners are in top 50% of fan votes'
        }
    
    def _check_early_elimination(self):
        """新增验证7：早期淘汰一致性"""
        # 检查第1-2周被淘汰的选手是否有较低的观众票
        
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Placement']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 找出每个选手的最后一周
        last_week = df.groupby(['Season', 'Name'])['Week'].max()
        
        early_eliminated = []
        for (season, name), max_week in last_week.items():
            if max_week <= 2:  # 第1-2周被淘汰
                # 获取该选手的平均观众票
                mask = (df['Season'] == season) & (df['Name'] == name)
                avg_vote = df.loc[mask, 'Estimated_Fan_Vote'].mean()
                
                # 获取该赛季第1-2周所有选手的平均观众票
                season_early = df[(df['Season'] == season) & (df['Week'] <= 2)]
                season_avg = season_early.groupby('Name')['Estimated_Fan_Vote'].mean()
                
                # 该选手的排名
                rank = (season_avg > avg_vote).sum() + 1
                total = len(season_avg)
                
                early_eliminated.append({
                    'season': season,
                    'name': name,
                    'avg_vote': avg_vote,
                    'rank': rank,
                    'total': total,
                    'percentile': 1 - (rank / total)
                })
        
        # 计算早期淘汰者在底部50%的比例
        bottom_half_rate = sum(1 for e in early_eliminated if e['percentile'] < 0.5) / len(early_eliminated) if early_eliminated else 0
        
        # 计算早期淘汰者的平均百分位
        avg_percentile = np.mean([e['percentile'] for e in early_eliminated]) if early_eliminated else 0
        
        status = 'PASS' if avg_percentile < 0.5 else 'FAIL'
        
        return {
            'bottom_half_rate': float(bottom_half_rate),
            'avg_percentile': float(avg_percentile),
            'num_early_eliminated': len(early_eliminated),
            'status': status,
            'interpretation': f'{bottom_half_rate:.1%} of early eliminated are in bottom 50% of fan votes'
        }
    
    def _check_vote_distribution(self):
        """新增验证8：投票分布合理性"""
        # 检查观众票的分布是否合理（不应该过于集中或过于分散）
        
        # 计算每周的投票分布熵
        entropies = []
        gini_coefficients = []
        
        for (season, week), group in self.df_fan.groupby(['Season', 'Week']):
            votes = group['Estimated_Fan_Vote'].values
            
            # 计算熵（信息熵）
            entropy = -np.sum(votes * np.log(votes + 1e-10))
            entropies.append(entropy)
            
            # 计算基尼系数
            sorted_votes = np.sort(votes)
            n = len(votes)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_votes)) / (n * np.sum(sorted_votes)) - (n + 1) / n
            gini_coefficients.append(gini)
        
        avg_entropy = np.mean(entropies)
        avg_gini = np.mean(gini_coefficients)
        
        # 合理范围：熵应该在1.5-2.5之间（不太集中也不太分散）
        # 基尼系数应该在0.1-0.4之间
        status = 'PASS' if 1.5 < avg_entropy < 2.5 and 0.1 < avg_gini < 0.4 else 'FAIL'
        
        return {
            'avg_entropy': float(avg_entropy),
            'avg_gini': float(avg_gini),
            'num_weeks': len(entropies),
            'status': status,
            'interpretation': f'Entropy={avg_entropy:.2f} (ideal: 1.5-2.5), Gini={avg_gini:.2f} (ideal: 0.1-0.4)'
        }
    
    def generate_validation_report(self):
        """生成验证报告"""
        results = self.validate_consistency()
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        # 计算通过率
        pass_count = sum(1 for r in results.values() if r.get('status') == 'PASS')
        total_tests = len(results)
        overall_pass_rate = pass_count / total_tests
        
        print(f"\nOverall Pass Rate: {pass_count}/{total_tests} ({overall_pass_rate:.1%})")
        print(f"\nDetailed Results:")
        
        for i, (test_name, test_result) in enumerate(results.items(), 1):
            status_symbol = "✅" if test_result.get('status') == 'PASS' else "❌"
            print(f"\n{i}. {test_name.replace('_', ' ').title()}: {status_symbol} {test_result.get('status')}")
            
            # 打印关键指标
            for key, value in test_result.items():
                if key not in ['status', 'interpretation']:
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
            
            # 打印解释
            if 'interpretation' in test_result:
                print(f"   → {test_result['interpretation']}")
        
        return results, overall_pass_rate
    
    def create_visualizations(self, results):
        """创建可视化"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 通过率总览
        ax1 = axes[0, 0]
        test_names = [name.replace('_', '\n').title() for name in results.keys()]
        statuses = [1 if r.get('status') == 'PASS' else 0 for r in results.values()]
        colors = ['green' if s == 1 else 'red' for s in statuses]
        
        ax1.barh(test_names, statuses, color=colors, alpha=0.7)
        ax1.set_xlabel('Status (1=PASS, 0=FAIL)')
        ax1.set_title('SMC Validation: Test Results')
        ax1.set_xlim([0, 1.2])
        
        for i, (name, status) in enumerate(zip(test_names, statuses)):
            ax1.text(status + 0.05, i, 'PASS' if status == 1 else 'FAIL', 
                    va='center', fontweight='bold')
        
        # 2. 观众票 vs 裁判分数
        ax2 = axes[0, 1]
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Judge_Avg_Score']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        ax2.scatter(df['Judge_Avg_Score'], df['Estimated_Fan_Vote'], 
                   alpha=0.3, s=10)
        ax2.set_xlabel('Judge Average Score')
        ax2.set_ylabel('Estimated Fan Vote')
        ax2.set_title(f'Fan Vote vs Judge Score (r={results["judge_correlation"]["spearman_correlation"]:.3f})')
        
        # 添加趋势线
        z = np.polyfit(df['Judge_Avg_Score'], df['Estimated_Fan_Vote'], 1)
        p = np.poly1d(z)
        ax2.plot(df['Judge_Avg_Score'].sort_values(), 
                p(df['Judge_Avg_Score'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
        
        # 3. 观众票分布
        ax3 = axes[1, 0]
        all_votes = self.df_fan['Estimated_Fan_Vote'].values
        ax3.hist(all_votes, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Estimated Fan Vote')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Fan Votes')
        ax3.axvline(all_votes.mean(), color='red', linestyle='--', 
                   label=f'Mean={all_votes.mean():.3f}')
        ax3.legend()
        
        # 4. 时间平滑性示例
        ax4 = axes[1, 1]
        # 选择几个有代表性的选手
        sample_contestants = self.df_fan.groupby(['Season', 'Name']).size()
        sample_contestants = sample_contestants[sample_contestants >= 8].index[:5]
        
        for season, name in sample_contestants:
            mask = (self.df_fan['Season'] == season) & (self.df_fan['Name'] == name)
            contestant_data = self.df_fan.loc[mask].sort_values('Week')
            ax4.plot(contestant_data['Week'], contestant_data['Estimated_Fan_Vote'], 
                    marker='o', label=f'{name[:15]}...', alpha=0.7)
        
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Estimated Fan Vote')
        ax4.set_title('Temporal Smoothness: Sample Contestants')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/SMC_Validation_Enhanced.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to ../results/SMC_Validation_Enhanced.png")
        
        return fig


def main():
    """主函数"""
    print("="*80)
    print("ENHANCED SMC VALIDATION")
    print("="*80)
    print("\nLoading data...")
    
    # 加载数据
    try:
        df_fan = pd.read_csv('../results/Q1_Estimated_Fan_Votes.csv')
        df_proc = pd.read_csv('../results/Processed_DWTS_Long_Format.csv')
        print(f"✓ Loaded {len(df_fan)} fan vote estimates")
        print(f"✓ Loaded {len(df_proc)} processed records")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return None
    
    # 创建验证器
    validator = EnhancedSMCValidator(df_fan, df_proc)
    
    # 生成验证报告
    results, pass_rate = validator.generate_validation_report()
    
    # 创建可视化
    validator.create_visualizations(results)
    
    # 保存结果
    print(f"\nSaving results...")
    
    # 转换结果为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in value.items()
        }
    
    with open('../results/SMC_Validation_Enhanced.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'overall_pass_rate': float(pass_rate),
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results.values() if r.get('status') == 'PASS')
        }, f, indent=2)
    
    print(f"✓ Results saved to ../results/SMC_Validation_Enhanced.json")
    
    return results, pass_rate


if __name__ == '__main__':
    results, pass_rate = main()
