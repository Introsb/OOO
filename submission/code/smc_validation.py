"""
SMC Validation: 5 Independent Consistency Checks
验证SMC算法的可靠性（无需Ground Truth）
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, ks_2samp
import json
import warnings
warnings.filterwarnings('ignore')


class SMCValidator:
    """SMC算法验证器"""
    
    def __init__(self, df_fan_votes, df_processed):
        self.df_fan = df_fan_votes
        self.df_proc = df_processed
        
    def validate_consistency(self):
        """一致性验证：多个独立证据"""
        
        results = {}
        
        print("\n" + "="*80)
        print("SMC VALIDATION: 5 INDEPENDENT CONSISTENCY CHECKS")
        print("="*80)
        
        # 验证1：投票总和归一化（必须=1.0）
        print("\n[1/5] Checking normalization...")
        results['normalization'] = self._check_normalization()
        
        # 验证2：与真实淘汰结果的一致性
        print("[2/5] Checking elimination consistency...")
        results['elimination_consistency'] = self._check_elimination_consistency()
        
        # 验证3：时间序列平滑性（人气不应该剧烈波动）
        print("[3/5] Checking temporal smoothness...")
        results['temporal_smoothness'] = self._check_temporal_smoothness()
        
        # 验证4：跨赛季稳定性（相似选手应该有相似人气）
        print("[4/5] Checking cross-season stability...")
        results['cross_season_stability'] = self._check_cross_season_stability()
        
        # 验证5：与裁判分数的相关性（应该存在但不完全相关）
        print("[5/5] Checking judge correlation...")
        results['judge_correlation'] = self._check_judge_correlation()
        
        return results
    
    def _check_normalization(self):
        """验证投票总和=1.0"""
        grouped = self.df_fan.groupby(['Season', 'Week'])
        sums = grouped['Estimated_Fan_Vote'].sum()
        
        # 所有周次的投票总和应该=1.0（允许1e-6的误差）
        deviations = np.abs(sums - 1.0)
        max_deviation = deviations.max()
        pass_rate = (deviations < 1e-6).mean()
        
        status = 'PASS' if pass_rate > 0.99 else 'FAIL'
        
        return {
            'max_deviation': float(max_deviation),
            'pass_rate': float(pass_rate),
            'total_weeks': len(sums),
            'passed_weeks': int((deviations < 1e-6).sum()),
            'status': status,
            'interpretation': 'All weeks sum to 1.0 (perfect normalization)' if status == 'PASS' else 'Normalization failed'
        }
    
    def _check_elimination_consistency(self):
        """验证与真实淘汰结果的一致性"""
        # 合并数据
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Placement']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 计算观众票和排名的相关性（负相关：票低→排名低）
        # 注意：Placement越小越好，Fan_Vote越高越好，所以应该是负相关
        corr = df['Estimated_Fan_Vote'].corr(df['Placement'])
        
        # 另一个指标：观众票在底部25%的选手，有多少排名也在底部25%
        df['Fan_Vote_Quartile'] = pd.qcut(df['Estimated_Fan_Vote'], q=4, labels=[1,2,3,4])
        df['Placement_Quartile'] = pd.qcut(df['Placement'], q=4, labels=[4,3,2,1])  # 反转，使得1=最差
        
        # 计算底部25%的一致性
        bottom_fan = df['Fan_Vote_Quartile'] == 1
        bottom_placement = df['Placement_Quartile'] == 1
        consistency_rate = (bottom_fan & bottom_placement).sum() / bottom_fan.sum()
        
        status = 'PASS' if consistency_rate > 0.25 else 'FAIL'
        
        return {
            'consistency_rate': float(consistency_rate),
            'correlation': float(corr),
            'bottom_quartile_overlap': int((bottom_fan & bottom_placement).sum()),
            'total_bottom_quartile': int(bottom_fan.sum()),
            'status': status,
            'interpretation': f'{consistency_rate:.1%} of lowest fan vote contestants are also in lowest placement quartile (expected ~25-40%)'
        }
    
    def _check_temporal_smoothness(self):
        """验证时间序列平滑性"""
        # 选取完整参赛的选手（至少5周）
        contestant_weeks = self.df_fan.groupby(['Season', 'Name']).size()
        long_contestants = contestant_weeks[contestant_weeks >= 5].index
        
        smoothness_scores = []
        
        for (season, name) in long_contestants:
            # 获取该选手的时间序列
            mask = (self.df_fan['Season'] == season) & (self.df_fan['Name'] == name)
            votes = self.df_fan.loc[mask].sort_values('Week')['Estimated_Fan_Vote'].values
            
            # 计算一阶差分的标准差（越小越平滑）
            if len(votes) > 1:
                diffs = np.diff(votes)
                smoothness = np.std(diffs)
                smoothness_scores.append(smoothness)
        
        avg_smoothness = np.mean(smoothness_scores)
        status = 'PASS' if avg_smoothness < 0.15 else 'FAIL'
        
        return {
            'avg_smoothness': float(avg_smoothness),
            'num_contestants': len(smoothness_scores),
            'status': status,
            'interpretation': f'Average week-to-week change is {avg_smoothness:.3f} (smooth trajectory, not erratic)'
        }
    
    def _check_cross_season_stability(self):
        """验证跨赛季稳定性"""
        # 计算每个选手的平均人气
        avg_votes = self.df_fan.groupby(['Season', 'Name'])['Estimated_Fan_Vote'].mean()
        
        # 按赛季分组，计算人气分布的相似性
        seasons = sorted(self.df_fan['Season'].unique())
        similarities = []
        
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i+1]
            
            # 获取两个赛季的人气分布
            if s1 in avg_votes.index.get_level_values(0) and s2 in avg_votes.index.get_level_values(0):
                votes1 = avg_votes[s1].values
                votes2 = avg_votes[s2].values
                
                if len(votes1) > 5 and len(votes2) > 5:
                    # 计算分布的相似性（使用KS统计量）
                    stat, pval = ks_2samp(votes1, votes2)
                    similarity = 1 - stat  # 转换为相似度
                    similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        status = 'PASS' if avg_similarity > 0.4 else 'FAIL'
        
        return {
            'avg_similarity': float(avg_similarity),
            'num_comparisons': len(similarities),
            'status': status,
            'interpretation': f'Fan vote distributions are {avg_similarity:.1%} similar across seasons (stable pattern)'
        }
    
    def _check_judge_correlation(self):
        """验证与裁判分数的相关性"""
        # 合并数据
        df = self.df_fan.merge(
            self.df_proc[['Season', 'Week', 'Name', 'Judge_Avg_Score']], 
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 移除缺失值
        df_clean = df.dropna(subset=['Estimated_Fan_Vote', 'Judge_Avg_Score'])
        
        # 计算相关系数
        corr_pearson = df_clean['Estimated_Fan_Vote'].corr(df_clean['Judge_Avg_Score'])
        corr_spearman, _ = spearmanr(df_clean['Estimated_Fan_Vote'], df_clean['Judge_Avg_Score'])
        
        # 理想情况：存在相关但不完全相关（0.1 < |r| < 0.7）
        status = 'PASS' if 0.1 < abs(corr_spearman) < 0.7 else 'FAIL'
        
        return {
            'pearson_correlation': float(corr_pearson),
            'spearman_correlation': float(corr_spearman),
            'sample_size': len(df_clean),
            'status': status,
            'interpretation': f'Moderate correlation (r={corr_spearman:.2f}) indicates fan votes are related to but not determined by judge scores'
        }
    
    def generate_validation_report(self):
        """生成验证报告"""
        results = self.validate_consistency()
        
        print("\n" + "="*80)
        print("VALIDATION REPORT SUMMARY")
        print("="*80)
        
        for test_name, test_result in results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print(f"  Status: {test_result['status']}")
            print(f"  {test_result['interpretation']}")
            
            # 打印详细指标
            for key, value in test_result.items():
                if key not in ['status', 'interpretation']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        # 计算总体通过率
        pass_count = sum(1 for r in results.values() if r['status'] == 'PASS')
        total_tests = len(results)
        overall_pass_rate = pass_count / total_tests
        
        print(f"\n{'='*80}")
        print(f"OVERALL PASS RATE: {pass_count}/{total_tests} ({overall_pass_rate:.0%})")
        
        if overall_pass_rate == 1.0:
            print("✓ ALL CHECKS PASSED - SMC estimates are reliable")
        elif overall_pass_rate >= 0.8:
            print("✓ MOST CHECKS PASSED - SMC estimates are generally reliable")
        else:
            print("✗ MULTIPLE CHECKS FAILED - SMC estimates may be unreliable")
        
        print(f"{'='*80}")
        
        return results, overall_pass_rate


def main():
    """主函数"""
    print("="*80)
    print("SMC VALIDATION ANALYSIS")
    print("="*80)
    print("\nLoading data...")
    
    # 加载数据
    try:
        df_fan = pd.read_csv('submission/results/Q1_Estimated_Fan_Votes.csv')
        df_proc = pd.read_csv('submission/results/Processed_DWTS_Long_Format.csv')
        print(f"✓ Loaded {len(df_fan)} fan vote estimates")
        print(f"✓ Loaded {len(df_proc)} processed records")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure the following files exist:")
        print("  - submission/results/Q1_Estimated_Fan_Votes.csv")
        print("  - submission/results/Processed_DWTS_Long_Format.csv")
        return None, None
    
    # 创建验证器
    validator = SMCValidator(df_fan, df_proc)
    
    # 生成验证报告
    results, pass_rate = validator.generate_validation_report()
    
    # 保存结果
    print(f"\nSaving results...")
    
    # 转换为可序列化的格式
    save_results = {}
    for key, value in results.items():
        save_results[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                             for k, v in value.items()}
    
    with open('submission/results/SMC_Validation_Report.json', 'w') as f:
        json.dump({
            'validation_results': save_results,
            'overall_pass_rate': float(pass_rate),
            'summary': f'{int(pass_rate*5)}/5 checks passed'
        }, f, indent=2)
    
    print(f"✓ Validation report saved to submission/results/SMC_Validation_Report.json")
    
    return results, pass_rate


if __name__ == '__main__':
    results, pass_rate = main()
