"""
SMC (Sequential Monte Carlo / Particle Filter) 观众投票反演系统
使用粒子滤波算法从裁判打分和淘汰结果反推观众投票分布
"""

import numpy as np
import pandas as pd
import re
from typing import Tuple, Dict, List


class SMCFanVoteEstimator:
    """基于SMC的观众投票反演器"""
    
    def __init__(self, n_particles: int = 5000, noise_std: float = 0.05, alpha: float = 2.0):
        """
        初始化SMC估计器
        
        参数:
            n_particles: 粒子数量
            noise_std: 随机游走噪声标准差
            alpha: 软约束指数衰减参数
        """
        self.n_particles = n_particles
        self.noise_std = noise_std
        self.alpha = alpha
    
    def initialize_particles(self, n_contestants: int) -> np.ndarray:
        """
        初始化粒子（使用Dirichlet分布）
        
        参数:
            n_contestants: 选手数量
            
        返回:
            shape (n_particles, n_contestants) 的粒子矩阵
        """
        # 使用Dirichlet分布初始化，假设大家票数比较平均
        alpha_prior = np.ones(n_contestants)
        particles = np.random.dirichlet(alpha_prior, size=self.n_particles)
        return particles
    
    def predict_step(self, particles: np.ndarray) -> np.ndarray:
        """
        预测步骤：添加随机噪声模拟人气波动
        
        参数:
            particles: 当前粒子矩阵
            
        返回:
            更新后的粒子矩阵
        """
        # 添加高斯噪声
        noise = np.random.normal(0, self.noise_std, particles.shape)
        new_particles = particles + noise
        
        # 确保非负
        new_particles = np.maximum(new_particles, 1e-10)
        
        # 重新归一化
        new_particles = new_particles / new_particles.sum(axis=1, keepdims=True)
        
        return new_particles
    
    def calculate_elimination_rank(self, judge_scores: np.ndarray, fan_votes: np.ndarray, 
                                   rule: str) -> int:
        """
        计算应该被淘汰的选手索引
        
        参数:
            judge_scores: 裁判分数数组
            fan_votes: 观众投票数组
            rule: 'rank' 或 'percent'
            
        返回:
            应该被淘汰的选手索引
        """
        n = len(judge_scores)
        
        if rule == 'rank':
            # 排名制：排名数值越大越差（1是最好）
            judge_ranks = n - np.argsort(np.argsort(judge_scores))  # 转换为1是最好的排名
            fan_ranks = n - np.argsort(np.argsort(fan_votes))
            total_ranks = judge_ranks + fan_ranks
            eliminated_idx = np.argmax(total_ranks)  # 总排名最大（最差）
        else:  # percent
            # 百分比制：百分比越低越差
            judge_pct = judge_scores / judge_scores.sum()
            fan_pct = fan_votes / fan_votes.sum()
            total_pct = judge_pct + fan_pct
            eliminated_idx = np.argmin(total_pct)  # 总百分比最低
        
        return eliminated_idx
    
    def update_step(self, particles: np.ndarray, judge_scores: np.ndarray, 
                   true_eliminated_idx: int, rule: str) -> np.ndarray:
        """
        更新步骤：根据真实淘汰结果计算粒子权重
        
        参数:
            particles: 粒子矩阵
            judge_scores: 裁判分数数组
            true_eliminated_idx: 真实被淘汰的选手索引
            rule: 'rank' 或 'percent'
            
        返回:
            粒子权重数组
        """
        n_particles = particles.shape[0]
        weights = np.zeros(n_particles)
        
        for i in range(n_particles):
            fan_votes = particles[i]
            predicted_eliminated = self.calculate_elimination_rank(judge_scores, fan_votes, rule)
            
            if predicted_eliminated == true_eliminated_idx:
                # 预测正确
                weights[i] = 1.0
            else:
                # 预测错误：软约束策略
                # 计算真实淘汰者在模拟排名中的位置
                if rule == 'rank':
                    n = len(judge_scores)
                    judge_ranks = n - np.argsort(np.argsort(judge_scores))
                    fan_ranks = n - np.argsort(np.argsort(fan_votes))
                    total_ranks = judge_ranks + fan_ranks
                    # 真实淘汰者的排名位置（从最差开始数）
                    true_rank_position = np.sum(total_ranks >= total_ranks[true_eliminated_idx])
                    rank_diff = true_rank_position - 1  # 0表示最差
                else:  # percent
                    judge_pct = judge_scores / judge_scores.sum()
                    fan_pct = fan_votes / fan_votes.sum()
                    total_pct = judge_pct + fan_pct
                    # 真实淘汰者的排名位置（从最差开始数）
                    true_rank_position = np.sum(total_pct <= total_pct[true_eliminated_idx])
                    rank_diff = true_rank_position - 1
                
                # 指数衰减权重
                weights[i] = np.exp(-self.alpha * rank_diff)
        
        return weights
    
    def resample(self, particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        重采样步骤
        
        参数:
            particles: 粒子矩阵
            weights: 粒子权重
            
        返回:
            重采样后的粒子矩阵
        """
        # 检查权重是否全为0
        if weights.sum() == 0:
            # 急救机制：重新初始化粒子
            print("    警告：所有粒子权重为0，触发急救机制")
            return self.initialize_particles(particles.shape[1])
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 系统重采样
        n_particles = particles.shape[0]
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        
        return particles[indices]
    
    def estimate_fan_votes(self, data_path: str, original_data_path: str) -> pd.DataFrame:
        """
        主函数：估计观众投票
        
        参数:
            data_path: 处理后的长表数据路径
            original_data_path: 原始数据路径（用于获取淘汰信息）
            
        返回:
            包含估计结果的DataFrame
        """
        # 加载数据
        df = pd.read_csv(data_path)
        df_original = pd.read_csv(original_data_path)
        
        # 提取淘汰信息
        elimination_info = self._extract_elimination_info(df_original)
        
        # 结果存储
        results = []
        
        # 按赛季处理
        seasons = sorted(df['Season'].unique())
        
        for season in seasons:
            print(f"\n处理 Season {season}...")
            season_data = df[df['Season'] == season].copy()
            
            # 确定规则
            rule = 'rank' if season <= 2 else 'percent'
            print(f"  使用规则: {rule}")
            
            # 按周处理
            weeks = sorted(season_data['Week'].unique())
            particles = None
            current_contestants = None
            
            for week in weeks:
                week_data = season_data[season_data['Week'] == week].copy()
                contestants = week_data['Name'].values
                judge_scores = week_data['Judge_Avg_Score'].values
                
                # 初始化粒子（第一周）
                if particles is None:
                    particles = self.initialize_particles(len(contestants))
                    current_contestants = list(contestants)
                    print(f"  Week {week}: 初始化 {self.n_particles} 个粒子，{len(contestants)} 位选手")
                else:
                    # 检查选手列表是否变化（处理数据不一致的情况）
                    if len(contestants) != len(current_contestants):
                        # 选手数量变化，需要调整粒子
                        print(f"  Week {week}: 选手数量从 {len(current_contestants)} 变为 {len(contestants)}")
                        
                        # 找出新的选手列表
                        new_indices = []
                        for name in contestants:
                            if name in current_contestants:
                                old_idx = current_contestants.index(name)
                                new_indices.append(old_idx)
                        
                        if len(new_indices) == len(contestants):
                            # 只是移除了选手
                            particles = particles[:, new_indices]
                            particles = particles / particles.sum(axis=1, keepdims=True)
                        else:
                            # 有新选手加入或其他复杂情况，重新初始化
                            print(f"    警告：选手列表不一致，重新初始化粒子")
                            particles = self.initialize_particles(len(contestants))
                        
                        current_contestants = list(contestants)
                    else:
                        # 预测步骤
                        particles = self.predict_step(particles)
                        print(f"  Week {week}: {len(contestants)} 位选手")
                
                # 获取本周淘汰者
                eliminated_name = elimination_info.get((season, week), None)
                
                if eliminated_name and eliminated_name in contestants:
                    # 更新步骤
                    eliminated_idx = np.where(contestants == eliminated_name)[0][0]
                    weights = self.update_step(particles, judge_scores, eliminated_idx, rule)
                    
                    # 重采样
                    particles = self.resample(particles, weights)
                    print(f"    淘汰: {eliminated_name}")
                
                # 计算估计值和不确定性（在淘汰前计算，使用当前周的完整选手列表）
                fan_vote_mean = particles.mean(axis=0)
                fan_vote_std = particles.std(axis=0)
                
                # 保存结果
                for i, name in enumerate(contestants):
                    results.append({
                        'Season': season,
                        'Week': week,
                        'Name': name,
                        'Estimated_Fan_Vote': fan_vote_mean[i],
                        'Uncertainty_Std': fan_vote_std[i],
                        'Rule_Used': rule
                    })
                
                # 淘汰后更新粒子矩阵（为下一周准备）
                if eliminated_name and eliminated_name in contestants:
                    surviving_indices = [i for i, name in enumerate(contestants) if name != eliminated_name]
                    particles = particles[:, surviving_indices]
                    particles = particles / particles.sum(axis=1, keepdims=True)
                    current_contestants = [name for name in contestants if name != eliminated_name]
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def _extract_elimination_info(self, df_original: pd.DataFrame) -> Dict[Tuple[int, int], str]:
        """
        从原始数据中提取淘汰信息
        
        参数:
            df_original: 原始数据DataFrame
            
        返回:
            字典 {(season, week): eliminated_name}
        """
        elimination_info = {}
        
        for _, row in df_original.iterrows():
            result_str = str(row['results'])
            season = row['season']
            name = row['celebrity_name']
            
            # 提取淘汰周次
            match = re.search(r'Eliminated Week (\d+)', result_str)
            if match:
                week = int(match.group(1))
                elimination_info[(season, week)] = name
        
        return elimination_info


def main():
    """主函数"""
    print("=" * 60)
    print("SMC观众投票反演系统")
    print("=" * 60)
    
    # 初始化估计器
    estimator = SMCFanVoteEstimator(n_particles=5000, noise_std=0.05, alpha=2.0)
    
    # 运行估计
    results_df = estimator.estimate_fan_votes(
        data_path='Processed_DWTS_Long_Format.csv',
        original_data_path='2026 MCM Problem C Data.csv'
    )
    
    # 保存结果
    output_path = 'Q1_Estimated_Fan_Votes.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"✓ 结果已保存至: {output_path}")
    print(f"  总行数: {len(results_df)}")
    print(f"  赛季数: {results_df['Season'].nunique()}")
    
    # 验证：检查每周的投票总和
    print(f"\n验证：检查每周投票总和...")
    for (season, week), group in results_df.groupby(['Season', 'Week']):
        vote_sum = group['Estimated_Fan_Vote'].sum()
        if abs(vote_sum - 1.0) > 0.01:
            print(f"  警告: Season {season} Week {week} 投票总和 = {vote_sum:.4f}")
    
    print(f"\n✓ 处理完成！")


if __name__ == '__main__':
    main()
