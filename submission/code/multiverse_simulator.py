"""
Project Multiverse - 平行宇宙仿真系统
模拟不同赛制规则下的淘汰结果，用于回答Q3和Q4
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


class MultiverseSimulator:
    """平行宇宙仿真器：模拟三种不同赛制规则"""
    
    def __init__(self):
        """初始化仿真器"""
        pass
    
    def load_and_merge_data(self, processed_path: str, fan_votes_path: str, 
                           original_path: str) -> pd.DataFrame:
        """
        加载并合并数据
        
        参数:
            processed_path: 预处理数据路径
            fan_votes_path: 观众投票估计路径
            original_path: 原始数据路径（用于获取真实淘汰信息）
            
        返回:
            合并后的DataFrame
        """
        # 加载数据
        df_processed = pd.read_csv(processed_path)
        df_fan_votes = pd.read_csv(fan_votes_path)
        df_original = pd.read_csv(original_path)
        
        # 合并裁判分数和观众投票
        df_merged = pd.merge(
            df_processed[['Season', 'Week', 'Name', 'Judge_Avg_Score', 'Placement']],
            df_fan_votes[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']],
            on=['Season', 'Week', 'Name'],
            how='inner'
        )
        
        # 提取真实淘汰信息
        elimination_info = self._extract_elimination_info(df_original)
        
        return df_merged, elimination_info
    
    def _extract_elimination_info(self, df_original: pd.DataFrame) -> Dict[Tuple[int, int], str]:
        """提取真实淘汰信息"""
        import re
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
    
    def simulate_rank_model(self, week_data: pd.DataFrame) -> str:
        """
        平行宇宙A：纯排名制
        
        参数:
            week_data: 某一周的数据
            
        返回:
            应该被淘汰的选手名字
        """
        # 计算排名（1是最好的）
        n = len(week_data)
        
        # 裁判分数排名：分数越高排名越好（1是最高分）
        judge_ranks = n + 1 - week_data['Judge_Avg_Score'].rank(method='min', ascending=False)
        
        # 观众投票排名：投票越高排名越好（1是最高票）
        fan_ranks = n + 1 - week_data['Estimated_Fan_Vote'].rank(method='min', ascending=False)
        
        # 总排名点数：越大越差
        total_points = judge_ranks + fan_ranks
        
        # 找出总点数最大的（最差的）
        max_points = total_points.max()
        worst_contestants = week_data[total_points == max_points]
        
        # 如果有平局，选择裁判分更低的
        if len(worst_contestants) > 1:
            eliminated_idx = worst_contestants['Judge_Avg_Score'].idxmin()
        else:
            eliminated_idx = worst_contestants.index[0]
        
        return week_data.loc[eliminated_idx, 'Name']
    
    def simulate_percentage_model(self, week_data: pd.DataFrame) -> str:
        """
        平行宇宙B：纯百分比制
        
        参数:
            week_data: 某一周的数据
            
        返回:
            应该被淘汰的选手名字
        """
        # 计算裁判分数占比
        judge_total = week_data['Judge_Avg_Score'].sum()
        judge_percent = week_data['Judge_Avg_Score'] / judge_total
        
        # 观众投票已经是占比（总和为1）
        fan_percent = week_data['Estimated_Fan_Vote']
        
        # 总分数
        total_score = judge_percent + fan_percent
        
        # 找出总分最低的
        eliminated_idx = total_score.idxmin()
        
        return week_data.loc[eliminated_idx, 'Name']
    
    def simulate_judges_save_model(self, week_data: pd.DataFrame) -> str:
        """
        平行宇宙C：裁判拯救机制
        
        参数:
            week_data: 某一周的数据
            
        返回:
            应该被淘汰的选手名字
        """
        # 步骤1：按百分比制找出Bottom 2
        judge_total = week_data['Judge_Avg_Score'].sum()
        judge_percent = week_data['Judge_Avg_Score'] / judge_total
        fan_percent = week_data['Estimated_Fan_Vote']
        total_score = judge_percent + fan_percent
        
        # 找出分数最低的两位
        bottom_2_indices = total_score.nsmallest(2).index
        bottom_2 = week_data.loc[bottom_2_indices]
        
        # 步骤2：比较Bottom 2的裁判分数
        # 步骤3：裁判拯救分数高的，淘汰分数低的
        eliminated_idx = bottom_2['Judge_Avg_Score'].idxmin()
        
        return week_data.loc[eliminated_idx, 'Name']
    
    def calculate_metrics(self, week_data: pd.DataFrame, 
                         actual_eliminated: str,
                         elim_rank: str, 
                         elim_percent: str, 
                         elim_save: str) -> Dict:
        """
        计算差异化分析指标
        
        参数:
            week_data: 某一周的数据
            actual_eliminated: 真实淘汰者
            elim_rank: 排名制淘汰者
            elim_percent: 百分比制淘汰者
            elim_save: 拯救机制淘汰者
            
        返回:
            指标字典
        """
        metrics = {}
        
        # 逆转标记：排名制 != 百分比制
        metrics['Is_Reversal'] = (elim_rank != elim_percent)
        
        # 拯救生效标记：拯救机制 != 百分比制
        metrics['Is_Saved'] = (elim_save != elim_percent)
        
        # 冤案指数：计算实际被淘汰者的排名
        if actual_eliminated in week_data['Name'].values:
            actual_data = week_data[week_data['Name'] == actual_eliminated].iloc[0]
            
            # 计算该选手的裁判排名和观众排名
            n = len(week_data)
            judge_rank = n + 1 - week_data['Judge_Avg_Score'].rank(method='min', ascending=False).loc[actual_data.name]
            fan_rank = n + 1 - week_data['Estimated_Fan_Vote'].rank(method='min', ascending=False).loc[actual_data.name]
            
            # 如果裁判排名和观众排名都不是倒数第一（都不是n），则是冤案
            metrics['Is_Injustice'] = (judge_rank < n) and (fan_rank < n)
            metrics['Judge_Rank'] = int(judge_rank)
            metrics['Fan_Rank'] = int(fan_rank)
        else:
            metrics['Is_Injustice'] = False
            metrics['Judge_Rank'] = None
            metrics['Fan_Rank'] = None
        
        return metrics
    
    def run_simulation(self, processed_path: str, fan_votes_path: str, 
                      original_path: str) -> pd.DataFrame:
        """
        运行完整的仿真
        
        参数:
            processed_path: 预处理数据路径
            fan_votes_path: 观众投票估计路径
            original_path: 原始数据路径
            
        返回:
            仿真结果DataFrame
        """
        print("=" * 60)
        print("Project Multiverse - 平行宇宙仿真系统")
        print("=" * 60)
        
        # 加载并合并数据
        print("\n1. 加载并合并数据...")
        df_merged, elimination_info = self.load_and_merge_data(
            processed_path, fan_votes_path, original_path
        )
        print(f"   合并后数据: {len(df_merged)} 行")
        
        # 仿真结果存储
        results = []
        
        # 按赛季和周次分组
        print("\n2. 开始仿真...")
        grouped = df_merged.groupby(['Season', 'Week'])
        
        for (season, week), week_data in grouped:
            # 获取真实淘汰者
            actual_eliminated = elimination_info.get((season, week), None)
            
            # 只有当有真实淘汰者时才进行仿真
            if actual_eliminated and len(week_data) >= 2:
                # 运行三个平行宇宙
                elim_rank = self.simulate_rank_model(week_data)
                elim_percent = self.simulate_percentage_model(week_data)
                elim_save = self.simulate_judges_save_model(week_data)
                
                # 计算指标
                metrics = self.calculate_metrics(
                    week_data, actual_eliminated, 
                    elim_rank, elim_percent, elim_save
                )
                
                # 保存结果
                results.append({
                    'Season': season,
                    'Week': week,
                    'Actual_Eliminated': actual_eliminated,
                    'Simulated_Elim_Rank': elim_rank,
                    'Simulated_Elim_Percent': elim_percent,
                    'Simulated_Elim_Save': elim_save,
                    'Is_Reversal': metrics['Is_Reversal'],
                    'Is_Saved': metrics['Is_Saved'],
                    'Is_Injustice': metrics['Is_Injustice'],
                    'Judge_Rank': metrics['Judge_Rank'],
                    'Fan_Rank': metrics['Fan_Rank'],
                    'Num_Contestants': len(week_data)
                })
                
                if len(results) % 50 == 0:
                    print(f"   已处理 {len(results)} 周...")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        print(f"\n3. 仿真完成！")
        print(f"   总周次数: {len(results_df)}")
        
        return results_df
    
    def generate_summary_statistics(self, results_df: pd.DataFrame) -> None:
        """
        生成汇总统计
        
        参数:
            results_df: 仿真结果DataFrame
        """
        print("\n" + "=" * 60)
        print("仿真结果统计")
        print("=" * 60)
        
        # Q3: 逆转分析
        print("\n【Q3: 赛制差异分析】")
        reversals = results_df['Is_Reversal'].sum()
        total = len(results_df)
        reversal_rate = reversals / total * 100
        print(f"  逆转次数: {reversals} / {total} ({reversal_rate:.2f}%)")
        print(f"  解释: 在 {reversal_rate:.2f}% 的情况下，排名制和百分比制会淘汰不同的人")
        
        # Q4: 拯救机制分析
        print("\n【Q4: 裁判拯救机制分析】")
        saves = results_df['Is_Saved'].sum()
        save_rate = saves / total * 100
        print(f"  拯救生效次数: {saves} / {total} ({save_rate:.2f}%)")
        print(f"  解释: 在 {save_rate:.2f}% 的情况下，裁判拯救机制改变了淘汰结果")
        
        # 冤案分析
        print("\n【冤案分析】")
        injustices = results_df['Is_Injustice'].sum()
        injustice_rate = injustices / total * 100
        print(f"  冤案次数: {injustices} / {total} ({injustice_rate:.2f}%)")
        print(f"  解释: 在 {injustice_rate:.2f}% 的情况下，被淘汰者的裁判和观众排名都不是最后")
        
        # 一致性分析
        print("\n【一致性分析】")
        # 三种模型都一致
        all_same = (
            (results_df['Simulated_Elim_Rank'] == results_df['Simulated_Elim_Percent']) &
            (results_df['Simulated_Elim_Percent'] == results_df['Simulated_Elim_Save'])
        ).sum()
        all_same_rate = all_same / total * 100
        print(f"  三种模型完全一致: {all_same} / {total} ({all_same_rate:.2f}%)")
        
        # 与真实结果的匹配度
        print("\n【与真实结果的匹配度】")
        rank_match = (results_df['Actual_Eliminated'] == results_df['Simulated_Elim_Rank']).sum()
        percent_match = (results_df['Actual_Eliminated'] == results_df['Simulated_Elim_Percent']).sum()
        save_match = (results_df['Actual_Eliminated'] == results_df['Simulated_Elim_Save']).sum()
        
        print(f"  排名制匹配: {rank_match} / {total} ({rank_match/total*100:.2f}%)")
        print(f"  百分比制匹配: {percent_match} / {total} ({percent_match/total*100:.2f}%)")
        print(f"  拯救机制匹配: {save_match} / {total} ({save_match/total*100:.2f}%)")


def main():
    """主函数"""
    # 初始化仿真器
    simulator = MultiverseSimulator()
    
    # 运行仿真
    results_df = simulator.run_simulation(
        processed_path='Processed_DWTS_Long_Format.csv',
        fan_votes_path='Q1_Estimated_Fan_Votes.csv',
        original_path='2026 MCM Problem C Data.csv'
    )
    
    # 保存结果
    output_path = 'Simulation_Results_Q3_Q4.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ 结果已保存至: {output_path}")
    
    # 生成统计报告
    simulator.generate_summary_statistics(results_df)
    
    print("\n" + "=" * 60)
    print("✓ 仿真完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
