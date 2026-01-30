"""
DWTS Data Preprocessing Pipeline
将"与星共舞"比赛数据从宽表格式转换为长表格式，并执行数据清洗和特征工程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any
import warnings


class DataLoader:
    """负责读取和验证输入文件"""
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        加载CSV文件
        
        参数:
            filepath: CSV文件路径
            
        返回:
            pd.DataFrame: 原始数据框
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式无效
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Input file '{filepath}' not found.")
        except Exception as e:
            raise ValueError(f"Error: Invalid CSV format in '{filepath}'. {str(e)}")
        
        # 验证必需列
        required_base_cols = ['season', 'celebrity_name', 'celebrity_age_during_season', 
                             'celebrity_industry', 'placement']
        
        # 检查基础列
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        
        # 检查week列（至少应该有week1的列）
        week_cols = [col for col in df.columns if col.startswith('week1_')]
        if not week_cols:
            missing_cols.append('week1-week11 columns')
        
        if missing_cols:
            raise ValueError(f"Error: Missing required columns: {missing_cols}")
        
        return df


class WideToLongTransformer:
    """将宽表格式转换为长表格式"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将宽表转换为长表
        
        参数:
            df: 宽格式数据框
            
        返回:
            pd.DataFrame: 长格式数据框
        """
        # 重命名列以匹配标准格式
        df_renamed = df.rename(columns={
            'celebrity_name': 'Name',
            'celebrity_age_during_season': 'Age',
            'celebrity_industry': 'Industry',
            'season': 'Season',
            'placement': 'Placement'
        })
        
        # 识别所有week列
        week_cols = {}
        for col in df_renamed.columns:
            if col.startswith('week') and '_judge' in col:
                # 提取week数字和judge数字
                parts = col.split('_')
                week_num = int(parts[0].replace('week', ''))
                judge_num = int(parts[1].replace('judge', ''))
                
                if week_num not in week_cols:
                    week_cols[week_num] = []
                week_cols[week_num].append(col)
        
        # 构建长表数据
        long_data = []
        
        for idx, row in df_renamed.iterrows():
            for week_num in sorted(week_cols.keys()):
                # 提取该周的所有裁判分数
                judge_scores = []
                for judge_col in sorted(week_cols[week_num]):
                    score = row[judge_col]
                    judge_scores.append(score)
                
                # 创建新行
                new_row = {
                    'Season': row['Season'],
                    'Week': week_num,
                    'Name': row['Name'],
                    'Age': row['Age'],
                    'Industry': row['Industry'],
                    'Placement': row['Placement']
                }
                
                # 添加裁判分数列
                for i, score in enumerate(judge_scores, 1):
                    new_row[f'judge{i}'] = score
                
                long_data.append(new_row)
        
        return pd.DataFrame(long_data)


class DataCleaner:
    """删除选手被淘汰后的无效数据"""
    
    def is_valid_score(self, score: Any) -> bool:
        """
        判断分数是否有效
        
        参数:
            score: 分数值
            
        返回:
            bool: True表示有效，False表示无效
        """
        # 检查是否为NaN
        if pd.isna(score):
            return False
        
        # 检查是否为0
        if score == 0:
            return False
        
        # 检查是否为空字符串
        if isinstance(score, str) and score.strip() == '':
            return False
        
        # 检查是否为'N/A'字符串
        if isinstance(score, str) and score.strip().upper() == 'N/A':
            return False
        
        return True
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据，删除无效记录
        
        参数:
            df: 长格式数据框
            
        返回:
            pd.DataFrame: 清洗后的数据框
        """
        # 找到所有judge列
        judge_cols = [col for col in df.columns if col.startswith('judge')]
        
        if not judge_cols:
            return df
        
        # 为每行检查是否有至少一个有效分数
        def has_valid_score(row):
            for col in judge_cols:
                if self.is_valid_score(row[col]):
                    return True
            return False
        
        # 按选手和赛季分组处理
        cleaned_rows = []
        
        for (season, name), group in df.groupby(['Season', 'Name'], sort=False):
            # 按Week排序
            group = group.sort_values('Week')
            
            # 找到第一个无效周次
            valid_weeks = []
            for idx, row in group.iterrows():
                if has_valid_score(row):
                    valid_weeks.append(idx)
                else:
                    # 遇到第一个无效周次，停止
                    break
            
            # 只保留有效周次的数据
            if valid_weeks:
                cleaned_rows.extend(valid_weeks)
        
        # 返回清洗后的数据框
        if cleaned_rows:
            return df.loc[cleaned_rows].reset_index(drop=True)
        else:
            return pd.DataFrame(columns=df.columns)


class FeatureEngineer:
    """计算派生特征"""
    
    def add_judge_avg_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算裁判平均分"""
        # 找到所有judge列
        judge_cols = [col for col in df.columns if col.startswith('judge')]
        
        if not judge_cols:
            raise ValueError("No judge columns found in dataframe")
        
        # 计算每行的平均分（自动忽略NaN）
        df['Judge_Avg_Score'] = df[judge_cols].mean(axis=1, skipna=True)
        
        return df
    
    def add_industry_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加行业编码"""
        # 使用pd.Categorical进行编码
        df['Industry_Code'] = pd.Categorical(df['Industry']).codes
        
        return df
    
    def add_score_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加标准化分数"""
        if 'Judge_Avg_Score' not in df.columns:
            raise ValueError("Judge_Avg_Score column must exist before scaling")
        
        # 计算全局均值和标准差
        mean_score = df['Judge_Avg_Score'].mean()
        std_score = df['Judge_Avg_Score'].std()
        
        # 处理标准差为0的边缘情况
        if std_score == 0 or pd.isna(std_score):
            warnings.warn("Warning: Standard deviation is 0, all scores are identical. Setting Score_Scaled to 0.")
            df['Score_Scaled'] = 0.0
        else:
            # 应用Z-score公式
            df['Score_Scaled'] = (df['Judge_Avg_Score'] - mean_score) / std_score
        
        return df


class TimeSeriesSplitter:
    """按时间顺序划分训练集和测试集"""
    
    def split(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据集
        
        参数:
            df: 完整数据框
            train_ratio: 训练集比例（默认0.8）
            
        返回:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
        """
        # 按Season排序
        df_sorted = df.sort_values('Season').reset_index(drop=True)
        
        # 获取唯一赛季并排序
        unique_seasons = sorted(df_sorted['Season'].unique())
        
        # 计算划分点
        split_point = int(len(unique_seasons) * train_ratio)
        
        # 划分赛季
        train_seasons = unique_seasons[:split_point]
        test_seasons = unique_seasons[split_point:]
        
        # 分配数据
        train_df = df_sorted[df_sorted['Season'].isin(train_seasons)].reset_index(drop=True)
        test_df = df_sorted[df_sorted['Season'].isin(test_seasons)].reset_index(drop=True)
        
        return train_df, test_df


class DataValidator:
    """验证输出并生成可视化"""
    
    def validate_and_report(self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """验证数据并生成报告"""
        print("\n=== 数据处理完成 ===")
        print(f"最终数据形状: {df.shape}")
        print(f"训练集行数: {len(train_df)}")
        print(f"测试集行数: {len(test_df)}")
        
        # 验证必需列
        required_cols = ['Season', 'Week', 'Name', 'Age', 'Industry', 'Industry_Code', 
                        'Judge_Avg_Score', 'Score_Scaled', 'Placement']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告：缺少必需列: {missing_cols}")
        else:
            print("✓ 所有必需列都存在")
    
    def plot_score_distribution(self, df: pd.DataFrame, output_path: str) -> None:
        """绘制分数分布直方图"""
        if 'Judge_Avg_Score' not in df.columns:
            print("警告：无法绘制直方图，缺少Judge_Avg_Score列")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(df['Judge_Avg_Score'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Judge Average Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Judge Average Scores')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 直方图已保存至: {output_path}")


class PreprocessingPipeline:
    """协调所有组件，执行完整的预处理流程"""
    
    def __init__(self):
        """初始化所有组件"""
        self.loader = DataLoader()
        self.transformer = WideToLongTransformer()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = TimeSeriesSplitter()
        self.validator = DataValidator()
    
    def run(self, input_path: str, output_path: str) -> None:
        """
        执行完整的预处理流程
        
        参数:
            input_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
        """
        try:
            print("开始数据预处理...")
            
            # 1. 加载数据
            print("1. 加载数据...")
            df = self.loader.load(input_path)
            print(f"   加载了 {len(df)} 行数据")
            
            # 2. 宽表转长表
            print("2. 宽表转长表...")
            df_long = self.transformer.transform(df)
            print(f"   转换后: {len(df_long)} 行")
            
            # 3. 数据清洗
            print("3. 数据清洗...")
            df_clean = self.cleaner.clean(df_long)
            print(f"   清洗后: {len(df_clean)} 行")
            
            if len(df_clean) == 0:
                raise ValueError("Error: No valid data remaining after cleaning")
            
            # 4. 特征工程
            print("4. 特征工程...")
            df_clean = self.engineer.add_judge_avg_score(df_clean)
            df_clean = self.engineer.add_industry_code(df_clean)
            df_clean = self.engineer.add_score_scaled(df_clean)
            print("   ✓ Judge_Avg_Score, Industry_Code, Score_Scaled 已添加")
            
            # 5. 数据集划分
            print("5. 数据集划分...")
            train_df, test_df = self.splitter.split(df_clean)
            print(f"   训练集: {len(train_df)} 行")
            print(f"   测试集: {len(test_df)} 行")
            
            # 6. 保存输出
            print(f"6. 保存输出到 {output_path}...")
            # 选择输出列
            output_cols = ['Season', 'Week', 'Name', 'Age', 'Industry_Code', 
                          'Judge_Avg_Score', 'Score_Scaled', 'Placement']
            df_output = df_clean[output_cols]
            df_output.to_csv(output_path, index=False, encoding='utf-8')
            print(f"   ✓ 数据已保存")
            
            # 7. 验证和可视化
            print("7. 验证和可视化...")
            self.validator.validate_and_report(df_clean, train_df, test_df)
            self.validator.plot_score_distribution(df_clean, 'judge_score_distribution.png')
            
            print("\n✓ 数据预处理完成！")
            
        except FileNotFoundError as e:
            print(f"\n错误: {e}")
            raise
        except ValueError as e:
            print(f"\n错误: {e}")
            raise
        except Exception as e:
            print(f"\n未预期的错误: {e}")
            raise
