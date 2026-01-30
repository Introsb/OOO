"""基于属性的测试（Property-Based Tests）"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from hypothesis import given, strategies as st, settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing_pipeline import DataLoader

# 配置Hypothesis
settings.register_profile("dwts", max_examples=100, deadline=None)
settings.load_profile("dwts")


@given(
    st.integers(min_value=1, max_value=10),  # 选手数量
    st.integers(min_value=1, max_value=11)   # 周次数量
)
@settings(max_examples=100, deadline=None)
def test_property_1_wide_to_long_row_expansion(num_contestants, num_weeks):
    """
    Feature: dwts-data-preprocessing, Property 1: 宽表到长表转换的行数扩展
    
    对于任意宽表数据框，如果一个选手有N个周次的数据（week1到weekN），
    那么转换后的长表应该为该选手创建N行数据。
    
    Validates: Requirements 2.1, 2.3
    """
    from src.preprocessing_pipeline import WideToLongTransformer
    
    # 构建宽表数据
    data = {
        'season': list(range(1, num_contestants + 1)),
        'celebrity_name': [f'Contestant{i}' for i in range(num_contestants)],
        'celebrity_age_during_season': [30] * num_contestants,
        'celebrity_industry': ['Actor'] * num_contestants,
        'placement': list(range(1, num_contestants + 1))
    }
    
    # 添加week列
    for week in range(1, num_weeks + 1):
        for judge in range(1, 5):
            data[f'week{week}_judge{judge}_score'] = [7.0] * num_contestants
    
    df_wide = pd.DataFrame(data)
    
    # 转换
    transformer = WideToLongTransformer()
    df_long = transformer.transform(df_wide)
    
    # 验证：输出行数应该等于选手数 × 周次数
    expected_rows = num_contestants * num_weeks
    assert len(df_long) == expected_rows, f"Expected {expected_rows} rows, got {len(df_long)}"
    
    # 验证每个选手都有正确的周次数
    for contestant_idx in range(num_contestants):
        contestant_name = f'Contestant{contestant_idx}'
        contestant_rows = df_long[df_long['Name'] == contestant_name]
        assert len(contestant_rows) == num_weeks, f"Contestant {contestant_name} should have {num_weeks} rows"


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=34),  # season
            st.text(alphabet=st.characters(blacklist_categories=('Cs', 'Cc')), min_size=1, max_size=20),  # name
            st.integers(min_value=18, max_value=80),  # age
            st.text(alphabet=st.characters(blacklist_categories=('Cs', 'Cc')), min_size=1, max_size=20),  # industry
            st.integers(min_value=1, max_value=15)  # placement
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100, deadline=None)
def test_property_11_data_loading_fidelity(data_rows):
    """
    Feature: dwts-data-preprocessing, Property 11: 数据加载保真性
    
    对于任意有效的CSV文件内容，加载后的数据框应该保留所有原始列的数据类型和值，
    不应有数据丢失或类型转换错误。
    
    Validates: Requirements 1.4
    """
    # 生成CSV内容
    csv_lines = ["season,celebrity_name,celebrity_age_during_season,celebrity_industry,placement,week1_judge1_score"]
    
    for season, name, age, industry, placement in data_rows:
        # 清理文本以避免CSV格式问题
        name = name.replace(',', '_').replace('\n', ' ').replace('\r', ' ').replace('"', "'").strip()
        industry = industry.replace(',', '_').replace('\n', ' ').replace('\r', ' ').replace('"', "'").strip()
        if not name:
            name = "Unknown"
        if not industry:
            industry = "Unknown"
        csv_lines.append(f"{season},{name},{age},{industry},{placement},7.0")
    
    csv_content = '\n'.join(csv_lines)
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # 加载数据
        loader = DataLoader()
        df = loader.load(temp_path)
        
        # 验证数据保真性
        assert len(df) == len(data_rows), "行数应该匹配"
        assert 'season' in df.columns
        assert 'celebrity_name' in df.columns
        assert 'celebrity_age_during_season' in df.columns
        assert 'celebrity_industry' in df.columns
        assert 'placement' in df.columns
        
        # 验证数据值保持一致
        for i, (season, name, age, industry, placement) in enumerate(data_rows):
            name = name.replace(',', '_').replace('\n', ' ').replace('\r', ' ').replace('"', "'").strip()
            industry = industry.replace(',', '_').replace('\n', ' ').replace('\r', ' ').replace('"', "'").strip()
            if not name:
                name = "Unknown"
            if not industry:
                industry = "Unknown"
            
            assert df.iloc[i]['season'] == season
            assert df.iloc[i]['celebrity_age_during_season'] == age
            assert df.iloc[i]['placement'] == placement
    
    finally:
        os.unlink(temp_path)


@given(st.integers(min_value=1, max_value=11))
@settings(max_examples=100, deadline=None)
def test_property_3_week_identifier_conversion(week_num):
    """
    Feature: dwts-data-preprocessing, Property 3: Week标识转换正确性
    
    对于任意week标识字符串（如"week1", "week5", "week11"），
    转换后的数字应该等于标识中的数字部分（1, 5, 11）。
    
    Validates: Requirements 2.5
    """
    from src.preprocessing_pipeline import WideToLongTransformer
    
    # 构建包含特定week的宽表数据
    data = {
        'season': [1],
        'celebrity_name': ['Test'],
        'celebrity_age_during_season': [30],
        'celebrity_industry': ['Actor'],
        'placement': [1]
    }
    
    # 只添加指定的week
    for judge in range(1, 5):
        data[f'week{week_num}_judge{judge}_score'] = [7.0]
    
    df_wide = pd.DataFrame(data)
    
    # 转换
    transformer = WideToLongTransformer()
    df_long = transformer.transform(df_wide)
    
    # 验证：Week列的值应该等于week_num
    assert len(df_long) == 1, "Should have exactly 1 row"
    assert df_long.iloc[0]['Week'] == week_num, f"Week should be {week_num}, got {df_long.iloc[0]['Week']}"


@given(
    st.one_of(
        st.floats(min_value=0.1, max_value=10.0),  # 有效分数
        st.just(0),  # 无效：0
        st.just(np.nan),  # 无效：NaN
        st.just('N/A'),  # 无效：N/A字符串
        st.just(''),  # 无效：空字符串
        st.just(None)  # 无效：None
    )
)
@settings(max_examples=100, deadline=None)
def test_property_4_invalid_score_identification(score):
    """
    Feature: dwts-data-preprocessing, Property 4: 无效分数识别
    
    对于任意分数值，如果它是N/A、0、空值或NaN，那么is_valid_score()函数应返回False；
    否则应返回True。
    
    Validates: Requirements 3.1
    """
    from src.preprocessing_pipeline import DataCleaner
    
    cleaner = DataCleaner()
    result = cleaner.is_valid_score(score)
    
    # 判断预期结果
    if pd.isna(score) or score == 0 or score == 'N/A' or score == '' or score is None:
        assert result is False, f"Score {score} should be invalid"
    else:
        assert result is True, f"Score {score} should be valid"


@given(
    st.integers(min_value=1, max_value=11),  # 淘汰周次
    st.integers(min_value=1, max_value=11)   # 总周次
)
@settings(max_examples=100, deadline=None)
def test_property_5_elimination_data_removal(elimination_week, total_weeks):
    """
    Feature: dwts-data-preprocessing, Property 5: 淘汰后数据清除
    
    对于任意选手的时序数据，如果第K周是第一个包含无效分数的周次，
    那么清洗后的数据应该只包含该选手第1周到第K-1周的数据，
    第K周及之后的所有数据应被删除。
    
    Validates: Requirements 3.2, 3.3
    """
    from src.preprocessing_pipeline import DataCleaner
    
    if elimination_week > total_weeks:
        elimination_week = total_weeks
    
    # 构建测试数据：选手在elimination_week被淘汰
    data = []
    for week in range(1, total_weeks + 1):
        row = {
            'Season': 1,
            'Week': week,
            'Name': 'TestContestant',
            'Age': 30,
            'Industry': 'Actor',
            'Placement': 5
        }
        
        # 在淘汰周及之后设置无效分数
        if week >= elimination_week:
            row['judge1'] = 0  # 无效分数
            row['judge2'] = 0
            row['judge3'] = 0
        else:
            row['judge1'] = 7.0  # 有效分数
            row['judge2'] = 8.0
            row['judge3'] = 7.5
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 清洗数据
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df)
    
    # 验证：应该只保留第1周到第(elimination_week-1)周的数据
    expected_weeks = elimination_week - 1
    assert len(df_cleaned) == expected_weeks, f"Should have {expected_weeks} weeks, got {len(df_cleaned)}"
    
    if expected_weeks > 0:
        assert df_cleaned['Week'].min() == 1
        assert df_cleaned['Week'].max() == expected_weeks


@given(st.integers(min_value=1, max_value=11))
@settings(max_examples=100, deadline=None)
def test_property_6_time_continuity(num_valid_weeks):
    """
    Feature: dwts-data-preprocessing, Property 6: 时间连续性保持
    
    对于任意选手在清洗后的数据中，该选手的Week列应该是连续的整数序列
    （如1,2,3,4或1,2,3），不应有跳跃（如1,3,5）。
    
    Validates: Requirements 3.4
    """
    from src.preprocessing_pipeline import DataCleaner
    
    # 构建测试数据：选手有num_valid_weeks周的有效数据
    data = []
    for week in range(1, num_valid_weeks + 1):
        data.append({
            'Season': 1,
            'Week': week,
            'Name': 'TestContestant',
            'Age': 30,
            'Industry': 'Actor',
            'Placement': 5,
            'judge1': 7.0,
            'judge2': 8.0,
            'judge3': 7.5
        })
    
    # 添加一些无效周次
    for week in range(num_valid_weeks + 1, 12):
        data.append({
            'Season': 1,
            'Week': week,
            'Name': 'TestContestant',
            'Age': 30,
            'Industry': 'Actor',
            'Placement': 5,
            'judge1': 0,
            'judge2': 0,
            'judge3': 0
        })
    
    df = pd.DataFrame(data)
    
    # 清洗数据
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df)
    
    if len(df_cleaned) > 0:
        weeks = sorted(df_cleaned['Week'].tolist())
        
        # 验证连续性：每个week应该比前一个week大1
        for i in range(1, len(weeks)):
            assert weeks[i] == weeks[i-1] + 1, f"Weeks should be continuous, got {weeks}"
        
        # 验证从1开始
        assert weeks[0] == 1, "Weeks should start from 1"


@given(st.lists(st.floats(min_value=0, max_value=10, allow_nan=False), min_size=3, max_size=4))
@settings(max_examples=100, deadline=None)
def test_property_7_judge_avg_score_calculation(judge_scores):
    """
    Feature: dwts-data-preprocessing, Property 7: 裁判平均分计算正确性
    
    对于任意一行数据的裁判分数集合，Judge_Avg_Score应该等于所有有效裁判分数的
    算术平均值（sum / count），自动排除无效值。
    
    Validates: Requirements 4.1, 4.4
    """
    from src.preprocessing_pipeline import FeatureEngineer
    
    # 构建测试数据
    data = {
        'Season': [1],
        'Week': [1],
        'Name': ['Test'],
        'Age': [30],
        'Industry': ['Actor'],
        'Placement': [1]
    }
    
    # 添加裁判分数
    for i, score in enumerate(judge_scores, 1):
        data[f'judge{i}'] = [score]
    
    df = pd.DataFrame(data)
    
    # 计算平均分
    engineer = FeatureEngineer()
    df = engineer.add_judge_avg_score(df)
    
    # 验证：Judge_Avg_Score应该等于有效分数的平均值
    expected_avg = sum(judge_scores) / len(judge_scores)
    actual_avg = df.iloc[0]['Judge_Avg_Score']
    
    assert abs(actual_avg - expected_avg) < 0.0001, f"Expected {expected_avg}, got {actual_avg}"


@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10))
@settings(max_examples=100, deadline=None)
def test_property_8_industry_encoding_bijection(industries):
    """
    Feature: dwts-data-preprocessing, Property 8: 行业编码的双射性
    
    对于任意两个行业文本值，如果它们相同，则它们的Industry_Code必须相同；
    如果它们不同，则它们的Industry_Code必须不同（一对一映射）。
    
    Validates: Requirements 5.1, 5.2
    """
    from src.preprocessing_pipeline import FeatureEngineer
    
    # 构建测试数据
    data = {
        'Season': list(range(1, len(industries) + 1)),
        'Week': [1] * len(industries),
        'Name': [f'Contestant{i}' for i in range(len(industries))],
        'Age': [30] * len(industries),
        'Industry': industries,
        'Placement': list(range(1, len(industries) + 1)),
        'judge1': [7.0] * len(industries)
    }
    
    df = pd.DataFrame(data)
    
    # 添加行业编码
    engineer = FeatureEngineer()
    df = engineer.add_industry_code(df)
    
    # 验证双射性
    industry_to_code = {}
    code_to_industry = {}
    
    for idx, row in df.iterrows():
        industry = row['Industry']
        code = row['Industry_Code']
        
        # 相同行业应该有相同编码
        if industry in industry_to_code:
            assert industry_to_code[industry] == code, f"Same industry {industry} has different codes"
        else:
            industry_to_code[industry] = code
        
        # 相同编码应该对应相同行业
        if code in code_to_industry:
            assert code_to_industry[code] == industry, f"Same code {code} maps to different industries"
        else:
            code_to_industry[code] = industry


@given(st.lists(st.floats(min_value=1.0, max_value=10.0, allow_nan=False), min_size=5, max_size=20))
@settings(max_examples=100, deadline=None)
def test_property_9_zscore_standardization(scores):
    """
    Feature: dwts-data-preprocessing, Property 9: Z-score标准化公式正确性
    
    对于任意Judge_Avg_Score值x，其对应的Score_Scaled值应该等于(x - μ) / σ，
    其中μ是所有Judge_Avg_Score的全局均值，σ是全局标准差。
    
    Validates: Requirements 6.1, 6.2
    """
    from src.preprocessing_pipeline import FeatureEngineer
    
    # 构建测试数据
    data = {
        'Season': list(range(1, len(scores) + 1)),
        'Week': [1] * len(scores),
        'Name': [f'Contestant{i}' for i in range(len(scores))],
        'Age': [30] * len(scores),
        'Industry': ['Actor'] * len(scores),
        'Placement': list(range(1, len(scores) + 1)),
        'Judge_Avg_Score': scores
    }
    
    df = pd.DataFrame(data)
    
    # 计算标准化分数
    engineer = FeatureEngineer()
    df = engineer.add_score_scaled(df)
    
    # 计算预期的均值和标准差
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)  # pandas uses ddof=1 by default
    
    # 验证每个Score_Scaled值
    for idx, row in df.iterrows():
        original_score = row['Judge_Avg_Score']
        scaled_score = row['Score_Scaled']
        
        if std_score == 0:
            # 标准差为0时，所有scaled值应该为0
            assert scaled_score == 0.0
        else:
            expected_scaled = (original_score - mean_score) / std_score
            assert abs(scaled_score - expected_scaled) < 0.0001, \
                f"Expected {expected_scaled}, got {scaled_score}"


@given(st.integers(min_value=5, max_value=34))
@settings(max_examples=100, deadline=None)
def test_property_10_timeseries_split_season_integrity(num_seasons):
    """
    Feature: dwts-data-preprocessing, Property 10: 时序数据集划分的赛季完整性
    
    对于任意处理后的数据框，按Season排序后，训练集应包含前80%赛季的所有行，
    测试集应包含后20%赛季的所有行，且训练集和测试集的赛季集合应该不相交（无重叠）。
    
    Validates: Requirements 7.2, 7.3, 7.4, 7.6
    """
    from src.preprocessing_pipeline import TimeSeriesSplitter
    
    # 构建多赛季数据
    data = []
    for season in range(1, num_seasons + 1):
        for week in range(1, 4):  # 每个赛季3周
            data.append({
                'Season': season,
                'Week': week,
                'Name': f'Contestant{season}',
                'Age': 30,
                'Industry': 'Actor',
                'Placement': 1,
                'Judge_Avg_Score': 7.0
            })
    
    df = pd.DataFrame(data)
    
    # 划分数据集
    splitter = TimeSeriesSplitter()
    train_df, test_df = splitter.split(df, train_ratio=0.8)
    
    # 获取训练集和测试集的赛季
    train_seasons = set(train_df['Season'].unique())
    test_seasons = set(test_df['Season'].unique())
    
    # 验证：训练集和测试集的赛季不相交
    assert len(train_seasons & test_seasons) == 0, "Train and test seasons should not overlap"
    
    # 验证：训练集包含前80%赛季
    all_seasons = sorted(df['Season'].unique())
    split_point = int(len(all_seasons) * 0.8)
    expected_train_seasons = set(all_seasons[:split_point])
    expected_test_seasons = set(all_seasons[split_point:])
    
    assert train_seasons == expected_train_seasons, "Train seasons mismatch"
    assert test_seasons == expected_test_seasons, "Test seasons mismatch"
    
    # 验证：训练集和测试集包含所有数据
    assert len(train_df) + len(test_df) == len(df), "Total rows should match"


@given(st.integers(min_value=5, max_value=20))
@settings(max_examples=100, deadline=None)
def test_property_12_data_order_preservation(num_seasons):
    """
    Feature: dwts-data-preprocessing, Property 12: 数据顺序保持
    
    对于任意数据框，在按Season排序后，训练集和测试集内部的行顺序应该与
    原始排序后的顺序一致，不应被随机打乱。
    
    Validates: Requirements 7.1, 7.5
    """
    from src.preprocessing_pipeline import TimeSeriesSplitter
    
    # 构建数据（故意不按顺序）
    data = []
    for season in range(num_seasons, 0, -1):  # 倒序
        for week in range(3, 0, -1):  # 倒序
            data.append({
                'Season': season,
                'Week': week,
                'Name': f'Contestant{season}_{week}',
                'Age': 30,
                'Industry': 'Actor',
                'Placement': 1,
                'Judge_Avg_Score': 7.0
            })
    
    df = pd.DataFrame(data)
    
    # 先按Season排序（这是预期的顺序）
    df_sorted = df.sort_values('Season').reset_index(drop=True)
    
    # 划分数据集
    splitter = TimeSeriesSplitter()
    train_df, test_df = splitter.split(df, train_ratio=0.8)
    
    # 验证训练集的顺序
    if len(train_df) > 1:
        for i in range(1, len(train_df)):
            # Season应该是非递减的
            assert train_df.iloc[i]['Season'] >= train_df.iloc[i-1]['Season'], \
                "Train data should be sorted by Season"
    
    # 验证测试集的顺序
    if len(test_df) > 1:
        for i in range(1, len(test_df)):
            # Season应该是非递减的
            assert test_df.iloc[i]['Season'] >= test_df.iloc[i-1]['Season'], \
                "Test data should be sorted by Season"
    
    # 验证训练集的所有Season都小于等于测试集的最小Season
    if len(train_df) > 0 and len(test_df) > 0:
        max_train_season = train_df['Season'].max()
        min_test_season = test_df['Season'].min()
        assert max_train_season < min_test_season, \
            "All train seasons should be before test seasons"


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100, deadline=None)
def test_property_2_output_data_structure_integrity(num_rows):
    """
    Feature: dwts-data-preprocessing, Property 2: 输出数据结构完整性
    
    对于任意处理后的数据框，它必须包含所有必需的列：Season, Week, Name, Age, 
    Industry, Industry_Code, Judge_Avg_Score, Score_Scaled, Placement，
    并且原始Industry列应被保留。
    
    Validates: Requirements 2.2, 4.5, 5.3, 5.4, 6.3, 8.2
    """
    from src.preprocessing_pipeline import FeatureEngineer
    
    # 构建测试数据
    data = {
        'Season': list(range(1, num_rows + 1)),
        'Week': [1] * num_rows,
        'Name': [f'Contestant{i}' for i in range(num_rows)],
        'Age': [30] * num_rows,
        'Industry': ['Actor'] * num_rows,
        'Placement': list(range(1, num_rows + 1)),
        'judge1': [7.0] * num_rows,
        'judge2': [8.0] * num_rows,
        'judge3': [7.5] * num_rows
    }
    
    df = pd.DataFrame(data)
    
    # 应用特征工程
    engineer = FeatureEngineer()
    df = engineer.add_judge_avg_score(df)
    df = engineer.add_industry_code(df)
    df = engineer.add_score_scaled(df)
    
    # 验证所有必需列存在
    required_cols = ['Season', 'Week', 'Name', 'Age', 'Industry', 'Industry_Code', 
                    'Judge_Avg_Score', 'Score_Scaled', 'Placement']
    
    for col in required_cols:
        assert col in df.columns, f"Required column {col} is missing"
    
    # 验证原始Industry列被保留
    assert 'Industry' in df.columns, "Original Industry column should be preserved"
