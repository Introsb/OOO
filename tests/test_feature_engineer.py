"""单元测试：FeatureEngineer边缘情况"""

import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing_pipeline import FeatureEngineer


def test_judge_avg_score_with_3_judges():
    """测试3个裁判的情况"""
    data = {
        'Season': [1],
        'Week': [1],
        'Name': ['Test'],
        'Age': [30],
        'Industry': ['Actor'],
        'Placement': [1],
        'judge1': [7.0],
        'judge2': [8.0],
        'judge3': [9.0]
    }
    
    df = pd.DataFrame(data)
    engineer = FeatureEngineer()
    df = engineer.add_judge_avg_score(df)
    
    expected_avg = (7.0 + 8.0 + 9.0) / 3
    assert abs(df.iloc[0]['Judge_Avg_Score'] - expected_avg) < 0.0001


def test_judge_avg_score_with_4_judges():
    """测试4个裁判的情况"""
    data = {
        'Season': [1],
        'Week': [1],
        'Name': ['Test'],
        'Age': [30],
        'Industry': ['Actor'],
        'Placement': [1],
        'judge1': [7.0],
        'judge2': [8.0],
        'judge3': [9.0],
        'judge4': [8.5]
    }
    
    df = pd.DataFrame(data)
    engineer = FeatureEngineer()
    df = engineer.add_judge_avg_score(df)
    
    expected_avg = (7.0 + 8.0 + 9.0 + 8.5) / 4
    assert abs(df.iloc[0]['Judge_Avg_Score'] - expected_avg) < 0.0001


def test_score_scaled_with_zero_std():
    """测试标准差为0的情况"""
    data = {
        'Season': [1, 2, 3],
        'Week': [1, 1, 1],
        'Name': ['A', 'B', 'C'],
        'Age': [30, 30, 30],
        'Industry': ['Actor', 'Actor', 'Actor'],
        'Placement': [1, 2, 3],
        'Judge_Avg_Score': [7.0, 7.0, 7.0]  # 所有分数相同
    }
    
    df = pd.DataFrame(data)
    engineer = FeatureEngineer()
    
    with pytest.warns(UserWarning, match="Standard deviation is 0"):
        df = engineer.add_score_scaled(df)
    
    # 所有Score_Scaled应该为0
    assert all(df['Score_Scaled'] == 0.0)
