"""
Property-Based Tests for Time Series Cross-Validation
时间序列交叉验证的属性测试
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

import sys
sys.path.insert(0, 'src')
from ml_optimization.time_series_cv import TimeSeriesCV


# 策略：生成带有Season列的数据框
@st.composite
def dataframes_with_seasons(draw, min_seasons=5, max_seasons=34, min_records_per_season=10, max_records_per_season=100):
    """生成带有Season列的数据框"""
    n_seasons = draw(st.integers(min_value=min_seasons, max_value=max_seasons))
    
    data = []
    for season in range(1, n_seasons + 1):
        n_records = draw(st.integers(min_value=min_records_per_season, max_value=max_records_per_season))
        for _ in range(n_records):
            data.append({
                'Season': season,
                'Week': draw(st.integers(min_value=1, max_value=11)),
                'feature1': draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)),
                'feature2': draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
            })
    
    return pd.DataFrame(data)


@given(df=dataframes_with_seasons(), n_splits=st.integers(min_value=3, max_value=10))
@settings(max_examples=100, deadline=None)
def test_property_3_temporal_ordering(df, n_splits):
    """
    Feature: ml-model-optimization, Property 3: Temporal Ordering in Cross-Validation
    
    For any cross-validation split, all training data indices must correspond 
    to time points that precede all validation data indices.
    
    验证：对于任何交叉验证分割，所有训练数据索引必须对应于在验证数据索引之前的时间点
    """
    # 如果季数不足，跳过
    n_seasons = df['Season'].nunique()
    if n_seasons < n_splits + 1:
        return
    
    # 创建时间序列交叉验证器
    cv = TimeSeriesCV(n_splits=n_splits, strategy='expanding')
    
    # 检查每个分割
    for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
        # 获取训练集和验证集的季数
        train_seasons = df.loc[train_idx, 'Season'].values
        val_seasons = df.loc[val_idx, 'Season'].values
        
        # 属性：最大训练季 <= 最小验证季
        max_train_season = train_seasons.max()
        min_val_season = val_seasons.min()
        
        assert max_train_season <= min_val_season, (
            f"Fold {fold + 1}: Temporal ordering violated! "
            f"max_train_season={max_train_season}, min_val_season={min_val_season}"
        )
        
        # 额外检查：训练集和验证集不应有重叠
        assert len(set(train_idx) & set(val_idx)) == 0, (
            f"Fold {fold + 1}: Train and validation sets overlap!"
        )


@given(df=dataframes_with_seasons(), n_splits=st.integers(min_value=3, max_value=10))
@settings(max_examples=100, deadline=None)
def test_property_9_expanding_window(df, n_splits):
    """
    Feature: ml-model-optimization, Property 9: Expanding Window Strategy
    
    For any time-series CV with expanding window strategy, the training set size 
    must monotonically increase with each fold while maintaining temporal ordering.
    
    验证：对于扩展窗口策略，训练集大小必须随着每折单调递增
    """
    # 如果季数不足，跳过
    n_seasons = df['Season'].nunique()
    if n_seasons < n_splits + 1:
        return
    
    # 创建时间序列交叉验证器（扩展窗口）
    cv = TimeSeriesCV(n_splits=n_splits, strategy='expanding')
    
    # 收集每折的训练集大小
    train_sizes = []
    
    for train_idx, val_idx in cv.split(df):
        train_sizes.append(len(train_idx))
    
    # 属性：训练集大小单调递增
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] >= train_sizes[i-1], (
            f"Training set size decreased from fold {i} to {i+1}: "
            f"{train_sizes[i-1]} -> {train_sizes[i]}"
        )


@given(df=dataframes_with_seasons(), n_splits=st.integers(min_value=3, max_value=10))
@settings(max_examples=100, deadline=None)
def test_property_10_metrics_aggregation(df, n_splits):
    """
    Feature: ml-model-optimization, Property 10: Cross-Validation Metrics Aggregation
    
    For any cross-validation run, the system must compute and report both the mean 
    and standard deviation of performance metrics across all folds.
    
    验证：对于任何交叉验证运行，系统必须计算并报告所有折的均值和标准差
    """
    # 如果季数不足，跳过
    n_seasons = df['Season'].nunique()
    if n_seasons < n_splits + 1:
        return
    
    # 创建时间序列交叉验证器
    cv = TimeSeriesCV(n_splits=n_splits, strategy='expanding')
    
    # 生成随机分数（模拟交叉验证结果）
    scores = np.random.uniform(0.5, 0.9, size=n_splits)
    
    # 获取指标汇总
    summary = cv.get_metrics_summary(scores)
    
    # 属性：汇总必须包含mean和std
    assert 'mean' in summary, "Summary must contain 'mean'"
    assert 'std' in summary, "Summary must contain 'std'"
    
    # 属性：mean和std必须正确计算
    expected_mean = scores.mean()
    expected_std = scores.std()
    
    assert abs(summary['mean'] - expected_mean) < 1e-6, (
        f"Mean calculation incorrect: expected {expected_mean}, got {summary['mean']}"
    )
    
    assert abs(summary['std'] - expected_std) < 1e-6, (
        f"Std calculation incorrect: expected {expected_std}, got {summary['std']}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
