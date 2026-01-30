"""
Property-Based Tests for Validation Framework
验证框架的属性测试
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st, assume

import sys
sys.path.insert(0, 'src')
from ml_optimization.validation_framework import ValidationFramework


# 策略：生成特征数据框和目标变量
@st.composite
def features_and_target(draw, n_samples=100, n_features=10):
    """生成特征数据框和目标变量"""
    data = {}
    
    # 添加必需的列
    data['Season'] = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=n_samples, max_size=n_samples))
    data['Week'] = draw(st.lists(st.integers(min_value=1, max_value=11), min_size=n_samples, max_size=n_samples))
    
    # 添加特征
    for i in range(n_features):
        data[f'feature_{i}'] = draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_samples, max_size=n_samples
        ))
    
    X = pd.DataFrame(data)
    
    # 生成目标变量
    y = pd.Series(
        draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_samples, max_size=n_samples
        )),
        name='target'
    )
    
    return X, y


@given(data=features_and_target(), threshold=st.floats(min_value=0.5, max_value=0.95))
@settings(max_examples=100, deadline=None)
def test_property_2_correlation_threshold(data, threshold):
    """
    Feature: ml-model-optimization, Property 2: Correlation Threshold Enforcement
    
    For any newly created feature, if its absolute correlation with the target 
    exceeds the threshold, the system must reject that feature and log a warning.
    
    验证：对于任何新创建的特征，如果其与目标的绝对相关性超过阈值，系统必须拒绝该特征
    """
    X, y = data
    
    # 创建验证框架
    validator = ValidationFramework(correlation_threshold=threshold)
    
    # 检查相关性
    correlations = validator.check_feature_correlation(X, y)
    
    # 属性1：所有特征都应该有相关性值
    feature_cols = [col for col in X.columns if col not in ['Season', 'Week', 'Name']]
    for feature in feature_cols:
        assert feature in correlations, f"Feature {feature} missing from correlations"
    
    # 属性2：高相关性特征应该被识别
    high_corr_features = [feat for feat, corr in correlations.items() if corr >= threshold]
    
    # 验证：如果有高相关性特征，它们的相关性确实 >= threshold
    for feature in high_corr_features:
        assert correlations[feature] >= threshold, (
            f"Feature {feature} marked as high correlation but corr={correlations[feature]} < {threshold}"
        )


@given(data=features_and_target(), min_lag=st.integers(min_value=1, max_value=5))
@settings(max_examples=100, deadline=None)
def test_property_4_lag_validation(data, min_lag):
    """
    Feature: ml-model-optimization, Property 4: Historical Feature Lag Validation
    
    For any historical feature used in the system, the lag value must be >= min_lag,
    ensuring that only past information is used.
    
    验证：对于任何历史特征，滞后值必须 >= min_lag
    """
    X, y = data
    
    # 创建验证框架
    validator = ValidationFramework(min_lag=min_lag)
    
    # 创建特征元数据（模拟一些有效和无效的滞后）
    feature_metadata = {}
    feature_cols = [col for col in X.columns if col not in ['Season', 'Week', 'Name']]
    
    for i, feature in enumerate(feature_cols):
        # 一半特征有有效滞后，一半有无效滞后
        if i % 2 == 0:
            feature_metadata[feature] = {'lag': min_lag + i}  # 有效
        else:
            feature_metadata[feature] = {'lag': max(0, min_lag - 1)}  # 可能无效
    
    # 检查未来泄露
    leakage_features = validator.check_future_leakage(X, feature_metadata)
    
    # 属性：所有被标记为泄露的特征，其lag必须 < min_lag
    for feature in leakage_features:
        lag = feature_metadata[feature]['lag']
        assert lag < min_lag, (
            f"Feature {feature} marked as leakage but lag={lag} >= {min_lag}"
        )
    
    # 属性：所有未被标记为泄露的特征，其lag必须 >= min_lag
    non_leakage_features = [f for f in feature_cols if f not in leakage_features and f in feature_metadata]
    for feature in non_leakage_features:
        lag = feature_metadata[feature]['lag']
        assert lag >= min_lag, (
            f"Feature {feature} not marked as leakage but lag={lag} < {min_lag}"
        )


@given(n_samples=st.integers(min_value=50, max_value=200), 
       n_features=st.integers(min_value=5, max_value=20))
@settings(max_examples=50, deadline=None)
def test_validation_report_completeness(n_samples, n_features):
    """
    测试验证报告的完整性
    
    验证：验证报告必须包含所有特征的信息
    """
    # 生成数据
    data = {}
    data['Season'] = np.random.randint(1, 10, size=n_samples)
    data['Week'] = np.random.randint(1, 11, size=n_samples)
    
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.uniform(0, 100, size=n_samples)
    
    X = pd.DataFrame(data)
    y = pd.Series(np.random.uniform(0, 100, size=n_samples), name='target')
    
    # 创建验证框架
    validator = ValidationFramework()
    
    # 生成报告
    report = validator.generate_validation_report(X, y)
    
    # 属性：报告必须包含所有特征（除了Season, Week, Name）
    feature_cols = [col for col in X.columns if col not in ['Season', 'Week', 'Name']]
    assert len(report) == len(feature_cols), (
        f"Report should contain {len(feature_cols)} features, got {len(report)}"
    )
    
    # 属性：报告必须包含必需的列
    required_columns = ['Feature', 'Correlation', 'Lag', 'Leakage_Status', 'Pass_Fail']
    for col in required_columns:
        assert col in report.columns, f"Report missing required column: {col}"
    
    # 属性：所有特征都应该有Pass或Fail状态
    assert report['Pass_Fail'].isin(['PASS', 'FAIL']).all(), (
        "All features must have PASS or FAIL status"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
