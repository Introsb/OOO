"""端到端集成测试"""

import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing_pipeline import PreprocessingPipeline


def test_end_to_end_pipeline():
    """测试完整的端到端流程"""
    pipeline = PreprocessingPipeline()
    
    input_file = "2026 MCM Problem C Data.csv"
    output_file = "test_output.csv"
    
    # 运行管道
    pipeline.run(input_file, output_file)
    
    # 验证输出文件存在
    assert os.path.exists(output_file), "Output file should exist"
    
    # 读取输出文件
    df_output = pd.read_csv(output_file)
    
    # 验证必需列
    required_cols = ['Season', 'Week', 'Name', 'Age', 'Industry_Code', 
                    'Judge_Avg_Score', 'Score_Scaled', 'Placement']
    for col in required_cols:
        assert col in df_output.columns, f"Column {col} should exist"
    
    # 验证数据行数在合理范围内（约2700-3000行，因为清洗后会删除无效数据）
    assert 2000 < len(df_output) < 4000, f"Expected ~2700 rows, got {len(df_output)}"
    
    # 验证没有NaN在关键列
    assert df_output['Judge_Avg_Score'].notna().all(), "Judge_Avg_Score should not have NaN"
    assert df_output['Score_Scaled'].notna().all(), "Score_Scaled should not have NaN"
    
    # 清理测试文件
    os.unlink(output_file)
    if os.path.exists('judge_score_distribution.png'):
        os.unlink('judge_score_distribution.png')


def test_output_utf8_encoding():
    """验证输出文件的UTF-8编码"""
    pipeline = PreprocessingPipeline()
    
    input_file = "2026 MCM Problem C Data.csv"
    output_file = "test_output_utf8.csv"
    
    # 运行管道
    pipeline.run(input_file, output_file)
    
    # 尝试用UTF-8读取
    try:
        df = pd.read_csv(output_file, encoding='utf-8')
        assert len(df) > 0
    except UnicodeDecodeError:
        pytest.fail("Output file is not UTF-8 encoded")
    finally:
        os.unlink(output_file)
        if os.path.exists('judge_score_distribution.png'):
            os.unlink('judge_score_distribution.png')


def test_file_overwrite():
    """验证文件覆盖行为"""
    pipeline = PreprocessingPipeline()
    
    input_file = "2026 MCM Problem C Data.csv"
    output_file = "test_output_overwrite.csv"
    
    # 第一次运行
    pipeline.run(input_file, output_file)
    first_mtime = os.path.getmtime(output_file)
    
    # 等待一小段时间
    import time
    time.sleep(0.1)
    
    # 第二次运行（应该覆盖）
    pipeline.run(input_file, output_file)
    second_mtime = os.path.getmtime(output_file)
    
    # 验证文件被覆盖
    assert second_mtime > first_mtime, "File should be overwritten"
    
    # 清理
    os.unlink(output_file)
    if os.path.exists('judge_score_distribution.png'):
        os.unlink('judge_score_distribution.png')
