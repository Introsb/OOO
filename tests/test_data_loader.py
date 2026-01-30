"""单元测试：DataLoader组件"""

import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing_pipeline import DataLoader


def test_load_valid_file():
    """测试成功加载有效文件"""
    loader = DataLoader()
    df = loader.load("2026 MCM Problem C Data.csv")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'season' in df.columns
    assert 'celebrity_name' in df.columns


def test_load_nonexistent_file():
    """测试文件不存在的错误处理"""
    loader = DataLoader()
    
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load("nonexistent_file.csv")
    
    assert "not found" in str(exc_info.value)


def test_load_invalid_format():
    """测试无效格式的错误处理"""
    # 创建一个临时的无效CSV文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("invalid,csv\n")
        f.write("missing,required,columns\n")
        temp_path = f.name
    
    try:
        loader = DataLoader()
        with pytest.raises(ValueError) as exc_info:
            loader.load(temp_path)
        
        assert "Missing required columns" in str(exc_info.value)
    finally:
        os.unlink(temp_path)
