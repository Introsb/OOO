#!/usr/bin/env python
"""
DWTS数据预处理主程序
将"与星共舞"比赛数据从宽表格式转换为长表格式
"""

import argparse
import sys
from src.preprocessing_pipeline import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description='DWTS数据预处理：将宽表转换为长表格式'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='2026 MCM Problem C Data.csv',
        help='输入CSV文件路径（默认: 2026 MCM Problem C Data.csv）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='Processed_DWTS_Long_Format.csv',
        help='输出CSV文件路径（默认: Processed_DWTS_Long_Format.csv）'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = PreprocessingPipeline()
        pipeline.run(args.input, args.output)
        return 0
    except Exception as e:
        print(f"\n处理失败: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
