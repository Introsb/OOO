"""
Run ML Optimization
运行ML优化管道
"""

import sys
sys.path.insert(0, 'src')

from ml_optimization.ml_optimizer import MLOptimizer


def main():
    """主函数"""
    print("\n" + "="*80)
    print("DWTS ML OPTIMIZATION PIPELINE")
    print("="*80)
    print("\n正在启动优化流程...")
    print("这可能需要15-30分钟，请耐心等待...\n")
    
    # 创建优化器
    optimizer = MLOptimizer(config_path='config/ml_optimization_config.yaml')
    
    # 运行优化
    results = optimizer.run_full_optimization(
        data_path='submission/results/Clean_Enhanced_Dataset.csv'
    )
    
    print("\n" + "="*80)
    print("✓ 优化完成！")
    print("="*80)
    print("\n结果已保存到：")
    print("  - 模型: models/optimized_judge_model.pkl, models/optimized_fan_model.pkl")
    print("  - 报告: reports/ml_optimization/optimization_summary.txt")
    print("  - 日志: logs/ml_optimization.log")
    print("\n" + "="*80)
    
    # 打印关键结果
    print("\n📊 关键结果：")
    print(f"\nJudge预测:")
    print(f"  最佳模型: {results['judge']['best_model_name']}")
    print(f"  测试集R²: {results['judge']['test_metrics']['r2']:.4f}")
    print(f"  测试集MAE: {results['judge']['test_metrics']['mae']:.4f}")
    
    print(f"\nFan预测:")
    print(f"  最佳模型: {results['fan']['best_model_name']}")
    print(f"  测试集R²: {results['fan']['test_metrics']['r2']:.4f}")
    print(f"  测试集MAE: {results['fan']['test_metrics']['mae']:.4f}")
    
    print("\n" + "="*80)
    
    return results


if __name__ == '__main__':
    results = main()
