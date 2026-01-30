"""
ML Optimizer - Main Pipeline
ML优化器 - 主管道，整合所有优化步骤
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

from .config_loader import ConfigLoader, setup_logging
from .time_series_cv import TimeSeriesCV
from .validation_framework import ValidationFramework
from .feature_engineer import AdvancedFeatureEngineer
from .hyperparameter_optimizer import HyperparameterOptimizer
from .ensemble_builder import EnsembleBuilder
from .model_interpreter import ModelInterpreter

logger = logging.getLogger(__name__)


class MLOptimizer:
    """
    ML优化器主类
    
    执行完整的优化管道：
    1. 加载配置和数据
    2. 特征工程
    3. 验证
    4. 超参数优化
    5. 集成构建
    6. 模型解释
    7. 生成报告
    """
    
    def __init__(self, config_path: str = 'config/ml_optimization_config.yaml'):
        """
        初始化ML优化器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = ConfigLoader.load_config(config_path)
        setup_logging(self.config)
        
        logger.info("="*80)
        logger.info("ML OPTIMIZER INITIALIZED")
        logger.info("="*80)
        
        # 初始化组件
        self.cv = TimeSeriesCV(
            n_splits=self.config['hyperparameter_optimization']['cv_splits'],
            strategy='expanding'
        )
        
        self.validator = ValidationFramework(
            correlation_threshold=self.config['validation']['correlation_threshold'],
            min_lag=self.config['validation']['min_lag']
        )
        
        self.feature_engineer = AdvancedFeatureEngineer(
            polynomial_degree=self.config['feature_engineering']['polynomial_degree'],
            interaction_depth=self.config['feature_engineering']['interaction_depth'],
            correlation_threshold=self.config['validation']['correlation_threshold']
        )
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            cv=self.cv,
            n_iter=self.config['hyperparameter_optimization']['n_iter']
        )
        
        self.ensemble_builder = EnsembleBuilder(cv=self.cv)
        self.model_interpreter = ModelInterpreter()
        
        # 结果存储
        self.results = {}
    
    def load_data(self, data_path: str = 'submission/results/Clean_Enhanced_Dataset.csv') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            (X, y_judge, y_fan)
        """
        logger.info(f"Loading data from {data_path}...")
        
        df = pd.read_csv(data_path)
        
        # 分离特征和目标
        target_cols = ['Judge_Avg_Score', 'Estimated_Fan_Vote']
        exclude_cols = target_cols + ['Score_Scaled', 'Placement', 'Industry_Code']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y_judge = df['Judge_Avg_Score'].copy()
        y_fan = df['Estimated_Fan_Vote'].copy()
        
        logger.info(f"  Loaded {len(df)} records, {len(feature_cols)} features")
        
        return X, y_judge, y_fan
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        分割训练集和测试集（最后2个季作为测试集）
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        test_seasons = self.config['validation']['test_seasons']
        
        seasons = sorted(X['Season'].unique())
        train_seasons = seasons[:-test_seasons]
        test_seasons_list = seasons[-test_seasons:]
        
        train_mask = X['Season'].isin(train_seasons)
        test_mask = X['Season'].isin(test_seasons_list)
        
        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[train_mask].copy()
        y_test = y[test_mask].copy()
        
        logger.info(f"Train: {len(X_train)} records, Test: {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test
    
    def optimize_for_target(self, X: pd.DataFrame, y: pd.Series, 
                           target_name: str) -> Dict[str, Any]:
        """
        为单个目标变量执行完整优化
        
        Args:
            X: 特征数据
            y: 目标变量
            target_name: 目标名称（'Judge' 或 'Fan'）
            
        Returns:
            优化结果字典
        """
        logger.info("="*80)
        logger.info(f"OPTIMIZING FOR {target_name.upper()}")
        logger.info("="*80)
        
        # 1. 分割数据
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 2. 特征工程
        allowed_features = [col for col in X_train.columns 
                           if col not in ['Season', 'Week', 'Name'] 
                           and not col.startswith('Judge') 
                           and not col.startswith('Estimated')]
        
        X_train_eng, selected_features = self.feature_engineer.engineer_features(
            X_train, y_train, allowed_features,
            selection_method=self.config['feature_engineering']['selection_method'],
            n_features_to_select=self.config['feature_engineering']['n_features_to_select']
        )
        
        # 对测试集应用相同的特征工程（不进行选择）
        # 先创建所有特征
        X_test_poly = self.feature_engineer.create_polynomial_features(X_test, allowed_features)
        X_test_interact = self.feature_engineer.create_interaction_features(X_test_poly, y_test, allowed_features)
        X_test_domain = self.feature_engineer.create_domain_features(X_test_interact)
        
        # 然后只保留训练集选中的特征
        required_cols = ['Season', 'Week', 'Name'] + selected_features
        # 只保留存在的列
        available_cols = [col for col in required_cols if col in X_test_domain.columns]
        X_test_eng = X_test_domain[available_cols].copy()
        
        # 3. 验证
        validation_passed, validation_report = self.validator.validate_all(
            X_train_eng, y_train
        )
        
        if not validation_passed:
            logger.warning("Validation failed! Proceeding with caution...")
        
        # 4. 准备训练数据（移除非特征列）
        # 只使用在测试集中也存在的特征
        common_features = [f for f in selected_features if f in X_test_eng.columns]
        
        # 保留Season列用于CV，但创建不含Season的版本用于模型训练
        X_train_with_season = X_train_eng[['Season', 'Week', 'Name'] + common_features].fillna(0)
        X_test_with_season = X_test_eng[['Season', 'Week', 'Name'] + common_features].fillna(0)
        
        X_train_model = X_train_with_season[common_features]
        X_test_model = X_test_with_season[common_features]
        
        # 5. 超参数优化（传入带Season的数据用于CV分割）
        optimization_results = self.hyperparameter_optimizer.optimize_all_models(
            X_train_with_season, y_train, common_features,
            model_types=self.config['ensemble']['base_models']
        )
        
        # 提取最佳模型
        base_models = {name: model for name, (params, score, model) in optimization_results.items()}
        
        if not base_models:
            logger.error("No models were successfully optimized!")
            raise ValueError("No models available for ensemble building")
        
        # 6. 集成构建（使用最后一折作为验证集）
        splits = list(self.cv.split(X_train_eng))
        train_idx, val_idx = splits[-1]
        
        X_train_fold = X_train_model.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train_model.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        ensembles = self.ensemble_builder.build_all_ensembles(
            base_models, X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )
        
        # 7. 选择最佳模型
        best_model_name = None
        best_model = None
        best_score = -np.inf
        
        # 评估所有模型
        all_models = {**base_models, **ensembles}
        
        for name, model in all_models.items():
            metrics = self.ensemble_builder.evaluate_ensemble(model, X_test_model, y_test)
            logger.info(f"{name} Test R²: {metrics['r2']:.4f}")
            
            if metrics['r2'] > best_score:
                best_score = metrics['r2']
                best_model = model
                best_model_name = name
        
        logger.info(f"Best model: {best_model_name} (R² = {best_score:.4f})")
        
        # 8. 模型解释
        interpretation = self.model_interpreter.generate_interpretation_report(
            best_model, selected_features
        )
        
        # 9. 最终评估
        final_metrics = self.ensemble_builder.evaluate_ensemble(best_model, X_test_model, y_test)
        
        results = {
            'target_name': target_name,
            'selected_features': selected_features,
            'validation_report': validation_report,
            'optimization_results': {name: {'params': params, 'cv_score': score} 
                                    for name, (params, score, model) in optimization_results.items()},
            'best_model_name': best_model_name,
            'best_model': best_model,
            'test_metrics': final_metrics,
            'interpretation': interpretation
        }
        
        return results
    
    def run_full_optimization(self, data_path: str = 'submission/results/Clean_Enhanced_Dataset.csv') -> Dict[str, Any]:
        """
        运行完整的优化管道
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            完整的优化结果
        """
        logger.info("="*80)
        logger.info("STARTING FULL ML OPTIMIZATION PIPELINE")
        logger.info("="*80)
        
        # 加载数据
        X, y_judge, y_fan = self.load_data(data_path)
        
        # 优化Judge预测
        judge_results = self.optimize_for_target(X, y_judge, 'Judge')
        
        # 优化Fan预测
        fan_results = self.optimize_for_target(X, y_fan, 'Fan')
        
        # 汇总结果
        final_results = {
            'judge': judge_results,
            'fan': fan_results,
            'config': self.config
        }
        
        # 保存结果
        self._save_results(final_results)
        
        # 生成总结报告
        self._generate_summary_report(final_results)
        
        logger.info("="*80)
        logger.info("ML OPTIMIZATION COMPLETE")
        logger.info("="*80)
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存结果"""
        output_dir = Path('results/ml_optimization')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'optimized_judge_model.pkl', 'wb') as f:
            pickle.dump(results['judge']['best_model'], f)
        
        with open(models_dir / 'optimized_fan_model.pkl', 'wb') as f:
            pickle.dump(results['fan']['best_model'], f)
        
        logger.info("Models saved to models/")
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """生成总结报告"""
        report_dir = Path('reports/ml_optimization')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建总结
        summary = []
        summary.append("="*80)
        summary.append("ML OPTIMIZATION SUMMARY REPORT")
        summary.append("="*80)
        summary.append("")
        
        for target in ['judge', 'fan']:
            target_results = results[target]
            summary.append(f"\n{target.upper()} PREDICTION:")
            summary.append(f"  Best Model: {target_results['best_model_name']}")
            summary.append(f"  Test R²: {target_results['test_metrics']['r2']:.4f}")
            summary.append(f"  Test MAE: {target_results['test_metrics']['mae']:.4f}")
            summary.append(f"  Test RMSE: {target_results['test_metrics']['rmse']:.4f}")
            summary.append(f"  Features Used: {len(target_results['selected_features'])}")
        
        summary.append("\n" + "="*80)
        
        summary_text = "\n".join(summary)
        
        with open(report_dir / 'optimization_summary.txt', 'w') as f:
            f.write(summary_text)
        
        logger.info(summary_text)
        logger.info(f"Summary report saved to {report_dir}/optimization_summary.txt")


def main():
    """主函数"""
    optimizer = MLOptimizer()
    results = optimizer.run_full_optimization()
    return results


if __name__ == '__main__':
    main()
