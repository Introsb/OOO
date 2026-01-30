"""
Ensemble Builder
集成构建器 - 创建和管理集成模型
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """集成构建器"""
    
    def __init__(self, cv):
        """
        初始化集成构建器
        
        Args:
            cv: 交叉验证对象
        """
        self.cv = cv
        logger.info("EnsembleBuilder initialized")
    
    def build_stacking_ensemble(self, base_models: Dict[str, Any], 
                               X: pd.DataFrame, y: pd.Series) -> Any:
        """
        构建Stacking集成
        
        Args:
            base_models: {model_name: model_instance}
            X: 特征数据
            y: 目标变量
            
        Returns:
            训练好的Stacking集成
        """
        logger.info("Building stacking ensemble...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        meta_model = Ridge(alpha=1.0)
        
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=self.cv,
            n_jobs=-1
        )
        
        stacking.fit(X, y)
        
        logger.info("  Stacking ensemble built")
        
        return stacking
    
    def build_voting_ensemble(self, base_models: Dict[str, Any],
                             X: pd.DataFrame, y: pd.Series,
                             weights: List[float] = None) -> Any:
        """
        构建Voting集成
        
        Args:
            base_models: {model_name: model_instance}
            X: 特征数据
            y: 目标变量
            weights: 权重列表（可选）
            
        Returns:
            训练好的Voting集成
        """
        logger.info("Building voting ensemble...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )
        
        voting.fit(X, y)
        
        logger.info("  Voting ensemble built")
        
        return voting
    
    def build_weighted_ensemble(self, base_models: Dict[str, Any],
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, List[float]]:
        """
        构建加权集成（基于验证集性能）
        
        Args:
            base_models: {model_name: model_instance}
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            (集成模型, 权重列表)
        """
        logger.info("Building weighted ensemble...")
        
        # 计算每个模型的验证误差
        errors = []
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            errors.append(rmse)
            logger.debug(f"  {name} RMSE: {rmse:.4f}")
        
        # 计算权重（误差的倒数，归一化）
        errors = np.array(errors)
        weights = 1 / (errors + 1e-6)
        weights = weights / weights.sum()
        
        logger.info(f"  Weights: {dict(zip(base_models.keys(), weights))}")
        
        # 创建加权Voting集成
        weighted_ensemble = self.build_voting_ensemble(base_models, X_train, y_train, weights.tolist())
        
        return weighted_ensemble, weights.tolist()
    
    def evaluate_ensemble(self, ensemble: Any, X_test: pd.DataFrame, 
                         y_test: pd.Series) -> Dict[str, float]:
        """
        评估集成模型
        
        Args:
            ensemble: 集成模型
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        y_pred = ensemble.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return metrics
    
    def build_all_ensembles(self, base_models: Dict[str, Any],
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        构建所有类型的集成
        
        Args:
            base_models: 基础模型字典
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            {ensemble_type: ensemble_model}
        """
        logger.info("="*80)
        logger.info("ENSEMBLE BUILDING")
        logger.info("="*80)
        
        ensembles = {}
        
        # Stacking
        try:
            stacking = self.build_stacking_ensemble(base_models, X_train, y_train)
            ensembles['stacking'] = stacking
        except Exception as e:
            logger.error(f"Failed to build stacking ensemble: {e}")
        
        # Voting
        try:
            voting = self.build_voting_ensemble(base_models, X_train, y_train)
            ensembles['voting'] = voting
        except Exception as e:
            logger.error(f"Failed to build voting ensemble: {e}")
        
        # Weighted
        try:
            weighted, weights = self.build_weighted_ensemble(base_models, X_train, y_train, X_val, y_val)
            ensembles['weighted'] = weighted
        except Exception as e:
            logger.error(f"Failed to build weighted ensemble: {e}")
        
        logger.info("="*80)
        
        return ensembles
