"""
Hyperparameter Optimizer
超参数优化器 - 使用贝叶斯优化调优模型参数
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """超参数优化器"""
    
    PARAM_SPACES = {
        'random_forest': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        },
        'gradient_boosting': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.6, 1.0)
        },
        'ridge': {
            'alpha': Real(0.01, 100.0, prior='log-uniform')
        },
        'lasso': {
            'alpha': Real(0.001, 10.0, prior='log-uniform')
        },
        'elasticnet': {
            'alpha': Real(0.001, 10.0, prior='log-uniform'),
            'l1_ratio': Real(0.0, 1.0)
        }
    }
    
    def __init__(self, cv, n_iter: int = 30):
        """
        初始化优化器
        
        Args:
            cv: 交叉验证对象
            n_iter: 优化迭代次数
        """
        self.cv = cv
        self.n_iter = n_iter
        self.best_params = {}
        
        logger.info(f"HyperparameterOptimizer initialized: n_iter={n_iter}")
    
    def _get_model(self, model_type: str) -> Any:
        """获取模型实例"""
        models = {
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42, max_iter=2000),
            'elasticnet': ElasticNet(random_state=42, max_iter=2000)
        }
        return models.get(model_type)
    
    def optimize_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                      feature_cols: List[str]) -> Tuple[Dict, float, Any]:
        """
        优化单个模型的超参数
        
        Args:
            model_type: 模型类型
            X: 特征数据（包含Season列用于CV）
            y: 目标变量
            feature_cols: 实际用于训练的特征列
            
        Returns:
            (最佳参数, 最佳分数, 最佳模型)
        """
        logger.info(f"Optimizing {model_type}...")
        
        model = self._get_model(model_type)
        param_space = self.PARAM_SPACES.get(model_type, {})
        
        if not param_space:
            logger.warning(f"No parameter space defined for {model_type}, using defaults")
            X_features = X[feature_cols]
            model.fit(X_features, y)
            
            # 手动计算CV分数
            scores = []
            for train_idx, val_idx in self.cv.split(X):
                X_train = X.iloc[train_idx][feature_cols]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx][feature_cols]
                y_val = y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            score = np.mean(scores)
            return {}, score, model
        
        # 手动实现贝叶斯优化的CV（避免Season列问题）
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        
        # 准备搜索空间（添加名字）
        dimensions = []
        param_names = []
        for param_name, param_range in param_space.items():
            # 为dimension添加名字
            if isinstance(param_range, Integer):
                dimensions.append(Integer(param_range.low, param_range.high, name=param_name))
            elif isinstance(param_range, Real):
                dimensions.append(Real(param_range.low, param_range.high, prior=param_range.prior, name=param_name))
            param_names.append(param_name)
        
        # 定义目标函数
        @use_named_args(dimensions)
        def objective(**params):
            # 设置参数
            model_instance = self._get_model(model_type)
            model_instance.set_params(**params)
            
            # 手动CV评估
            scores = []
            for train_idx, val_idx in self.cv.split(X):
                X_train = X.iloc[train_idx][feature_cols]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx][feature_cols]
                y_val = y.iloc[val_idx]
                
                model_instance.fit(X_train, y_train)
                score = model_instance.score(X_val, y_val)
                scores.append(score)
            
            # 返回负分数（因为gp_minimize是最小化）
            return -np.mean(scores)
        
        # 运行贝叶斯优化
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_iter,
            random_state=42,
            verbose=False,
            n_jobs=1  # 避免并行问题
        )
        
        # 提取最佳参数
        best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
        best_score = -result.fun  # 转回正分数
        
        # 用最佳参数训练最终模型
        best_model = self._get_model(model_type)
        best_model.set_params(**best_params)
        X_features = X[feature_cols]
        best_model.fit(X_features, y)
        
        self.best_params[model_type] = best_params
        
        logger.info(f"  Best {model_type} R²: {best_score:.4f}")
        logger.info(f"  Best params: {best_params}")
        
        return best_params, best_score, best_model
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series, 
                           feature_cols: List[str],
                           model_types: list = None) -> Dict[str, Tuple[Dict, float, Any]]:
        """
        优化所有模型
        
        Args:
            X: 特征数据（包含Season列）
            y: 目标变量
            feature_cols: 实际用于训练的特征列
            model_types: 要优化的模型类型列表
            
        Returns:
            {model_type: (best_params, best_score, best_model)}
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'elasticnet']
        
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        results = {}
        
        for model_type in model_types:
            try:
                params, score, model = self.optimize_model(model_type, X, y, feature_cols)
                results[model_type] = (params, score, model)
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
        
        logger.info("="*80)
        
        return results
