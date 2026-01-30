"""
Configuration Loader
配置加载器 - 加载和验证配置文件
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""
    
    DEFAULT_CONFIG = {
        'feature_engineering': {
            'polynomial_degree': 2,
            'interaction_depth': 2,
            'selection_method': 'mutual_info',
            'n_features_to_select': 50
        },
        'hyperparameter_optimization': {
            'search_method': 'bayesian',
            'n_iter': 30,
            'cv_splits': 5
        },
        'ensemble': {
            'base_models': ['random_forest', 'gradient_boosting', 'ridge'],
            'ensemble_types': ['stacking', 'voting', 'weighted'],
            'meta_model': 'ridge'
        },
        'validation': {
            'correlation_threshold': 0.85,
            'min_lag': 1,
            'test_seasons': 2
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/ml_optimization.log'
        }
    }
    
    @staticmethod
    def load_config(config_path: str = 'config/ml_optimization_config.yaml') -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                logger.warning("Using default configuration")
                return ConfigLoader.DEFAULT_CONFIG.copy()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置
            ConfigLoader._validate_config(config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            return ConfigLoader.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """验证配置有效性"""
        
        # 验证特征工程配置
        if 'feature_engineering' in config:
            fe_config = config['feature_engineering']
            
            if 'polynomial_degree' in fe_config:
                degree = fe_config['polynomial_degree']
                if not isinstance(degree, int) or degree < 1 or degree > 3:
                    raise ValueError(f"Invalid polynomial_degree: {degree}. Must be 1, 2, or 3")
            
            if 'interaction_depth' in fe_config:
                depth = fe_config['interaction_depth']
                if not isinstance(depth, int) or depth < 1 or depth > 3:
                    raise ValueError(f"Invalid interaction_depth: {depth}. Must be 1, 2, or 3")
        
        # 验证超参数优化配置
        if 'hyperparameter_optimization' in config:
            hpo_config = config['hyperparameter_optimization']
            
            if 'search_method' in hpo_config:
                method = hpo_config['search_method']
                if method not in ['grid', 'random', 'bayesian']:
                    raise ValueError(f"Invalid search_method: {method}")
            
            if 'cv_splits' in hpo_config:
                splits = hpo_config['cv_splits']
                if not isinstance(splits, int) or splits < 3 or splits > 10:
                    raise ValueError(f"Invalid cv_splits: {splits}. Must be between 3 and 10")
        
        # 验证验证配置
        if 'validation' in config:
            val_config = config['validation']
            
            if 'correlation_threshold' in val_config:
                threshold = val_config['correlation_threshold']
                if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1:
                    raise ValueError(f"Invalid correlation_threshold: {threshold}. Must be between 0 and 1")
            
            if 'min_lag' in val_config:
                lag = val_config['min_lag']
                if not isinstance(lag, int) or lag < 1:
                    raise ValueError(f"Invalid min_lag: {lag}. Must be >= 1")
        
        logger.info("Configuration validation passed")


def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志配置
    
    Args:
        config: 配置字典
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', 'logs/ml_optimization.log')
    
    # 创建日志目录
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")
