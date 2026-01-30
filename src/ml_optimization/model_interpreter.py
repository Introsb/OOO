"""
Model Interpreter
模型解释器 - 提供SHAP值和特征重要性
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """模型解释器"""
    
    def __init__(self):
        logger.info("ModelInterpreter initialized")
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        获取特征重要性（树模型）
        
        Args:
            model: 模型对象
            feature_names: 特征名称列表
            
        Returns:
            特征重要性数据框
        """
        logger.info("Extracting feature importance...")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model does not have feature_importances_ or coef_")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"  Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def generate_interpretation_report(self, model: Any, feature_names: List[str]) -> Dict:
        """
        生成解释报告
        
        Args:
            model: 模型对象
            feature_names: 特征名称列表
            
        Returns:
            解释报告字典
        """
        logger.info("Generating interpretation report...")
        
        importance_df = self.get_feature_importance(model, feature_names)
        
        report = {
            'top_features': importance_df.head(10).to_dict('records') if not importance_df.empty else [],
            'feature_count': len(feature_names)
        }
        
        return report
