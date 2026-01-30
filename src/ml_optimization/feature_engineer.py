"""
Advanced Feature Engineer
高级特征工程器 - 创建高级特征而不产生数据泄露
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    高级特征工程器
    
    创建多项式、交互和领域特定特征，同时确保无数据泄露
    """
    
    def __init__(self, polynomial_degree: int = 2, interaction_depth: int = 2,
                 correlation_threshold: float = 0.85):
        """
        初始化特征工程器
        
        Args:
            polynomial_degree: 多项式度数（2或3）
            interaction_depth: 交互深度（2或3）
            correlation_threshold: 相关性阈值
        """
        self.polynomial_degree = polynomial_degree
        self.interaction_depth = interaction_depth
        self.correlation_threshold = correlation_threshold
        self.feature_metadata = {}
        
        logger.info(f"AdvancedFeatureEngineer initialized: degree={polynomial_degree}, "
                   f"depth={interaction_depth}, threshold={correlation_threshold}")
    
    def create_polynomial_features(self, X: pd.DataFrame, 
                                   allowed_features: List[str]) -> pd.DataFrame:
        """
        创建多项式特征
        
        Args:
            X: 特征数据框
            allowed_features: 允许使用的特征列表（外部+历史特征）
            
        Returns:
            包含原始+多项式特征的数据框
        """
        logger.info(f"Creating polynomial features (degree={self.polynomial_degree})...")
        
        # 只使用允许的特征
        X_allowed = X[allowed_features].copy()
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
        X_poly = poly.fit_transform(X_allowed)
        
        # 获取特征名称
        feature_names = poly.get_feature_names_out(allowed_features)
        
        # 创建数据框
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # 只保留新特征（度数>1）
        new_features = [col for col in X_poly_df.columns if col not in allowed_features]
        X_new = X_poly_df[new_features]
        
        # 更新元数据
        for feat in new_features:
            self.feature_metadata[feat] = {
                'category': 'polynomial',
                'degree': self.polynomial_degree,
                'source_features': allowed_features
            }
        
        logger.info(f"  Created {len(new_features)} polynomial features")
        
        # 合并到原始数据框
        X_result = pd.concat([X, X_new], axis=1)
        
        return X_result
    
    def create_interaction_features(self, X: pd.DataFrame, y: pd.Series,
                                    allowed_features: List[str]) -> pd.DataFrame:
        """
        创建交互特征（特征对的乘积）
        
        Args:
            X: 特征数据框
            y: 目标变量（用于验证相关性）
            allowed_features: 允许使用的特征列表
            
        Returns:
            包含原始+交互特征的数据框
        """
        logger.info(f"Creating interaction features (depth={self.interaction_depth})...")
        
        X_result = X.copy()
        new_features_count = 0
        rejected_count = 0
        
        # 创建两两交互
        if self.interaction_depth >= 2:
            for i, feat1 in enumerate(allowed_features):
                for feat2 in allowed_features[i+1:]:
                    # 创建交互特征
                    interaction_name = f"{feat1}_x_{feat2}"
                    interaction_values = X[feat1] * X[feat2]
                    
                    # 检查相关性
                    corr = abs(interaction_values.corr(y))
                    
                    if corr < self.correlation_threshold:
                        X_result[interaction_name] = interaction_values
                        self.feature_metadata[interaction_name] = {
                            'category': 'interaction',
                            'depth': 2,
                            'source_features': [feat1, feat2],
                            'correlation': corr
                        }
                        new_features_count += 1
                    else:
                        rejected_count += 1
                        logger.debug(f"  Rejected {interaction_name}: corr={corr:.4f} >= {self.correlation_threshold}")
        
        # 创建三阶交互（如果depth=3）
        if self.interaction_depth >= 3:
            # 只选择最重要的特征进行三阶交互（避免组合爆炸）
            top_features = allowed_features[:min(5, len(allowed_features))]
            
            for i, feat1 in enumerate(top_features):
                for j, feat2 in enumerate(top_features[i+1:], start=i+1):
                    for feat3 in top_features[j+1:]:
                        interaction_name = f"{feat1}_x_{feat2}_x_{feat3}"
                        interaction_values = X[feat1] * X[feat2] * X[feat3]
                        
                        corr = abs(interaction_values.corr(y))
                        
                        if corr < self.correlation_threshold:
                            X_result[interaction_name] = interaction_values
                            self.feature_metadata[interaction_name] = {
                                'category': 'interaction',
                                'depth': 3,
                                'source_features': [feat1, feat2, feat3],
                                'correlation': corr
                            }
                            new_features_count += 1
                        else:
                            rejected_count += 1
        
        logger.info(f"  Created {new_features_count} interaction features")
        logger.info(f"  Rejected {rejected_count} features (high correlation)")
        
        return X_result
    
    def create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        创建DWTS领域特定特征
        
        Args:
            X: 特征数据框
            
        Returns:
            包含原始+领域特征的数据框
        """
        logger.info("Creating domain-specific features...")
        
        X_result = X.copy()
        new_features = []
        
        # 1. Week动量（如果有score_lag特征）
        if 'score_lag_1' in X.columns and 'score_lag_2' in X.columns:
            X_result['week_momentum'] = (X['score_lag_1'] - X['score_lag_2']) / (X['score_lag_2'] + 1e-6)
            new_features.append('week_momentum')
            self.feature_metadata['week_momentum'] = {
                'category': 'domain',
                'description': 'Score momentum from previous weeks'
            }
        
        # 2. 一致性（如果有hist_std特征）
        if 'score_hist_std' in X.columns:
            X_result['consistency_score'] = 1 / (X['score_hist_std'] + 1e-6)
            new_features.append('consistency_score')
            self.feature_metadata['consistency_score'] = {
                'category': 'domain',
                'description': 'Consistency based on historical std'
            }
        
        # 3. 相对表现（如果有hist_mean特征）
        if 'score_hist_mean' in X.columns and 'score_lag_1' in X.columns:
            X_result['relative_performance'] = X['score_lag_1'] / (X['score_hist_mean'] + 1e-6)
            new_features.append('relative_performance')
            self.feature_metadata['relative_performance'] = {
                'category': 'domain',
                'description': 'Current score relative to historical average'
            }
        
        # 4. 淘汰压力（基于Week和竞争者数量）
        if 'Week' in X.columns and 'n_competitors' in X.columns:
            # 假设平均每周淘汰1-2人
            typical_weeks_remaining = 11 - X['Week']
            X_result['elimination_pressure'] = X['n_competitors'] / (typical_weeks_remaining + 1)
            new_features.append('elimination_pressure')
            self.feature_metadata['elimination_pressure'] = {
                'category': 'domain',
                'description': 'Elimination pressure based on competitors and weeks remaining'
            }
        
        # 5. 经验因子（Age × Week交互的变体）
        if 'Age' in X.columns and 'Week' in X.columns:
            X_result['experience_factor'] = X['Age'] * np.log1p(X['Week'])
            new_features.append('experience_factor')
            self.feature_metadata['experience_factor'] = {
                'category': 'domain',
                'description': 'Experience factor combining age and week progression'
            }
        
        logger.info(f"  Created {len(new_features)} domain-specific features")
        
        return X_result
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info',
                       n_features: int = 50) -> List[str]:
        """
        选择最重要的特征
        
        Args:
            X: 特征数据框
            y: 目标变量
            method: 选择方法 ('mutual_info', 'recursive', 'lasso')
            n_features: 要选择的特征数量
            
        Returns:
            选中的特征名称列表
        """
        logger.info(f"Selecting features using {method} method...")
        
        # 排除非特征列
        exclude_cols = ['Season', 'Week', 'Name']
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        X_features = X[feature_cols].copy()
        
        # 填充缺失值
        X_features = X_features.fillna(X_features.mean())
        
        # 限制特征数量
        n_features = min(n_features, len(feature_cols))
        
        if method == 'mutual_info':
            # 互信息选择
            mi_scores = mutual_info_regression(X_features, y, random_state=42)
            mi_scores_df = pd.DataFrame({
                'feature': feature_cols,
                'score': mi_scores
            }).sort_values('score', ascending=False)
            
            selected_features = mi_scores_df.head(n_features)['feature'].tolist()
            
        elif method == 'recursive':
            # 递归特征消除
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rfe = RFE(estimator, n_features_to_select=n_features, step=5)
            rfe.fit(X_features, y)
            
            selected_features = [feat for feat, selected in zip(feature_cols, rfe.support_) if selected]
            
        elif method == 'lasso':
            # Lasso特征选择
            lasso = Lasso(alpha=0.01, random_state=42, max_iter=1000)
            lasso.fit(X_features, y)
            
            # 选择系数非零的特征
            coef_df = pd.DataFrame({
                'feature': feature_cols,
                'coef': np.abs(lasso.coef_)
            }).sort_values('coef', ascending=False)
            
            selected_features = coef_df.head(n_features)['feature'].tolist()
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        logger.info(f"  Selected {len(selected_features)} features")
        
        return selected_features
    
    def engineer_features(self, X: pd.DataFrame, y: pd.Series,
                         allowed_features: List[str],
                         selection_method: str = 'mutual_info',
                         n_features_to_select: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        执行完整的特征工程流程
        
        Args:
            X: 特征数据框
            y: 目标变量
            allowed_features: 允许使用的特征列表
            selection_method: 特征选择方法
            n_features_to_select: 要选择的特征数量
            
        Returns:
            (工程化的数据框, 选中的特征列表)
        """
        logger.info("="*80)
        logger.info("ADVANCED FEATURE ENGINEERING")
        logger.info("="*80)
        
        # 1. 创建多项式特征
        X_poly = self.create_polynomial_features(X, allowed_features)
        
        # 2. 创建交互特征
        X_interact = self.create_interaction_features(X_poly, y, allowed_features)
        
        # 3. 创建领域特定特征
        X_domain = self.create_domain_features(X_interact)
        
        # 4. 特征选择
        selected_features = self.select_features(X_domain, y, selection_method, n_features_to_select)
        
        # 5. 保留必需的列
        required_cols = ['Season', 'Week', 'Name']
        final_cols = required_cols + selected_features
        X_final = X_domain[final_cols].copy()
        
        logger.info(f"Final feature count: {len(selected_features)}")
        logger.info("="*80)
        
        return X_final, selected_features
