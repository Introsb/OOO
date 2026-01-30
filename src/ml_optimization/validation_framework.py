"""
Validation Framework
验证框架 - 确保无数据泄露的严格验证
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class ValidationFramework:
    """
    验证框架
    
    检查特征是否存在数据泄露：
    1. 相关性检查（< 0.85）
    2. 滞后验证（lag >= 1）
    3. 目标泄露检查
    4. 交叉验证分割验证
    """
    
    def __init__(self, correlation_threshold: float = 0.85, min_lag: int = 1):
        """
        初始化验证框架
        
        Args:
            correlation_threshold: 相关性阈值
            min_lag: 最小滞后值
        """
        if correlation_threshold <= 0 or correlation_threshold > 1:
            raise ValueError(f"correlation_threshold must be between 0 and 1, got {correlation_threshold}")
        
        if min_lag < 1:
            raise ValueError(f"min_lag must be >= 1, got {min_lag}")
        
        self.correlation_threshold = correlation_threshold
        self.min_lag = min_lag
        logger.info(f"ValidationFramework initialized: threshold={correlation_threshold}, min_lag={min_lag}")
    
    def check_feature_correlation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        检查所有特征与目标的相关性
        
        Args:
            X: 特征数据框
            y: 目标变量
            
        Returns:
            特征名到相关性的字典
        """
        correlations = {}
        
        for col in X.columns:
            if col in ['Season', 'Week', 'Name']:
                continue
            
            try:
                corr = X[col].corr(y)
                correlations[col] = abs(corr)
            except Exception as e:
                logger.warning(f"Could not compute correlation for {col}: {e}")
                correlations[col] = 0.0
        
        # 检查是否有超过阈值的特征
        high_corr_features = {k: v for k, v in correlations.items() if v >= self.correlation_threshold}
        
        if high_corr_features:
            logger.warning(f"Found {len(high_corr_features)} features with correlation >= {self.correlation_threshold}")
            for feat, corr in sorted(high_corr_features.items(), key=lambda x: x[1], reverse=True):
                logger.warning(f"  {feat}: {corr:.4f}")
        else:
            logger.info(f"All features have correlation < {self.correlation_threshold}")
        
        return correlations
    
    def check_future_leakage(self, X: pd.DataFrame, feature_metadata: Dict[str, Dict]) -> List[str]:
        """
        检查是否有特征使用了未来信息（lag < 1）
        
        Args:
            X: 特征数据框
            feature_metadata: 特征元数据字典 {feature_name: {'lag': int, ...}}
            
        Returns:
            有潜在泄露的特征列表
        """
        leakage_features = []
        
        for feature in X.columns:
            if feature in ['Season', 'Week', 'Name']:
                continue
            
            if feature in feature_metadata:
                metadata = feature_metadata[feature]
                lag = metadata.get('lag', None)
                
                if lag is not None and lag < self.min_lag:
                    leakage_features.append(feature)
                    logger.error(f"Future leakage detected in {feature}: lag={lag} < {self.min_lag}")
        
        if leakage_features:
            logger.error(f"Found {len(leakage_features)} features with future leakage")
        else:
            logger.info("No future leakage detected")
        
        return leakage_features
    
    def check_target_leakage(self, X: pd.DataFrame, y: pd.Series, 
                            feature_definitions: Dict[str, str]) -> List[str]:
        """
        检查是否有特征直接来自当前目标值
        
        Args:
            X: 特征数据框
            y: 目标变量
            feature_definitions: 特征定义字典 {feature_name: definition_string}
            
        Returns:
            有目标泄露的特征列表
        """
        leakage_features = []
        target_name = y.name if hasattr(y, 'name') else 'target'
        
        for feature in X.columns:
            if feature in ['Season', 'Week', 'Name']:
                continue
            
            if feature in feature_definitions:
                definition = feature_definitions[feature].lower()
                
                # 检查定义中是否包含目标变量名（但不是lag版本）
                if target_name.lower() in definition and 'lag' not in definition and 'shift' not in definition:
                    leakage_features.append(feature)
                    logger.error(f"Target leakage detected in {feature}: definition contains '{target_name}'")
        
        if leakage_features:
            logger.error(f"Found {len(leakage_features)} features with target leakage")
        else:
            logger.info("No target leakage detected")
        
        return leakage_features
    
    def validate_cv_splits(self, cv_splits: List[Tuple[np.ndarray, np.ndarray]], 
                          X: pd.DataFrame) -> bool:
        """
        验证交叉验证分割是否遵守时间顺序
        
        Args:
            cv_splits: 交叉验证分割列表 [(train_idx, val_idx), ...]
            X: 特征数据框（必须包含Season列）
            
        Returns:
            True if valid, False otherwise
        """
        if 'Season' not in X.columns:
            logger.error("Cannot validate CV splits: X does not contain 'Season' column")
            return False
        
        all_valid = True
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            # 获取训练集和验证集的季数
            train_seasons = X.loc[train_idx, 'Season'].values
            val_seasons = X.loc[val_idx, 'Season'].values
            
            # 检查最大训练季 <= 最小验证季
            max_train_season = train_seasons.max()
            min_val_season = val_seasons.min()
            
            if max_train_season > min_val_season:
                logger.error(f"Fold {fold + 1}: Temporal ordering violated! "
                           f"max_train_season={max_train_season}, min_val_season={min_val_season}")
                all_valid = False
        
        if all_valid:
            logger.info(f"All {len(cv_splits)} CV splits respect temporal ordering")
        
        return all_valid
    
    def generate_validation_report(self, X: pd.DataFrame, y: pd.Series,
                                   feature_metadata: Dict[str, Dict] = None,
                                   feature_definitions: Dict[str, str] = None) -> pd.DataFrame:
        """
        生成综合验证报告
        
        Args:
            X: 特征数据框
            y: 目标变量
            feature_metadata: 特征元数据（可选）
            feature_definitions: 特征定义（可选）
            
        Returns:
            验证报告数据框
        """
        logger.info("Generating validation report...")
        
        # 1. 相关性检查
        correlations = self.check_feature_correlation(X, y)
        
        # 2. 未来泄露检查
        future_leakage = []
        if feature_metadata:
            future_leakage = self.check_future_leakage(X, feature_metadata)
        
        # 3. 目标泄露检查
        target_leakage = []
        if feature_definitions:
            target_leakage = self.check_target_leakage(X, y, feature_definitions)
        
        # 构建报告
        report_data = []
        
        for feature in X.columns:
            if feature in ['Season', 'Week', 'Name']:
                continue
            
            corr = correlations.get(feature, 0.0)
            lag = feature_metadata.get(feature, {}).get('lag', 'N/A') if feature_metadata else 'N/A'
            
            # 确定泄露状态
            leakage_status = []
            if corr >= self.correlation_threshold:
                leakage_status.append('High Correlation')
            if feature in future_leakage:
                leakage_status.append('Future Leakage')
            if feature in target_leakage:
                leakage_status.append('Target Leakage')
            
            leakage_str = ', '.join(leakage_status) if leakage_status else 'None'
            pass_fail = 'FAIL' if leakage_status else 'PASS'
            
            report_data.append({
                'Feature': feature,
                'Correlation': f'{corr:.4f}',
                'Lag': lag,
                'Leakage_Status': leakage_str,
                'Pass_Fail': pass_fail
            })
        
        report_df = pd.DataFrame(report_data)
        
        # 统计
        n_total = len(report_df)
        n_pass = (report_df['Pass_Fail'] == 'PASS').sum()
        n_fail = n_total - n_pass
        
        logger.info(f"Validation report: {n_pass}/{n_total} features passed, {n_fail} failed")
        
        return report_df
    
    def validate_all(self, X: pd.DataFrame, y: pd.Series,
                    feature_metadata: Dict[str, Dict] = None,
                    feature_definitions: Dict[str, str] = None,
                    cv_splits: List[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[bool, pd.DataFrame]:
        """
        执行所有验证检查
        
        Args:
            X: 特征数据框
            y: 目标变量
            feature_metadata: 特征元数据（可选）
            feature_definitions: 特征定义（可选）
            cv_splits: 交叉验证分割（可选）
            
        Returns:
            (是否通过验证, 验证报告)
        """
        logger.info("="*80)
        logger.info("VALIDATION FRAMEWORK - COMPREHENSIVE CHECK")
        logger.info("="*80)
        
        # 生成报告
        report = self.generate_validation_report(X, y, feature_metadata, feature_definitions)
        
        # 检查是否有失败的特征
        n_fail = (report['Pass_Fail'] == 'FAIL').sum()
        features_passed = n_fail == 0
        
        # 检查CV分割（如果提供）
        cv_passed = True
        if cv_splits:
            cv_passed = self.validate_cv_splits(cv_splits, X)
        
        # 总体结果
        all_passed = features_passed and cv_passed
        
        if all_passed:
            logger.info("✓ VALIDATION PASSED: No data leakage detected")
        else:
            logger.error("✗ VALIDATION FAILED: Data leakage detected")
            if not features_passed:
                logger.error(f"  {n_fail} features failed validation")
            if not cv_passed:
                logger.error("  CV splits violate temporal ordering")
        
        logger.info("="*80)
        
        return all_passed, report
