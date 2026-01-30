"""
Time Series Cross-Validation
时间序列交叉验证 - 防止数据泄露的交叉验证策略
"""

import numpy as np
import pandas as pd
import logging
from typing import Iterator, Tuple, List
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class TimeSeriesCV(BaseCrossValidator):
    """
    时间序列交叉验证器
    
    实现扩展窗口策略，确保训练数据始终在验证数据之前
    """
    
    def __init__(self, n_splits: int = 5, strategy: str = 'expanding'):
        """
        初始化时间序列交叉验证器
        
        Args:
            n_splits: 折数（3-10）
            strategy: 策略 ('expanding' 或 'sliding')
        """
        if n_splits < 3 or n_splits > 10:
            raise ValueError(f"n_splits must be between 3 and 10, got {n_splits}")
        
        if strategy not in ['expanding', 'sliding']:
            raise ValueError(f"strategy must be 'expanding' or 'sliding', got {strategy}")
        
        self.n_splits = n_splits
        self.strategy = strategy
        logger.info(f"TimeSeriesCV initialized: n_splits={n_splits}, strategy={strategy}")
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """返回折数"""
        return self.n_splits
    
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成时间序列交叉验证分割
        
        Args:
            X: 特征数据框（必须包含Season列）
            y: 目标变量（可选）
            groups: 分组（可选）
            
        Yields:
            (train_indices, val_indices) 元组
        """
        if 'Season' not in X.columns:
            raise ValueError("X must contain 'Season' column for time-series splitting")
        
        # 获取唯一的季数并排序
        seasons = sorted(X['Season'].unique())
        n_seasons = len(seasons)
        
        if n_seasons < self.n_splits + 1:
            logger.warning(f"Not enough seasons ({n_seasons}) for {self.n_splits} splits")
            logger.warning(f"Reducing n_splits to {n_seasons - 1}")
            self.n_splits = n_seasons - 1
        
        # 计算每个验证集的季数
        val_size = max(1, n_seasons // (self.n_splits + 1))
        
        logger.info(f"Splitting {n_seasons} seasons into {self.n_splits} folds")
        logger.info(f"Validation size: {val_size} seasons per fold")
        
        for fold in range(self.n_splits):
            if self.strategy == 'expanding':
                # 扩展窗口：训练集逐渐增大
                train_end_idx = n_seasons - (self.n_splits - fold) * val_size
                train_seasons = seasons[:train_end_idx]
                val_seasons = seasons[train_end_idx:train_end_idx + val_size]
            else:
                # 滑动窗口：训练集大小固定
                val_start_idx = (fold + 1) * val_size
                val_end_idx = val_start_idx + val_size
                train_start_idx = max(0, val_start_idx - n_seasons // 2)
                train_seasons = seasons[train_start_idx:val_start_idx]
                val_seasons = seasons[val_start_idx:val_end_idx]
            
            if len(val_seasons) == 0:
                continue
            
            # 获取索引
            train_idx = X[X['Season'].isin(train_seasons)].index.to_numpy()
            val_idx = X[X['Season'].isin(val_seasons)].index.to_numpy()
            
            # 验证时间顺序
            if not self._validate_temporal_order(X, train_idx, val_idx):
                raise ValueError(f"Temporal ordering violated in fold {fold + 1}")
            
            logger.debug(f"Fold {fold + 1}: Train seasons {train_seasons[0]}-{train_seasons[-1]}, "
                        f"Val seasons {val_seasons[0]}-{val_seasons[-1]}")
            logger.debug(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            yield train_idx, val_idx
    
    def _validate_temporal_order(self, X: pd.DataFrame, 
                                 train_idx: np.ndarray, 
                                 val_idx: np.ndarray) -> bool:
        """
        验证训练数据在验证数据之前
        
        Args:
            X: 特征数据框
            train_idx: 训练集索引
            val_idx: 验证集索引
            
        Returns:
            True if valid, False otherwise
        """
        if len(train_idx) == 0 or len(val_idx) == 0:
            return False
        
        # 获取训练集和验证集的季数
        train_seasons = X.loc[train_idx, 'Season'].values
        val_seasons = X.loc[val_idx, 'Season'].values
        
        # 检查最大训练季 < 最小验证季
        max_train_season = train_seasons.max()
        min_val_season = val_seasons.min()
        
        is_valid = max_train_season <= min_val_season
        
        if not is_valid:
            logger.error(f"Temporal ordering violated: max_train_season={max_train_season}, "
                        f"min_val_season={min_val_season}")
        
        return is_valid
    
    def cross_val_score(self, model, X: pd.DataFrame, y: pd.Series, 
                       scoring='r2') -> np.ndarray:
        """
        使用时间序列交叉验证计算分数
        
        Args:
            model: 模型对象
            X: 特征数据框
            y: 目标变量
            scoring: 评分方法
            
        Returns:
            每折的分数数组
        """
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.split(X, y)):
            # 分割数据
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_val)
            
            # 计算分数
            if scoring == 'r2':
                score = r2_score(y_val, y_pred)
            elif scoring == 'mae':
                score = -mean_absolute_error(y_val, y_pred)  # 负数使其越大越好
            elif scoring == 'rmse':
                score = -np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")
            
            scores.append(score)
            logger.debug(f"Fold {fold + 1} {scoring}: {score:.4f}")
        
        scores = np.array(scores)
        logger.info(f"Cross-validation {scoring}: mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        return scores
    
    def get_metrics_summary(self, scores: np.ndarray) -> dict:
        """
        获取指标汇总
        
        Args:
            scores: 分数数组
            
        Returns:
            包含均值和标准差的字典
        """
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'scores': scores.tolist()
        }
