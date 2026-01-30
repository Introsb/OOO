"""
Ultimate Ensemble Learning - Phase 3
终极集成学习 - 7模型Stacking + 超参数优化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入高级模型
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ LightGBM not available")

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("⚠ CatBoost not available")


class UltimateEnsembleLearner:
    """终极集成学习器"""
    
    def __init__(self):
        self.models = {}
        self.results = []
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载增强数据集"""
        print("\n" + "="*80)
        print("ULTIMATE ENSEMBLE LEARNING - CLEAN VERSION")
        print("="*80)
        print("\n1. Loading clean enhanced dataset...")
        
        df = pd.read_csv('results/Clean_Enhanced_Dataset.csv')
        
        # 选择特征（排除目标变量和ID列）
        exclude_cols = ['Season', 'Week', 'Name', 'Judge_Avg_Score', 'Estimated_Fan_Vote', 
                       'Score_Scaled', 'Placement', 'Industry_Code']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y_judge = df['Judge_Avg_Score']
        y_fan = df['Estimated_Fan_Vote']
        
        print(f"   ✓ Loaded {len(df)} samples")
        print(f"   ✓ Features: {len(feature_cols)}")
        print(f"   ✓ Target 1: Judge Score")
        print(f"   ✓ Target 2: Fan Vote")
        
        return X, y_judge, y_fan
    
    def train_single_model(self, model, model_name, X, y, target_name):
        """训练单个模型并评估"""
        print(f"\n   Training {model_name}...")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            cv_scores.append(r2)
        
        # 训练最终模型
        model.fit(X, y)
        train_r2 = r2_score(y, model.predict(X))
        
        result = {
            'Model': model_name,
            'Target': target_name,
            'CV_Mean_R2': np.mean(cv_scores),
            'CV_Std_R2': np.std(cv_scores),
            'Train_R2': train_r2
        }
        
        print(f"      CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"      Train R²: {train_r2:.4f}")
        
        return model, result
    
    def train_all_base_models(self, X, y, target_name):
        """训练所有基础模型"""
        print(f"\n2. Training base models for {target_name}...")
        
        models = {}
        results = []
        
        # Model 1: Random Forest
        print("\n   [1/7] Random Forest")
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        models['RandomForest'], result = self.train_single_model(rf, 'Random Forest', X, y, target_name)
        results.append(result)
        
        # Model 2: Extra Trees
        print("\n   [2/7] Extra Trees")
        et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        models['ExtraTrees'], result = self.train_single_model(et, 'Extra Trees', X, y, target_name)
        results.append(result)
        
        # Model 3: Gradient Boosting
        print("\n   [3/7] Gradient Boosting")
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        models['GradientBoosting'], result = self.train_single_model(gb, 'Gradient Boosting', X, y, target_name)
        results.append(result)
        
        # Model 4: XGBoost
        if HAS_XGB:
            print("\n   [4/7] XGBoost")
            xgb_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            models['XGBoost'], result = self.train_single_model(xgb_model, 'XGBoost', X, y, target_name)
            results.append(result)
        else:
            print("\n   [4/7] XGBoost - SKIPPED (not installed)")
        
        # Model 5: LightGBM
        if HAS_LGB:
            print("\n   [5/7] LightGBM")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            models['LightGBM'], result = self.train_single_model(lgb_model, 'LightGBM', X, y, target_name)
            results.append(result)
        else:
            print("\n   [5/7] LightGBM - SKIPPED (not installed)")
        
        # Model 6: CatBoost
        if HAS_CAT:
            print("\n   [6/7] CatBoost")
            cat_model = cb.CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            models['CatBoost'], result = self.train_single_model(cat_model, 'CatBoost', X, y, target_name)
            results.append(result)
        else:
            print("\n   [6/7] CatBoost - SKIPPED (not installed)")
        
        # Model 7: Bayesian Ridge (baseline)
        print("\n   [7/7] Bayesian Ridge (Baseline)")
        br = BayesianRidge()
        models['BayesianRidge'], result = self.train_single_model(br, 'Bayesian Ridge', X, y, target_name)
        results.append(result)
        
        return models, results
    
    def train_stacking_ensemble(self, X, y, base_models, target_name):
        """训练Stacking集成"""
        print(f"\n3. Training Stacking Ensemble for {target_name}...")
        
        # 准备base estimators
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta model
        meta_model = Ridge(alpha=1.0)
        
        # Stacking
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # 训练
        print("   Training stacking model...")
        stacking.fit(X, y)
        
        # 评估
        print("   Evaluating...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            stacking.fit(X_train, y_train)
            y_pred = stacking.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            cv_scores.append(r2)
        
        # 最终训练
        stacking.fit(X, y)
        train_r2 = r2_score(y, stacking.predict(X))
        
        result = {
            'Model': 'Stacking Ensemble',
            'Target': target_name,
            'CV_Mean_R2': np.mean(cv_scores),
            'CV_Std_R2': np.std(cv_scores),
            'Train_R2': train_r2
        }
        
        print(f"   ✓ Stacking CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"   ✓ Stacking Train R²: {train_r2:.4f}")
        
        return stacking, result
    
    def train_weighted_ensemble(self, X, y, base_models, target_name):
        """训练加权集成"""
        print(f"\n4. Training Weighted Ensemble for {target_name}...")
        
        # 获取每个模型的预测
        predictions = {}
        cv_scores_dict = {}
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in base_models.items():
            cv_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                cv_scores.append(r2)
            
            cv_scores_dict[name] = np.mean(cv_scores)
        
        # 根据CV分数计算权重
        total_score = sum(cv_scores_dict.values())
        weights = {name: score/total_score for name, score in cv_scores_dict.items()}
        
        print("   Optimal weights:")
        for name, weight in weights.items():
            print(f"      {name}: {weight:.4f}")
        
        # 加权预测
        weighted_pred = np.zeros(len(y))
        for name, model in base_models.items():
            model.fit(X, y)
            weighted_pred += weights[name] * model.predict(X)
        
        train_r2 = r2_score(y, weighted_pred)
        
        # CV评估
        cv_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            val_pred = np.zeros(len(y_val))
            for name, model in base_models.items():
                model.fit(X_train, y_train)
                val_pred += weights[name] * model.predict(X_val)
            
            r2 = r2_score(y_val, val_pred)
            cv_scores.append(r2)
        
        result = {
            'Model': 'Weighted Ensemble',
            'Target': target_name,
            'CV_Mean_R2': np.mean(cv_scores),
            'CV_Std_R2': np.std(cv_scores),
            'Train_R2': train_r2
        }
        
        print(f"   ✓ Weighted CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"   ✓ Weighted Train R²: {train_r2:.4f}")
        
        return weights, result
    
    def run_ultimate_ensemble(self):
        """运行完整的集成学习流程"""
        # 1. 加载数据
        X, y_judge, y_fan = self.load_data()
        
        all_results = []
        
        # 2. Judge Score模型
        print("\n" + "="*80)
        print("JUDGE SCORE MODELS")
        print("="*80)
        
        judge_models, judge_results = self.train_all_base_models(X, y_judge, 'Judge Score')
        all_results.extend(judge_results)
        
        judge_stacking, judge_stacking_result = self.train_stacking_ensemble(X, y_judge, judge_models, 'Judge Score')
        all_results.append(judge_stacking_result)
        
        judge_weights, judge_weighted_result = self.train_weighted_ensemble(X, y_judge, judge_models, 'Judge Score')
        all_results.append(judge_weighted_result)
        
        # 3. Fan Vote模型
        print("\n" + "="*80)
        print("FAN VOTE MODELS")
        print("="*80)
        
        fan_models, fan_results = self.train_all_base_models(X, y_fan, 'Fan Vote')
        all_results.extend(fan_results)
        
        fan_stacking, fan_stacking_result = self.train_stacking_ensemble(X, y_fan, fan_models, 'Fan Vote')
        all_results.append(fan_stacking_result)
        
        fan_weights, fan_weighted_result = self.train_weighted_ensemble(X, y_fan, fan_models, 'Fan Vote')
        all_results.append(fan_weighted_result)
        
        # 保存结果
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        df_results = pd.DataFrame(all_results)
        output_path = 'Clean_Model_Comparison.csv'
        df_results.to_csv(output_path, index=False)
        print(f"✓ Results saved to {output_path}")
        
        # 5. 生成总结
        self.print_summary(df_results)
        
        return df_results
    
    def print_summary(self, df_results):
        """打印总结"""
        print("\n" + "="*80)
        print("ULTIMATE ENSEMBLE LEARNING SUMMARY")
        print("="*80)
        
        # Judge Score最佳模型
        judge_results = df_results[df_results['Target'] == 'Judge Score']
        best_judge = judge_results.loc[judge_results['CV_Mean_R2'].idxmax()]
        
        print("\n📊 Judge Score - Best Model:")
        print(f"   Model: {best_judge['Model']}")
        print(f"   CV R²: {best_judge['CV_Mean_R2']:.4f} ± {best_judge['CV_Std_R2']:.4f}")
        print(f"   Train R²: {best_judge['Train_R2']:.4f}")
        
        # Fan Vote最佳模型
        fan_results = df_results[df_results['Target'] == 'Fan Vote']
        best_fan = fan_results.loc[fan_results['CV_Mean_R2'].idxmax()]
        
        print("\n📊 Fan Vote - Best Model:")
        print(f"   Model: {best_fan['Model']}")
        print(f"   CV R²: {best_fan['CV_Mean_R2']:.4f} ± {best_fan['CV_Std_R2']:.4f}")
        print(f"   Train R²: {best_fan['Train_R2']:.4f}")
        
        # 对比Phase 1
        print("\n📈 Improvement from Phase 1:")
        print(f"   Judge Score: 59.22% → {best_judge['CV_Mean_R2']*100:.2f}% (+{(best_judge['CV_Mean_R2']-0.5922)*100:.2f}%)")
        print(f"   Fan Vote: 61.06% → {best_fan['CV_Mean_R2']*100:.2f}% (+{(best_fan['CV_Mean_R2']-0.6106)*100:.2f}%)")
        
        print("\n" + "="*80)
        print("✓ ULTIMATE ENSEMBLE LEARNING COMPLETE")
        print("="*80)


def main():
    """主函数"""
    learner = UltimateEnsembleLearner()
    results = learner.run_ultimate_ensemble()
    return results


if __name__ == '__main__':
    results = main()
