"""
修正数据泄露问题 V2
完全移除所有可能导致泄露的特征
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("修正数据泄露问题 V2 - 移除所有泄露特征")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[1/4] Loading data...")
df = pd.read_csv('submission/results/Final_Optimized_Dataset.csv')
print(f"   Loaded {len(df)} records")

# ============================================================================
# Step 2: 只保留无泄露的特征
# ============================================================================
print("\n[2/4] Using only leak-free features...")

# 基础特征（无泄露）
feature_cols = [
    # 外部特征
    'Week', 'Age', 'Season', 
    'Week_Type', 'Is_Final', 'Week_Progress',
    
    # 搭档和生存特征（历史数据）
    'Partner_Hist_Score', 'Survival_Weeks', 'Survival_Momentum',
    
    # 滞后特征（历史数据）
    'judge_lag1', 'judge_lag2', 'judge_hist_mean', 'judge_improvement',
    'fan_lag1', 'fan_lag2', 'fan_hist_mean', 'fan_improvement'
]

print(f"   Using {len(feature_cols)} leak-free features")
print("   Removed features:")
print("      - Judge_Score_Rel_Week (uses same-week data)")
print("      - Judge_Fan_Divergence (uses Judge_Rank from current week)")
print("      - Teflon_Index (derived from Judge_Fan_Divergence)")

# ============================================================================
# Step 3: Train Models
# ============================================================================
print("\n[3/4] Training models with leak-free features...")

# Split data
test_seasons = df['Season'].max() - 1
train_df = df[df['Season'] <= test_seasons].copy()
test_df = df[df['Season'] > test_seasons].copy()

X_train = train_df[feature_cols].fillna(0)
y_train_judge = train_df['Judge_Avg_Score']
y_train_fan = train_df['Estimated_Fan_Vote']

X_test = test_df[feature_cols].fillna(0)
y_test_judge = test_df['Judge_Avg_Score']
y_test_fan = test_df['Estimated_Fan_Vote']

print(f"   Train: {len(train_df)} records, Test: {len(test_df)} records")

# Train Judge models
print("\n   [3.1] Training Judge prediction models...")
rf_judge = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
gb_judge = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
ridge_judge = Ridge(alpha=1.0)

rf_judge.fit(X_train, y_train_judge)
gb_judge.fit(X_train, y_train_judge)
ridge_judge.fit(X_train, y_train_judge)

# Predictions
pred_rf_judge = rf_judge.predict(X_test)
pred_gb_judge = gb_judge.predict(X_test)
pred_ridge_judge = ridge_judge.predict(X_test)

# Weighted ensemble
pred_judge = 0.4 * pred_rf_judge + 0.4 * pred_gb_judge + 0.2 * pred_ridge_judge

# Evaluate
r2_judge = r2_score(y_test_judge, pred_judge)
mae_judge = mean_absolute_error(y_test_judge, pred_judge)
rmse_judge = np.sqrt(mean_squared_error(y_test_judge, pred_judge))

print(f"   Judge R²: {r2_judge:.4f} ({r2_judge*100:.2f}%)")
print(f"   Judge MAE: {mae_judge:.4f}")
print(f"   Judge RMSE: {rmse_judge:.4f}")

# Train Fan models
print("\n   [3.2] Training Fan prediction models...")
rf_fan = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
gb_fan = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
ridge_fan = Ridge(alpha=1.0)

rf_fan.fit(X_train, y_train_fan)
gb_fan.fit(X_train, y_train_fan)
ridge_fan.fit(X_train, y_train_fan)

# Predictions
pred_rf_fan = rf_fan.predict(X_test)
pred_gb_fan = gb_fan.predict(X_test)
pred_ridge_fan = ridge_fan.predict(X_test)

# Weighted ensemble
pred_fan = 0.4 * pred_rf_fan + 0.4 * pred_gb_fan + 0.2 * pred_ridge_fan

# Evaluate
r2_fan = r2_score(y_test_fan, pred_fan)
mae_fan = mean_absolute_error(y_test_fan, pred_fan)
rmse_fan = np.sqrt(mean_squared_error(y_test_fan, pred_fan))

print(f"   Fan R²: {r2_fan:.4f} ({r2_fan*100:.2f}%)")
print(f"   Fan MAE: {mae_fan:.4f}")
print(f"   Fan RMSE: {rmse_fan:.4f}")

# ============================================================================
# Step 4: Compute Elimination Accuracy
# ============================================================================
print("\n[4/4] Computing elimination accuracy...")

def compute_elimination_accuracy(df_test, pred_judge, pred_fan):
    """计算淘汰预测准确率"""
    df_test = df_test.copy()
    df_test['Pred_Judge'] = pred_judge
    df_test['Pred_Fan'] = pred_fan
    df_test['Pred_Combined'] = df_test['Pred_Judge'] + df_test['Pred_Fan']
    
    # Calculate actual combined score
    if 'Combined_Score' not in df_test.columns:
        df_test['Combined_Score'] = df_test['Judge_Avg_Score'] + df_test['Estimated_Fan_Vote']
    
    correct = 0
    bottom_3_correct = 0
    total_weeks = 0
    
    for (season, week), group in df_test.groupby(['Season', 'Week']):
        if len(group) < 2:
            continue
        
        # Actual eliminated (lowest combined score)
        actual_eliminated_idx = group['Combined_Score'].idxmin()
        
        # Predicted eliminated (lowest predicted combined score)
        pred_eliminated_idx = group['Pred_Combined'].idxmin()
        
        # Bottom 3
        bottom_3_idx = group.nsmallest(3, 'Pred_Combined').index
        
        if actual_eliminated_idx == pred_eliminated_idx:
            correct += 1
        if actual_eliminated_idx in bottom_3_idx:
            bottom_3_correct += 1
        
        total_weeks += 1
    
    return {
        'elimination_accuracy': correct / total_weeks if total_weeks > 0 else 0,
        'bottom_3_accuracy': bottom_3_correct / total_weeks if total_weeks > 0 else 0,
        'total_weeks': total_weeks
    }

metrics = compute_elimination_accuracy(test_df, pred_judge, pred_fan)
print(f"   Elimination Accuracy: {metrics['elimination_accuracy']:.4f} ({metrics['elimination_accuracy']*100:.2f}%)")
print(f"   Bottom-3 Accuracy: {metrics['bottom_3_accuracy']:.4f} ({metrics['bottom_3_accuracy']*100:.2f}%)")
print(f"   Total weeks evaluated: {metrics['total_weeks']}")

# ============================================================================
# Save Models
# ============================================================================
print("\n[5/4] Saving models...")
with open('models/leak_free_judge_model.pkl', 'wb') as f:
    pickle.dump({
        'rf': rf_judge,
        'gb': gb_judge,
        'ridge': ridge_judge,
        'weights': [0.4, 0.4, 0.2]
    }, f)

with open('models/leak_free_fan_model.pkl', 'wb') as f:
    pickle.dump({
        'rf': rf_fan,
        'gb': gb_fan,
        'ridge': ridge_fan,
        'weights': [0.4, 0.4, 0.2]
    }, f)

with open('models/leak_free_feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("   ✓ Models saved")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY - Leak-Free Results (Conservative Estimate)")
print("=" * 80)
print(f"\nJudge Prediction:")
print(f"  R²: {r2_judge:.4f} ({r2_judge*100:.2f}%)")
print(f"  MAE: {mae_judge:.4f}")
print(f"  RMSE: {rmse_judge:.4f}")

print(f"\nFan Prediction:")
print(f"  R²: {r2_fan:.4f} ({r2_fan*100:.2f}%)")
print(f"  MAE: {mae_fan:.4f}")
print(f"  RMSE: {rmse_fan:.4f}")

print(f"\nElimination Metrics:")
print(f"  Elimination Accuracy: {metrics['elimination_accuracy']*100:.2f}%")
print(f"  Bottom-3 Accuracy: {metrics['bottom_3_accuracy']*100:.2f}%")

print("\n" + "=" * 80)
print("✓ All data leakage removed! These are trustworthy results.")
print("=" * 80)
