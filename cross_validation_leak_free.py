"""
无泄露模型的交叉验证
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("5-Fold Cross-Validation - Leak-Free Model")
print("=" * 80)

# Load data
df = pd.read_csv('submission/results/Final_Optimized_Dataset.csv')

# Leak-free features
feature_cols = [
    'Week', 'Age', 'Season', 
    'Week_Type', 'Is_Final', 'Week_Progress',
    'Partner_Hist_Score', 'Survival_Weeks', 'Survival_Momentum',
    'judge_lag1', 'judge_lag2', 'judge_hist_mean', 'judge_improvement',
    'fan_lag1', 'fan_lag2', 'fan_hist_mean', 'fan_improvement'
]

# 5-fold CV on last 5 seasons
test_seasons = [30, 31, 32, 33, 34]
results_judge = []
results_fan = []
results_elim = []

print(f"\nTest seasons: {test_seasons}")
print(f"Features: {len(feature_cols)}")

for test_season in test_seasons:
    print(f"\n[Fold {test_season-29}/5] Testing on Season {test_season}...")
    
    # Split
    train_df = df[df['Season'] < test_season].copy()
    test_df = df[df['Season'] == test_season].copy()
    
    X_train = train_df[feature_cols].fillna(0)
    y_train_judge = train_df['Judge_Avg_Score']
    y_train_fan = train_df['Estimated_Fan_Vote']
    
    X_test = test_df[feature_cols].fillna(0)
    y_test_judge = test_df['Judge_Avg_Score']
    y_test_fan = test_df['Estimated_Fan_Vote']
    
    # Train Judge
    rf_judge = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb_judge = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    ridge_judge = Ridge(alpha=1.0)
    
    rf_judge.fit(X_train, y_train_judge)
    gb_judge.fit(X_train, y_train_judge)
    ridge_judge.fit(X_train, y_train_judge)
    
    pred_judge = 0.4 * rf_judge.predict(X_test) + 0.4 * gb_judge.predict(X_test) + 0.2 * ridge_judge.predict(X_test)
    
    r2_judge = r2_score(y_test_judge, pred_judge)
    mae_judge = mean_absolute_error(y_test_judge, pred_judge)
    rmse_judge = np.sqrt(mean_squared_error(y_test_judge, pred_judge))
    
    results_judge.append({'season': test_season, 'r2': r2_judge, 'mae': mae_judge, 'rmse': rmse_judge})
    print(f"   Judge R²: {r2_judge:.4f} ({r2_judge*100:.2f}%)")
    
    # Train Fan
    rf_fan = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb_fan = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    ridge_fan = Ridge(alpha=1.0)
    
    rf_fan.fit(X_train, y_train_fan)
    gb_fan.fit(X_train, y_train_fan)
    ridge_fan.fit(X_train, y_train_fan)
    
    pred_fan = 0.4 * rf_fan.predict(X_test) + 0.4 * gb_fan.predict(X_test) + 0.2 * ridge_fan.predict(X_test)
    
    r2_fan = r2_score(y_test_fan, pred_fan)
    mae_fan = mean_absolute_error(y_test_fan, pred_fan)
    rmse_fan = np.sqrt(mean_squared_error(y_test_fan, pred_fan))
    
    results_fan.append({'season': test_season, 'r2': r2_fan, 'mae': mae_fan, 'rmse': rmse_fan})
    print(f"   Fan R²: {r2_fan:.4f} ({r2_fan*100:.2f}%)")
    
    # Elimination accuracy
    test_df_copy = test_df.copy()
    test_df_copy['Pred_Combined'] = pred_judge + pred_fan
    if 'Combined_Score' not in test_df_copy.columns:
        test_df_copy['Combined_Score'] = test_df_copy['Judge_Avg_Score'] + test_df_copy['Estimated_Fan_Vote']
    
    correct = 0
    total = 0
    for (season, week), group in test_df_copy.groupby(['Season', 'Week']):
        if len(group) >= 2:
            actual_idx = group['Combined_Score'].idxmin()
            pred_idx = group['Pred_Combined'].idxmin()
            if actual_idx == pred_idx:
                correct += 1
            total += 1
    
    elim_acc = correct / total if total > 0 else 0
    results_elim.append({'season': test_season, 'accuracy': elim_acc})
    print(f"   Elimination Accuracy: {elim_acc:.4f} ({elim_acc*100:.2f}%)")

# Summary
print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)

df_judge = pd.DataFrame(results_judge)
df_fan = pd.DataFrame(results_fan)
df_elim = pd.DataFrame(results_elim)

print(f"\nJudge R²:")
print(f"  Mean: {df_judge['r2'].mean():.4f} ({df_judge['r2'].mean()*100:.2f}%)")
print(f"  Std: {df_judge['r2'].std():.4f} ({df_judge['r2'].std()*100:.2f}%)")
print(f"  Range: [{df_judge['r2'].min():.4f}, {df_judge['r2'].max():.4f}]")

print(f"\nFan R²:")
print(f"  Mean: {df_fan['r2'].mean():.4f} ({df_fan['r2'].mean()*100:.2f}%)")
print(f"  Std: {df_fan['r2'].std():.4f} ({df_fan['r2'].std()*100:.2f}%)")
print(f"  Range: [{df_fan['r2'].min():.4f}, {df_fan['r2'].max():.4f}]")

print(f"\nElimination Accuracy:")
print(f"  Mean: {df_elim['accuracy'].mean():.4f} ({df_elim['accuracy'].mean()*100:.2f}%)")
print(f"  Std: {df_elim['accuracy'].std():.4f} ({df_elim['accuracy'].std()*100:.2f}%)")

print("\n" + "=" * 80)
print("✓ Cross-validation complete!")
print("=" * 80)
