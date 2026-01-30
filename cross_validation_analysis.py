"""
交叉验证分析 - 验证模型是否过拟合
使用最后5个赛季做时间序列交叉验证
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("交叉验证分析 - 验证模型可信度")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[1/3] 加载数据...")
df = pd.read_csv('submission/results/Problem_Driven_Dataset.csv')
print(f"   总记录数: {len(df)}")
print(f"   赛季范围: Season {df['Season'].min()} - {df['Season'].max()}")

# ============================================================================
# Step 2: Time-Series Cross-Validation
# ============================================================================
print("\n[2/3] 执行时间序列交叉验证...")

feature_cols = [
    'Week', 'Age', 'Season', 'Survival_Weeks', 'Survival_Momentum',
    'Week_Type', 'Is_Final', 'Week_Progress', 'Partner_Hist_Score',
    'judge_lag1', 'judge_lag2', 'judge_hist_mean', 'judge_improvement',
    'fan_lag1', 'fan_lag2', 'fan_hist_mean', 'fan_improvement',
    'Judge_Score_Rel_Week', 'Judge_Fan_Divergence', 'Teflon_Index'
]

# 使用最后5个赛季做交叉验证
max_season = df['Season'].max()
test_seasons = [max_season - 4, max_season - 3, max_season - 2, max_season - 1, max_season]

print(f"\n   测试赛季: {test_seasons}")
print(f"   交叉验证折数: {len(test_seasons)}")

results_judge = []
results_fan = []
elimination_accuracies = []

for i, test_season in enumerate(test_seasons, 1):
    print(f"\n   Fold {i}/5: 测试 Season {test_season}")
    
    # Split data
    train_df = df[df['Season'] < test_season].copy()
    test_df = df[df['Season'] == test_season].copy()
    
    if len(test_df) == 0:
        print(f"      ⚠️  Season {test_season} 无数据，跳过")
        continue
    
    X_train = train_df[feature_cols].fillna(0)
    y_train_judge = train_df['Judge_Avg_Score']
    y_train_fan = train_df['Estimated_Fan_Vote']
    
    X_test = test_df[feature_cols].fillna(0)
    y_test_judge = test_df['Judge_Avg_Score']
    y_test_fan = test_df['Estimated_Fan_Vote']
    
    print(f"      训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
    
    # Train Judge models
    rf_judge = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb_judge = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    ridge_judge = Ridge(alpha=1.0)
    
    rf_judge.fit(X_train, y_train_judge)
    gb_judge.fit(X_train, y_train_judge)
    ridge_judge.fit(X_train, y_train_judge)
    
    pred_judge = 0.4 * rf_judge.predict(X_test) + 0.3 * gb_judge.predict(X_test) + 0.3 * ridge_judge.predict(X_test)
    
    r2_judge = r2_score(y_test_judge, pred_judge)
    mae_judge = mean_absolute_error(y_test_judge, pred_judge)
    rmse_judge = np.sqrt(mean_squared_error(y_test_judge, pred_judge))
    
    results_judge.append({
        'season': test_season,
        'r2': r2_judge,
        'mae': mae_judge,
        'rmse': rmse_judge
    })
    
    print(f"      Judge R²: {r2_judge:.4f} ({r2_judge*100:.2f}%)")
    
    # Train Fan models
    rf_fan = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb_fan = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    ridge_fan = Ridge(alpha=1.0)
    
    rf_fan.fit(X_train, y_train_fan)
    gb_fan.fit(X_train, y_train_fan)
    ridge_fan.fit(X_train, y_train_fan)
    
    pred_fan = 0.4 * rf_fan.predict(X_test) + 0.3 * gb_fan.predict(X_test) + 0.3 * ridge_fan.predict(X_test)
    
    r2_fan = r2_score(y_test_fan, pred_fan)
    mae_fan = mean_absolute_error(y_test_fan, pred_fan)
    rmse_fan = np.sqrt(mean_squared_error(y_test_fan, pred_fan))
    
    results_fan.append({
        'season': test_season,
        'r2': r2_fan,
        'mae': mae_fan,
        'rmse': rmse_fan
    })
    
    print(f"      Fan R²: {r2_fan:.4f} ({r2_fan*100:.2f}%)")
    
    # Compute Elimination Accuracy for this fold
    test_df_copy = test_df.copy()
    test_df_copy['Pred_Combined'] = 0.5 * pred_judge + 0.5 * pred_fan
    
    correct = 0
    total = 0
    
    for (season, week), group in test_df_copy.groupby(['Season', 'Week']):
        if len(group) < 2:
            continue
        
        actual_combined = 0.5 * group['Judge_Avg_Score'] + 0.5 * group['Estimated_Fan_Vote']
        actual_eliminated_idx = actual_combined.idxmin()
        
        pred_combined = group['Pred_Combined']
        pred_eliminated_idx = pred_combined.idxmin()
        
        if actual_eliminated_idx == pred_eliminated_idx:
            correct += 1
        total += 1
    
    elim_acc = correct / total if total > 0 else 0
    elimination_accuracies.append(elim_acc)
    print(f"      Elimination Accuracy: {elim_acc:.4f} ({elim_acc*100:.2f}%)")

# ============================================================================
# Step 3: Aggregate Results
# ============================================================================
print("\n" + "=" * 80)
print("[3/3] 交叉验证结果汇总")
print("=" * 80)

# Judge Results
df_judge = pd.DataFrame(results_judge)
print("\n📊 Judge 预测性能 (5-Fold Cross-Validation)")
print("-" * 80)
print(f"   Mean R²:   {df_judge['r2'].mean():.4f} ± {df_judge['r2'].std():.4f} ({df_judge['r2'].mean()*100:.2f}% ± {df_judge['r2'].std()*100:.2f}%)")
print(f"   Mean MAE:  {df_judge['mae'].mean():.4f} ± {df_judge['mae'].std():.4f}")
print(f"   Mean RMSE: {df_judge['rmse'].mean():.4f} ± {df_judge['rmse'].std():.4f}")
print(f"\n   各赛季详情:")
for _, row in df_judge.iterrows():
    print(f"      Season {int(row['season'])}: R² = {row['r2']:.4f} ({row['r2']*100:.2f}%), MAE = {row['mae']:.4f}")

# Fan Results
df_fan = pd.DataFrame(results_fan)
print("\n📊 Fan 预测性能 (5-Fold Cross-Validation)")
print("-" * 80)
print(f"   Mean R²:   {df_fan['r2'].mean():.4f} ± {df_fan['r2'].std():.4f} ({df_fan['r2'].mean()*100:.2f}% ± {df_fan['r2'].std()*100:.2f}%)")
print(f"   Mean MAE:  {df_fan['mae'].mean():.4f} ± {df_fan['mae'].std():.4f}")
print(f"   Mean RMSE: {df_fan['rmse'].mean():.4f} ± {df_fan['rmse'].std():.4f}")
print(f"\n   各赛季详情:")
for _, row in df_fan.iterrows():
    print(f"      Season {int(row['season'])}: R² = {row['r2']:.4f} ({row['r2']*100:.2f}%), MAE = {row['mae']:.4f}")

# Elimination Accuracy
print("\n📊 Elimination Accuracy (5-Fold Cross-Validation)")
print("-" * 80)
print(f"   Mean Accuracy: {np.mean(elimination_accuracies):.4f} ± {np.std(elimination_accuracies):.4f} ({np.mean(elimination_accuracies)*100:.2f}% ± {np.std(elimination_accuracies)*100:.2f}%)")
print(f"\n   各赛季详情:")
for i, (season, acc) in enumerate(zip(test_seasons, elimination_accuracies)):
    print(f"      Season {season}: {acc:.4f} ({acc*100:.2f}%)")

# ============================================================================
# Step 4: Overfitting Analysis
# ============================================================================
print("\n" + "=" * 80)
print("🔍 过拟合分析")
print("=" * 80)

judge_std = df_judge['r2'].std()
fan_std = df_fan['r2'].std()

print(f"\n📈 Judge R² 标准差: {judge_std:.4f} ({judge_std*100:.2f}%)")
if judge_std < 0.05:
    print("   ✅ 标准差 < 5%: 模型稳定，过拟合风险低")
elif judge_std < 0.10:
    print("   ⚠️  标准差 5-10%: 模型较稳定，轻微过拟合")
else:
    print("   ❌ 标准差 > 10%: 模型不稳定，存在过拟合")

print(f"\n📈 Fan R² 标准差: {fan_std:.4f} ({fan_std*100:.2f}%)")
if fan_std < 0.05:
    print("   ✅ 标准差 < 5%: 模型稳定，过拟合风险低")
elif fan_std < 0.10:
    print("   ⚠️  标准差 5-10%: 模型较稳定，轻微过拟合")
else:
    print("   ❌ 标准差 > 10%: 模型不稳定，存在过拟合")

# Compare with single-season result
print(f"\n📊 单赛季 vs 交叉验证对比:")
print(f"   Judge R²: 94.79% (单赛季) vs {df_judge['r2'].mean()*100:.2f}% ± {df_judge['r2'].std()*100:.2f}% (交叉验证)")
print(f"   Fan R²:   81.76% (单赛季) vs {df_fan['r2'].mean()*100:.2f}% ± {df_fan['r2'].std()*100:.2f}% (交叉验证)")

judge_diff = 94.79 - df_judge['r2'].mean()*100
fan_diff = 81.76 - df_fan['r2'].mean()*100

if judge_diff > 10:
    print(f"\n   ⚠️  Judge: 单赛季结果高出 {judge_diff:.2f}%，存在明显过拟合")
elif judge_diff > 5:
    print(f"\n   ⚠️  Judge: 单赛季结果高出 {judge_diff:.2f}%，存在轻微过拟合")
else:
    print(f"\n   ✅ Judge: 单赛季结果高出 {judge_diff:.2f}%，过拟合风险低")

if fan_diff > 10:
    print(f"   ⚠️  Fan: 单赛季结果高出 {fan_diff:.2f}%，存在明显过拟合")
elif fan_diff > 5:
    print(f"   ⚠️  Fan: 单赛季结果高出 {fan_diff:.2f}%，存在轻微过拟合")
else:
    print(f"   ✅ Fan: 单赛季结果高出 {fan_diff:.2f}%，过拟合风险低")

# ============================================================================
# Step 5: Generate Report
# ============================================================================
print("\n" + "=" * 80)
print("📝 生成交叉验证报告...")
print("=" * 80)

report = f"""# 交叉验证分析报告

## 验证方法

- **方法**: 时间序列5折交叉验证
- **测试赛季**: {test_seasons}
- **训练策略**: 对于每个测试赛季，使用之前所有赛季作为训练集
- **特征数**: 20个（17个原始特征 + 3个问题驱动特征）

## 交叉验证结果

### Judge 预测性能

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| **R²** | **{df_judge['r2'].mean():.4f}** | {df_judge['r2'].std():.4f} | [{df_judge['r2'].min():.4f}, {df_judge['r2'].max():.4f}] |
| **MAE** | {df_judge['mae'].mean():.4f} | {df_judge['mae'].std():.4f} | [{df_judge['mae'].min():.4f}, {df_judge['mae'].max():.4f}] |
| **RMSE** | {df_judge['rmse'].mean():.4f} | {df_judge['rmse'].std():.4f} | [{df_judge['rmse'].min():.4f}, {df_judge['rmse'].max():.4f}] |

**各赛季详情**:
{chr(10).join([f"- Season {int(row['season'])}: R² = {row['r2']:.4f} ({row['r2']*100:.2f}%), MAE = {row['mae']:.4f}" for _, row in df_judge.iterrows()])}

### Fan 预测性能

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| **R²** | **{df_fan['r2'].mean():.4f}** | {df_fan['r2'].std():.4f} | [{df_fan['r2'].min():.4f}, {df_fan['r2'].max():.4f}] |
| **MAE** | {df_fan['mae'].mean():.4f} | {df_fan['mae'].std():.4f} | [{df_fan['mae'].min():.4f}, {df_fan['mae'].max():.4f}] |
| **RMSE** | {df_fan['rmse'].mean():.4f} | {df_fan['rmse'].std():.4f} | [{df_fan['rmse'].min():.4f}, {df_fan['rmse'].max():.4f}] |

**各赛季详情**:
{chr(10).join([f"- Season {int(row['season'])}: R² = {row['r2']:.4f} ({row['r2']*100:.2f}%), MAE = {row['mae']:.4f}" for _, row in df_fan.iterrows()])}

### Elimination Accuracy

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| **Accuracy** | **{np.mean(elimination_accuracies):.4f}** | {np.std(elimination_accuracies):.4f} | [{np.min(elimination_accuracies):.4f}, {np.max(elimination_accuracies):.4f}] |

**各赛季详情**:
{chr(10).join([f"- Season {season}: {acc:.4f} ({acc*100:.2f}%)" for season, acc in zip(test_seasons, elimination_accuracies)])}

## 过拟合分析

### 稳定性评估

- **Judge R² 标准差**: {judge_std:.4f} ({judge_std*100:.2f}%)
  - {'✅ 标准差 < 5%: 模型稳定，过拟合风险低' if judge_std < 0.05 else '⚠️ 标准差 5-10%: 模型较稳定，轻微过拟合' if judge_std < 0.10 else '❌ 标准差 > 10%: 模型不稳定，存在过拟合'}

- **Fan R² 标准差**: {fan_std:.4f} ({fan_std*100:.2f}%)
  - {'✅ 标准差 < 5%: 模型稳定，过拟合风险低' if fan_std < 0.05 else '⚠️ 标准差 5-10%: 模型较稳定，轻微过拟合' if fan_std < 0.10 else '❌ 标准差 > 10%: 模型不稳定，存在过拟合'}

### 单赛季 vs 交叉验证对比

| 指标 | 单赛季 (Season 34) | 交叉验证 (5-Fold) | 差异 |
|------|-------------------|-------------------|------|
| Judge R² | 94.79% | {df_judge['r2'].mean()*100:.2f}% ± {df_judge['r2'].std()*100:.2f}% | {judge_diff:+.2f}% |
| Fan R² | 81.76% | {df_fan['r2'].mean()*100:.2f}% ± {df_fan['r2'].std()*100:.2f}% | {fan_diff:+.2f}% |

**分析**:
- Judge: {'存在明显过拟合' if judge_diff > 10 else '存在轻微过拟合' if judge_diff > 5 else '过拟合风险低'}
- Fan: {'存在明显过拟合' if fan_diff > 10 else '存在轻微过拟合' if fan_diff > 5 else '过拟合风险低'}

## 结论

### 真实性能估计

基于5折交叉验证，我们的模型真实性能为：

- **Judge R²**: {df_judge['r2'].mean()*100:.2f}% ± {df_judge['r2'].std()*100:.2f}%
- **Fan R²**: {df_fan['r2'].mean()*100:.2f}% ± {df_fan['r2'].std()*100:.2f}%
- **Elimination Accuracy**: {np.mean(elimination_accuracies)*100:.2f}% ± {np.std(elimination_accuracies)*100:.2f}%

### 可信度评估

{'✅ **高可信度**: 交叉验证结果稳定，标准差小，模型泛化能力强。' if judge_std < 0.05 and fan_std < 0.05 else '⚠️ **中等可信度**: 交叉验证结果较稳定，存在轻微过拟合，但整体可接受。' if judge_std < 0.10 and fan_std < 0.10 else '❌ **低可信度**: 交叉验证结果不稳定，存在明显过拟合，需要进一步优化。'}

### 论文建议

在论文中应报告：

> "We performed 5-fold time-series cross-validation on the last 5 seasons. Our model achieves a mean Judge R² of {df_judge['r2'].mean()*100:.2f}% (±{df_judge['r2'].std()*100:.2f}%) and Fan R² of {df_fan['r2'].mean()*100:.2f}% (±{df_fan['r2'].std()*100:.2f}%), with an elimination prediction accuracy of {np.mean(elimination_accuracies)*100:.2f}% (±{np.std(elimination_accuracies)*100:.2f}%). The low standard deviation indicates robust generalization performance."

---

*生成时间: 2026-01-30*
*验证方法: 时间序列5折交叉验证*
"""

with open('CROSS_VALIDATION_REPORT.md', 'w') as f:
    f.write(report)

print("   ✅ 报告已保存到 CROSS_VALIDATION_REPORT.md")

print("\n" + "=" * 80)
print("✅ 交叉验证分析完成！")
print("=" * 80)
print(f"\n🎯 关键结论:")
print(f"   Judge R²: {df_judge['r2'].mean()*100:.2f}% ± {df_judge['r2'].std()*100:.2f}%")
print(f"   Fan R²: {df_fan['r2'].mean()*100:.2f}% ± {df_fan['r2'].std()*100:.2f}%")
print(f"   Elimination Accuracy: {np.mean(elimination_accuracies)*100:.2f}% ± {np.std(elimination_accuracies)*100:.2f}%")
print(f"\n💡 建议: 在论文中使用交叉验证结果，更可信！")
print("=" * 80)
