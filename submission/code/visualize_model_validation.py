"""
模型检验可视化
生成交叉验证、残差分析、鲁棒性测试的图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("生成模型检验可视化图表")
print("="*80)

# 加载数据
df_processed = pd.read_csv('Processed_DWTS_Long_Format.csv')
df_fan = pd.read_csv('Q1_Estimated_Fan_Votes.csv')
df_raw = pd.read_csv('2026 MCM Problem C Data.csv')

# 合并数据
df = df_processed.merge(
    df_fan[['Season', 'Week', 'Name', 'Estimated_Fan_Vote']], 
    on=['Season', 'Week', 'Name'],
    how='inner'
)

partner_info = df_raw[['celebrity_name', 'ballroom_partner', 'season']].copy()
partner_info.columns = ['Name', 'Partner', 'Season']
df = df.merge(partner_info, on=['Name', 'Season'], how='left')

# 准备特征
X_numeric = df[['Age', 'Season']].copy()
industry_dummies = pd.get_dummies(df['Industry_Code'], prefix='Industry')
partner_counts = df['Partner'].value_counts()
frequent_partners = partner_counts[partner_counts >= 5].index
df['Partner_Grouped'] = df['Partner'].apply(
    lambda x: x if x in frequent_partners else 'Other'
)
partner_dummies = pd.get_dummies(df['Partner_Grouped'], prefix='Partner')
X = pd.concat([X_numeric, industry_dummies, partner_dummies], axis=1)
y_judge = df['Judge_Avg_Score'].values
y_fan = df['Estimated_Fan_Vote'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 图1：交叉验证结果对比
# ============================================================================
print("\n生成图1：交叉验证结果对比...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 裁判分数模型
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model_judge = BayesianRidge()

r2_scores_judge = []
for train_idx, test_idx in kfold.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_judge[train_idx], y_judge[test_idx]
    model_judge.fit(X_train, y_train)
    r2_scores_judge.append(model_judge.score(X_test, y_test))

ax1.plot(range(1, 11), r2_scores_judge, marker='o', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(y=np.mean(r2_scores_judge), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores_judge):.4f}')
ax1.fill_between(range(1, 11), 
                  np.mean(r2_scores_judge) - np.std(r2_scores_judge),
                  np.mean(r2_scores_judge) + np.std(r2_scores_judge),
                  alpha=0.2, color='steelblue')
ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Judge Score Model - 10-Fold CV', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_ylim([0.15, 0.30])

# 观众投票模型
model_fan = BayesianRidge()
r2_scores_fan = []
for train_idx, test_idx in kfold.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_fan[train_idx], y_fan[test_idx]
    model_fan.fit(X_train, y_train)
    r2_scores_fan.append(model_fan.score(X_test, y_test))

ax2.plot(range(1, 11), r2_scores_fan, marker='o', linewidth=2, markersize=8, color='coral')
ax2.axhline(y=np.mean(r2_scores_fan), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores_fan):.4f}')
ax2.fill_between(range(1, 11), 
                  np.mean(r2_scores_fan) - np.std(r2_scores_fan),
                  np.mean(r2_scores_fan) + np.std(r2_scores_fan),
                  alpha=0.2, color='coral')
ax2.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Fan Vote Model - 10-Fold CV', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 0.15])

plt.tight_layout()
plt.savefig('model_validation_cv.png', dpi=300, bbox_inches='tight')
print("✓ 保存: model_validation_cv.png")
plt.close()

# ============================================================================
# 图2：残差分析
# ============================================================================
print("\n生成图2：残差分析...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 裁判分数模型
model_judge = BayesianRidge()
model_judge.fit(X_scaled, y_judge)
y_pred_judge = model_judge.predict(X_scaled)
residuals_judge = y_judge - y_pred_judge

# 残差vs预测值
ax1.scatter(y_pred_judge, residuals_judge, alpha=0.3, s=10, color='steelblue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax1.set_title('Judge Model: Residuals vs Predicted', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# 残差直方图
ax2.hist(residuals_judge, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Residuals', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Judge Model: Residual Distribution', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 观众投票模型
model_fan = BayesianRidge()
model_fan.fit(X_scaled, y_fan)
y_pred_fan = model_fan.predict(X_scaled)
residuals_fan = y_fan - y_pred_fan

# 残差vs预测值
ax3.scatter(y_pred_fan, residuals_fan, alpha=0.3, s=10, color='coral')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax3.set_title('Fan Model: Residuals vs Predicted', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# 残差直方图
ax4.hist(residuals_fan, bins=50, alpha=0.7, color='coral', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuals', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Fan Model: Residual Distribution', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_validation_residuals.png', dpi=300, bbox_inches='tight')
print("✓ 保存: model_validation_residuals.png")
plt.close()

# ============================================================================
# 图3：鲁棒性测试
# ============================================================================
print("\n生成图3：鲁棒性测试...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

noise_levels = [0, 0.01, 0.05, 0.10, 0.15, 0.20]

# 裁判分数模型
base_model_judge = BayesianRidge()
base_model_judge.fit(X_scaled, y_judge)
base_score_judge = base_model_judge.score(X_scaled, y_judge)

r2_scores_noise_judge = [base_score_judge]
for noise in noise_levels[1:]:
    noise_data = np.random.normal(0, noise, X_scaled.shape)
    X_noisy = X_scaled + noise_data
    noisy_model = BayesianRidge()
    noisy_model.fit(X_noisy, y_judge)
    r2_scores_noise_judge.append(noisy_model.score(X_noisy, y_judge))

ax1.plot([n*100 for n in noise_levels], r2_scores_noise_judge, marker='o', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(y=base_score_judge, color='red', linestyle='--', linewidth=2, label=f'Baseline: {base_score_judge:.4f}')
ax1.fill_between([n*100 for n in noise_levels], 
                  base_score_judge * 0.95, base_score_judge * 1.05,
                  alpha=0.2, color='green', label='±5% Range')
ax1.set_xlabel('Noise Level (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Judge Model: Robustness to Noise', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 观众投票模型
base_model_fan = BayesianRidge()
base_model_fan.fit(X_scaled, y_fan)
base_score_fan = base_model_fan.score(X_scaled, y_fan)

r2_scores_noise_fan = [base_score_fan]
for noise in noise_levels[1:]:
    noise_data = np.random.normal(0, noise, X_scaled.shape)
    X_noisy = X_scaled + noise_data
    noisy_model = BayesianRidge()
    noisy_model.fit(X_noisy, y_fan)
    r2_scores_noise_fan.append(noisy_model.score(X_noisy, y_fan))

ax2.plot([n*100 for n in noise_levels], r2_scores_noise_fan, marker='o', linewidth=2, markersize=8, color='coral')
ax2.axhline(y=base_score_fan, color='red', linestyle='--', linewidth=2, label=f'Baseline: {base_score_fan:.4f}')
ax2.fill_between([n*100 for n in noise_levels], 
                  base_score_fan * 0.95, base_score_fan * 1.05,
                  alpha=0.2, color='green', label='±5% Range')
ax2.set_xlabel('Noise Level (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Fan Model: Robustness to Noise', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_validation_robustness.png', dpi=300, bbox_inches='tight')
print("✓ 保存: model_validation_robustness.png")
plt.close()

print("\n" + "="*80)
print("所有可视化图表生成完成")
print("="*80)
print("\n生成的文件：")
print("  1. model_validation_cv.png - 交叉验证结果")
print("  2. model_validation_residuals.png - 残差分析")
print("  3. model_validation_robustness.png - 鲁棒性测试")
