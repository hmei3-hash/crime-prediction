# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# %%
df = pd.read_csv(r'C:\CSE163\eda\Crimes_-_2001_to_Present.csv', nrows=20000)

# %%
#cleaning for random forest--->focused on block, hour and aarested flage
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Hour'] = df['Date'].dt.hour
df_clean = df.dropna(subset=['Block', 'Hour', 'Arrest'])
df_clean

# %%

crime_counts = df_clean.groupby(['Block', 'Hour']).size().reset_index(name='crime_count')
crime_counts

# %%
block_total_crimes = df_clean.groupby('Block').size().reset_index(name='block_total_crimes')

block_total_crimes

# %%

crime_counts = crime_counts.merge(block_total_crimes, on='Block')
crime_counts

# %%
arrest_rates = df_clean.groupby(['Block', 'Hour'])['Arrest'].apply(
    lambda x: (x == 'Y').sum() / len(x) if len(x) > 0 else 0
).reset_index(name='arrest_rate')
crime_counts = crime_counts.merge(arrest_rates, on=['Block', 'Hour'])
print(f"✓ 特征工程完成，共 {len(crime_counts)} 行数据")
print(f"\n特征统计:")
print(f"  Block 种类: {crime_counts['Block'].nunique()}")
print(f"  Hour 范围: {crime_counts['Hour'].min()}-{crime_counts['Hour'].max()}")
print(f"  逮捕率范围: {crime_counts['arrest_rate'].min():.2f}-{crime_counts['arrest_rate'].max():.2f}")


# %%
crime_counts = crime_counts.merge(block_total_crimes, on='Block', how='left')
crime_counts = crime_counts.merge(arrest_rates, on=['Block', 'Hour'], how='left') 
crime_counts

# %%
le_block = LabelEncoder()
crime_counts['Block_encoded'] = le_block.fit_transform(crime_counts['Block'])
print(f"✓ Block 编码: {crime_counts['Block'].nunique()} 种 → 数字")
crime_counts


# %%

X = crime_counts[['Block_encoded', 'Hour', 'arrest_rate_x', 'block_total_crimes_x']]
y = crime_counts['crime_count']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)



# %%

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 计算指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n📊 训练集性能:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  R²:   {train_r2:.4f}")

print(f"\n📊 测试集性能:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")

if test_r2 > 0.7:
    print("\n✅ 模型性能良好 (R² > 0.7)")
elif test_r2 > 0.5:
    print("\n⚠️  模型性能中等 (R² > 0.5)")
else:
    print("\n❌ 模型性能不理想 (R² < 0.5)")


# %%




# ===== 步骤6: 特征重要性分析 (RQ2的答案) =====
print("\n步骤6️⃣: 特征重要性分析")
print("-" * 70)

feature_importance = pd.DataFrame({
    'feature': ['Block', 'Hour', 'Arrest Rate', 'Block Total Crimes'],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n影响犯罪数量的关键因素 (RQ2答案):")
for idx, row in feature_importance.iterrows():
    pct = row['importance'] * 100
    bar = '█' * int(pct / 2)
    print(f"  {row['feature']:20s}: {pct:5.1f}% {bar}")

# ===== 步骤7: 可视化 =====
print("\n步骤7️⃣: 可视化结果")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Random Forest Regression: Crime Count Prediction', fontsize=14, fontweight='bold')

# 图1: 特征重要性
ax1 = axes[0, 0]
ax1.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance (RQ2: Key Factors)')
for i, v in enumerate(feature_importance['importance']):
    ax1.text(v, i, f' {v:.4f}', va='center')

# 图2: 实际 vs 预测
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_test, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Crime Count')
ax2.set_ylabel('Predicted Crime Count')
ax2.set_title('Actual vs Predicted (Test Set)')
ax2.text(0.05, 0.95, f'R² = {test_r2:.4f}\nRMSE = {test_rmse:.4f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 图3: 残差分析
ax3 = axes[1, 0]
residuals = y_test - y_pred_test
ax3.scatter(y_pred_test, residuals, alpha=0.5)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')

# 图4: 模型性能对比
ax4 = axes[1, 1]
metrics = ['RMSE', 'MAE', 'R²']
train_vals = [train_rmse, train_mae, train_r2]
test_vals = [test_rmse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.35

ax4.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
ax4.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
ax4.set_ylabel('Score')
ax4.set_title('Model Performance Metrics')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rf_crime_prediction.png', dpi=300, bbox_inches='tight')
print("✓ 可视化已保存为 'rf_crime_prediction.png'")
plt.show()

# ===== 步骤8: 样本预测 =====
print("\n步骤8️⃣: 样本预测")
print("-" * 70)

# 预测例子：Block "001", Hour 12, 逮捕率 0.5
sample_block = "001"
sample_hour = 12
sample_arrest_rate = 0.5

if sample_block in le_block.classes_:
    sample_X = pd.DataFrame({
        'Block_encoded': [le_block.transform([sample_block])[0]],
        'Hour': [sample_hour],
        'arrest_rate': [sample_arrest_rate],
        'block_total_crimes': [block_total_crimes[block_total_crimes['Block']==sample_block]['block_total_crimes'].values[0]]
    })
    
    prediction = rf_model.predict(sample_X)[0]
    print(f"\n示例预测:")
    print(f"  Block: {sample_block}")
    print(f"  Hour: {sample_hour}:00")
    print(f"  Arrest Rate: {sample_arrest_rate:.1%}")
    print(f"  预测犯罪数: {prediction:.1f} 起")

print("\n" + "="*70)
print("✅ 所有步骤完成!")
print("="*70)

# ===== 总结 =====
print("\n📋 总结")
print("-" * 70)
print(f"""
模型: 随机森林回归 (Random Forest Regressor)
数据: {len(df)} → {len(crime_counts)} (Block×Hour 组合)
特征: Block, Hour, Arrest Rate, Block Total Crimes
目标: 每个Block每小时的犯罪数

性能:
  • 测试集 R² = {test_r2:.4f} (模型解释{test_r2*100:.1f}%的方差)
  • 测试集 RMSE = {test_rmse:.4f} (平均预测误差)

关键发现 (RQ2 - 影响因素):
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['importance']*100:.1f}%
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['importance']*100:.1f}%
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['importance']*100:.1f}%
  4. {feature_importance.iloc[3]['feature']}: {feature_importance.iloc[3]['importance']*100:.1f}%
""")

# %%



