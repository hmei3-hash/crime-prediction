"""
优化版: 只预测Top 30个Block，速度快10倍
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML预测可视化 + 警力部署推荐 (优化版)")
print("="*70)

# ===== 加载和预处理 =====
print("\n📂 加载数据...")
df = pd.read_csv(r'C:\CSE163\eda\Crimes_-_2001_to_Present.csv', nrows=20000)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Hour'] = df['Date'].dt.hour

tier1 = ['ROBBERY', 'THEFT', 'BATTERY', 'ASSAULT']
tier2 = ['BURGLARY', 'MOTOR VEHICLE THEFT', 'CRIM SEXUAL ASSAULT', 
         'CRIMINAL SEXUAL ASSAULT', 'KIDNAPPING', 'STALKING', 'HUMAN TRAFFICKING']
all_tourism = tier1 + tier2

df_tourism = df[df['Primary Type'].isin(all_tourism)].copy()
df_clean = df_tourism.dropna(subset=['Block', 'Hour', 'Arrest', 'Latitude', 'Longitude'])

print(f"✓ 旅游犯罪: {len(df_clean)}")

# ===== 特征工程 =====
print("🔧 特征工程...")
crime_counts = df_clean.groupby(['Block', 'Hour']).size().reset_index(name='crime_count')
block_crime_total = df_clean.groupby('Block').size().reset_index(name='block_total_crimes')
arrest_data = df_clean.groupby(['Block', 'Hour']).agg({
    'Arrest': lambda x: (x == 'Y').sum() / len(x)
}).reset_index()
arrest_data.columns = ['Block', 'Hour', 'arrest_rate']

crime_counts = crime_counts.merge(block_crime_total, on='Block', how='left')
crime_counts = crime_counts.merge(arrest_data, on=['Block', 'Hour'], how='left')

block_coords = df_clean.groupby('Block').agg({
    'Latitude': 'mean',
    'Longitude': 'mean'
}).reset_index()
crime_counts = crime_counts.merge(block_coords, on='Block', how='left')

print(f"✓ 特征数据: {len(crime_counts)}")

# ===== 训练模型 =====
print("🤖 训练模型...")
le_block = LabelEncoder()
crime_counts['Block_encoded'] = le_block.fit_transform(crime_counts['Block'])
X = crime_counts[['Block_encoded', 'Hour', 'arrest_rate', 'block_total_crimes']]
y = crime_counts['crime_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

test_r2 = r2_score(y_test, rf_model.predict(X_test))
print(f"✓ 模型R²: {test_r2:.4f}")

# ===== 优化: 只预测Top 30个Block =====
print("\n🔮 生成预测 (Top 30 Blocks)...")
top_blocks = crime_counts.groupby('Block')['crime_count'].sum().nlargest(30).index.tolist()
print(f"✓ 预测 {len(top_blocks)} 个Block × 24小时")

predictions = []
for i, block in enumerate(top_blocks):
    if (i+1) % 10 == 0:
        print(f"   处理中... {i+1}/{len(top_blocks)}")
    
    block_encoded = le_block.transform([block])[0]
    block_total = crime_counts[crime_counts['Block']==block]['block_total_crimes'].iloc[0]
    arrest_rate = crime_counts[crime_counts['Block']==block]['arrest_rate'].mean()
    lat = crime_counts[crime_counts['Block']==block]['Latitude'].iloc[0]
    lon = crime_counts[crime_counts['Block']==block]['Longitude'].iloc[0]
    
    for hour in range(24):
        sample = pd.DataFrame({
            'Block_encoded': [block_encoded],
            'Hour': [hour],
            'arrest_rate': [arrest_rate],
            'block_total_crimes': [block_total]
        })
        
        pred = rf_model.predict(sample)[0]
        predictions.append({
            'Block': block,
            'Hour': hour,
            'Predicted': max(0, pred),  # 不能是负数
            'Latitude': lat,
            'Longitude': lon
        })

pred_df = pd.DataFrame(predictions)
print(f"✓ 完成 {len(pred_df)} 个预测")

# ===== 警力部署推荐 =====
print("\n👮 警力部署推荐...")
deployment = pred_df.nlargest(15, 'Predicted')

print(f"\n🚨 Top 15 警力部署地点:")
for idx, (_, row) in enumerate(deployment.iterrows(), 1):
    police = int(np.ceil(row['Predicted'] / 2))
    print(f"{idx:2d}. Block {row['Block']:6s} Hour {int(row['Hour']):02d}:00 "
          f"| 预计 {row['Predicted']:5.1f} 起 | 需要 {police} 警察")

# ===== 可视化 =====
print("\n🎨 生成可视化...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
fig.suptitle('ML Crime Prediction & Police Deployment', fontsize=14, fontweight='bold')

# 图1: 按小时
ax1 = fig.add_subplot(gs[0, 0])
hourly = pred_df.groupby('Hour')['Predicted'].sum()
ax1.bar(hourly.index, hourly.values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Total Predicted Crimes')
ax1.set_title('Hourly Crime Distribution')
ax1.set_xticks(range(0, 24, 2))
ax1.grid(axis='y', alpha=0.3)

# 图2: Top 10 Blocks
ax2 = fig.add_subplot(gs[0, 1])
block_pred = pred_df.groupby('Block')['Predicted'].sum().nlargest(10)
ax2.barh(range(len(block_pred)), block_pred.values, color='crimson', alpha=0.7)
ax2.set_yticks(range(len(block_pred)))
ax2.set_yticklabels([f"Block {b}" for b in block_pred.index], fontsize=8)
ax2.set_xlabel('Total Predicted Crimes')
ax2.set_title('Top 10 High-Risk Blocks')
ax2.invert_yaxis()

# 图3: 地理热力
ax3 = fig.add_subplot(gs[1, :])
scatter = ax3.scatter(pred_df['Longitude'], pred_df['Latitude'],
                      c=pred_df['Predicted'], s=80, cmap='RdYlBn_r', 
                      alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Predicted Crimes')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.set_title('Geographic Distribution (All Hours Combined)')
ax3.grid(alpha=0.3)

plt.savefig('prediction_deployment_optimized.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: prediction_deployment_optimized.png")
plt.show()
